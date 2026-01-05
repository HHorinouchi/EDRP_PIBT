#!/usr/bin/env python3
"""
Sweep each PriorityParams field around the best learned vector from sweep logs,
run rollouts, and plot reward sensitivity per environment.

Usage example:
  python policy/train/plot_param_sweep.py --episodes 100 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import concurrent.futures as futures

# Repo root on path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from drp_env.drp_env import DrpEnv  # noqa: E402
from drp_env.EE_map import UNREAL_MAP  # noqa: E402
from policy.my_policy import (  # noqa: E402
    PriorityParams,
    get_priority_params,
    policy as rollout_policy,
    set_priority_params,
)


DEFAULT_SPEED = 5.0
MAP_DIMENSION_CACHE: Dict[str, Tuple[float, float]] = {}


def _get_map_dimensions(map_name: Optional[str]) -> Tuple[float, float]:
    if not map_name:
        return 0.0, 0.0
    cached = MAP_DIMENSION_CACHE.get(map_name)
    if cached is not None:
        return cached
    node_csv_path = ROOT_DIR / "drp_env" / "map" / map_name / "node.csv"
    if not node_csv_path.exists():
        MAP_DIMENSION_CACHE[map_name] = (0.0, 0.0)
        return 0.0, 0.0

    scale = 1.0 if map_name in UNREAL_MAP else 1e5
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    try:
        with node_csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header_skipped = False
            for row in reader:
                if not header_skipped:
                    header_skipped = True
                    continue
                if len(row) < 3:
                    continue
                try:
                    x_val = float(row[1]) * scale
                    y_val = float(row[2]) * scale
                except (ValueError, TypeError):
                    continue
                min_x = min(min_x, x_val)
                max_x = max(max_x, x_val)
                min_y = min(min_y, y_val)
                max_y = max(max_y, y_val)
    except Exception:
        MAP_DIMENSION_CACHE[map_name] = (0.0, 0.0)
        return 0.0, 0.0

    if not (
        math.isfinite(min_x)
        and math.isfinite(max_x)
        and math.isfinite(min_y)
        and math.isfinite(max_y)
    ):
        MAP_DIMENSION_CACHE[map_name] = (0.0, 0.0)
        return 0.0, 0.0

    width = max(max_x - min_x, 0.0)
    height = max(max_y - min_y, 0.0)
    dims = (width, height)
    MAP_DIMENSION_CACHE[map_name] = dims
    return dims


def _compute_time_limit(
    agent_num: Optional[int], map_name: Optional[str], speed: Optional[float]
) -> int:
    speed_val = DEFAULT_SPEED
    if speed is not None:
        try:
            speed_val = float(speed)
        except (TypeError, ValueError):
            speed_val = DEFAULT_SPEED
    speed_val = max(speed_val, 1e-6)

    width, height = _get_map_dimensions(map_name)
    if width <= 0.0 and height <= 0.0:
        return 200
    diagonal = width + height
    if diagonal <= 0.0:
        return 200
    time_limit = diagonal * 6.0 / speed_val
    return max(int(math.ceil(time_limit)), 1)


def _normalized_env_config(cfg: dict) -> dict:
    normalized = dict(cfg)
    agent_num = normalized.get("agent_num")
    map_name = normalized.get("map_name")
    speed = normalized.get("speed")
    normalized["time_limit"] = _compute_time_limit(agent_num, map_name, speed)
    return normalized


def _tasks_cleared(env: DrpEnv) -> bool:
    if not getattr(env, "is_tasklist", False):
        return False
    total_tasks = len(getattr(env, "alltasks_flat", []) or [])
    if total_tasks == 0:
        return False
    completed = getattr(env, "task_completion", 0)
    if completed < total_tasks:
        return False
    pending = len(getattr(env, "current_tasklist", []) or [])
    if pending > 0:
        return False
    assigned = getattr(env, "assigned_tasks", []) or []
    return not any(len(task) > 0 for task in assigned)


BASE_ENV_CONFIG = dict(
    agent_num=10,
    speed=5.0,
    start_ori_array=[],
    goal_array=[],
    visu_delay=0.0,
    state_repre_flag="onehot",
    collision="bounceback",
    task_flag=True,
    reward_list={"goal": 100, "collision": -100, "wait": -10, "move": -1},
    map_name="map_shibuya",
    task_density=1.0,
)


LOG_PARAM_ORDER = [
    "goal_weight",
    "pick_weight",
    "drop_weight",
    "idle_bias",
    "idle_penalty",
    "assign_pick_weight",
    "assign_drop_weight",
    "congestion_weight",
    "step_tolerance",
]

SWEEP_PARAM_ORDER = [
    "goal_weight",
    "pick_weight",
    "drop_weight",
    "idle_bias",
    "idle_penalty",
    "step_tolerance",
    "assign_pick_weight",
    "assign_drop_weight",
    "assign_idle_bias",
    "congestion_weight",
]


def _parse_log_best_params(log_path: Path) -> Tuple[Dict[str, float], float]:
    best_row: Optional[Dict[str, str]] = None
    best_reward = float("-inf")
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                reward_mean = float(row.get("reward_mean", "nan"))
            except Exception:
                continue
            if not math.isfinite(reward_mean):
                continue
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_row = row
    if best_row is None:
        raise ValueError(f"No valid rows in log: {log_path}")

    params: Dict[str, float] = {}
    for idx, name in enumerate(LOG_PARAM_ORDER):
        key = f"v{idx}"
        raw = best_row.get(key)
        if raw is None:
            raise ValueError(f"Missing {key} in log: {log_path}")
        params[name] = float(raw)
    return params, best_reward


def _make_env_config(map_name: str, agent_num: int) -> dict:
    cfg = dict(BASE_ENV_CONFIG)
    cfg["map_name"] = map_name
    cfg["agent_num"] = agent_num
    return _normalized_env_config(cfg)


def _run_episode(env_config: dict, step_limit: Optional[int]) -> float:
    env = DrpEnv(**_normalized_env_config(env_config))
    obs = env.reset()
    done_flags = [False for _ in range(env.agent_num)]
    steps = 0
    max_steps = step_limit if step_limit is not None else getattr(env, "time_limit", None)
    if max_steps is None or max_steps <= 0:
        max_steps = 10**9
    last_info: dict = {
        "task_completion": 0.0,
        "collision": False,
        "goal": False,
        "timeup": False,
    }
    while not all(done_flags) and steps < max_steps:
        policy_output = rollout_policy(obs, env)
        if not isinstance(policy_output, tuple) or len(policy_output) < 2:
            raise ValueError("Policy output must be tuple (actions, task_assign, ...)")
        actions, task_assign = policy_output[0], policy_output[1]
        obs, step_rewards, done_flags, info = env.step({"pass": actions, "task": task_assign})
        steps += 1
        if isinstance(info, dict):
            last_info = info
        if isinstance(info, dict) and info.get("collision", False):
            last_info = dict(last_info)
            last_info["collision"] = True
            break
        if _tasks_cleared(env):
            last_info = dict(last_info)
            last_info["goal"] = True
            break

    tasks_completed = float(last_info.get("task_completion", getattr(env, "task_completion", 0)))
    r_goal = float(getattr(env, "r_goal", 0.0))
    r_move = float(getattr(env, "r_move", 0.0))
    r_coll = float(getattr(env, "r_coll", 0.0))
    speed = float(getattr(env, "speed", 1.0))
    goal_reward = tasks_completed * r_goal / 10
    step_penalty = steps * r_move
    penalty_term = 0.0
    time_cap = getattr(env, "time_limit", None)
    if time_cap is None or time_cap <= 0:
        time_cap = max(steps, 1)
    timeup_triggered = bool(last_info.get("timeup", False)) or (steps >= max_steps)
    if last_info.get("collision", False):
        remaining = max(float(time_cap - steps), 0.0)
        scale = 1.0 + (remaining / max(float(time_cap), 1.0))
        penalty_term = (r_coll * speed) * scale
    elif timeup_triggered:
        penalty_term = (r_coll * speed)

    env.close()
    return float(goal_reward + step_penalty + penalty_term)


def _evaluate_params(
    env_config: dict,
    params_dict: Dict[str, float],
    episodes: int,
    max_steps: Optional[int],
    seed: int,
) -> float:
    params = PriorityParams.from_dict(params_dict)
    set_priority_params(params)
    rewards = []
    for i in range(episodes):
        np.random.seed(seed + i)
        rewards.append(_run_episode(env_config, max_steps))
    return float(np.mean(rewards)) if rewards else 0.0


def _evaluate_worker(payload: dict) -> Tuple[str, float, float]:
    env_config = payload["env_config"]
    params_dict = payload["params_dict"]
    episodes = payload["episodes"]
    max_steps = payload["max_steps"]
    seed = payload["seed"]
    param_name = payload["param_name"]
    delta = payload["delta"]
    mean_reward = _evaluate_params(env_config, params_dict, episodes, max_steps, seed)
    return param_name, delta, mean_reward


def _iter_log_files(log_dir: Path) -> Iterable[Path]:
    for path in sorted(log_dir.glob("train_log_*_agents_*.csv")):
        if path.is_file():
            yield path


def _parse_env_from_log_name(path: Path) -> Tuple[str, int]:
    match = re.match(r"train_log_(.+)_agents_(\\d+)\\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected log filename: {path.name}")
    map_name = match.group(1)
    agent_num = int(match.group(2))
    return map_name, agent_num


def _plot_param_result(
    out_path: Path,
    env_name: str,
    param_name: str,
    delta_values: List[float],
    rewards: List[float],
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=200)
    ax.plot(delta_values, rewards, color="#1f77b4")
    ax.axvline(0.0, color="#999999", linewidth=1)
    ax.set_title(f"{env_name}: {param_name}")
    ax.set_xlabel("delta")
    ax.set_ylabel("reward")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=str(ROOT_DIR / "policy" / "train" / "sweep_results" / "logs"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "policy" / "train" / "sweep_results" / "param_sweep"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log_dir = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_values = np.arange(-2.5, 2.5, 0.1, dtype=np.float32).tolist()
    max_steps = args.max_steps if args.max_steps > 0 else None

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    base_params_template = asdict(get_priority_params())

    for log_path in _iter_log_files(log_dir):
        map_name, agent_num = _parse_env_from_log_name(log_path)
        env_config = _make_env_config(map_name, agent_num)

        params_json_path = (log_dir.parent / f"priority_params_{map_name}_agents_{agent_num}.json")
        if not params_json_path.exists():
            raise FileNotFoundError(f"Priority params json not found: {params_json_path}")
        base_payload = json.loads(params_json_path.read_text())
        base_params = dict(base_params_template)
        base_params.update(base_payload)
        log_params, best_reward = _parse_log_best_params(log_path)

        payloads = []
        for param_name in SWEEP_PARAM_ORDER:
            base_val = float(base_params.get(param_name, 0.0))
            for delta in delta_values:
                params_dict = dict(base_params)
                params_dict[param_name] = base_val + float(delta)
                payloads.append(
                    dict(
                        env_config=env_config,
                        params_dict=params_dict,
                        episodes=args.episodes,
                        max_steps=max_steps,
                        seed=args.seed,
                        param_name=param_name,
                        delta=float(delta),
                    )
                )

        results: Dict[str, List[Tuple[float, float]]] = {
            name: [] for name in SWEEP_PARAM_ORDER
        }
        if args.workers <= 1:
            for payload in payloads:
                param_name, delta, reward = _evaluate_worker(payload)
                results[param_name].append((delta, reward))
        else:
            with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                future_map = [ex.submit(_evaluate_worker, payload) for payload in payloads]
                for fut in futures.as_completed(future_map):
                    param_name, delta, reward = fut.result()
                    results[param_name].append((delta, reward))

        results_by_param: Dict[str, List[float]] = {}
        for param_name in SWEEP_PARAM_ORDER:
            pairs = sorted(results[param_name], key=lambda x: x[0])
            results_by_param[param_name] = [reward for _, reward in pairs]

        env_label = f"{map_name}_agents_{agent_num}"
        for param_name, rewards in results_by_param.items():
            plot_path = out_dir / f"param_sweep_{env_label}_{param_name}.png"
            _plot_param_result(
                plot_path,
                env_label,
                param_name,
                delta_values,
                rewards,
            )

        csv_path = out_dir / f"param_sweep_{env_label}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["param", "delta", "reward_mean"])
            for param_name in SWEEP_PARAM_ORDER:
                for delta, reward in sorted(results[param_name], key=lambda x: x[0]):
                    writer.writerow([param_name, f"{delta:.2f}", f"{reward:.6f}"])

        base_json_path = out_dir / f"param_sweep_{env_label}_base.json"
        base_payload = dict(base_params)
        base_payload["best_reward_mean"] = best_reward
        base_json_path.write_text(json.dumps(base_payload, indent=2))


if __name__ == "__main__":
    main()
