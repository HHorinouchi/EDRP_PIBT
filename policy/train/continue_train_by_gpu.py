# GPU-accelerated ES training for MAPD×PIBT priority parameters
# - Strategy mirrors policy/train/train.py but moves ES math (noise, normalization, updates) to GPU (CUDA/MPS) via PyTorch
# - Environment rollouts remain Python-side (CPU). Optionally parallelized with process workers.

import argparse
import os
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import time
import math
import numpy as np
import concurrent.futures as futures
import json
from dataclasses import asdict, dataclass



# Local imports (repo root on path)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import torch

from drp_env.drp_env import DrpEnv  # noqa: E402
from drp_env.EE_map import UNREAL_MAP  # noqa: E402
from policy.my_policy import (  # noqa: E402
    PriorityParams,
    get_priority_params,
    policy as rollout_policy,
    save_priority_params,
    set_priority_params,
)


def _coerce_cuda_device(token: str) -> Optional[str]:
    """Normalize various CUDA device strings to cuda:<idx>."""
    if not torch.cuda.is_available():
        return None
    token = token.strip().lower()
    if not token:
        return None
    if token == "cuda":
        idx = 0
    elif token.isdigit():
        idx = int(token)
    elif token.startswith("cuda:"):
        try:
            idx = int(token.split(":", 1)[1])
        except (IndexError, ValueError):
            return None
    else:
        return None
    if 0 <= idx < torch.cuda.device_count():
        return f"cuda:{idx}"
    return None


def _parse_device_list(device_arg: Optional[str]) -> List[str]:
    if not device_arg:
        return []
    devices: List[str] = []
    for raw in device_arg.split(","):
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in ("cpu", "mps"):
            devices.append(lowered)
            continue
        coerced = _coerce_cuda_device(lowered)
        if coerced is not None:
            devices.append(coerced)
        else:
            devices.append(token)
    return devices


def _resolve_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        lowered = device_override.strip().lower()
        if lowered == "cpu":
            return torch.device("cpu")
        if lowered == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        coerced = _coerce_cuda_device(lowered)
        if coerced is not None:
            device = torch.device(coerced)
            idx = device.index if device.index is not None else 0
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
        try:
            device = torch.device(device_override)
            if device.type == "cuda" and torch.cuda.is_available():
                idx = device.index if device.index is not None else 0
                torch.cuda.set_device(idx)
                return torch.device(f"cuda:{idx}")
            if device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return device
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
        except Exception:
            idx = 0
        idx = int(idx)
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_time_limit(agent_num: Optional[int]) -> int:
    try:
        value = int(agent_num)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        value = 1
    # Time limit = agent_num × 5 × 50
    return max(value * 5 * 50, 1)


def _normalized_env_config(cfg: dict) -> dict:
    normalized = dict(cfg)
    agent_num = normalized.get("agent_num")
    normalized["time_limit"] = _compute_time_limit(agent_num)
    return normalized

# Base environment configuration (single-map specialization by default)
ENV_CONFIG = dict(
    agent_num=10,
    speed=5.0,
    start_ori_array=[],
    goal_array=[],
    visu_delay=0.0,
    state_repre_flag="onehot",
    time_limit=_compute_time_limit(10),
    collision="bounceback",
    task_flag=True,
    reward_list={"goal": 100, "collision": -100, "wait": -10, "move": -1},
    map_name="map_shibuya",
    task_density=1.0,
)

def make_env() -> DrpEnv:
    # EE_map.MapMake(enable_render=False) is default via constructor
    return DrpEnv(**_normalized_env_config(ENV_CONFIG))

# Domain randomization helpers
MAP_POOL = list(UNREAL_MAP)

# Canonical sweep targets requested for batch training runs.
SWEEP_MAP_LIST = [
    "map_3x3",
    "map_5x4",
    "map_8x5",
    "map_10x6",
    "map_10x8",
    "map_10x10",
    "map_aoba00",
    "map_aoba01",
    "map_kyodai",
    "map_osaka",
    "map_paris",
    "map_shibuya",
    "map_shijo",
]

def _get_map_node_count(map_name: str) -> int:
    try:
        node_csv = ROOT_DIR / "drp_env" / "map" / map_name / "node.csv"
        with node_csv.open("r") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


# def _agent_range_for_map(map_name: str) -> Tuple[int, int]:
#     """Return the inclusive agent-count range [min_agents, max_agents] for a map."""
#     n_nodes = _get_map_node_count(map_name)
#     if n_nodes <= 0:
#         return (0, 0)
#     # Cap at 3/4 of nodes, but never exceed available nodes.
#     max_agents = min(int(math.floor(n_nodes * 0.75)), n_nodes)
#     if max_agents < 3:
#         return (0, 0)
#     return (3, max_agents)


def _agent_counts_for_map(map_name: str) -> List[int]:
    """Return specific agent counts (25%, 50%, 75% of nodes) for sweep runs."""
    n_nodes = _get_map_node_count(map_name)
    if n_nodes <= 0:
        return []
    ratios = (0.25, 0.5, 0.75)
    counts = set()
    for ratio in ratios:
        value = int(math.floor(n_nodes * ratio))
        if value <= 0:
            continue
        value = min(value, n_nodes)
        if value < 3:
            value = 3
        counts.add(value)
    filtered = sorted(val for val in counts if 3 <= val <= n_nodes)
    return filtered


@dataclass
class EpisodeStats:
    reward: float
    steps: int
    task_completion: float
    collision: bool
    goal: bool
    timeup: bool


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

def sample_env_config(rng: np.random.Generator) -> dict:
    map_name = rng.choice(MAP_POOL)
    n_nodes = _get_map_node_count(map_name) or 9
    max_agents = max(2, min(5, max(2, n_nodes - 1)))
    agent_num = int(rng.integers(2, max_agents + 1))
    speed = float(rng.uniform(0.8, 2.0))
    collision = "bounceback" if rng.random() < 0.7 else "terminated"
    task_density = float(rng.uniform(0.3, 1.5))
    time_limit = _compute_time_limit(agent_num)
    return dict(
        agent_num=agent_num,
        speed=speed,
        start_ori_array=[],
        goal_array=[],
        visu_delay=0.0,
        state_repre_flag="onehot",
        time_limit=time_limit,
        collision=collision,
        task_flag=True,
        task_density=task_density,
        reward_list={"goal": 20*time_limit, "collision": -10*agent_num*time_limit, "wait": -10, "move": -1},
        map_name=map_name,
    )

def make_env_from_config(cfg: dict) -> DrpEnv:
    return DrpEnv(**_normalized_env_config(cfg))

# Parameter vectorization: use exactly the 8 parameters the trainer should learn.
# Ordered vector (len=8):
# 0: goal_weight
# 1: pick_weight
# 2: drop_weight
# 3: idle_penalty
# 4: assign_pick_weight
# 5: assign_drop_weight
# 6: assign_idle_bias
# 7: congestion_weight
def params_to_vector(params: PriorityParams) -> np.ndarray:
    return np.array(
        [
            params.goal_weight,
            params.pick_weight,
            params.drop_weight,
            params.idle_penalty,
            params.assign_pick_weight,
            params.assign_drop_weight,
            params.assign_idle_bias,
            params.congestion_weight,
        ],
        dtype=np.float32,
    )


def vector_to_params(vec: np.ndarray) -> PriorityParams:
    # Convert a length-8 (or shorter) vector back into a PriorityParams object.
    # For fields not present in the learned vector (idle_bias, load_balance_weight),
    # preserve the current in-memory params so training does not unexpectedly change them.
    raw = np.asarray(vec, dtype=np.float32)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    safe_len = 8
    z = np.zeros(safe_len, dtype=np.float32)
    z[: min(raw.size, safe_len)] = raw[: min(raw.size, safe_len)]

    goal_w, pick_w, drop_w = np.clip(z[:3], 0.0, 10.0)
    idle_penalty = float(np.clip(z[3], 0.0, 5000.0))
    assign_pick = float(np.clip(z[4], 0.0, 10.0))
    assign_drop = float(np.clip(z[5], 0.0, 10.0))
    assign_idle_bias = float(np.clip(z[6], -5000.0, 5000.0))
    congestion_w = float(np.clip(z[7], 0.0, 10.0))

    # Preserve non-learned fields from current in-memory params (if any)
    current = get_priority_params()
    idle_bias = float(getattr(current, "idle_bias", 0.0))
    load_balance_w = float(getattr(current, "load_balance_weight", 0.0))

    return PriorityParams(
        goal_weight=float(goal_w),
        pick_weight=float(pick_w),
        drop_weight=float(drop_w),
        idle_bias=idle_bias,
        idle_penalty=idle_penalty,
        assign_pick_weight=assign_pick,
        assign_drop_weight=assign_drop,
        assign_idle_bias=assign_idle_bias,
        congestion_weight=congestion_w,
        load_balance_weight=load_balance_w,
    )

# Rollout (CPU env). Optionally end on collision and overwrite reward.
def rollout_once(
    params_vec: np.ndarray,
    max_steps: Optional[int],
    domain_randomize: bool,
    collision_penalty: Optional[float],
    seed: Optional[int] = None,
) -> EpisodeStats:
    if seed is not None:
        # Make env seeding deterministic-ish per worker
        np.random.seed(seed)
    params = vector_to_params(params_vec)
    set_priority_params(params)
    if domain_randomize:
        rng = np.random.default_rng(seed)
        cfg = sample_env_config(rng)
        env = make_env_from_config(cfg)
    else:
        env = make_env()
    obs = env.reset()
    done_flags = [False for _ in range(env.agent_num)]
    steps = 0
    step_limit = max_steps if max_steps is not None else getattr(env, "time_limit", None)
    if step_limit is None or step_limit <= 0:
        step_limit = 10**9
    last_info: dict = {
        "task_completion": 0.0,
        "collision": False,
        "goal": False,
        "timeup": False,
    }
    while not all(done_flags) and steps < step_limit:
        policy_output = rollout_policy(obs, env)
        if isinstance(policy_output, tuple):
            if len(policy_output) >= 3:
                actions, task_assign, count = policy_output[0], policy_output[1], policy_output[2]
            else:
                raise ValueError("Policy returned tuple with fewer than three elements")
        else:
            raise TypeError("Policy output must be a tuple of (actions, task_assign, count, ...)")
        obs, step_rewards, done_flags, info = env.step({"pass": actions, "task": task_assign})
        steps += 1
        if isinstance(info, dict):
            last_info = info
        if isinstance(info, dict) and info.get("collision", False):
            last_info = dict(last_info)
            last_info["collision"] = True
            # print("Episode ended due to collision.")
            break
        if _tasks_cleared(env):
            last_info = dict(last_info)
            last_info["goal"] = True
            # print("Episode ended after completing all tasks.")
            break
        # for i in range(env.agent_num):
        #     print(f" Agent {i} action: {actions[i]}")
        #     print(f" Agent {i} start: {env.current_start[i]}, goal: {env.goal_array[i]}")
        #     print(f" Agent {i} avail actions: {env.get_avail_agent_actions(i, env.n_actions)[1]}")
        #     # タスクをアサインされている場合、そのタスクも表示
        #     if i < len(env.assigned_tasks):
        #         print(f" Agent {i} assigned task: {env.assigned_tasks[i]}")
    env.close()
    tasks_completed = float(last_info.get("task_completion", getattr(env, "task_completion", 0)))
    r_goal = float(getattr(env, "r_goal", 0.0))
    r_move = float(getattr(env, "r_move", 0.0))
    r_coll = float(getattr(env, "r_coll", 0.0))
    speed = float(getattr(env, "speed", 1.0))
    goal_reward = tasks_completed * r_goal / 10
    step_penalty = steps * r_move
    collision_term = (r_coll * speed) if last_info.get("collision", False) else 0.0
    final_reward = goal_reward + step_penalty + collision_term
    if collision_penalty is not None and last_info.get("collision", False):
        final_reward += float(collision_penalty)

    print(
        " Episode stats -> reward: {:.2f}, tasks: {:.0f}, steps: {}, collision: {}".format(
            final_reward,
            tasks_completed,
            steps,
            bool(last_info.get("collision", False)),
        )
    )

    episode_stats = EpisodeStats(
        reward=float(final_reward),
        steps=int(steps),
        task_completion=float(last_info.get("task_completion", 0.0)),
        collision=bool(last_info.get("collision", False)),
        goal=bool(last_info.get("goal", False)),
        timeup=bool(last_info.get("timeup", False)),
    )
    return episode_stats

def rollout_mean(
    params_vec: np.ndarray,
    episodes: int,
    max_steps: Optional[int],
    domain_randomize: bool,
    collision_penalty: Optional[float],
    workers: int = 0,
    base_seed: int = 0,
) -> Tuple[float, Dict[str, float]]:
    # Evaluate mean episode reward with optional multiprocessing
    if episodes <= 1 or workers <= 0:
        ep_stats = [
            rollout_once(params_vec, max_steps, domain_randomize, collision_penalty, seed=base_seed + i)
            for i in range(episodes)
        ]
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as ex:
            tasks = [
                ex.submit(
                    rollout_once,
                    params_vec.copy(),
                    max_steps,
                    domain_randomize,
                    collision_penalty,
                    base_seed + i,
                )
                for i in range(episodes)
            ]
            ep_stats = [t.result() for t in tasks]

    rewards = np.asarray([s.reward for s in ep_stats], dtype=np.float32)
    mean_reward = float(np.mean(rewards)) if rewards.size else 0.0

    def _mean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=np.float32)
        return float(np.mean(arr)) if arr.size else 0.0

    metrics: Dict[str, float] = {
        "avg_steps": _mean([s.steps for s in ep_stats]),
        "avg_task_completion": _mean([s.task_completion for s in ep_stats]),
        "goal_rate": _mean([1.0 if s.goal else 0.0 for s in ep_stats]),
        "collision_rate": _mean([1.0 if s.collision else 0.0 for s in ep_stats]),
        "timeup_rate": _mean([1.0 if s.timeup else 0.0 for s in ep_stats]),
        "episodes": float(len(ep_stats)),
    }
    return mean_reward, metrics


def _save_learning_curve(history: List[float], plot_path: Optional[str]) -> None:
    if not plot_path or not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting failed for {plot_path}: {exc}")
        return

    try:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(history) + 1), history, label="reward_mean")
        plt.xlabel("iteration")
        plt.ylabel("mean reward")
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved plot to: {plot_path}")
    except Exception as exc:
        print(f"Plotting failed for {plot_path}: {exc}")
    finally:
        try:
            plt.close()
        except Exception:
            pass


def _evaluate_candidate_worker(payload: dict) -> Tuple[float, Dict[str, float]]:
    params_vec = np.asarray(payload["params_vec"], dtype=np.float32)
    episodes = int(payload.get("episodes", 1))
    max_steps = payload.get("max_steps")
    domain_randomize = bool(payload.get("domain_randomize", False))
    collision_penalty = payload.get("collision_penalty")
    workers = int(payload.get("workers", 0) or 0)
    base_seed = int(payload.get("base_seed", 0))
    return rollout_mean(
        params_vec,
        episodes=episodes,
        max_steps=max_steps,
        domain_randomize=domain_randomize,
        collision_penalty=collision_penalty,
        workers=workers,
        base_seed=base_seed,
    )

def train_priority_params_gpu(
    iterations: int = 40,
    population: int = 16,
    sigma: float = 0.1,
    lr: float = 0.05,
    episodes_per_candidate: int = 5,
    eval_episodes: int = 5,
    seed: int = 0,
    domain_randomize: bool = False,
    collision_penalty: Optional[float] = None,
    save_params_json: Optional[str] = None,
    log_csv: Optional[str] = "policy/train/train_log_gpu.csv",
    plot_png: Optional[str] = None,
    clip_step_norm: float = 0.0,
    best_update_mode: str = "max",  # ["max", "mean_gap", "moving_avg"]
    best_update_alpha: float = 0.1,
    best_update_gap: float = 0.0,
    max_steps: Optional[int] = None,
    workers: int = 0,  # number of CPU processes for per-candidate episode rollouts
    candidate_workers: int = 0,  # process pool for evaluating distinct ES candidates
    device_override: Optional[str] = None,
) -> Tuple[PriorityParams, float, List[float], str, Dict[str, float]]:
    # Select device
    device = _resolve_device(device_override)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Initialize vector from saved params
    mean_vec_np = params_to_vector(get_priority_params()).astype(np.float32)
    dim = mean_vec_np.size
    mean_vec = torch.tensor(mean_vec_np, device=device)

    # Baseline reward
    base_mean, base_metrics = rollout_mean(
        mean_vec.detach().cpu().numpy(),
        episodes=episodes_per_candidate,
        max_steps=max_steps,
        domain_randomize=domain_randomize,
        collision_penalty=collision_penalty,
        workers=workers,
        base_seed=seed * 1000 + 1,
    )
    best_reward = float(base_mean)
    best_vector = mean_vec.detach().clone()
    hist_means: List[float] = []
    best_reward_ema = float(best_reward)

    assigned_candidate_workers = int(candidate_workers or 0)
    if assigned_candidate_workers < 0:
        assigned_candidate_workers = max(os.cpu_count() or 1, 1)
    candidate_pool: Optional[futures.ProcessPoolExecutor] = None
    if assigned_candidate_workers > 0:
        candidate_pool = futures.ProcessPoolExecutor(max_workers=assigned_candidate_workers)
        if workers and workers > 0:
            print(
                "[WARN] Both candidate_workers and workers are >0; this may oversubscribe CPU cores."
            )

    # print(
    #     "[Init] reward_mean={:.2f}, goal_rate={:.3f}, collision_rate={:.3f}, avg_steps={:.1f}".format(
    #         base_mean,
    #         base_metrics.get("goal_rate", float("nan")),
    #         base_metrics.get("collision_rate", float("nan")),
    #         base_metrics.get("avg_steps", float("nan")),
    #     )
    # )

    # CSV header
    metric_keys = [
        ("goal_rate", "best_goal_rate"),
        ("collision_rate", "best_collision_rate"),
        ("timeup_rate", "best_timeup_rate"),
        ("avg_steps", "best_avg_steps"),
        ("avg_task_completion", "best_avg_task_completion"),
    ]

    def _log_iter(
        it_idx: int,
        r_mean: float,
        r_std: float,
        r_max: float,
        vec_np: np.ndarray,
        metrics: Optional[dict] = None,
    ):
        if not log_csv:
            return
        try:
            Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
            write_header = not os.path.exists(log_csv)
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    header = ["iter", "reward_mean", "reward_std", "reward_max"]
                    header += [alias for _, alias in metric_keys]
                    header += [f"v{i}" for i in range(dim)]
                    w.writerow(header)
                row = [it_idx, r_mean, r_std, r_max]
                for key, _ in metric_keys:
                    value = float(metrics.get(key, float("nan"))) if metrics else float("nan")
                    row.append(value)
                row += list(map(float, vec_np.tolist()))
                w.writerow(row)
        except Exception:
            pass

    try:
        for it in range(1, iterations + 1):
            # Noise: (population, dim) on device
            noise = torch.randn((population, dim), device=device, dtype=mean_vec.dtype)
            candidates = mean_vec.unsqueeze(0) + sigma * noise

            # Evaluate rewards for each candidate (CPU env). We must move vectors to CPU numpy
            candidate_payloads = []
            for idx in range(population):
                vec_np = candidates[idx].detach().cpu().numpy()
                candidate_payloads.append(
                    {
                        "params_vec": vec_np,
                        "episodes": episodes_per_candidate,
                        "max_steps": max_steps,
                        "domain_randomize": domain_randomize,
                        "collision_penalty": collision_penalty,
                        "workers": workers,
                        "base_seed": seed * 100000 + it * 1000 + idx,
                    }
                )

            candidate_results: List[Tuple[float, Dict[str, float]]] = []
            if candidate_pool is not None:
                futures_list = [candidate_pool.submit(_evaluate_candidate_worker, payload) for payload in candidate_payloads]
                for fut in futures_list:
                    candidate_results.append(fut.result())
            else:
                for payload in candidate_payloads:
                    candidate_results.append(_evaluate_candidate_worker(payload))

            rewards = [res[0] for res in candidate_results]
            candidate_metrics = [res[1] for res in candidate_results]
            rewards_np = np.asarray(rewards, dtype=np.float32)

            # ES update on GPU
            r_mean = float(rewards_np.mean())
            r_std = float(rewards_np.std())
            r_max = float(rewards_np.max())
            rewards_t = torch.tensor(rewards_np, device=device, dtype=mean_vec.dtype)
            normalized = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            step = (lr / (population * sigma)) * (noise.transpose(0, 1) @ normalized)  # (dim,)
            if clip_step_norm and clip_step_norm > 0.0:
                step_norm = float(torch.linalg.vector_norm(step).item())
                if step_norm > clip_step_norm:
                    step = step * (clip_step_norm / (step_norm + 1e-8))
            new_mean_vec = mean_vec + step
            mean_vec = new_mean_vec

            # Best tracking (configurable)
            best_idx = int(np.argmax(rewards_np))
            current_best = float(rewards_np[best_idx])
            best_reward_ema = (best_update_alpha * r_mean) + ((1.0 - best_update_alpha) * best_reward_ema)
            if best_update_mode == "mean_gap":
                threshold = r_mean + best_update_gap
            elif best_update_mode == "moving_avg":
                threshold = best_reward_ema + best_update_gap
            else:
                threshold = best_reward
            if current_best > threshold:
                best_reward = current_best
                best_vector = candidates[best_idx].detach().clone()

            # Log and history
            best_metrics = candidate_metrics[best_idx] if candidate_metrics else {}

            _log_iter(
                it,
                r_mean,
                r_std,
                r_max,
                mean_vec.detach().cpu().numpy(),
                metrics=best_metrics,
            )
            hist_means.append(r_mean)
    finally:
        if candidate_pool is not None:
            candidate_pool.shutdown(wait=True)
        # print(
        #     f"[GPU-ES Iter {it:03d}] reward_mean={r_mean:.2f} reward_std={r_std:.2f} reward_max={r_max:.2f} "
        #     f"eps={episodes_per_candidate} device={device.type}"
        # )

    # Save best params
    best_params = vector_to_params(best_vector.detach().cpu().numpy())
    set_priority_params(best_params)
    save_priority_params(best_params)

    # Final eval
    final_score, final_stats = rollout_mean(
        params_to_vector(best_params),
        episodes=eval_episodes,
        max_steps=max_steps,
        domain_randomize=domain_randomize,
        collision_penalty=collision_penalty,
        workers=workers,
        base_seed=seed * 999999 + 4242,
    )
    # Optionally save best params and metadata to JSON
    if save_params_json:
        try:
            # Save as flat dict matching the example format: keys -> numeric values
            payload = asdict(best_params)
            Path(save_params_json).parent.mkdir(parents=True, exist_ok=True)
            with open(save_params_json, "w") as jf:
                json.dump(payload, jf, indent=2)
        except Exception:
            pass

    _save_learning_curve(hist_means, plot_png)
    return best_params, float(final_score), hist_means, device.type, final_stats

def main():
    parser = argparse.ArgumentParser(description="GPU ES training for my_policy priority parameters (PyTorch).")
    # ES hyperparams
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--episodes-per-candidate", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    # Env config overrides (single-map specialization)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--agent-num", type=int, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--collision", type=str, choices=["bounceback", "terminated"], default=None)
    parser.add_argument("--task-density", type=float, default=None)
    # Options
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument(
        "--collision-penalty",
        type=str,
        default="none",
        help="Collision penalty for an episode. Use a numeric value to overwrite episode reward on collision, or 'none' to keep per-step reward_list based rewards.",
    )
    parser.add_argument("--log-csv", type=str, default="policy/train/train_log_gpu.csv")
    parser.add_argument(
        "--plot-png",
        type=str,
        default=None,
        help="Path to save the learning curve PNG (defaults to '<log-csv stem>_learning_curve.png').",
    )
    parser.add_argument("--clip-step-norm", type=float, default=0.0)
    parser.add_argument("--best-update-mode", type=str, choices=["max", "mean_gap", "moving_avg"], default="max")
    parser.add_argument("--best-update-alpha", type=float, default=0.1)
    parser.add_argument("--best-update-gap", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0, help="Process workers for rollout parallelization (CPU).")
    parser.add_argument(
        "--candidate-workers",
        type=int,
        default=0,
        help="Process pool size for concurrent ES candidate evaluations (CPU). Use a negative value to auto-set to CPU count.",
    )
    parser.add_argument(
        "--sweep-workers",
        type=int,
        default=0,
        help="Number of concurrent sweep jobs. Defaults to the number of --gpu-devices if provided, otherwise 1.",
    )
    parser.add_argument(
        "--sweep-plot-dir",
        type=str,
        default=None,
        help="If set, save per-run reward plots under this directory during --sweep.",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Comma-separated device list (e.g. '0,1', 'cuda:1,cpu') to cycle across training jobs.",
    )
    parser.add_argument("--save-params-json", type=str, default=None, help="Path to save best priority params and metadata as JSON.")
    parser.add_argument("--sweep", action="store_true", help="Run sequential training across the predefined map list and agent counts.")
    parser.add_argument(
        "--sweep-output-dir",
        type=str,
        default="policy/train/sweep_results",
        help="Directory to store JSON parameter files and logs when --sweep is enabled.",
    )
    args = parser.parse_args()

    device_list = _parse_device_list(args.gpu_devices)
    candidate_workers = args.candidate_workers
    if candidate_workers < 0:
        candidate_workers = max(os.cpu_count() or 1, 1)
    sweep_plot_dir = Path(args.sweep_plot_dir).resolve() if args.sweep_plot_dir else None

    # Apply ENV overrides
    overrides = {}
    if args.speed is not None:
        overrides["speed"] = args.speed
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.collision is not None:
        overrides["collision"] = args.collision
    if args.task_density is not None:
        overrides["task_density"] = args.task_density
    if not args.sweep:
        if args.map_name is not None:
            overrides["map_name"] = args.map_name
        if args.agent_num is not None:
            overrides["agent_num"] = args.agent_num
    else:
        if args.map_name is not None or args.agent_num is not None:
            print("[SWEEP] Ignoring --map-name/--agent-num because --sweep controls these values.")
    if overrides:
        ENV_CONFIG.update(overrides)
        ENV_CONFIG["time_limit"] = _compute_time_limit(ENV_CONFIG.get("agent_num"))

    # interpret collision_penalty (allow 'none' to disable overwrite and use per-step rewards)
    cp_arg = args.collision_penalty
    try:
        if isinstance(cp_arg, str) and cp_arg.lower() in ("none", "null", "nil", "off"):
            collision_penalty_val = None
        else:
            collision_penalty_val = float(cp_arg)
    except Exception:
        collision_penalty_val = float(-1000.0)

    if args.sweep:
        run_sweep(args, collision_penalty_val, candidate_workers, device_list, sweep_plot_dir)
        return

    t0 = time.time()
    device_override = device_list[0] if device_list else None
    plot_path = _derive_plot_path(args.log_csv, args.plot_png)
    params, final_score, hist, device_type, final_stats = train_priority_params_gpu(
        iterations=args.iterations,
        population=args.population,
        sigma=args.sigma,
        lr=args.lr,
        episodes_per_candidate=args.episodes_per_candidate,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        domain_randomize=args.domain_randomize,
        collision_penalty=collision_penalty_val,
        save_params_json=args.save_params_json,
        log_csv=args.log_csv,
        plot_png=str(plot_path) if plot_path else None,
        clip_step_norm=args.clip_step_norm,
        best_update_mode=args.best_update_mode,
        best_update_alpha=args.best_update_alpha,
        best_update_gap=args.best_update_gap,
        max_steps=args.max_steps,
        workers=args.workers,
        candidate_workers=candidate_workers,
        device_override=device_override,
    )

    print("==== Final evaluation (GPU ES) ====")
    print(f"PriorityParams: {params}")
    print(f"Average episode reward (@{args.eval_episodes} episodes): {final_score:.2f}")
    if final_stats:
        print(
            "Final metrics -> goal_rate: {goal:.3f}, collision_rate: {coll:.3f}, timeup_rate: {timeup:.3f}, avg_steps: {steps:.1f}, avg_task_completion: {tasks:.2f}".format(
                goal=final_stats.get("goal_rate", float("nan")),
                coll=final_stats.get("collision_rate", float("nan")),
                timeup=final_stats.get("timeup_rate", float("nan")),
                steps=final_stats.get("avg_steps", float("nan")),
                tasks=final_stats.get("avg_task_completion", float("nan")),
            )
        )
    elapsed = time.time() - t0
    print("==== Execution time ====")
    print(f"Device: {device_type}, Elapsed: {elapsed:.2f} seconds")
def run_sweep(
    args,
    collision_penalty_val: Optional[float],
    candidate_workers: int,
    device_list: List[str],
    sweep_plot_dir: Optional[Path],
) -> None:
    """Run training across the requested map/agent combinations (optionally in parallel)."""
    output_dir = Path(args.sweep_output_dir or "policy/train/sweep_results")
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env_config = dict(ENV_CONFIG)
    plot_dir: Optional[Path] = None
    if sweep_plot_dir is not None:
        plot_dir = Path(sweep_plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[dict] = []

    for map_name in SWEEP_MAP_LIST:
        agent_counts = _agent_counts_for_map(map_name)
        if not agent_counts:
            continue
        for agent_num in agent_counts:
            order = len(tasks)
            run_seed = args.seed + order
            run_config = dict(base_env_config)
            run_config.update({
                "map_name": map_name,
                "agent_num": agent_num,
            })
            run_config["time_limit"] = _compute_time_limit(run_config.get("agent_num"))

            json_path = output_dir / f"priority_params_{map_name}_agents_{agent_num}.json"
            log_csv_path = logs_dir / f"train_log_{map_name}_agents_{agent_num}.csv"
            if plot_dir is not None:
                plot_png_path = plot_dir / f"reward_{map_name}_agents_{agent_num}.png"
            else:
                plot_png_path = None

            trainer_kwargs = dict(
                iterations=args.iterations,
                population=args.population,
                sigma=args.sigma,
                lr=args.lr,
                episodes_per_candidate=args.episodes_per_candidate,
                eval_episodes=args.eval_episodes,
                seed=run_seed,
                domain_randomize=args.domain_randomize,
                collision_penalty=collision_penalty_val,
                save_params_json=str(json_path),
                log_csv=str(log_csv_path),
                plot_png=str(plot_png_path.resolve()) if plot_png_path else None,
                clip_step_norm=args.clip_step_norm,
                best_update_mode=args.best_update_mode,
                best_update_alpha=args.best_update_alpha,
                best_update_gap=args.best_update_gap,
                max_steps=args.max_steps,
                workers=args.workers,
                candidate_workers=candidate_workers,
            )

            summary_entry = {
                "map_name": map_name,
                "agent_num": agent_num,
                "params_path": str(json_path),
                "log_csv": str(log_csv_path),
                "iterations": args.iterations,
                "population": args.population,
                "sigma": args.sigma,
                "lr": args.lr,
                "episodes_per_candidate": args.episodes_per_candidate,
                "eval_episodes": args.eval_episodes,
                "seed": run_seed,
                "_order": order,
            }

            tasks.append(
                {
                    "run_config": run_config,
                    "trainer_kwargs": trainer_kwargs,
                    "summary_entry": summary_entry,
                }
            )

    if not tasks:
        print("[SWEEP] No valid map/agent combinations found.")
        return

    device_cycle = device_list or [None]
    requested_workers = args.sweep_workers if args.sweep_workers and args.sweep_workers > 0 else None
    sweep_workers = requested_workers or len(device_cycle)
    sweep_workers = max(1, min(sweep_workers, len(tasks)))

    summary: List[dict] = []

    if sweep_workers <= 1:
        try:
            for idx, task in enumerate(tasks):
                ENV_CONFIG.clear()
                ENV_CONFIG.update(task["run_config"])
                trainer_kwargs = dict(task["trainer_kwargs"])
                device_override = device_cycle[idx % len(device_cycle)]
                trainer_kwargs["device_override"] = device_override
                _, final_score, _, device_type, final_stats = train_priority_params_gpu(**trainer_kwargs)
                entry = dict(task["summary_entry"])
                entry.update(
                    {
                        "final_score": final_score,
                        "device": device_type,
                        "goal_rate": final_stats.get("goal_rate") if final_stats else None,
                        "collision_rate": final_stats.get("collision_rate") if final_stats else None,
                        "timeup_rate": final_stats.get("timeup_rate") if final_stats else None,
                        "avg_steps": final_stats.get("avg_steps") if final_stats else None,
                        "avg_task_completion": final_stats.get("avg_task_completion") if final_stats else None,
                    }
                )
                summary.append(entry)
        finally:
            ENV_CONFIG.clear()
            ENV_CONFIG.update(base_env_config)
    else:
        payloads = []
        for idx, task in enumerate(tasks):
            payload = {
                "run_config": task["run_config"],
                "trainer_kwargs": dict(task["trainer_kwargs"]),
                "summary_entry": dict(task["summary_entry"]),
                "base_env_config": base_env_config,
                "device_override": device_cycle[idx % len(device_cycle)],
            }
            payloads.append(payload)

        with futures.ProcessPoolExecutor(max_workers=sweep_workers) as pool:
            future_to_payload = [pool.submit(_run_sweep_job, payload) for payload in payloads]
            for fut in futures.as_completed(future_to_payload):
                summary.append(fut.result())

    summary.sort(key=lambda x: x.get("_order", 0))
    for entry in summary:
        entry.pop("_order", None)

    summary_path = output_dir / "sweep_summary.json"
    ENV_CONFIG.clear()
    ENV_CONFIG.update(base_env_config)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SWEEP] Completed {len(tasks)} runs. Summary saved to {summary_path}.")


def _run_sweep_job(payload: dict) -> dict:
    run_config = payload["run_config"]
    trainer_kwargs = dict(payload["trainer_kwargs"])
    summary_entry = dict(payload["summary_entry"])
    base_env_config = dict(payload.get("base_env_config", ENV_CONFIG))
    device_override = payload.get("device_override")

    try:
        ENV_CONFIG.clear()
        ENV_CONFIG.update(run_config)
        trainer_kwargs["device_override"] = device_override
        _, final_score, _, device_type, final_stats = train_priority_params_gpu(**trainer_kwargs)
    finally:
        ENV_CONFIG.clear()
        ENV_CONFIG.update(base_env_config)

    summary_entry.update(
        {
            "final_score": final_score,
            "device": device_type,
            "goal_rate": final_stats.get("goal_rate") if final_stats else None,
            "collision_rate": final_stats.get("collision_rate") if final_stats else None,
            "timeup_rate": final_stats.get("timeup_rate") if final_stats else None,
            "avg_steps": final_stats.get("avg_steps") if final_stats else None,
            "avg_task_completion": final_stats.get("avg_task_completion") if final_stats else None,
        }
    )
    return summary_entry


def _derive_plot_path(log_csv: Optional[str], plot_png: Optional[str]) -> Optional[Path]:
    """Return a target path for the learning curve image."""
    if plot_png:
        return Path(plot_png)
    if not log_csv:
        return None
    log_path = Path(log_csv)
    stem = log_path.stem or "training"
    return log_path.parent / f"{stem}_learning_curve.png"


if __name__ == "__main__":
    main()
