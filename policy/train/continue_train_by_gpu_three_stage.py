#!/usr/bin/env python3
"""
Three-stage training:
1) Learn step_tolerance with agent_num randomized in [5, 10].
2) Learn pick_weight and drop_weight only.
3) Learn step_tolerance, assign_pick_weight, goal_weight.

assign_drop_weight is never trained.

Example commands:
# python policy/train/continue_train_by_gpu_three_stage.py \
#   --map-name map_5x4 \
#   --stage3-ratios 0.25,0.5 \
#   --output-dir policy/train/three_stage
#
# python policy/train/continue_train_by_gpu_three_stage.py \
#   --sweep-maps \
#   --stage3-ratios 0.25,0.5 \
#   --iterations 100 \
#   --population 32 \
#   --episodes-per-candidate 20 \
#   --eval-episodes 20 \
#   --candidate-workers 16 \
#   --workers 0 \
#   --output-dir policy/train/three_stage
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import policy.train.continue_train_by_gpu as base
from policy.my_policy import PriorityParams, get_priority_params, set_priority_params


def _clip_step_tol(value: float) -> float:
    val = float(np.clip(value, 0.0, 100.0))
    if not np.isfinite(val) or val <= 0.0:
        val = base._default_step_tolerance_for_speed(float(base.ENV_CONFIG.get("speed", 5.0) or 5.0))
    return val


def _params_to_vector_step_only(params: PriorityParams) -> np.ndarray:
    return np.array([float(getattr(params, "step_tolerance", 0.0) or 0.0)], dtype=np.float32)


def _vector_to_params_step_only(vec: np.ndarray) -> PriorityParams:
    raw = np.asarray(vec, dtype=np.float32)
    current = get_priority_params()
    step_val = float(raw[0]) if raw.size > 0 else float(getattr(current, "step_tolerance", 0.0) or 0.0)
    step_val = _clip_step_tol(step_val)
    return PriorityParams(
        goal_weight=float(getattr(current, "goal_weight", 1.0)),
        pick_weight=float(getattr(current, "pick_weight", 1.0)),
        drop_weight=float(getattr(current, "drop_weight", 1.0)),
        assign_pick_weight=float(getattr(current, "assign_pick_weight", 1.0)),
        assign_drop_weight=float(getattr(current, "assign_drop_weight", 1.0)),
        congestion_weight=float(getattr(current, "congestion_weight", 0.0)),
        assign_spread_weight=float(getattr(current, "assign_spread_weight", 1.0)),
        step_tolerance=step_val,
    )


def _params_to_vector_pick_drop(params: PriorityParams) -> np.ndarray:
    return np.array(
        [
            float(getattr(params, "pick_weight", 1.0)),
            float(getattr(params, "drop_weight", 1.0)),
        ],
        dtype=np.float32,
    )


def _vector_to_params_pick_drop(vec: np.ndarray) -> PriorityParams:
    raw = np.asarray(vec, dtype=np.float32)
    current = get_priority_params()
    pick = float(raw[0]) if raw.size > 0 else float(getattr(current, "pick_weight", 1.0))
    drop = float(raw[1]) if raw.size > 1 else float(getattr(current, "drop_weight", 1.0))
    pick = float(np.clip(pick, 0.0, 10.0))
    drop = float(np.clip(drop, 0.0, 10.0))
    return PriorityParams(
        goal_weight=float(getattr(current, "goal_weight", 1.0)),
        pick_weight=pick,
        drop_weight=drop,
        assign_pick_weight=float(getattr(current, "assign_pick_weight", 1.0)),
        assign_drop_weight=float(getattr(current, "assign_drop_weight", 1.0)),
        congestion_weight=float(getattr(current, "congestion_weight", 0.0)),
        assign_spread_weight=float(getattr(current, "assign_spread_weight", 1.0)),
        step_tolerance=_clip_step_tol(float(getattr(current, "step_tolerance", 0.0) or 0.0)),
    )


def _params_to_vector_stage3(params: PriorityParams) -> np.ndarray:
    return np.array(
        [
            float(getattr(params, "step_tolerance", 0.0) or 0.0),
            float(getattr(params, "assign_pick_weight", 1.0)),
            float(getattr(params, "goal_weight", 1.0)),
        ],
        dtype=np.float32,
    )


def _vector_to_params_stage3(vec: np.ndarray) -> PriorityParams:
    raw = np.asarray(vec, dtype=np.float32)
    current = get_priority_params()
    step_val = float(raw[0]) if raw.size > 0 else float(getattr(current, "step_tolerance", 0.0) or 0.0)
    assign_pick = float(raw[1]) if raw.size > 1 else float(getattr(current, "assign_pick_weight", 1.0))
    goal_w = float(raw[2]) if raw.size > 2 else float(getattr(current, "goal_weight", 1.0))
    step_val = _clip_step_tol(step_val)
    assign_pick = float(np.clip(assign_pick, 0.0, 10.0))
    goal_w = float(np.clip(goal_w, 0.0, 10.0))
    return PriorityParams(
        goal_weight=goal_w,
        pick_weight=float(getattr(current, "pick_weight", 1.0)),
        drop_weight=float(getattr(current, "drop_weight", 1.0)),
        assign_pick_weight=assign_pick,
        assign_drop_weight=float(getattr(current, "assign_drop_weight", 1.0)),
        congestion_weight=float(getattr(current, "congestion_weight", 0.0)),
        assign_spread_weight=float(getattr(current, "assign_spread_weight", 1.0)),
        step_tolerance=step_val,
    )


def _sample_env_config_agent_range(rng: np.random.Generator) -> dict:
    map_name = base.ENV_CONFIG.get("map_name", "map_shibuya")
    n_nodes = base._get_map_node_count(map_name) or 0
    min_agents = 5
    max_agents = 10
    if n_nodes > 0:
        max_agents = min(max_agents, max(2, n_nodes - 1))
        min_agents = min(min_agents, max_agents)
    agent_num = int(rng.integers(min_agents, max_agents + 1))
    cfg = dict(base.ENV_CONFIG)
    cfg["agent_num"] = agent_num
    return base._normalized_env_config(cfg)


def _run_stage(
    name: str,
    args: argparse.Namespace,
    log_csv: Path,
    plot_png: Optional[Path],
    save_json: Optional[Path],
    max_iterations: Optional[int] = None,
    early_stop_collision: Optional[float] = None,
    early_stop_patience: int = 0,
) -> PriorityParams:
    max_steps = args.max_steps if args.max_steps > 0 else None
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)

    start_time = time.time()
    total_iterations = max_iterations if max_iterations is not None else args.iterations
    patience = 0
    best_params = None
    hist_means = []
    hist_collision = []
    for it in range(int(total_iterations)):
        best_params, _, hist_means, hist_collision, _, _ = base.train_priority_params_gpu(
            iterations=1,
            population=args.population,
            sigma=args.sigma,
            lr=args.lr,
            episodes_per_candidate=args.episodes_per_candidate,
            eval_episodes=args.eval_episodes,
            seed=args.seed + it,
            domain_randomize=args.domain_randomize,
            collision_penalty=args.collision_penalty,
            save_params_json=None,
            log_csv=str(log_csv),
            plot_png=str(plot_png) if plot_png is not None else None,
            clip_step_norm=args.clip_step_norm,
            best_update_mode=args.best_update_mode,
            best_update_alpha=args.best_update_alpha,
            best_update_gap=args.best_update_gap,
            max_steps=max_steps,
            workers=workers,
            candidate_workers=args.candidate_workers,
            device_override=args.device,
            verbose=False,
            reuse_env=False,
            resume_from_log=True,
        )
        if early_stop_collision is not None and hist_collision:
            last_coll = hist_collision[-1]
            if np.isfinite(last_coll) and last_coll < early_stop_collision:
                patience += 1
            else:
                patience = 0
            if patience >= max(early_stop_patience, 1):
                break
    if best_params is None:
        best_params, _, _, _, _, _ = base.train_priority_params_gpu(
            iterations=1,
            population=args.population,
            sigma=args.sigma,
            lr=args.lr,
            episodes_per_candidate=args.episodes_per_candidate,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            domain_randomize=args.domain_randomize,
            collision_penalty=args.collision_penalty,
            save_params_json=None,
            log_csv=str(log_csv),
            plot_png=str(plot_png) if plot_png is not None else None,
            clip_step_norm=args.clip_step_norm,
            best_update_mode=args.best_update_mode,
            best_update_alpha=args.best_update_alpha,
            best_update_gap=args.best_update_gap,
            max_steps=max_steps,
            workers=workers,
            candidate_workers=args.candidate_workers,
            device_override=args.device,
            verbose=False,
            reuse_env=False,
            resume_from_log=True,
        )
    elapsed = time.time() - start_time
    actual_iterations = max(len(hist_means), 1)
    total_rollouts = int(actual_iterations) * int(args.population) * int(args.episodes_per_candidate)
    if save_json is not None:
        payload = {
            "stage": name,
            "params": {
                "goal_weight": float(getattr(best_params, "goal_weight", 0.0)),
                "pick_weight": float(getattr(best_params, "pick_weight", 0.0)),
                "drop_weight": float(getattr(best_params, "drop_weight", 0.0)),
                "assign_pick_weight": float(getattr(best_params, "assign_pick_weight", 0.0)),
                "assign_drop_weight": float(getattr(best_params, "assign_drop_weight", 0.0)),
                "congestion_weight": float(getattr(best_params, "congestion_weight", 0.0)),
                "assign_spread_weight": float(getattr(best_params, "assign_spread_weight", 0.0)),
                "step_tolerance": float(getattr(best_params, "step_tolerance", 0.0)),
            },
            "training_time_seconds": float(elapsed),
            "rollout_count": total_rollouts,
            "episodes_per_candidate": int(args.episodes_per_candidate),
            "iterations": int(actual_iterations),
            "population": int(args.population),
        }
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(payload, indent=2))
    set_priority_params(best_params)
    print(f"[STAGE] {name} done.")
    return best_params


def _agent_num_from_ratio(map_name: str, ratio: float) -> int:
    n_nodes = base._get_map_node_count(map_name) or 0
    if n_nodes <= 0:
        return max(2, int(math.floor(10)))
    return max(2, int(math.floor(n_nodes * ratio)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-name", type=str, default="map_5x4")
    parser.add_argument("--agent-num", type=int, default=10)
    parser.add_argument("--stage3-ratios", type=str, default="0.25,0.5")
    parser.add_argument("--sweep-maps", action="store_true")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--episodes-per-candidate", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--clip-step-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--candidate-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume-from-log", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("policy/train/three_stage")),
    )
    args = parser.parse_args()

    maps = [args.map_name]
    if args.sweep_maps:
        maps = list(base.SWEEP_MAP_LIST)

    ratios = []
    for token in args.stage3_ratios.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ratios.append(float(token))
        except Exception:
            continue
    if not ratios:
        ratios = [0.25, 0.5]

    for map_name in maps:
        base.ENV_CONFIG["map_name"] = map_name
        base.ENV_CONFIG["agent_num"] = args.agent_num
        base.ENV_CONFIG = base._normalized_env_config(base.ENV_CONFIG)

        out_dir = Path(args.output_dir) / map_name
        logs_dir = out_dir / "logs"
        plots_dir = out_dir / "plots"
        logs_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: step_tolerance only, agent_num random in [5, 10]
        base.params_to_vector = _params_to_vector_step_only
        base.vector_to_params = _vector_to_params_step_only
        base.sample_env_config = _sample_env_config_agent_range
        args.domain_randomize = True
        _run_stage(
            "stage1_step_tolerance",
            args,
            logs_dir / "train_log_stage1_step_tolerance.csv",
            plots_dir / "reward_stage1_step_tolerance.png",
            logs_dir / "stage1_step_tolerance_params.json",
            max_iterations=150,
            early_stop_collision=0.1,
            early_stop_patience=10,
        )

        # Stage 2: pick/drop only, agent_num random in [5, 10]
        base.params_to_vector = _params_to_vector_pick_drop
        base.vector_to_params = _vector_to_params_pick_drop
        base.sample_env_config = _sample_env_config_agent_range
        args.domain_randomize = True
        _run_stage(
            "stage2_pick_drop",
            args,
            logs_dir / "train_log_stage2_pick_drop.csv",
            plots_dir / "reward_stage2_pick_drop.png",
            logs_dir / "stage2_pick_drop_params.json",
            max_iterations=100,
        )

        # Stage 3: step_tolerance, assign_pick_weight, goal_weight (agent_num ratios)
        base.params_to_vector = _params_to_vector_stage3
        base.vector_to_params = _vector_to_params_stage3
        base.sample_env_config = base.sample_env_config
        args.domain_randomize = False
        for ratio in ratios:
            base.ENV_CONFIG["agent_num"] = _agent_num_from_ratio(map_name, ratio)
            base.ENV_CONFIG = base._normalized_env_config(base.ENV_CONFIG)
            ratio_tag = f"{ratio:.2f}".replace(".", "p")
            _run_stage(
                f"stage3_step_assign_goal_ratio_{ratio_tag}",
                args,
                logs_dir / f"train_log_stage3_step_assign_goal_{ratio_tag}.csv",
                plots_dir / f"reward_stage3_step_assign_goal_{ratio_tag}.png",
                logs_dir / f"stage3_step_assign_goal_{ratio_tag}_params.json",
                max_iterations=100,
            )


if __name__ == "__main__":
    main()
