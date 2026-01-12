#!/usr/bin/env python3
"""
continue_train_by_gpu.py の仕組みを使い、step_tolerance のみを学習する。
各マップで agent_num=3..15 を順に学習する。
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

import policy.train.continue_train_by_gpu as base
from policy.my_policy import PriorityParams, get_priority_params


def _params_to_vector_step_only(params: PriorityParams) -> np.ndarray:
    return np.array([float(getattr(params, "step_tolerance", 0.0) or 0.0)], dtype=np.float32)


def _vector_to_params_step_only(vec: np.ndarray) -> PriorityParams:
    raw = np.asarray(vec, dtype=np.float32)
    current = get_priority_params()
    step_val = float(raw[0]) if raw.size > 0 else float(getattr(current, "step_tolerance", 0.0) or 0.0)
    step_val = float(np.clip(step_val, 0.0, 100.0))
    if not np.isfinite(step_val) or step_val <= 0.0:
        step_val = base._default_step_tolerance_for_speed(float(base.ENV_CONFIG.get("speed", 5.0) or 5.0))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--episodes-per-candidate", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
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
        default=str(Path("policy/train/step_tolerance_results")),
    )
    args = parser.parse_args()

    # override vectorization in base module
    base.params_to_vector = _params_to_vector_step_only
    base.vector_to_params = _vector_to_params_step_only

    max_steps = args.max_steps if args.max_steps > 0 else None
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    plots_dir = out_dir / "plots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for map_name in base.SWEEP_MAP_LIST:
        for agent_num in range(3, 16):
            base.ENV_CONFIG["map_name"] = map_name
            base.ENV_CONFIG["agent_num"] = agent_num
            base.ENV_CONFIG = base._normalized_env_config(base.ENV_CONFIG)

            log_csv = logs_dir / f"train_log_step_tol_{map_name}_agents_{agent_num}.csv"
            plot_png = plots_dir / f"reward_step_tol_{map_name}_agents_{agent_num}.png"

            base.train_priority_params_gpu(
                iterations=args.iterations,
                population=args.population,
                sigma=args.sigma,
                lr=args.lr,
                episodes_per_candidate=args.episodes_per_candidate,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                domain_randomize=False,
                collision_penalty=args.collision_penalty,
                save_params_json=None,
                log_csv=str(log_csv),
                plot_png=str(plot_png),
                clip_step_norm=0.0,
                best_update_mode="max",
                best_update_alpha=0.1,
                best_update_gap=0.0,
                max_steps=max_steps,
                workers=workers,
                candidate_workers=args.candidate_workers,
                device_override=args.device,
                verbose=False,
                reuse_env=True,
                resume_from_log=args.resume_from_log,
            )


if __name__ == "__main__":
    main()
