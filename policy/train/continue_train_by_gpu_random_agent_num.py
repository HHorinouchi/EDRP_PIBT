#!/usr/bin/env python3
"""
continue_train_by_gpu.py をベースに、各エピソードで agent_num を
2〜(ノード数の0.5倍) の範囲でランダム化して学習を行う。
"""

import argparse
import math
import os
from typing import Optional

import numpy as np

import policy.train.continue_train_by_gpu as base


def _agent_num_range(map_name: str) -> int:
    n_nodes = base._get_map_node_count(map_name) or 0
    if n_nodes <= 0:
        return 2
    max_agents = max(2, min(int(math.floor(n_nodes * 0.5)), 15))
    return max_agents


def _sample_env_config_random_agent(rng: np.random.Generator) -> dict:
    if getattr(_sample_env_config_random_agent, "random_map", False):
        map_name = rng.choice(base.MAP_POOL)
    else:
        map_name = base.ENV_CONFIG.get("map_name", "map_shibuya")
    max_agents = _agent_num_range(map_name)
    agent_num = int(rng.integers(2, max_agents + 1))
    cfg = dict(base.ENV_CONFIG)
    cfg["map_name"] = map_name
    cfg["agent_num"] = agent_num
    return base._normalized_env_config(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--episodes-per-candidate", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--log-csv", type=str, default="policy/train/train_log_gpu_random_agent.csv")
    parser.add_argument("--plot-png", type=str, default=None)
    parser.add_argument("--clip-step-norm", type=float, default=0.0)
    parser.add_argument("--best-update-mode", type=str, default="max")
    parser.add_argument("--best-update-alpha", type=float, default=0.1)
    parser.add_argument("--best-update-gap", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--candidate-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume-from-log", action="store_true")
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--random-map", action="store_true")
    args = parser.parse_args()

    if args.map_name is not None:
        base.ENV_CONFIG["map_name"] = args.map_name
    base.ENV_CONFIG = base._normalized_env_config(base.ENV_CONFIG)

    _sample_env_config_random_agent.random_map = bool(args.random_map)
    base.sample_env_config = _sample_env_config_random_agent

    max_steps = args.max_steps if args.max_steps > 0 else None
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)

    base.train_priority_params_gpu(
        iterations=args.iterations,
        population=args.population,
        sigma=args.sigma,
        lr=args.lr,
        episodes_per_candidate=args.episodes_per_candidate,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        domain_randomize=True,
        collision_penalty=args.collision_penalty,
        save_params_json=None,
        log_csv=args.log_csv,
        plot_png=args.plot_png,
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
        resume_from_log=args.resume_from_log,
    )


if __name__ == "__main__":
    main()
