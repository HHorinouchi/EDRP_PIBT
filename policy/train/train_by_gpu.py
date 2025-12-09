# GPU-accelerated ES training for MAPD×PIBT priority parameters
# - Strategy mirrors policy/train/train.py but moves ES math (noise, normalization, updates) to GPU (CUDA/MPS) via PyTorch
# - Environment rollouts remain Python-side (CPU). Optionally parallelized with process workers.

import argparse
import os
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import time
import math
import numpy as np
import concurrent.futures as futures
import json
from dataclasses import asdict

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

# Base environment configuration (single-map specialization by default)
ENV_CONFIG = dict(
    agent_num=10,
    speed=5.0,
    start_ori_array=[],
    goal_array=[],
    visu_delay=0.0,
    state_repre_flag="onehot",
    time_limit=300,
    collision="bounceback",
    task_flag=True,
    reward_list={"goal": 10000, "collision": -5*10*300, "wait": -10, "move": -1},
    map_name="map_shibuya",
    task_density=1.0,
)

def make_env() -> DrpEnv:
    # EE_map.MapMake(enable_render=False) is default via constructor
    return DrpEnv(**ENV_CONFIG)

# Domain randomization helpers
MAP_POOL = list(UNREAL_MAP)

def _get_map_node_count(map_name: str) -> int:
    try:
        node_csv = ROOT_DIR / "drp_env" / "map" / map_name / "node.csv"
        with node_csv.open("r") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0

def sample_env_config(rng: np.random.Generator) -> dict:
    map_name = rng.choice(MAP_POOL)
    n_nodes = _get_map_node_count(map_name) or 9
    max_agents = max(2, min(5, max(2, n_nodes - 1)))
    agent_num = int(rng.integers(2, max_agents + 1))
    speed = float(rng.uniform(0.8, 2.0))
    time_limit = int(rng.integers(150, 401))
    collision = "bounceback" if rng.random() < 0.7 else "terminated"
    task_density = float(rng.uniform(0.3, 1.5))
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
    return DrpEnv(**cfg)

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
    max_steps: int,
    domain_randomize: bool,
    collision_penalty: Optional[float],
    seed: Optional[int] = None,
) -> float:
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
    total_reward = 0.0
    obs = env.reset()
    done_flags = [False for _ in range(env.agent_num)]
    steps = 0
    while not all(done_flags) and steps < max_steps:
        actions, task_assign = rollout_policy(obs, env)
        obs, step_rewards, done_flags, info = env.step({"pass": actions, "task": task_assign})
        total_reward += float(sum(step_rewards))
        if isinstance(info, dict) and info.get("collision", False):
            if collision_penalty is not None:
                total_reward = float(collision_penalty)
            print("Episode ended due to collision.")
            break
        steps += 1
        # print(f"未割り当てタスク数: {len(env.current_tasklist)}")
        # for i in range(env.agent_num):
        #     print(f" Agent {i} action: {actions[i]}")
        #     print(f" Agent {i} start: {env.current_start[i]}, goal: {env.goal_array[i]}")
        #     print(f" Agent {i} avail actions: {env.get_avail_agent_actions(i, env.n_actions)[1]}")
        #     # タスクをアサインされている場合、そのタスクも表示
        #     if i < len(env.assigned_tasks):
        #         print(f" Agent {i} assigned task: {env.assigned_tasks[i]}")
    env.close()
    print(f" Total reward: {total_reward}")
    return total_reward

def rollout_mean(
    params_vec: np.ndarray,
    episodes: int,
    max_steps: int,
    domain_randomize: bool,
    collision_penalty: Optional[float],
    workers: int = 0,
    base_seed: int = 0,
) -> float:
    # Evaluate mean episode reward with optional multiprocessing
    if episodes <= 1 or workers <= 0:
        vals = [
            rollout_once(params_vec, max_steps, domain_randomize, collision_penalty, seed=base_seed + i)
            for i in range(episodes)
        ]
        return float(np.mean(vals))
    # Use process pool
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
        vals = [t.result() for t in tasks]
    return float(np.mean(vals))

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
    max_steps: int = 500,
    workers: int = 0,  # number of CPU processes for parallel rollouts
) -> Tuple[PriorityParams, float, List[float], str]:
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Initialize vector from saved params
    mean_vec_np = params_to_vector(get_priority_params()).astype(np.float32)
    dim = mean_vec_np.size
    mean_vec = torch.tensor(mean_vec_np, device=device)

    # Baseline reward
    base_mean = rollout_mean(
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

    # CSV header
    def _log_iter(it_idx: int, r_mean: float, r_std: float, r_max: float, vec_np: np.ndarray):
        if not log_csv:
            return
        try:
            Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
            write_header = not os.path.exists(log_csv)
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    header = ["iter", "reward_mean", "reward_std", "reward_max"] + [f"v{i}" for i in range(dim)]
                    w.writerow(header)
                row = [it_idx, r_mean, r_std, r_max] + list(map(float, vec_np.tolist()))
                w.writerow(row)
        except Exception:
            pass

    for it in range(1, iterations + 1):
        # Noise: (population, dim) on device
        noise = torch.randn((population, dim), device=device, dtype=mean_vec.dtype)
        candidates = mean_vec.unsqueeze(0) + sigma * noise

        # Evaluate rewards for each candidate (CPU env). We must move vectors to CPU numpy
        rewards = []
        for idx in range(population):
            vec_np = candidates[idx].detach().cpu().numpy()
            r = rollout_mean(
                vec_np,
                episodes=episodes_per_candidate,
                max_steps=max_steps,
                domain_randomize=domain_randomize,
                collision_penalty=collision_penalty,
                workers=workers,
                base_seed=seed * 100000 + it * 1000 + idx,
            )
            rewards.append(r)
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
        _log_iter(it, r_mean, r_std, r_max, mean_vec.detach().cpu().numpy())
        hist_means.append(r_mean)
        # print(
        #     f"[GPU-ES Iter {it:03d}] reward_mean={r_mean:.2f} reward_std={r_std:.2f} reward_max={r_max:.2f} "
        #     f"eps={episodes_per_candidate} device={device.type}"
        # )

    # Save best params
    best_params = vector_to_params(best_vector.detach().cpu().numpy())
    set_priority_params(best_params)
    save_priority_params(best_params)

    # Final eval
    final_score = rollout_mean(
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
    return best_params, float(final_score), hist_means, device.type

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
    parser.add_argument("--plot-png", type=str, default=None)
    parser.add_argument("--clip-step-norm", type=float, default=0.0)
    parser.add_argument("--best-update-mode", type=str, choices=["max", "mean_gap", "moving_avg"], default="max")
    parser.add_argument("--best-update-alpha", type=float, default=0.1)
    parser.add_argument("--best-update-gap", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0, help="Process workers for rollout parallelization (CPU).")
    parser.add_argument("--save-params-json", type=str, default=None, help="Path to save best priority params and metadata as JSON.")
    args = parser.parse_args()

    # Apply ENV overrides
    overrides = {}
    if args.map_name is not None:
        overrides["map_name"] = args.map_name
    if args.agent_num is not None:
        overrides["agent_num"] = args.agent_num
    if args.speed is not None:
        overrides["speed"] = args.speed
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.collision is not None:
        overrides["collision"] = args.collision
    if args.task_density is not None:
        overrides["task_density"] = args.task_density
    if overrides:
        ENV_CONFIG.update(overrides)

    t0 = time.time()
    # interpret collision_penalty (allow 'none' to disable overwrite and use per-step rewards)
    cp_arg = args.collision_penalty
    try:
        if isinstance(cp_arg, str) and cp_arg.lower() in ("none", "null", "nil", "off"):
            collision_penalty_val = None
        else:
            collision_penalty_val = float(cp_arg)
    except Exception:
        collision_penalty_val = float(-1000.0)

    params, final_score, hist, device_type = train_priority_params_gpu(
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
        plot_png=args.plot_png,
        clip_step_norm=args.clip_step_norm,
        best_update_mode=args.best_update_mode,
        best_update_alpha=args.best_update_alpha,
        best_update_gap=args.best_update_gap,
        max_steps=args.max_steps,
        workers=args.workers,
    )

    print("==== Final evaluation (GPU ES) ====")
    print(f"PriorityParams: {params}")
    print(f"Average episode reward (@{args.eval_episodes} episodes): {final_score:.2f}")
    if args.plot_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(hist, label="reward_mean")
            plt.xlabel("iteration")
            plt.ylabel("mean reward")
            plt.grid(True)
            plt.legend()
            Path(args.plot_png).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot_png, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to: {args.plot_png}")
        except Exception as e:
            print(f"Plotting failed: {e}")
    elapsed = time.time() - t0
    print("==== Execution time ====")
    print(f"Device: {device_type}, Elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
