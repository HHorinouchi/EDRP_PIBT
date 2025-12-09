# MAPD×PIBT パラメータ学習スクリプト
# - my_policy.detect_actions の優先順位計算に使う係数を進化戦略で最適化する

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import os
import csv

# ルートディレクトリ（リポジトリ直下）を sys.path に追加してローカルモジュールを解決
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from drp_env.drp_env import DrpEnv  # noqa: E402
from drp_env.EE_map import UNREAL_MAP  # noqa: E402
from policy.my_policy import (  # noqa: E402
    PriorityParams,
    get_priority_params,
    policy as rollout_policy,
    save_priority_params,
    set_priority_params,
)

# DrpEnv の基本設定。train/eval で同じ環境を用いる。
ENV_CONFIG = dict(
    agent_num=4,
    speed=1.0,
    start_ori_array=[],
    goal_array=[],
    visu_delay=0.0,
    state_repre_flag="onehot",
    time_limit=300,
    collision="bounceback",
    task_flag=True,
    reward_list={"goal": 100, "collision": -10, "wait": -10, "move": -1},
    map_name="map_3x3",
    task_density=1.0,
)


def make_env() -> DrpEnv:
    return DrpEnv(**ENV_CONFIG)


# Domain randomization helpers
MAP_POOL = list(UNREAL_MAP)


def get_map_node_count(map_name: str) -> int:
    """Count nodes in the specified map by reading its node.csv."""
    try:
        node_csv = ROOT_DIR / "drp_env" / "map" / map_name / "node.csv"
        with node_csv.open("r") as f:
            # subtract header
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def sample_env_config(rng: np.random.Generator) -> dict:
    """Sample an environment configuration to diversify training scenarios."""
    map_name = rng.choice(MAP_POOL)
    n_nodes = get_map_node_count(map_name) or 9  # sensible fallback

    # Ensure agent_num never exceeds number of nodes (MapMake hard-exits otherwise)
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
        reward_list={"goal": 100, "collision": -10, "wait": -10, "move": -1},
        map_name=map_name,
    )


def make_env_from_config(cfg: dict) -> DrpEnv:
    return DrpEnv(**cfg)


def params_to_vector(params: PriorityParams) -> np.ndarray:
    """PriorityParams -> ndarray の順序はここで一元管理する。"""
    return np.array(
        [
            params.goal_weight,
            params.pick_weight,
            params.drop_weight,
            params.idle_bias,
            params.idle_penalty,
            # task assignment weights
            params.assign_pick_weight,
            params.assign_drop_weight,
            params.assign_idle_bias,
            # global/system-level weights
            params.congestion_weight,
            params.load_balance_weight,
        ],
        dtype=np.float32,
    )


def vector_to_params(vec: np.ndarray) -> PriorityParams:
    """学習時の探索ベクトルを PriorityParams に戻す。"""
    raw_vec = np.asarray(vec, dtype=np.float32)
    raw_vec = np.where(np.isfinite(raw_vec), raw_vec, 0.0)

    # Pad to 10 dims if shorter (backward compatibility)
    safe_len = 10
    safe_vec = np.zeros(safe_len, dtype=np.float32)
    safe_vec[: min(raw_vec.size, safe_len)] = raw_vec[: min(raw_vec.size, safe_len)]

    # 距離重みは非負・過度な値をクリップ
    goal_w, pick_w, drop_w = np.clip(safe_vec[:3], 0.0, 10.0)
    idle_bias = float(np.clip(safe_vec[3], -5000.0, 5000.0))
    idle_penalty = float(np.clip(safe_vec[4], 0.0, 5000.0))

    assign_pick = float(np.clip(safe_vec[5], 0.0, 10.0))
    assign_drop = float(np.clip(safe_vec[6], 0.0, 10.0))
    assign_idle_bias = float(np.clip(safe_vec[7], -5000.0, 5000.0))

    # global/system-level
    congestion_w = float(np.clip(safe_vec[8], 0.0, 10.0))
    load_balance_w = float(np.clip(safe_vec[9], -10.0, 10.0))

    return PriorityParams(
        goal_weight=float(goal_w),
        pick_weight=float(pick_w),
        drop_weight=float(drop_w),
        idle_bias=idle_bias,
        idle_penalty=idle_penalty,
        # task assignment
        assign_pick_weight=assign_pick,
        assign_drop_weight=assign_drop,
        assign_idle_bias=assign_idle_bias,
        # global/system-level
        congestion_weight=congestion_w,
        load_balance_weight=load_balance_w,
    )


def rollout(
    params: PriorityParams,
    episodes: int = 1,
    max_steps: int = 500,
    domain_randomize: bool = False,
    rng: Optional[np.random.Generator] = None,
    collision_penalty: Optional[float] = None,
) -> float:
    """
    与えられた優先順位パラメータで複数エピソードを実行し、平均エピソード報酬を返す。
    domain_randomize=True のとき、各エピソードで環境構成（マップ・ノード/エージェント数・速度等）をランダム化する。
    """
    set_priority_params(params)
    rewards = []
    for _ in range(episodes):
        if domain_randomize:
            rgen = rng or np.random.default_rng()
            cfg = sample_env_config(rgen)
            env = make_env_from_config(cfg)
        else:
            env = make_env()

        obs = env.reset()
        done_flags = [False for _ in range(env.agent_num)]
        total_reward = 0.0
        steps = 0
        while not all(done_flags) and steps < max_steps:
            actions, task_assign = rollout_policy(obs, env)
            obs, step_rewards, done_flags, info = env.step({"pass": actions, "task": task_assign})
            total_reward += float(sum(step_rewards))
            # End episode immediately on collision, regardless of env termination mode
            if isinstance(info, dict) and info.get("collision", False):
                if collision_penalty is not None:
                    total_reward = float(collision_penalty)
                done_flags = [True for _ in range(env.agent_num)]
                break
            steps += 1
        env.close()
        rewards.append(total_reward)
    return float(np.mean(rewards))


def train_priority_params(
    iterations: int = 40,
    population: int = 8,
    sigma: float = 0.2,
    lr: float = 0.05,
    episodes_per_candidate: int = 2,
    seed: int = 0,
    domain_randomize: bool = False,
    collision_penalty: Optional[float] = None,
    log_csv: Optional[str] = None,
    clip_step_norm: float = 0.0,
    best_update_mode: str = "max",
    best_update_alpha: float = 0.1,
    best_update_gap: float = 0.0,
) -> Tuple[PriorityParams, float, list]:
    """
    OpenAI-ES 風の進化戦略で優先順位パラメータを最適化する。
    """
    rng = np.random.default_rng(seed)
    mean_vector = params_to_vector(get_priority_params()).astype(np.float32)
    best_reward = rollout(
        vector_to_params(mean_vector),
        episodes=episodes_per_candidate,
        domain_randomize=domain_randomize,
        rng=rng,
        collision_penalty=collision_penalty,
    )
    best_vector = mean_vector.copy()
    reward_mean_history = []
    best_reward_ema = float(best_reward)

    for it in range(iterations):
        noise = rng.standard_normal((population, mean_vector.size), dtype=np.float32)
        rewards = np.zeros(population, dtype=np.float32)
        candidates = mean_vector + sigma * noise

        for idx in range(population):
            candidate_params = vector_to_params(candidates[idx])
            rewards[idx] = rollout(
                candidate_params,
                episodes=episodes_per_candidate,
                domain_randomize=domain_randomize,
                rng=rng,
                collision_penalty=collision_penalty,
            )

        # 報酬を正規化して更新方向を計算
        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())
        reward_max = float(rewards.max())
        normalized = (rewards - reward_mean) / (reward_std + 1e-8)
        step = (lr / (population * sigma)) * noise.T @ normalized
        # 進化方向のノルムをクリップ（任意）
        if clip_step_norm and clip_step_norm > 0.0:
            step_norm = float(np.linalg.norm(step))
            if step_norm > clip_step_norm:
                step = step * (clip_step_norm / (step_norm + 1e-8))
        # ログ: CSV に現在の mean_vector と統計量を出力
        try:
            if log_csv:
                os.makedirs(str(Path(log_csv).parent), exist_ok=True)
                write_header = not os.path.exists(log_csv)
                with open(log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        header = ["iter", "reward_mean", "reward_std", "reward_max"] + [f"v{i}" for i in range(mean_vector.size)]
                        w.writerow(header)
                    row = [it + 1, reward_mean, reward_std, reward_max] + list(map(float, mean_vector.tolist()))
                    w.writerow(row)
        except Exception as _e:
            pass
        # mean_vector を更新
        mean_vector = mean_vector + step

        # 最良個体を記録（更新条件はオプション）
        best_idx = int(np.argmax(rewards))
        current_best = float(rewards[best_idx])
        # EMA を更新（moving_avg 用）
        best_reward_ema = (best_update_alpha * reward_mean) + ((1.0 - best_update_alpha) * best_reward_ema)
        if best_update_mode == "mean_gap":
            threshold = reward_mean + best_update_gap
        elif best_update_mode == "moving_avg":
            threshold = best_reward_ema + best_update_gap
        else:  # "max"
            threshold = best_reward
        if current_best > threshold:
            best_reward = current_best
            best_vector = candidates[best_idx].copy()

        # 進捗出力
        print(
            f"[Iter {it+1:03d}] reward_mean={reward_mean:.2f} "
            f"reward_std={reward_std:.2f} reward_max={reward_max:.2f} eps={episodes_per_candidate}"
        )
        reward_mean_history.append(reward_mean)

    best_params = vector_to_params(best_vector)
    set_priority_params(best_params)
    save_priority_params(best_params)
    return best_params, best_reward, reward_mean_history


def main():
    parser = argparse.ArgumentParser(description="Train my_policy priority parameters.")
    parser.add_argument("--iterations", type=int, default=40, help="ES iterations")
    parser.add_argument("--population", type=int, default=8, help="Samples per iteration")
    parser.add_argument("--sigma", type=float, default=0.2, help="Noise scale for ES")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for ES")
    parser.add_argument(
        "--episodes-per-candidate",
        type=int,
        default=2,
        help="Rollouts per sampled parameter set during training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Rollouts for the final evaluation of the best parameters",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="Enable domain randomization (multi-environment training). Disabled by default for single-map specialization.",
    )
    # Single-map overrides (applied to ENV_CONFIG)
    parser.add_argument("--map-name", type=str, default=None, help="Map name, e.g., map_3x3")
    parser.add_argument("--agent-num", type=int, default=None, help="Number of agents")
    parser.add_argument("--speed", type=float, default=None, help="Agent speed")
    parser.add_argument("--time-limit", type=int, default=None, help="Episode time limit")
    parser.add_argument(
        "--collision",
        type=str,
        choices=["bounceback", "terminated"],
        default=None,
        help="Collision handling mode",
    )
    parser.add_argument("--task-density", type=float, default=None, help="Average tasks per step (Poisson mean)")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate current saved parameters",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=-1000.0,
        help="Reward to assign immediately when a collision occurs (episode ends).",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="policy/train/train_log.csv",
        help="Path to CSV file to log iteration stats and parameter vector.",
    )
    parser.add_argument(
        "--plot-png",
        type=str,
        default=None,
        help="Optional path to save a reward_mean curve as PNG after training.",
    )
    parser.add_argument(
        "--clip-step-norm",
        type=float,
        default=0.0,
        help="If > 0, clip ES update step L2 norm to this value for stability.",
    )
    parser.add_argument(
        "--best-update-mode",
        type=str,
        choices=["max", "mean_gap", "moving_avg"],
        default="max",
        help="Criterion to accept new best params: 'max' (default), 'mean_gap' (compare to reward_mean), 'moving_avg' (EMA baseline).",
    )
    parser.add_argument(
        "--best-update-alpha",
        type=float,
        default=0.1,
        help="EMA alpha for moving_avg mode (0-1). Larger = faster tracking.",
    )
    parser.add_argument(
        "--best-update-gap",
        type=float,
        default=0.0,
        help="Additional margin over baseline (mean or EMA) to accept best params.",
    )
    args = parser.parse_args()

    # Apply single-map overrides to ENV_CONFIG (in-place) so make_env() uses them.
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

    if args.eval_only:
        params = get_priority_params()
        print("Loaded saved parameters; skipping training.")
    else:
        params, best_reward, hist_means = train_priority_params(
            iterations=args.iterations,
            population=args.population,
            sigma=args.sigma,
            lr=args.lr,
            episodes_per_candidate=args.episodes_per_candidate,
            seed=args.seed,
            domain_randomize=args.domain_randomize,
            collision_penalty=args.collision_penalty,
            log_csv=args.log_csv,
            clip_step_norm=args.clip_step_norm,
            best_update_mode=args.best_update_mode,
            best_update_alpha=args.best_update_alpha,
            best_update_gap=args.best_update_gap,
        )
        print(f"Training best reward: {best_reward:.2f}")
        if args.plot_png:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(hist_means, label="reward_mean")
                plt.xlabel("iteration")
                plt.ylabel("mean reward")
                plt.grid(True)
                plt.legend()
                Path(args.plot_png).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(args.plot_png, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Plotting failed: {e}")

    final_score = rollout(params, episodes=args.eval_episodes, domain_randomize=args.domain_randomize, collision_penalty=args.collision_penalty)
    print("==== Final evaluation ====")
    print(f"PriorityParams: {params}")
    print(f"Average episode reward (@{args.eval_episodes} episodes): {final_score:.2f}")


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    _elapsed = time.time() - _t0
    print("==== Execution time ====")
    print(f"Elapsed: {_elapsed:.2f} seconds")
