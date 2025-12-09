#!/usr/bin/env python3
"""
並列実行版の task_test。
このスクリプトは複数のエピソードを並列（マルチプロセス）で実行します。
注意:
 - 環境（gym env）は各ワーカー内で作成されます。
 - 元の `policy.my_policy.policy` を再利用します（policy は CPU ベースのまま）。
 - GPU を明示的に使う処理は環境側や policy 側が対応していないため行っていません。
   （多くの場合 env.step は CPU で動くため、GPU が直接役立つ場面は限定的です。）

使い方の例:
  python task_test_by_gpu.py --episodes 100 --workers 8

コマンドラインオプション:
  --episodes: 実行するエピソード数 (デフォルト 1000)
  --workers: 並列ワーカー数 (デフォルト: CPU コア数)
  --max-steps: 各エピソード内の最大ステップ数 (デフォルト 300)
  --env-id: 使用する gym 環境 ID (デフォルト は task_test と同じ)

戻り値: 各エピソードの完了タスク数と衝突による終了回数を集計して表示。

このスクリプトはまずシンプルなマルチプロセス並列化を提供します。
将来的に policy の主要な数値計算を PyTorch に移植して一括で GPU に載せることで
更なる加速が可能です。
"""

import argparse
import multiprocessing as mp
import os
import time
from typing import Tuple

import gym

# policy は既存実装を流用
from policy.my_policy2 import policy


def run_episode(_idx: int, env_id: str, max_steps: int) -> Tuple[int, bool]:
    """ワーカーが一つのエピソードを走らせる関数。

    戻り値: (tasks_completed, collision_ended)
    """
    try:
        env = gym.make(env_id, state_repre_flag="onehot_fov", task_flag=True)
    except Exception as e:
        # 環境が作れない場合は 0 を返す
        print(f"Failed to create env in worker: {e}")
        return 0, False

    n_obs = env.reset()
    last_completion = 0

    for step in range(max_steps):
        actions, task, count = policy(n_obs, env)
        joint_action = {"pass": actions, "task": task}
        n_obs, reward, done, info = env.step(joint_action)

        completion = info.get("task_completion")
        if completion is not None and completion != last_completion:
            last_completion = completion

        if all(done):
            # collision が True なら衝突終了
            collision = bool(info.get("collision", False))
            env.close()
            return last_completion, collision

    # max_steps 経過
    env.close()
    return last_completion, False


def worker_init():
    # 各ワーカーで必要なら初期化処理をここに置く（例: GPU デバイス設定など）
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--env-id", type=str, default="drp_env:drp-8agent_map_shibuya-v2")
    args = parser.parse_args()

    print(f"Episodes: {args.episodes}, workers: {args.workers}, env: {args.env_id}")

    # GPU 情報表示（存在するなら）
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA available: device_count={torch.cuda.device_count()}, current_device={torch.cuda.current_device()}")
        else:
            print("CUDA not available: running on CPU")
    except Exception:
        print("PyTorch not installed or failed to import; skipping GPU check")

    total_tasks = 0
    total_collision = 0
    # 衝突しなかったエピソードの合計タスク数とカウント
    total_tasks_no_collision = 0
    count_no_collision = 0

    # マルチプロセスプールでエピソードを平行実行
    # Linux ではデフォルト spawn は 'fork'。問題があれば 'spawn' に変更してください。
    with mp.Pool(processes=args.workers, initializer=worker_init) as pool:
        # pool.starmap を使って各エピソードを実行
        jobs = [ (i, args.env_id, args.max_steps) for i in range(args.episodes) ]
        # imap_unordered で結果を逐次集計
        for tasks_completed, collision in pool.starmap(run_episode, jobs):
            total_tasks += tasks_completed
            if collision:
                total_collision += 1
            else:
                total_tasks_no_collision += tasks_completed
                count_no_collision += 1

    avg_completed = float(total_tasks) / float(args.episodes) if args.episodes > 0 else 0.0
    print(f"Final total tasks average completed: {avg_completed}")
    print(f"Episodes ended due to collision: {total_collision} / {args.episodes}")
    if count_no_collision > 0:
        avg_no_collision = float(total_tasks_no_collision) / float(count_no_collision)
    else:
        avg_no_collision = 0.0
    print(f"Average tasks completed (no collision): {avg_no_collision} (count: {count_no_collision})")


if __name__ == "__main__":
    main()

# 学習済みモデル：
# Final total tasks average completed: 9.0647
# Episodes ended due to collision: 443 / 10000
# Average tasks completed (no collision): 9.154755676467511 (count: 9557)

# 学習なしモデル：
# Final total tasks average completed: 8.1365
# Episodes ended due to collision: 3860 / 10000
# Average tasks completed (no collision): 10.640553745928338 (count: 6140)

# 学習済みモデル：drp_env:drp-5agent_map_3x3-v2
# Final total tasks average completed: 19.8302
# Episodes ended due to collision: 702 / 10000
# Average tasks completed (no collision): 20.246827274682726 (count: 9298)

# 学習なしモデル：drp_env:drp-5agent_map_3x3-v2
# Final total tasks average completed: 18.3569
# Episodes ended due to collision: 5788 / 10000
# Average tasks completed (no collision): 27.993827160493826 (count: 4212)