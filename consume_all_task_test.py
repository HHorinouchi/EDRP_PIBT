import gym
import numpy as np
import yaml
import time
from argparse import Namespace
from policy.my_policy2 import policy

total_task_completed = 0
total_collision_ended = 0
loopnum = 10
max_count = 0


def _tasks_remaining(environment) -> bool:
    current = getattr(environment, "current_tasklist", [])
    assigned = getattr(environment, "assigned_tasks", [])
    pending = getattr(environment, "alltasks_flat", [])
    next_idx = getattr(environment, "next_task_idx", len(pending))
    has_current = any(len(task) > 0 for task in current)
    has_assigned = any(len(task) > 0 for task in assigned)
    has_future = next_idx < len(pending)
    return has_current or has_assigned or has_future


for _ in range(loopnum):
    env = gym.make(
        "drp_env:drp-8agent_map_shibuya-v2",
        state_repre_flag="onehot_fov",
        task_flag=True,
    )
    n_obs = env.reset()
    print("action_space", env.action_space)
    print("observation_space", env.observation_space)

    print("obs", env.start_ori_array, env.goal_array)

    last_completion = 0
    step = 0
    while True:
        # env.render()
        # print(f"\nStep {step + 1}")
        env.render()
        actions, task, count = policy(n_obs, env)
        joint_action = {"pass": actions, "task": task}
        n_obs, reward, done, info = env.step(joint_action)
        step += 1

        completion = info.get("task_completion")
        if completion is not None and completion != last_completion:
            # print(f"Step {step + 1}: tasks completed {completion}")
            last_completion = completion

        # 現在の割り当てられていないタスクを表示
        # print(f"Unassigned tasks: {len(env.current_tasklist)}")
        # 各エージェントの行動を表示
        # for i in range(env.agent_num):
        #     print(f" Agent {i} action: {actions[i]}")
        #     print(f" Agent {i} start: {env.current_start[i]}, goal: {env.goal_array[i]}")
        #     print(f" Agent {i} avail actions: {env.get_avail_agent_actions(i, env.n_actions)[1]}")
        #     # タスクをアサインされている場合、そのタスクも表示
        #     if i < len(env.assigned_tasks):
        #         print(f" Agent {i} assigned task: {env.assigned_tasks[i]}")

        collision_detected = bool(info.get("collision"))
        remaining = _tasks_remaining(env)

        max_count = max(count, max_count)
        print(f"Total tasks completed: {last_completion}")

        if collision_detected:
            total_collision_ended += 1
            print("Episode ended due to collision.")
            break

        if not remaining:
            print("All tasks consumed. Episode finished.")
            break

        if all(done):
            if info.get("timeup"):
                print("Episode reached time limit with remaining tasks.")
            elif info.get("goal"):
                print("Episode ended after reaching goals.")
            else:
                print("Episode ended unexpectedly despite pending tasks.")
            break
    total_task_completed += last_completion
    env.close()

print(f"Final total tasks average completed: {total_task_completed / loopnum}")
print(f"Max count of while loop: {max_count}")
print(f"Episodes ended due to collision: {total_collision_ended} / {loopnum}")

# 1000回平均結果
# タスク振り分けを、ピックアップノードからの距離のみで取ったとき：16.75
# タスク振り分けを、ピックアップノードからの距離とドロップオフノードからの距離の両方を考慮したとき：22.31
