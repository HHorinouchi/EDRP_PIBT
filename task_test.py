import gym
import numpy as np
import yaml
import time
from argparse import Namespace
from policy.my_policy import policy

total_task_completed = 0
total_collision_ended = 0
loopnum = 10
total_steps = 300
max_count = 0
for i in range(0, loopnum):
    env=gym.make("drp_env:drp-12agent_map_5x4-v2", state_repre_flag = "onehot_fov", task_flag = True)
    n_obs=env.reset()
    print("action_space", env.action_space)
    print("observation_space", env.observation_space)

    print("obs", env.start_ori_array, env.goal_array)

    last_completion = 0
    total_count = 0

    for step in range(total_steps):
        env.render()
        # print(f"\nStep {step + 1}")
        actions, task, count = policy(n_obs, env)
        joint_action = {"pass": actions, "task": task}
        n_obs, reward, done, info = env.step(joint_action)

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

        if all(done):
            if info.get("collision"):
                total_collision_ended += 1
                print("Episode ended due to collision.")
            elif info.get("timeup"):
                print("Episode ended due to timeup.")
            elif info.get("goal"):
                print("Episode ended by reaching goals.")
            else:
                print("Episode ended for unknown reason.")
            break
        print(f"Total tasks completed: {last_completion}")
        max_count = max(count, max_count)
    total_task_completed += last_completion
    env.close()

print(f"Final total tasks average completed: {total_task_completed / loopnum}")
print(f"Max count of while loop: {max_count}")
print(f"Episodes ended due to collision: {total_collision_ended} / {loopnum}")

# 1000回平均結果
# タスク振り分けを、ピックアップノードからの距離のみで取ったとき：16.75
# タスク振り分けを、ピックアップノードからの距離とドロップオフノードからの距離の両方を考慮したとき：22.31
