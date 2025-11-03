import gym
import numpy as np
import yaml
import time
from argparse import Namespace
from policy.my_policy import policy

step_limit = 1000
env=gym.make("drp_env:drp-2agent_map_shijo-v2", state_repre_flag = "onehot_fov", task_flag = True, time_limit=step_limit, speed=100)

total_task_completed = 0
loopnum = 1
for i in range(0, loopnum):
    n_obs=env.reset()
    print("action_space", env.action_space)
    print("observation_space", env.observation_space)

    print("obs", env.start_ori_array, env.goal_array)

    last_completion = 0


    for step in range(step_limit):
        # env.render()
        # print(f"\nStep {step + 1}")
        actions, task = policy(n_obs, env)
        joint_action = {"pass": actions, "task": task}
        n_obs, reward, done, info = env.step(joint_action)

        completion = info.get("task_completion")
        if completion is not None and completion != last_completion:
            # print(f"Step {step + 1}: tasks completed {completion}")
            last_completion = completion

        # 現在の割り当てられていないタスクを表示
            print(f"Unassigned tasks: {len(env.current_tasklist)}")
        # 各エージェントの行動を表示
        for i in range(env.agent_num):
            print(f" Agent {i} action: {actions[i]}")
            print(f" Agent {i} start: {env.current_start[i]}, goal: {env.goal_array[i]}")
            # タスクをアサインされている場合、そのタスクも表示
            if i < len(env.assigned_tasks):
                print(f" Agent {i} assigned task: {env.assigned_tasks[i]}")

        if all(done):
            break

        print(f"Total tasks completed: {last_completion}")
    total_task_completed += last_completion

print(f"Final total tasks average completed: {total_task_completed / loopnum}")

# 1000回平均結果
# タスク振り分けを、ピックアップノードからの距離のみで取ったとき：16.357
# タスク振り分けを、ピックアップノードからの距離とドロップオフノードからの距離の両方を考慮したとき：