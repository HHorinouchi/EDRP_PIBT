import gym
import numpy as np
import yaml
import time
from argparse import Namespace
from policy.my_policy import policy

env=gym.make("drp_env:drp-2agent_map_3x3-v2", state_repre_flag = "onehot_fov", task_flag = True)

n_obs=env.reset()
#print("action_space", env.action_space)
#print("observation_space", env.observation_space)

#print("obs", env.start_ori_array, env.goal_array)

last_completion = 0

for step in range(50):
    # env.render()

    actions, task = policy(n_obs, env)
    joint_action = {"pass": actions, "task": task}
    n_obs, reward, done, info = env.step(joint_action)

    completion = info.get("task_completion")
    if completion is not None and completion != last_completion:
        print(f"Step {step + 1}: tasks completed {completion}")
        last_completion = completion
    
    # 現在の割り当てられていないタスクを表示
    print(f"Unassigned tasks: {len(env.current_tasklist)}")
    # 各エージェントの行動を表示
    for i in range(env.agent_num):
        print(f" Agent {i} action: {actions[i]}")

    if all(done):
        break

print(f"Total tasks completed: {last_completion}")

#"""