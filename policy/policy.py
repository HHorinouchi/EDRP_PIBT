import gym
import random

### submission information ####
TEAM_NAME = "your_team_name" 
#TEAM_NAME must be the same as the name registered on the DRP website 
#(or the team name if participating as a team).
##############################

import gym
import copy
import random

def policy(obs, env):
    actions = []
    for i in range(env.agent_num):
        _, avail_actions = env.get_avail_agent_actions(i, env.n_actions)
        actions.append(random.choice(avail_actions))

    task_assign = assign_task(env)
    return actions, task_assign

def assign_task(env):
    current_tasklist = copy.deepcopy(env.current_tasklist)
    assigned_tasklist = copy.deepcopy(env.assigned_tasks)
    task_assign = []

    for i in range(env.agent_num):
        best_task = -1
        if len(assigned_tasklist[i]) == 0 and len(current_tasklist) > 0:
            shortest_path_length = float('inf')

            for j in range(len(current_tasklist)):
                if env.assigned_list[j] == -1:    
                    path_length = env.get_path_length(env.goal_array[i], current_tasklist[j][0])

                    if shortest_path_length > path_length:
                        shortest_path_length = path_length
                        best_task = j

            current_tasklist.pop(best_task)

        task_assign.append(best_task)
                
    return task_assign