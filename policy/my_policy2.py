import copy
import math
import random
from typing import Dict, List, Optional, Set

import gym
import networkx as nx

### submission information ####
TEAM_NAME = "your_team_name"
# TEAM_NAME must be the same as the name registered on the DRP website
# (or the team name if participating as a team).
##############################


def policy(obs, env):
    actions, count = detect_actions(env)
    task_assign = assign_task(env)
    return actions, task_assign, count

def assign_task(env):
    """Greedy task assignment based on travel distance to the next pick node."""
    # Use original task indices to avoid index-shift bugs when popping from local copies.
    # We will select tasks by their index in env.current_tasklist (and env.assigned_list)
    task_assign = [-1] * env.agent_num

    if not hasattr(env, 'current_tasklist') or len(env.current_tasklist) == 0:
        print("No tasks available for assignment")
        return task_assign

    # available_indices are indices in env.current_tasklist that are unassigned
    available_indices = [idx for idx, a in enumerate(env.assigned_list) if a == -1]

    for i in range(env.agent_num):
        # skip if agent already has an assigned task
        if i < len(env.assigned_tasks) and len(env.assigned_tasks[i]) != 0:
            continue

        best_task_idx = -1
        best_dist = float('inf')

        for idx in list(available_indices):
            # idx indexes into env.current_tasklist
            try:
                task = env.current_tasklist[idx]
            except Exception:
                continue

            # task expected as [pick_node, drop_node]
            pick_node = task[0]
            drop_node = task[1]
            # measure distance from agent's current goal (or location) to pick_node
            # use env.get_path_length which may return None
            path_to_pick = env.get_path_length(env.current_start[i], pick_node)
            path_pick_to_drop = env.get_path_length(pick_node, drop_node)
            path_length = path_to_pick + path_pick_to_drop
            if path_length is None:
                print("Path length is None")
                continue
            if path_length < best_dist:
                best_dist = path_length
                best_task_idx = idx
    
        if best_task_idx == -1:
            print(f"No suitable task found for agent {i}")

        if best_task_idx != -1:
            task_assign[i] = best_task_idx
            # reserve this task locally so another agent won't pick it in this loop
            available_indices.remove(best_task_idx)

    return task_assign

def detect_actions(env):
    """
    PIBTをもとに、各エージェントの行動を決定する
    各エージェントの割り当てられたタスクに基づき、そのタスク完了にかかる最短経路で優先順位を決定する
    """
    actions = []
    shortest_path_distances = []
    if env.goal_array is None:
        return [-1]*env.agent_num
    # 各エージェントの最短経路距離を計算
    for i in range(env.agent_num):
        assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else assign_task(env)[i]
        has_task = bool(assigned_task)
        current_goal = env.goal_array[i] if i < len(env.goal_array) else None
        path_length = float("inf")
        if has_task and len(assigned_task) >= 2:
            pick_node = assigned_task[0]
            drop_node = assigned_task[1]
            if current_goal is not None and current_goal == drop_node:
                path_length = calculate_goal_path_length(env, i, assigned_task)
            elif current_goal is not None and current_goal == pick_node:
                path_length = calculate_task_path_length(env, i, assigned_task)
        shortest_path_distances.append((i, path_length))

    # 最短経路距離に基づき優先順位を決定（距離が短いほど高優先度）
    priority_order = sorted(shortest_path_distances, key=lambda x: x[1])

    # 優先度順に行動を決定。高優先度のエージェントと衝突しない行動を選択する
    # 基本的には最短経路上の行動を選択するが、衝突する場合は他の行動を選択
    # i番目の可能な行動の中でそれ以上の優先度と衝突しない行動がなかった場合、一つ上の優先度の行動を再選択する
    # 占有されるノードを管理（何ステップ後に占有されるかも一緒に）List[(node, step)]
    occupied_nodes = [(None, None)] * env.agent_num
    # 占有されるエッジを管理 List[ (from_node, to_node) ]
    occupied_edges = [(None, None)] * env.agent_num
    current_priority = 0 # 行動が決定したエージェント数
    most_high_priority = 0  # 最も高い優先度のエージェントインデックス
    max_wait = 4  # 最大停止ステップ数
    # 各エージェントが可能な行動をリスト化し、行動を変更するときに同じ行動を繰り返さないようにする
    avail_actions_list = []
    for i in range(env.agent_num):
        _, avail_actions = env.get_avail_agent_actions(i, env.n_actions)
        # avail_actions_listをその行動を選択したときの割り当てられたタスクを完了するまでの最短経路距離でソート
        path_length_list = []
        for action in avail_actions:
            assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
            has_task = bool(assigned_task)
            current_goal = env.goal_array[i] if i < len(env.goal_array) else None
            next_node = action
            if has_task and len(assigned_task) >= 2:
                pick_node = assigned_task[0]
                drop_node = assigned_task[1]
                if current_goal is not None and current_goal == drop_node:
                    path = env.get_path_length(next_node, drop_node)
                elif current_goal is not None and current_goal == pick_node:
                    path = env.get_path_length(next_node, pick_node)
                else:
                    path = None
            else:
                path = None
            path_length = path if path is not None else float("inf")
            path_length_list.append((action, path_length))

        sorted_avail_actions = [action for action, _ in sorted(path_length_list, key=lambda x: x[1])]
        for i in range(1, 4):
            # -1, -2, ... を後ろに回す
            # actionが-の場合、停止するステップ数が1,2,3,...と増えることを意味するため
            sorted_avail_actions.append(-i)
        avail_actions_list.append(sorted_avail_actions)
        actions.append(None)

    count = 0
    row_avail_actions_list = copy.deepcopy(avail_actions_list)
    while current_priority < env.agent_num:
        count += 1
        # 優先度順にエージェントの行動を決定
        # 衝突しない可能な行動が見つからなかった場合、current_priorityを減らして一つ上の優先度のエージェントの行動を再選択する
        agent_idx, _ = priority_order[current_priority]
        # 最短経路上の行動を優先的に選択
        # ノードとエッジの占有状況を考慮して行動を決定
        avail_actions = avail_actions_list[agent_idx]
        action_selected = False
        for action in list(avail_actions):
            if len(row_avail_actions_list[agent_idx]) == max_wait+1: # これは、エッジ上にいるとき
                next_node = row_avail_actions_list[agent_idx][0]  # エッジ上にいるときは進行方向のノードにしか行けない
            elif action < 0: # ノード上にいて、行き先がなく停止を選択するとき,next_nodeは現在ノード
                next_node = env.current_start[agent_idx]
            else:
                next_node = action
            needed_step = calculate_steps_to_node(env, agent_idx, next_node) # エージェントが次のノードに到達するまでに必要なステップ数
            conflict = False
            # ノードの占有状況を確認
            # occupied_nodesにnext_nodeが存在し、かつその占有ステップ数がneeded_stepと5tep以内の誤差しかない場合、衝突するので次の行動を評価
            # occupied_nodes: List[(node:int, step:float)]
            for occupied_node, occupied_step in occupied_nodes:
                if occupied_node == next_node and (abs(occupied_step - (needed_step + max(0, -action))) <= 3 or occupied_step == -1):
                    # 衝突が発生した場合、次の行動を評価
                    conflict = True
                    avail_actions.remove(action)
                    break
            for start, end in occupied_edges:
                # 逆向きにエッジが占有されている場合に、衝突するので次の行動を評価(進む向きが一緒なら問題ない)
                if end == env.current_start[agent_idx] and start == next_node and conflict == False:
                    conflict = True
                    avail_actions.remove(action)
                    break
            if not conflict:
                actions[agent_idx] = action
                action_selected = True
                current_priority += 1
                # ノードの占有状況を更新
                occupied_nodes[agent_idx] = (next_node, needed_step + max(0, -action))  # 停止の場合、needed_stepに停止ステップ数を加える
                # エッジの占有状況を更新
                occupied_edges[agent_idx] = (env.current_start[agent_idx], next_node)
                avail_actions.remove(action)
                break
        if not action_selected:
            # 衝突が避けられない場合可能な行動を補填し、一つ上の優先度のエージェントの行動を再選択

            # 最上位優先度エージェントの場合、停止を選択
            if current_priority <= most_high_priority:
                # 最優先エージェントで可能な行動がない場合、停止を選択
                actions[agent_idx] = -3
                current_priority += 1
                most_high_priority += 1
                # ノードの占有状況を更新
                occupied_nodes[agent_idx] = (env.current_start[agent_idx], needed_step + 3)
                # エッジの占有状況を更新
                if len(row_avail_actions_list[agent_idx]) == max_wait+1:
                    occupied_edges[agent_idx] = (env.current_start[agent_idx], row_avail_actions_list[agent_idx][0])
            else:
                # 現在のagent_idxの可能行動をリセット
                avail_actions_list[agent_idx] = copy.deepcopy(row_avail_actions_list[agent_idx])
                # 一つ上の優先度のエージェントの行動を再
                prev_agent_idx, _ = priority_order[current_priority - 1]
                current_priority -= 1
                # 一つ上の優先度のエージェントが占有しているノードとエッジの占有状況を解除
                # occupied_nodesとoccupied_edgesから、最も後に追加された該当エージェントの占有状況を削除
                occupied_nodes[prev_agent_idx] = (None, None)
                occupied_edges[prev_agent_idx] = (None, None)
                continue
    
    return actions, count

# 現在ノードとタスクスタートノード、タスクゴールノードを用いた最短経路距離を計算（未ピックアップ）
def calculate_task_path_length(env, agent_idx, assigned_task):
    if not assigned_task or len(assigned_task) < 2:
        return float("inf")

    pick_node = assigned_task[0]
    drop_node = assigned_task[1]
    base_node = env.current_start[agent_idx] if env.current_start[agent_idx] is not None else env.goal_array[agent_idx]

    path_length_to_pick = env.get_path_length(base_node, pick_node)
    path_pick_to_drop = env.get_path_length(pick_node, drop_node)

    if path_length_to_pick is None or path_pick_to_drop is None:
        return float("inf")

    return path_length_to_pick + path_pick_to_drop

# 現在ノードとタスクゴールノードを用いた最短経路距離を計算（ピックアップ済み）
def calculate_goal_path_length(env, agent_idx, assigned_task):
    if not assigned_task or len(assigned_task) < 2:
        return float("inf")

    drop_node = assigned_task[1]
    base_node = env.current_start[agent_idx] if env.current_start[agent_idx] is not None else env.goal_array[agent_idx]

    path_length = env.get_path_length(base_node, drop_node)
    if path_length is None:
        return float("inf")

    return path_length

def calculate_steps_to_node(env, agent_idx, target_node):
    agent_x, agent_y = env.obs[agent_idx][:2]
    target_x, target_y = env.pos[target_node]

    euclidean_distance = math.hypot(target_x - agent_x, target_y - agent_y)
    if env.speed <= 0:
        return float("inf")

    steps_needed = math.ceil(euclidean_distance / env.speed)
    return steps_needed
