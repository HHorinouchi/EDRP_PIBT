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
    actions = detect_actions(env)
    task_assign = assign_task(env)
    return actions, task_assign

def assign_task(env):
    """Greedy task assignment based on travel distance to the next pick node."""

    # env.step() 外で環境の状態を変えないようリストをコピー
    current_tasklist = copy.deepcopy(env.current_tasklist)
    assigned_tasklist = copy.deepcopy(env.assigned_tasks)
    task_assign = []

    for i in range(env.agent_num):
        best_task = -1
        if len(assigned_tasklist[i]) == 0 and len(current_tasklist) > 0:
            shortest_path_length = float("inf")

            # 未割り当てタスクを評価し最も近いピック地点を選択
            for j in range(len(current_tasklist)):
                if env.assigned_list[j] == -1:
                    path_length = env.get_path_length(env.goal_array[i], current_tasklist[j][0])

                    if path_length is not None and shortest_path_length > path_length:
                        shortest_path_length = path_length
                        best_task = j

            # このローカルコピー内で選択済みタスクを消費済みとして扱う
            if best_task != -1:
                current_tasklist.pop(best_task)

        task_assign.append(best_task)

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
        assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
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
    occupied_nodes = []
    # 占有されるエッジを管理 List[ (from_node, to_node) ]
    occupied_edges = []
    current_priority = 0 # 行動が決定したエージェント数
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
        avail_actions_list.append(sorted_avail_actions)
        actions.append(None)

    row_avail_actions_list = copy.deepcopy(avail_actions_list)
    while current_priority < env.agent_num:
        # 優先度順にエージェントの行動を決定
        # 衝突しない可能な行動が見つからなかった場合、current_priorityを減らして一つ上の優先度のエージェントの行動を再選択する
        agent_idx, _ = priority_order[current_priority]
        # 最短経路上の行動を優先的に選択
        # ノードとエッジの占有状況を考慮して行動を決定
        avail_actions = avail_actions_list[agent_idx]
        action_selected = False
        for action in list(avail_actions):
            next_node = action
            needed_step = calculate_steps_to_node(env, agent_idx, next_node) # エージェントが次のノードに到達するまでに必要なステップ数
            conflict = False
            # ノードの占有状況を確認
            # occupied_nodesにnext_nodeが存在し、かつその占有ステップ数がneeded_stepと5tep以内の誤差しかない場合、衝突するので次の行動を評価
            # occupied_nodes: List[(node:int, step:float)]
            for occupied_node, occupied_step in occupied_nodes:
                if occupied_node == next_node and abs(occupied_step - needed_step) <= 5:
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
                occupied_nodes.append((next_node, needed_step))
                # エッジの占有状況を更新
                occupied_edges.append((env.current_start[agent_idx], next_node))
                avail_actions.remove(action)
                break
        if not action_selected:
            # 衝突が避けられない場合可能な行動を補填し、一つ上の優先度のエージェントの行動を再選択
            if current_priority > 0:
                # もしひとつ上の優先度エージェントが停止していたら、現在の優先度エージェントも停止
                if actions[priority_order[current_priority - 1][0]] == -1:
                        actions[agent_idx] = -1
                        current_priority += 1
                else:
                    avail_actions_list[agent_idx] = row_avail_actions_list[agent_idx]
                    current_priority -= 1
            else:
                # 最優先エージェントで衝突が避けられない場合、停止
                    actions[agent_idx] = -1
                    current_priority += 1
    return actions

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
