import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import gym
import networkx as nx

### submission information ####
TEAM_NAME = "your_team_name"
# TEAM_NAME must be the same as the name registered on the DRP website
# (or the team name if participating as a team).
##############################


@dataclass
class PriorityParams:
    """
    Parameters for detect_actions priority calculation.

    goal_weight:     Weight for distance to drop when currently heading to drop.
    pick_weight:     Weight for distance to pick when still before pickup.
    drop_weight:     Weight for distance pick->drop when still before pickup.
    idle_bias:       Additive bias applied only when the agent has no task.
    idle_penalty:    Base score given to agents with no task (keeps them last).
    """

    goal_weight: float = 1.0
    pick_weight: float = 1.0
    drop_weight: float = 1.0
    idle_bias: float = 0.0
    idle_penalty: float = 1000.0

    # Task assignment scoring weights
    assign_pick_weight: float = 1.0
    assign_drop_weight: float = 0.0
    assign_idle_bias: float = 0.0

    # Global/system-level weights
    congestion_weight: float = 0.0  # penalize local congestion around an agent
    load_balance_weight: float = 0.0  # bias assignment based on system unassigned ratio

    @classmethod
    def from_dict(cls, data: Dict) -> "PriorityParams":
        return cls(
            goal_weight=float(data.get("goal_weight", cls.goal_weight)),
            pick_weight=float(data.get("pick_weight", cls.pick_weight)),
            drop_weight=float(data.get("drop_weight", cls.drop_weight)),
            idle_bias=float(data.get("idle_bias", cls.idle_bias)),
            idle_penalty=float(data.get("idle_penalty", cls.idle_penalty)),
            assign_pick_weight=float(data.get("assign_pick_weight", cls.assign_pick_weight)),
            assign_drop_weight=float(data.get("assign_drop_weight", cls.assign_drop_weight)),
            assign_idle_bias=float(data.get("assign_idle_bias", cls.assign_idle_bias)),
            congestion_weight=float(data.get("congestion_weight", cls.congestion_weight)),
            load_balance_weight=float(data.get("load_balance_weight", cls.load_balance_weight)),
        )


_PRIORITY_PARAMS_PATH = Path(__file__).with_name("priority_params.json")
_PRIORITY_PARAMS = None


def _load_priority_params() -> PriorityParams:
    if not _PRIORITY_PARAMS_PATH.exists():
        return PriorityParams()
    try:
        data = json.loads(_PRIORITY_PARAMS_PATH.read_text())
        return PriorityParams.from_dict(data)
    except Exception:
        # Fall back to defaults if the file is malformed
        return PriorityParams()


def get_priority_params() -> PriorityParams:
    """Return the in-memory priority parameters (defaults if unset)."""
    return _PRIORITY_PARAMS or PriorityParams()


def set_priority_params(params: PriorityParams) -> None:
    """Update the priority parameters used inside detect_actions."""
    global _PRIORITY_PARAMS
    _PRIORITY_PARAMS = params


def save_priority_params(params: PriorityParams) -> None:
    """Persist priority parameters so that future runs can reuse them."""
    payload = json.dumps(asdict(params), indent=2)
    _PRIORITY_PARAMS_PATH.write_text(payload)


# Load parameters once on import so that training outputs are picked up automatically.
set_priority_params(_load_priority_params())


def policy(obs, env):
    actions = detect_actions(env)
    task_assign = assign_task(env)
    return actions, task_assign

def assign_task(env):
    """Task assignment with tunable priority parameters."""
    task_assign = [-1] * env.agent_num

    if not hasattr(env, "current_tasklist") or len(env.current_tasklist) == 0:
        return task_assign

    params = get_priority_params()

    # available_indices are indices in env.current_tasklist that are unassigned
    available_indices = [idx for idx, a in enumerate(env.assigned_list) if a == -1]

    # system-level unassigned task ratio (0.0 if no tasks)
    total_slots = len(env.assigned_list) if hasattr(env, "assigned_list") else 0
    unassigned_cnt = sum(1 for a in env.assigned_list if a == -1) if total_slots > 0 else 0
    unassigned_ratio = float(unassigned_cnt) / float(total_slots) if total_slots > 0 else 0.0

    for i in range(env.agent_num):
        # skip if agent already has an assigned task
        if i < len(env.assigned_tasks) and len(env.assigned_tasks[i]) != 0:
            continue

        best_task_idx = -1
        best_score = float("inf")

        base_node = (
            env.current_start[i]
            if i < len(env.current_start) and env.current_start[i] is not None
            else env.goal_array[i]
        )

        for idx in list(available_indices):
            # idx indexes into env.current_tasklist
            try:
                task = env.current_tasklist[idx]
            except Exception:
                continue

            # task expected as [pick_node, drop_node, (optional deadline)]
            if len(task) < 2:
                continue

            pick_node = task[0]
            drop_node = task[1]

            # distances for scoring
            dist_to_pick = env.get_path_length(base_node, pick_node)
            dist_pick_to_drop = env.get_path_length(pick_node, drop_node)

            if dist_to_pick is None:
                continue
            if dist_pick_to_drop is None:
                dist_pick_to_drop = float("inf")

            score = (
                params.assign_pick_weight * float(dist_to_pick)
                + params.assign_drop_weight * float(dist_pick_to_drop)
                + params.assign_idle_bias
                + params.load_balance_weight * float(unassigned_ratio)
            )

            if score < best_score:
                best_score = score
                best_task_idx = idx

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
    if env.goal_array is None:
        return [-1]*env.agent_num
    # 各エージェントの最短経路距離を計算
    priority_scores = []
    for i in range(env.agent_num):
        assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
        score = _priority_score(env, i, assigned_task)
        priority_scores.append((i, score))

    # 最短経路距離（または重みづけ距離）に基づき優先順位を決定（距離が短いほど高優先度）
    priority_order = sorted(priority_scores, key=lambda x: x[1])

    # 優先度順に行動を決定。高優先度のエージェントと衝突しない行動を選択する
    # 基本的には最短経路上の行動を選択するが、衝突する場合は他の行動を選択
    # i番目の可能な行動の中でそれ以上の優先度と衝突しない行動がなかった場合、一つ上の優先度の行動を再選択する
    # 占有されるノードを管理（何ステップ後に占有されるかも一緒に）List[(node, step)]
    occupied_nodes = [(None, None)] * env.agent_num
    # 占有されるエッジを管理 List[ (from_node, to_node) ]
    occupied_edges = [(None, None)] * env.agent_num
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
    best_priority = 0
    while current_priority < env.agent_num:
        # 優先度順にエージェントの行動を決定
        # 衝突しない可能な行動が見つからなかった場合、current_priorityを減らして一つ上の優先度のエージェントの行動を再選択する
        agent_idx, _ = priority_order[current_priority]
        # 最短経路上の行動を優先的に選択
        # ノードとエッジの占有状況を考慮して行動を決定
        avail_actions = avail_actions_list[agent_idx]
        action_selected = False
        for action in list(avail_actions):
            if len(row_avail_actions_list[agent_idx]) == 2: # これは、エッジ上にいるとき
                next_node = row_avail_actions_list[agent_idx][0]  # エッジ上にいるときは進行方向のノードにしか行けない
            elif action == -1: # ノード上にいて、行き先がなく停止を選択するとき,next_nodeは現在ノード
                next_node = env.current_start[agent_idx]
            else:
                next_node = action
            needed_step = calculate_steps_to_node(env, agent_idx, next_node) # エージェントが次のノードに到達するまでに必要なステップ数
            conflict = False
            # ノードの占有状況を確認
            # occupied_nodesにnext_nodeが存在し、かつその占有ステップ数がneeded_stepと5tep以内の誤差しかない場合、衝突するので次の行動を評価
            # occupied_nodes: List[(node:int, step:float)]
            for occupied_node, occupied_step in occupied_nodes:
                if occupied_node == next_node and (abs(occupied_step - needed_step) <= 3 or occupied_step == -1):
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
                occupied_nodes[agent_idx] = (next_node, needed_step)
                # エッジの占有状況を更新
                occupied_edges[agent_idx] = (env.current_start[agent_idx], next_node)
                avail_actions.remove(action)
                break
        if not action_selected:
            # 衝突が避けられない場合可能な行動を補填し、一つ上の優先度のエージェントの行動を再選択

            # 最上位優先度エージェントの場合、停止を選択
            if current_priority == best_priority:
                # 最優先エージェントで可能な行動がない場合、停止を選択
                actions[agent_idx] = -1
                current_priority += 1
                best_priority += 1
                # ノードの占有状況を更新
                occupied_nodes[agent_idx] = (env.current_start[agent_idx], -1)
                # エッジの占有状況は更新しない（停止しているため）
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


def _agent_congestion(env, agent_idx: int, radius: float = 5.0) -> float:
    """Count other agents within given radius of agent_idx."""
    ax, ay = env.obs[agent_idx][:2]
    cnt = 0
    for j in range(env.agent_num):
        if j == agent_idx:
            continue
        bx, by = env.obs[j][:2]
        if math.hypot(bx - ax, by - ay) <= radius:
            cnt += 1
    return float(cnt)


def _priority_score(env, agent_idx: int, assigned_task: List[int]) -> float:
    """Compute the priority score for an agent given the current policy parameters."""
    params = get_priority_params()
    has_task = bool(assigned_task)
    if not has_task or len(assigned_task) < 2:
        return params.idle_penalty + params.idle_bias

    pick_node = assigned_task[0]
    drop_node = assigned_task[1]
    base_node = env.current_start[agent_idx] if env.current_start[agent_idx] is not None else env.goal_array[agent_idx]
    current_goal = env.goal_array[agent_idx] if agent_idx < len(env.goal_array) else None

    if current_goal is not None and current_goal == drop_node:
        dist_goal = env.get_path_length(base_node, drop_node)
        dist_goal = dist_goal if dist_goal is not None else float("inf")
        return params.goal_weight * dist_goal + params.congestion_weight * _agent_congestion(env, agent_idx)

    dist_pick = env.get_path_length(base_node, pick_node)
    dist_drop = env.get_path_length(pick_node, drop_node)
    dist_pick = dist_pick if dist_pick is not None else float("inf")
    dist_drop = dist_drop if dist_drop is not None else float("inf")
    return params.pick_weight * dist_pick + params.drop_weight * dist_drop + params.idle_bias + params.congestion_weight * _agent_congestion(env, agent_idx)
