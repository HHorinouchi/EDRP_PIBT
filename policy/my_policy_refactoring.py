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


_PRIORITY_PARAMS_PATH = Path(__file__).with_name("priority_params_shibuya_10.json")
_PRIORITY_PARAMS = None
_PRIORITY_PARAMS_LOADED_FROM_FILE = False


def _load_priority_params() -> PriorityParams:
    # パスが存在するかどうかを確認し、存在する場合はファイルから読み込む。
    # ファイルが存在しない場合はデフォルト値を返すが、"ファイル読み込みフラグ"は立てない。
    global _PRIORITY_PARAMS_LOADED_FROM_FILE
    if not _PRIORITY_PARAMS_PATH.exists():
        _PRIORITY_PARAMS_LOADED_FROM_FILE = False
        return PriorityParams()
    try:
        data = json.loads(_PRIORITY_PARAMS_PATH.read_text())
        _PRIORITY_PARAMS_LOADED_FROM_FILE = True
        return PriorityParams.from_dict(data)
    except Exception as e:
        print(f"Failed to load priority parameters: {e}")
        _PRIORITY_PARAMS_LOADED_FROM_FILE = False
        # Fall back to defaults if the file is malformed
        return PriorityParams()


def get_priority_params() -> PriorityParams:
    """Return the in-memory priority parameters (defaults if unset)."""
    return _PRIORITY_PARAMS or PriorityParams()


def priority_params_exists() -> bool:
    """Return True if priority parameters were successfully loaded from file.

    This distinguishes between using default parameters (no file) and having
    an explicit configuration file present. The requested behavior is to run
    the policy only when such a file exists.
    """
    # Treat programmatically set params as present as well. During training
    # ES rollouts call set_priority_params(...) in-memory, so require the
    # policy to run when _PRIORITY_PARAMS is set even if no file exists.
    if _PRIORITY_PARAMS_LOADED_FROM_FILE:
        return True
    return _PRIORITY_PARAMS is not None


def set_priority_params(params: PriorityParams) -> None:
    """Update the priority parameters used inside detect_actions."""
    global _PRIORITY_PARAMS
    _PRIORITY_PARAMS = params
    # When parameters are set programmatically (e.g. during ES training), mark
    # them as loaded so policy() will not early-return. This allows in-memory
    # updates via set_priority_params(...) to take effect for rollouts.
    try:
        global _PRIORITY_PARAMS_LOADED_FROM_FILE
        _PRIORITY_PARAMS_LOADED_FROM_FILE = True
    except Exception:
        pass


def save_priority_params(params: PriorityParams) -> None:
    """Persist priority parameters so that future runs can reuse them."""
    payload = json.dumps(asdict(params), indent=2)
    _PRIORITY_PARAMS_PATH.write_text(payload)


# Load parameters once on import so that training outputs are picked up automatically.
set_priority_params(_load_priority_params())


def policy(obs, env):
    # If priority parameters were not loaded from a file, do not run the
    # full policy. Return a safe no-op: all agents stop and no tasks assigned.
    if not priority_params_exists():
        agent_num = getattr(env, "agent_num", None)
        if agent_num is None:
            return [], []
        return [-1] * agent_num, [-1] * agent_num

    actions, count = detect_actions(env)
    task_assign = assign_task(env)
    return actions, task_assign, count

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
    (リファクタリング版: ロジック変更なし、構造化による可読性向上)
    """
    if env.goal_array is None:
        return [-1] * env.agent_num

    # --- 1. 初期設定 ---
    step_tolerance = _calculate_step_tolerance(env)
    max_wait = 1
    
    # 優先順位の決定
    priority_order = _calculate_priority_order(env)

    # 行動候補の生成
    avail_actions_list = _generate_avail_actions(env, max_wait)
    # バックトラック時のリセット用に初期状態を保存
    initial_avail_actions = copy.deepcopy(avail_actions_list)

    # 占有管理テーブル
    # occupied_nodes: List[(node, release_step)]
    occupied_nodes = [(None, None)] * env.agent_num
    # occupied_edges: List[(from, to, release_step)] (後半はデッドロック回避の予約用)
    occupied_edges = [(None, None, 0)] * (env.agent_num * 2)

    # 探索用変数
    actions = [None] * env.agent_num
    current_priority = 0
    most_high_priority = 0
    count = 0

    # --- 2. メインループ (バックトラック探索) ---
    while current_priority < env.agent_num:
        count += 1
        agent_idx, _ = priority_order[current_priority]
        avail_actions = avail_actions_list[agent_idx]
        
        action_selected = False
        
        # 可能な行動を順に試行
        for action in list(avail_actions):
            # 次のノードと到達ステップを計算
            next_node = _get_next_node(env, agent_idx, action, initial_avail_actions, max_wait)
            needed_step = calculate_steps_to_node(env, agent_idx, next_node)

            # 衝突判定 (上位優先度のエージェントとの比較)
            if _is_conflict(env, agent_idx, next_node, needed_step, step_tolerance, 
                            current_priority, priority_order, occupied_nodes, occupied_edges):
                avail_actions.remove(action)
                continue

            # 行動確定時の処理 (予約テーブルの更新)
            actions[agent_idx] = action
            action_selected = True
            current_priority += 1
            
            _update_reservations(env, agent_idx, action, next_node, needed_step, step_tolerance,
                                 current_priority, priority_order, occupied_nodes, occupied_edges)
            
            avail_actions.remove(action)
            break

        # --- 3. 行動が決まらなかった場合 (バックトラック or 強制待機) ---
        if not action_selected:
            if current_priority <= most_high_priority:
                # これ以上戻れない場合（最上位）、強制的に待機を選択して進める
                actions[agent_idx] = -max_wait
                current_priority += 1
                most_high_priority += 1
                
                # 強制待機の予約更新
                occupied_nodes[agent_idx] = (env.current_start[agent_idx], 0)
                if len(initial_avail_actions[agent_idx]) == max_wait + 1:
                    # エッジ上にいる場合の特殊処理
                    occupied_edges[agent_idx] = (env.current_start[agent_idx], initial_avail_actions[agent_idx][0], step_tolerance)
            
            else:
                # 一つ前のエージェントに戻ってやり直し
                avail_actions_list[agent_idx] = copy.deepcopy(initial_avail_actions[agent_idx])
                
                prev_agent_idx, _ = priority_order[current_priority - 1]
                current_priority -= 1
                
                # 前のエージェントの予約を解除
                occupied_nodes[prev_agent_idx] = (None, None)
                occupied_edges[prev_agent_idx] = (None, None, 0)
                occupied_edges[prev_agent_idx + env.agent_num] = (None, None, 0)

    return actions, count


# --- 以下、ヘルパー関数 ---

def _calculate_step_tolerance(env):
    speed = float(getattr(env, "speed", 5.0))
    if speed <= 0:
        return float("inf")
    return max(1, math.ceil(5.0 / speed)) + 2

def _calculate_priority_order(env):
    priority_scores = []
    for i in range(env.agent_num):
        assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
        score = _priority_score(env, i, assigned_task)
        priority_scores.append((i, score))
    return sorted(priority_scores, key=lambda x: x[1])

def _generate_avail_actions(env, max_wait):
    avail_actions_list = []
    for i in range(env.agent_num):
        _, avail_actions = env.get_avail_agent_actions(i, env.n_actions)
        
        # 最短経路距離に基づいてソート
        path_length_list = []
        for action in avail_actions:
            assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
            has_task = bool(assigned_task)
            current_goal = env.goal_array[i] if i < len(env.goal_array) else None
            next_node = action
            
            path = None
            if has_task and len(assigned_task) >= 2:
                pick_node, drop_node = assigned_task[0], assigned_task[1]
                if current_goal == drop_node:
                    path = env.get_path_length(next_node, drop_node)
                elif current_goal == pick_node:
                    path = env.get_path_length(next_node, pick_node)
            
            path_length = path if path is not None else float("inf")
            path_length_list.append((action, path_length))

        sorted_avail_actions = [a for a, _ in sorted(path_length_list, key=lambda x: x[1])]
        
        # 待機行動の追加
        for w in range(1, max_wait + 1):
            sorted_avail_actions.append(-w)
        
        avail_actions_list.append(sorted_avail_actions)
    return avail_actions_list

def _get_next_node(env, agent_idx, action, initial_avail_actions, max_wait):
    """行動に応じた次のノードを特定する"""
    if len(initial_avail_actions[agent_idx]) == max_wait + 1:
        # エッジ上にいる場合は進行方向のみ
        return initial_avail_actions[agent_idx][0]
    elif action < 0:
        # 待機の場合は現在地
        return env.current_start[agent_idx]
    else:
        # 移動
        return action

def _is_conflict(env, agent_idx, next_node, needed_step, step_tolerance, 
                 current_priority, priority_order, occupied_nodes, occupied_edges):
    """
    指定された行動が上位優先度のエージェントと衝突するか判定する
    """
    for check_idx in range(current_priority):
        higher_agent_idx, _ = priority_order[check_idx]
        
        # 1. ノード競合の確認
        occ_node, occ_needed_step = occupied_nodes[higher_agent_idx]
        if occ_node == next_node and abs(occ_needed_step - needed_step) <= step_tolerance:
            return True
        
        # 2. エッジ競合の確認 (メインのエッジ予約)
        if _check_edge_conflict(env, agent_idx, next_node, needed_step, step_tolerance,
                                occupied_edges[higher_agent_idx]):
            return True
            
        # 3. デッドロック防止予約との競合確認
        if _check_edge_conflict(env, agent_idx, next_node, needed_step, step_tolerance,
                                occupied_edges[higher_agent_idx + env.agent_num]):
            return True
            
    return False

def _check_edge_conflict(env, agent_idx, next_node, needed_step, step_tolerance, occ_edge_info):
    """単一のエッジ予約情報との競合をチェック"""
    start, end, release = occ_edge_info
    if start is not None and end is not None:
        same_direction = (start == env.current_start[agent_idx] and end == next_node)
        reverse_direction = (start == next_node and end == env.current_start[agent_idx])
        
        if reverse_direction:
            return True
        if same_direction and needed_step <= release + step_tolerance:
            return True
    return False

def _update_reservations(env, agent_idx, action, next_node, needed_step, step_tolerance,
                         current_priority, priority_order, occupied_nodes, occupied_edges):
    """
    行動確定後に予約テーブル(nodes, edges)を更新し、
    将来のデッドロック防止のために1手先の経路確保を行う
    """
    # 1. 基本的な予約更新
    if action < 0:
        wait_steps = abs(action)
        release_node = needed_step + wait_steps
        occupied_nodes[agent_idx] = (None, None) # ※元のロジック通り
        occupied_edges[agent_idx] = (env.current_start[agent_idx], next_node, release_node)
    else:
        release_node = needed_step
        occupied_nodes[agent_idx] = (next_node, release_node)
        occupied_edges[agent_idx] = (env.current_start[agent_idx], next_node, release_node)

    # 2. デッドロック回避のための先読み予約
    # 次のノードに到着した後、進める経路が「1つ」しかなければ、その経路も予約しておく
    graph = env.G
    neighbors = list(graph.neighbors(next_node))
    potential_edges = [(next_node, nb) for nb in neighbors]
    valid_edges = []

    for edge in potential_edges:
        occ_start, occ_end = edge
        blocked = False
        # 上位のエージェントがそのエッジを塞いでいないか確認
        for check_idx in range(current_priority):
            higher_agent_idx, _ = priority_order[check_idx]
            
            # メイン予約との比較
            if _is_blocked_by_reservation(occ_start, occ_end, needed_step, step_tolerance, 
                                          occupied_edges[higher_agent_idx]):
                blocked = True
                break
            
            # デッドロック予約との比較
            # ※元のコードではデッドロック予約との逆走のみチェックしていたためそれに従う
            e_start2, e_end2, _ = occupied_edges[higher_agent_idx + env.agent_num]
            if e_start2 is not None and e_end2 is not None:
                if occ_start == e_end2 and occ_end == e_start2: # Reverse
                    blocked = True
                    break
        
        if not blocked:
            valid_edges.append(edge)

    # 選択肢が1つしかない場合、そのエッジを「後半」の予約リストに登録
    if len(valid_edges) == 1:
        occ_start, occ_end = valid_edges[0]
        occupied_edges[agent_idx + env.agent_num] = (occ_start, occ_end, 0)

def _is_blocked_by_reservation(occ_start, occ_end, needed_step, step_tolerance, res_info):
    """先読み用のエッジ競合判定"""
    res_start, res_end, res_release = res_info
    if res_start is not None and res_end is not None:
        reverse = (occ_start == res_end and occ_end == res_start)
        same = (occ_start == res_start and occ_end == res_end)
        
        if reverse:
            return True
        if same and needed_step <= res_release + step_tolerance:
            return True
    return False

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


def _agent_congestion(env, agent_idx: int, max_steps: int = 5) -> float:
    """Count agents reachable within ``max_steps`` steps from current start or next node."""

    # current_start と現在向かっている next node (current_goal) を起点候補として列挙する

    def _candidate_nodes(idx: int) -> List[int]:
        nodes: List[int] = []
        if idx < len(env.current_start):
            start_node = env.current_start[idx]
            if start_node is not None:
                nodes.append(int(start_node))
        if idx < len(env.current_goal):
            next_node = env.current_goal[idx]
            if next_node is not None and int(next_node) not in nodes:
                nodes.append(int(next_node))
        if not nodes and idx < len(env.obs):
            try:
                nodes.append(int(env.obs[idx][2]))
            except Exception:
                pass
        return nodes

    # 起点候補が得られない場合は混雑度 0 とする
    origins = _candidate_nodes(agent_idx)
    if not origins:
        return 0.0

    # speed から距離をステップ数に換算するための係数を取得
    speed = float(getattr(env, "speed", 1.0))
    if speed <= 0:
        return 0.0

    # 他エージェントの current_start / current_goal を目的地候補とみなし、5 ステップ以内に到達できる数を数える
    count = 0
    for other_idx in range(env.agent_num):
        if other_idx == agent_idx:
            continue
        targets = _candidate_nodes(other_idx)
        if not targets:
            continue
        reached = False
        for origin in origins:
            for target in targets:
                if target == origin:
                    count += 1
                    reached = True
                    break
                distance = env.get_path_length(origin, target)
                if distance is None:
                    continue
                steps_needed = math.ceil(float(distance) / speed)
                if steps_needed <= max_steps:
                    count += 1
                    reached = True
                    break
            if reached:
                break
    return float(count)


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
