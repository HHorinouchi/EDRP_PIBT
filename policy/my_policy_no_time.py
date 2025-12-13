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
    衝突回避を最優先したPIBT (Priority Inheritance with Backtracking) 実装
    再帰的な探索により、高優先度のエージェントの経路を確保しつつ、
    低優先度のエージェントがそれを回避できるか探索する。
    """
    
    # --- 1. 初期設定と優先度計算 ---
    if env.goal_array is None:
        return [-1] * env.agent_num, 0

    speed = float(getattr(env, "speed", 5.0))
    # ステップ許容誤差（移動にかかるステップ数）
    if speed <= 0:
        step_tolerance = float("inf")
    else:
        step_tolerance = max(1, math.ceil(5.0 / speed)) + 2

    # 各エージェントの優先度スコア計算（ゴールに近いほど優先）
    priority_scores = []
    for i in range(env.agent_num):
        assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
        # _priority_score関数がない場合は簡易的に距離計算などを実装してください
        # ここでは元のコードに準拠して外部関数を呼んでいる想定ですが、
        # なければ以下の行を有効化してください
        # dist = env.get_path_length(env.current_start[i], env.goal_array[i]) if env.goal_array[i] else float('inf')
        # score = dist
        score = _priority_score(env, i, assigned_task) if '_priority_score' in globals() else 0
        priority_scores.append((i, score))

    # 距離が短い（スコアが小さい）順にソート
    priority_order = sorted(priority_scores, key=lambda x: x[1])
    sorted_agent_indices = [p[0] for p in priority_order]

    # --- 2. 行動候補の事前生成 ---
    # 各エージェントの有効な行動リストを作成し、ゴールに近い順にソートしておく
    agent_avail_actions = {}
    for i in range(env.agent_num):
        _, avail = env.get_avail_agent_actions(i, env.n_actions)
        
        # 行動の評価（ゴールに近づく行動を優先）
        action_scores = []
        for action in avail:
            if action < 0: # 待機
                next_node = env.current_start[i]
                # 待機はペナルティを少し与えて、移動を優先させる（デッドロック防止）
                dist = float('inf') # 暫定
            else:
                next_node = action
                # タスクやゴールまでの距離を取得
                # (簡易化のためゴールへの距離のみ記述していますが、元のロジックがあればそれを使用)
                if env.goal_array[i] is not None:
                    dist = env.get_path_length(next_node, env.goal_array[i])
                    if dist is None: dist = float('inf')
                else:
                    dist = float('inf')
            action_scores.append((action, dist))
        
        # 距離昇順にソート
        sorted_actions = [a for a, _ in sorted(action_scores, key=lambda x: x[1])]
        
        # 待機行動(-1)をリストの最後に追加（移動できない場合の最終手段）
        # 元のコードのように -1, -2... と増やす必要は衝突回避の観点では薄いため、-1のみでも可
        if -1 not in sorted_actions:
            sorted_actions.append(-1)
            
        agent_avail_actions[i] = sorted_actions

    # --- 3. 予約テーブル (Reservation Table) ---
    # key: node_id, value: (agent_idx, arrival_step)
    reserved_nodes = {} 
    # key: (from_node, to_node), value: agent_idx
    reserved_edges = {}

    final_actions = [None] * env.agent_num
    
    # バックトラックの深さ制限（計算時間爆発防止）
    # 全探索すると時間がかかるため、ある程度で諦めるカウンタなどを入れても良い
    backtrack_count = 0 

    # --- 4. 再帰的な経路探索関数 ---
    def solve(priority_idx):
        nonlocal backtrack_count
        
        # 全員分の行動が決まったら成功
        if priority_idx >= env.agent_num:
            return True

        agent_idx = sorted_agent_indices[priority_idx]
        current_node = env.current_start[agent_idx]
        avail_actions = agent_avail_actions[agent_idx]

        for action in avail_actions:
            # --- 次のノードと必要なステップ数の計算 ---
            if action < 0: # 待機
                next_node = current_node
                # 待機の場合、占有時間は現在のステップ数 + 待機時間
                # 安全のため、次の1ステップ分は確実にその場を占有するとみなす
                arrival_step = calculate_steps_to_node(env, agent_idx, current_node) + 1
            else: # 移動
                next_node = action
                arrival_step = calculate_steps_to_node(env, agent_idx, next_node)

            # --- 衝突判定 ---
            conflict = False

            # A. ノード競合 (Vertex Conflict)
            # 誰かが既にそのノードを予約していて、かつタイミングが被る場合
            if next_node in reserved_nodes:
                res_agent, res_arrival = reserved_nodes[next_node]
                # タイミングの許容誤差内であれば衝突とみなす
                if abs(res_arrival - arrival_step) <= step_tolerance:
                    conflict = True

            # B. エッジ競合 (Edge/Swap Conflict)
            # すれ違い（Swap）の禁止: A->B に行くとき、誰かが B->A に来ていないか
            if not conflict:
                if (next_node, current_node) in reserved_edges:
                    conflict = True
            
            # C. 同一エッジの同時利用禁止（オプション）
            if not conflict:
                if (current_node, next_node) in reserved_edges:
                    conflict = True

            # --- 行動の仮予約 ---
            if not conflict:
                # 予約を実行
                reserved_nodes[next_node] = (agent_idx, arrival_step)
                if action >= 0: # 移動する場合のみエッジを予約
                    reserved_edges[(current_node, next_node)] = agent_idx
                
                final_actions[agent_idx] = action

                # 次の優先度のエージェントへ
                if solve(priority_idx + 1):
                    return True # 全員成功

                # --- バックトラック (失敗したので予約を解除) ---
                del reserved_nodes[next_node]
                if action >= 0:
                    del reserved_edges[(current_node, next_node)]
                
                final_actions[agent_idx] = None
        
        return False # どの行動もNGだった場合

    # --- 5. 実行 ---
    success = solve(0)

    # 万が一解が見つからなかった場合（デッドロック等）、全員待機させるなどのフェイルセーフ
    if not success:
        print("Warning: PIBT deadlock detected. Force wait.")
        return [-1] * env.agent_num, 0

    return final_actions, 0

# --- 補助関数 (元のコードに依存するため、環境に合わせて調整してください) ---
def _priority_score(env, agent_idx, task):
    # 優先度スコア計算ロジック（適宜実装）
    # 小さい方が優先度が高いとする
    if env.goal_array[agent_idx] is None:
        return float('inf')
    return env.get_path_length(env.current_start[agent_idx], env.goal_array[agent_idx])

# calculate_steps_to_node は元のコードにある想定で使用しています


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
