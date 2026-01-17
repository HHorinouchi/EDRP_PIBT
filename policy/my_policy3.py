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

# priority_scoreは高いほうが優先度が低い
@dataclass
class PriorityParams:
    """
    Parameters for detect_actions priority calculation.

    goal_weight:     Weight for distance to drop when currently heading to drop.
    pick_weight:     Weight for distance to pick when still before pickup.
    drop_weight:     Weight for distance pick->drop when still before pickup.
    step_tolerance:  Collision timing tolerance (None uses speed-based default).
    """

    goal_weight: float = 1.0
    pick_weight: float = 1.0
    drop_weight: float = 1.0
    step_tolerance: Optional[float] = None

    # Task assignment scoring weights
    assign_pick_weight: float = 1.0
    assign_drop_weight: float = 1.0
    congestion_weight: float = -1.0  # penalize local congestion around an agent
    assign_spread_weight: float = 1.0  # penalize assignment to nearby drop nodes
    @classmethod
    def from_dict(cls, data: Dict) -> "PriorityParams":
        return cls(
            goal_weight=float(data.get("goal_weight", cls.goal_weight)),
            pick_weight=float(data.get("pick_weight", cls.pick_weight)),
            drop_weight=float(data.get("drop_weight", cls.drop_weight)),
            step_tolerance=float(data.get("step_tolerance", cls.step_tolerance))
            if data.get("step_tolerance", cls.step_tolerance) is not None
            else None,
            assign_pick_weight=float(data.get("assign_pick_weight", cls.assign_pick_weight)),
            assign_drop_weight=float(data.get("assign_drop_weight", cls.assign_drop_weight)),
            congestion_weight=float(data.get("congestion_weight", cls.congestion_weight)),
            assign_spread_weight=float(data.get("assign_spread_weight", cls.assign_spread_weight)),
        )


ROOT_DIR = Path(__file__).resolve().parents[1]
_PRIORITY_PARAMS_PATH = (
    ROOT_DIR
    / "policy"
    / "train"
    / "sweep_results"
    / "priority_params_map_5x4_agents_10.json"
)
_PRIORITY_PARAMS = None
_PRIORITY_PARAMS_LOADED_FROM_FILE = False

DEFAULT_LOAD_BALANCE_WEIGHT = 0.0


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

    def _assigned_goal(idx: int) -> Optional[int]:
        if idx < len(env.assigned_tasks) and len(env.assigned_tasks[idx]) >= 2:
            return env.assigned_tasks[idx][1]
        if idx < len(env.goal_array):
            return env.goal_array[idx]
        return None

    other_targets = []
    for j in range(env.agent_num):
        target = _assigned_goal(j)
        if target is not None:
            other_targets.append(int(target))

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

            spread_penalty = 0.0
            if other_targets:
                for target in other_targets:
                    dist = env.get_path_length(drop_node, target)
                    if dist is None:
                        continue
                    spread_penalty += 1.0 / (float(dist) + 1.0)

            score = (
                params.assign_pick_weight * float(dist_to_pick)
                + params.assign_drop_weight * float(dist_pick_to_drop)
                + DEFAULT_LOAD_BALANCE_WEIGHT * float(unassigned_ratio)
                + params.assign_spread_weight * spread_penalty
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
    再帰的なPIBT (Priority Inheritance with Backtracking) による行動決定
    """
    params = get_priority_params()
    
    # 1. 優先度の決定 (計算量は最小限に)
    priority_scores = []
    for i in range(env.agent_num):
        assigned_task = env.assigned_tasks[i] if i < len(env.assigned_tasks) else []
        # _priority_score は重いので、簡易的なスコア計算推奨。
        # ここでは既存関数を使うが、実運用では _agent_congestion の get_path_length を削除すべき。
        score = _priority_score(env, i, assigned_task)
        priority_scores.append((i, score))
    
    # スコアが小さい方が高優先度
    priority_order = sorted(priority_scores, key=lambda x: x[1])
    sorted_agents = [x[0] for x in priority_order]
    
    # 2. 変数初期化
    actions = [None] * env.agent_num
    # 各ノードを誰が使うか (node_id -> agent_id)
    occupied_next_nodes = {} 
    # 各ノードに誰がいるか (node_id -> agent_id)
    current_node_occupants = {env.current_start[i]: i for i in range(env.agent_num)}

    # 3. 再帰関数 PIBT
    def func_pibt(agent_i, inherit_priority_agent=None):
        # 既に行動決定済みなら終了
        if actions[agent_i] is not None:
            return True
        
        # 可能な行動を取得 (優先度順にソート済みであることを期待)
        # get_avail_agent_actions は "現在のノード" も含める必要がある (Waitアクション)
        _, avail_actions = env.get_avail_agent_actions(agent_i, env.n_actions)
        
        # 行動の候補を作成 (タスクへの距離などでソート)
        # ここでは簡易的に、既存のロジックでソートされたものと仮定
        # ★重要: 候補には必ず「現在地(Wait)」を含めること
        candidates = sorted(avail_actions, key=lambda a: random.random()) # 本来はゴールへの距離でソート
        
        # タスクがある場合、ゴール方向を優先するソートを行う
        assigned_task = env.assigned_tasks[agent_i] if agent_i < len(env.assigned_tasks) else []
        if assigned_task and len(assigned_task) >= 2:
            target = assigned_task[1] if env.goal_array[agent_i] == assigned_task[1] else assigned_task[0]
            def dist_sort(action):
                d = env.get_path_length(action, target)
                return d if d is not None else float('inf')
            candidates = sorted(avail_actions, key=dist_sort)

        # 現在地を維持するアクション(Wait)を候補の末尾に追加（またはavail_actionsに含まれているか確認）
        current_node = env.current_start[agent_i]
        if current_node not in candidates:
             candidates.append(current_node)

        for next_node in candidates:
            # 衝突チェック1: 既に誰かが予約しているノードには行けない
            if next_node in occupied_next_nodes:
                continue
            
            # 衝突チェック2: エッジ衝突 (Swap衝突) の防止
            # next_node に今いるエージェント j が、agent_i の現在地に来ようとしていないか？
            agent_j = current_node_occupants.get(next_node)
            
            # まだ行動が決まっていない相手がいる場合
            if agent_j is not None and agent_j != agent_i and actions[agent_j] is None:
                # 相手に「自分はここに行きたいから、そこからどいてくれ」と再帰的に依頼
                # 優先度継承: agent_j は agent_i の優先度を継承して行動決定を試みる
                if func_pibt(agent_j, agent_i):
                    # 相手がどいてくれた -> 相手の行き先チェック (Swapになっていないか)
                    # actions[agent_j] が agent_i の現在地(current_node)だとSwap衝突
                    if actions[agent_j] == current_node:
                        continue # SwapになるのでこのノードはNG
                else:
                    # 相手がどけなかった -> このノードは諦める
                    continue
            
            # すでに行動決定済みの相手がいる場合
            elif agent_j is not None and agent_j != agent_i:
                 if actions[agent_j] == current_node:
                     continue # Swap衝突

            # ここまで来れば next_node は確保可能
            occupied_next_nodes[next_node] = agent_i
            actions[agent_i] = next_node
            return True
        
        # どの行動も取れない場合 (ここに来るのは稀だが、Waitすらできない場合)
        # 通常はWait(現在地)が通るはず。
        # どうしても無理なら「現在地」を強制予約 (衝突覚悟だがNoneよりマシ)
        actions[agent_i] = current_node 
        occupied_next_nodes[current_node] = agent_i
        return False

    # 4. 優先度の高い順に実行
    for agent_idx in sorted_agents:
        if actions[agent_idx] is None:
            func_pibt(agent_idx)
            
    # 5. 万が一 None が残っていた場合のフェイルセーフ
    for i in range(env.agent_num):
        if actions[i] is None:
            actions[i] = env.current_start[i] # その場停止

    return actions, 0 # countは使わないので0

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


def _default_step_tolerance(speed: float) -> float:
    if speed <= 0:
        return float("inf")
    return max(1, math.ceil(5.0 / speed))+ 1


def _resolve_step_tolerance(env, params: PriorityParams, speed: float) -> float:
    value = getattr(params, "step_tolerance", None)
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    return _default_step_tolerance(speed)


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
    # 各エージェントの可能な行動数
    _, avail_actions = env.get_avail_agent_actions(agent_idx, env.n_actions)
    avail_actions_num = len(avail_actions)
    if not has_task or len(assigned_task) < 2:
        return params.congestion_weight * _agent_congestion(env, agent_idx)

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
    return (
        params.pick_weight * dist_pick
        + params.drop_weight * dist_drop
        + params.congestion_weight * _agent_congestion(env, agent_idx)
    )
