"""多エージェント経路計画のための Conflict-Based Search ポリシー。

``task_test.py`` からインポートされ、CBS に基づく協調行動を出力する。
評価用スクリプトが処理フローを理解しやすいよう、各所を日本語で記述している。
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

### 提出情報 ####
TEAM_NAME = "your_team_name"
# TEAM_NAME は DRP の登録名（チーム参加ならチーム名）と一致させる。
##############################

# 直前ステップで得た CBS の残り経路を保持し、次ステップのウォームスタートに再利用する。
_LAST_CBS_PATHS: Dict[int, List[int]] = {}


@dataclass(frozen=True)
class Constraint:
    agent: int
    time: int
    node: Optional[int] = None
    edge: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class Conflict:
    time: int
    agent1: int
    agent2: int
    node: Optional[int] = None
    edge: Optional[Tuple[int, int]] = None


@dataclass(order=True)
class CTNode:
    cost: int
    seq: int = field(compare=True)
    constraints: Tuple[Constraint, ...] = field(compare=False)
    paths: Dict[int, List[int]] = field(compare=False)


_CT_COUNTER = count()
_CBS_MAX_HORIZON = 120
_CBS_MIN_SLACK = 4


def policy(obs, env):
    """`task_test.py` から呼ばれるエントリーポイント。"""

    # `obs` は環境インスタンスが完全な状態を保持しているため利用しない。
    # (actions, task_assign) を返すインタフェースを保ちつつ、移動部分だけ CBS に置き換える。

    actions = detect_actions(env)
    task_assign = assign_task(env)
    return actions, task_assign

def assign_task(env):
    """次のピック地点までの距離に基づき貪欲にタスクを割り当てる。"""
    # env.current_tasklist を直接参照し、インデックスずれによるバグを避ける。
    # 削除は行わず、割り当てるタスクの添字だけを控えることで整合性を保つ。
    task_assign = [-1] * env.agent_num

    if not hasattr(env, 'current_tasklist') or len(env.current_tasklist) < env.agent_num:
        return task_assign

    # available_indices には未割り当てタスクの添字を保持する。
    available_indices = [idx for idx, a in enumerate(env.assigned_list) if a == -1]

    for i in range(env.agent_num):
        # すでにタスクを保持しているエージェントはスキップ。
        if i < len(env.assigned_tasks) and len(env.assigned_tasks[i]) != 0:
            continue

        best_task_idx = -1
        best_dist = float('inf')

        for idx in list(available_indices):
            # idx は env.current_tasklist の添字として扱う。
            try:
                task = env.current_tasklist[idx]
            except Exception:
                continue

            # タスクは [pick_node, drop_node] 形式を想定。
            pick_node = task[0]
            # エージェントの現在ゴール（もしくは位置）から pick_node までの距離を取得。
            # env.get_path_length は None を返す場合がある点に注意。
            path_length = env.get_path_length(env.goal_array[i], pick_node)
            if path_length is None:
                continue
            if path_length < best_dist:
                best_dist = path_length
                best_task_idx = idx

        if best_task_idx != -1:
            task_assign[i] = best_task_idx
            # ループ内で他エージェントが同じタスクを選ばないようローカルに確保。
            available_indices.remove(best_task_idx)

    return task_assign

def detect_actions(env):
    """CBS ベースで全エージェントの次行動を決定する。"""
    # 新しいエピソード開始時はウォームスタート用バッファをクリア。
    if getattr(env, "step_account", 0) <= 1:
        _LAST_CBS_PATHS.clear()

    starts = list(getattr(env, "current_start", []))
    if not starts or getattr(env, "agent_num", 0) != len(starts):
        # フォールバック: ランダム行動に委譲。
        return _random_feasible_actions(env)

    goals = _resolve_goals(env, starts)

    # 全員がすでにゴールに居るならその場待機。
    if all(s == g for s, g in zip(starts, goals)):
        return [starts[i] if _is_action_available(env, i, starts[i]) else _fallback_action(env, i)
                for i in range(env.agent_num)]

    # マップ密度と既存経路を元に探索ウィンドウを見積もり、CBS を実行。
    horizon = _estimate_planning_window(env, starts, goals)
    plan = _compute_cbs_plan(env, starts, goals, horizon)

    if plan is None:
        # CBS が失敗した場合はフォールバック: 現在地に留まる。
        return [starts[i] if _is_action_available(env, i, starts[i]) else _fallback_action(env, i)
                for i in range(env.agent_num)]

    actions: List[int] = []
    for agent_id in range(env.agent_num):
        path = plan.get(agent_id, [starts[agent_id]])
        next_node = path[1] if len(path) > 1 else starts[agent_id]
        if not _is_action_available(env, agent_id, next_node):
            next_node = starts[agent_id]
            if not _is_action_available(env, agent_id, next_node):
                next_node = _fallback_action(env, agent_id)
        actions.append(next_node)

        # 次ステップ用に経路をウォームスタートとして保存。
        if len(path) > 1:
            _LAST_CBS_PATHS[agent_id] = path[1:]
        else:
            _LAST_CBS_PATHS.pop(agent_id, None)

    return actions


def _random_feasible_actions(env) -> List[int]:
    actions: List[int] = []
    for i in range(getattr(env, "agent_num", 0)):
        _, avail = env.get_avail_agent_actions(i, env.n_actions)
        if avail:
            actions.append(avail[0])
        else:
            actions.append(0)
    return actions


def _is_action_available(env, agent_id: int, node: int) -> bool:
    _, avail = env.get_avail_agent_actions(agent_id, env.n_actions)
    return node in avail


def _fallback_action(env, agent_id: int) -> int:
    _, avail = env.get_avail_agent_actions(agent_id, env.n_actions)
    return avail[0] if avail else 0


def _resolve_goals(env, starts: Sequence[int]) -> List[int]:
    goals: List[int] = []
    goal_array = getattr(env, "goal_array", None)
    for idx, start in enumerate(starts):
        goal = None
        if goal_array is not None and idx < len(goal_array):
            goal = goal_array[idx]
        # ゴールが未設定なら現在地をゴール扱いとする。
        goals.append(start if goal is None else goal)
    return goals


def _estimate_planning_window(env, starts: Sequence[int], goals: Sequence[int]) -> int:
    n_nodes = getattr(env, "n_nodes", max(len(env.G.nodes), 1))
    density = getattr(env, "agent_num", 1) / max(1, n_nodes)
    if density <= 0.1:
        base = 20
    elif density <= 0.3:
        base = 35
    else:
        base = 55

    longest = 0
    graph = env.G
    for node in set(list(starts) + list(goals)):
        lengths = nx.single_source_shortest_path_length(graph, node)
        if lengths:
            longest = max(longest, max(lengths.values()))

    horizon = max(base, longest + _CBS_MIN_SLACK)
    return min(_CBS_MAX_HORIZON, horizon)


def _compute_cbs_plan(env, starts: Sequence[int], goals: Sequence[int], horizon: int) -> Optional[Dict[int, List[int]]]:
    graph = env.G
    num_agents = len(starts)

    constraints: Tuple[Constraint, ...] = tuple()
    paths: Dict[int, List[int]] = {}

    for agent_id in range(num_agents):
        path = _low_level_search(graph, starts[agent_id], goals[agent_id], constraints, agent_id, horizon)
        if path is None:
            return None
        paths[agent_id] = path

    _synchronise_path_lengths(paths)
    cost = _solution_cost(paths)
    open_list: List[CTNode] = [CTNode(cost=cost, seq=next(_CT_COUNTER), constraints=constraints, paths=paths)]

    while open_list:
        node = heapq.heappop(open_list)
        conflict = _detect_first_conflict(node.paths)
        if conflict is None:
            return node.paths

        for agent in (conflict.agent1, conflict.agent2):
            new_constraint = _materialise_constraint(conflict, agent)
            new_constraints = node.constraints + (new_constraint,)

            new_paths = {aid: path[:] for aid, path in node.paths.items()}
            replanned = _low_level_search(graph, starts[agent], goals[agent], new_constraints, agent, horizon)
            if replanned is None:
                continue
            new_paths[agent] = replanned
            _synchronise_path_lengths(new_paths)
            new_cost = _solution_cost(new_paths)
            heapq.heappush(open_list, CTNode(cost=new_cost, seq=next(_CT_COUNTER),
                                            constraints=new_constraints, paths=new_paths))

    return None


def _materialise_constraint(conflict: Conflict, agent: int) -> Constraint:
    if conflict.node is not None:
        return Constraint(agent=agent, time=conflict.time, node=conflict.node)
    assert conflict.edge is not None
    u, v = conflict.edge
    if agent == conflict.agent1:
        return Constraint(agent=agent, time=conflict.time, edge=(u, v))
    else:
        return Constraint(agent=agent, time=conflict.time, edge=(v, u))


def _low_level_search(graph: nx.Graph, start: int, goal: int,
                      constraints: Tuple[Constraint, ...], agent: int, max_time: int) -> Optional[List[int]]:
    if start == goal:
        # ゴールと一致していればその場待機のみで十分。
        return [start]

    vertex_constraints, edge_constraints = _build_constraint_tables(constraints, agent)
    if start in vertex_constraints.get(0, set()):
        return None

    heuristic = nx.single_source_shortest_path_length(graph, goal)
    if start not in heuristic:
        return None

    frontier: List[Tuple[int, int, int, int, List[int]]] = []  # (評価値, 実コスト, 現在ノード, 時刻, 経路)
    start_h = heuristic[start]
    heapq.heappush(frontier, (start_h, 0, start, 0, [start]))
    best_cost: Dict[Tuple[int, int], int] = {(start, 0): 0}

    while frontier:
        f_score, g_cost, node, time, path = heapq.heappop(frontier)

        if node == goal:
            return path

        if time >= max_time:
            continue

        for nxt in _neighbours_with_wait(graph, node):
            next_time = time + 1
            if nxt in vertex_constraints.get(next_time, set()):
                continue
            if (node, nxt) in edge_constraints.get(time, set()):
                continue

            next_g = g_cost + 1
            state = (nxt, next_time)
            if best_cost.get(state, float('inf')) <= next_g:
                continue

            h_val = heuristic.get(nxt)
            if h_val is None:
                continue
            f_val = next_g + h_val
            best_cost[state] = next_g
            heapq.heappush(frontier, (f_val, next_g, nxt, next_time, path + [nxt]))

    return None


def _build_constraint_tables(constraints: Tuple[Constraint, ...], agent: int) -> Tuple[DefaultDict[int, set], DefaultDict[int, set]]:
    vertex: DefaultDict[int, set] = defaultdict(set)
    edge: DefaultDict[int, set] = defaultdict(set)
    for cons in constraints:
        if cons.agent != agent:
            continue
        if cons.node is not None:
            vertex[cons.time].add(cons.node)
        elif cons.edge is not None:
            edge[cons.time].add(cons.edge)
    return vertex, edge


def _neighbours_with_wait(graph: nx.Graph, node: int) -> Iterable[int]:
    for neighbour in graph.neighbors(node):
        yield neighbour
    # 待機アクション
    yield node


def _synchronise_path_lengths(paths: Dict[int, List[int]]) -> None:
    if not paths:
        return
    max_len = max(len(path) for path in paths.values())
    for path in paths.values():
        if not path:
            continue
        last = path[-1]
        if len(path) < max_len:
            path.extend([last] * (max_len - len(path)))


def _solution_cost(paths: Dict[int, List[int]]) -> int:
    cost = 0
    for path in paths.values():
        if path:
            cost += len(path) - 1
    return cost


def _detect_first_conflict(paths: Dict[int, List[int]]) -> Optional[Conflict]:
    if not paths:
        return None

    agents = list(paths.keys())
    max_len = max(len(path) for path in paths.values())

    # 頂点衝突の検出
    for t in range(max_len):
        occupancy: Dict[int, List[int]] = defaultdict(list)
        for agent in agents:
            path = paths[agent]
            node = path[t] if t < len(path) else path[-1]
            occupancy[node].append(agent)
        for node, occupants in occupancy.items():
            if len(occupants) > 1:
                return Conflict(time=t, agent1=occupants[0], agent2=occupants[1], node=node)

    # エッジ（入れ替わり）衝突の検出
    for t in range(max_len - 1):
        transitions: Dict[int, Tuple[int, int]] = {}
        for agent in agents:
            path = paths[agent]
            curr = path[t] if t < len(path) else path[-1]
            nxt = path[t + 1] if (t + 1) < len(path) else path[-1]
            transitions[agent] = (curr, nxt)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a_i = agents[i]
                a_j = agents[j]
                u, v = transitions[a_i]
                u2, v2 = transitions[a_j]
                if u == v2 and v == u2 and u != v:
                    return Conflict(time=t, agent1=a_i, agent2=a_j, edge=(u, v))

    return None