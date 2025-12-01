from pathlib import Path
import sys

# Ensure repository root is on sys.path so that `drp_env` can be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from drp_env.drp_env import DrpEnv


def main():
    # 小さな環境を立ち上げ、ノード/エッジ/隣接情報を取得してから、エッジにペナルティを適用して変化を確認するデモ
    env = DrpEnv(
        agent_num=2,
        speed=1.0,
        start_ori_array=[],
        goal_array=[],
        visu_delay=0.0,
        state_repre_flag="onehot",
        time_limit=50,
        collision="bounceback",
        task_flag=False,
        map_name="map_3x3",
        task_density=0.5,
    )

    env.reset()

    # 情報取得
    nodes = env.get_node_info()
    edges_before = env.get_edge_info()
    adj = env.get_adjacency_list()

    print("=== Map info BEFORE penalty ===")
    print("Total nodes:", len(nodes))
    print("First 5 nodes:", nodes[:5])
    print("Total edges:", len(edges_before))
    print("First 5 edges:", edges_before[:5])
    print("Adjacency (first 3 nodes):", {n["id"]: adj[n["id"]] for n in nodes[:3]})

    # 先頭のエッジにペナルティを適用
    if edges_before:
        u, v = edges_before[0]["u"], edges_before[0]["v"]
        base_w = edges_before[0].get("base_weight", edges_before[0]["weight"])
        scale_factor = 1.2  # 重みを 1.2 倍にする
        additive_penalty = base_w * (scale_factor - 1.0)
        print(f"\nScaling edge ({u}, {v}) weight by x{scale_factor:.2f} ( +{additive_penalty:.3f} ) ...")
        env.add_edge_penalty([(u, v)], penalty=additive_penalty)

        edges_after = env.get_edge_info()
        changed = next(
            (e for e in edges_after if (e["u"] == u and e["v"] == v) or (e["u"] == v and e["v"] == u)),
            None,
        )
        print("Edge after penalty:", changed)

    env.close()


if __name__ == "__main__":
    main()
