#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_node_counts(map_dir: Path) -> dict:
    node_counts = {}
    for map_path in map_dir.iterdir():
        if not map_path.is_dir():
            continue
        node_csv = map_path / "node.csv"
        if not node_csv.exists():
            continue
        try:
            with node_csv.open() as f:
                rows = list(csv.reader(f))
            node_counts[map_path.name] = max(len(rows) - 1, 0)
        except Exception:
            continue
    return node_counts


def extract_metrics(vec: np.ndarray) -> list:
    # Use raw v1..v8 as requested (legacy schema)
    return [float(v) for v in vec]


def main() -> None:
    log_dir = Path("sweep_results/logs")
    map_dir = Path("drp_env/map")
    out_dir = Path("assets/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    node_counts = load_node_counts(map_dir)
    records = []
    for path in log_dir.glob("train_log_map_*_agents_*.csv"):
        name = path.stem
        parts = name.split("_")
        if "agents" not in parts:
            continue
        agents_idx = parts.index("agents")
        map_name = "_".join(parts[2:agents_idx])
        agent_num = int(parts[agents_idx + 1])
        if map_name not in node_counts or node_counts[map_name] <= 0:
            continue
        last_row = None
        with path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                last_row = row
        if last_row is None:
            continue
        vec = []
        for i in range(1, 9):
            vec.append(float(last_row.get(f"v{i}", "nan")))
        if any(np.isnan(vec)):
            continue
        n_nodes = node_counts[map_name]
        ratio = agent_num / float(n_nodes)
        if ratio >= 0.3:
            continue
        records.append((map_name, agent_num, n_nodes, np.array(vec, dtype=float)))

    if not records:
        print("No records to plot.")
        return

    param_names = [f"v{i}" for i in range(1, 9)]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    for map_name, agent_num, n_nodes, vec in sorted(records, key=lambda x: x[0]):
        y = extract_metrics(vec)
        x = list(range(len(param_names)))
        ax.plot(x, y, marker="o", label=f"{map_name} ({agent_num}/{n_nodes})")

    ax.set_title("agent_num / nodes group ≈ 0.25")
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=25, ha="right")
    ax.set_ylabel("value / ratio")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    plot_path = out_dir / "param_by_ratio_line_0.25.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(plot_path)


if __name__ == "__main__":
    main()
