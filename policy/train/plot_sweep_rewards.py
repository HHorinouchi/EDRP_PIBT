#!/usr/bin/env python3
"""Plot reward_mean curves from all sweep_results logs on a single chart."""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def _read_log(csv_path: Path, max_iter: int, metric_column: str) -> Tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        iters: List[float] = []
        rewards: List[float] = []
        for row in reader:
            it_val = _safe_float(row.get("iter", "nan"))
            reward = _safe_float(row.get(metric_column, "nan"))
            if math.isnan(it_val) or math.isnan(reward):
                continue
            if int(it_val) > max_iter:
                continue
            iters.append(it_val)
            rewards.append(reward)
    if not iters:
        return np.array([]), np.array([])
    order = np.argsort(np.array(iters))
    it_sorted = np.array(iters)[order]
    reward_sorted = np.array(rewards)[order]
    return it_sorted, reward_sorted


def _parse_map_agents(path: Path) -> Tuple[str, float]:
    match = re.match(r"train_log_(.+)_agents_(\d+)\.csv", path.name)
    if match:
        map_name = match.group(1)
        agents = float(match.group(2))
        return map_name, agents
    return path.stem, float("inf")


def plot_all_logs(
    logs_dir: Path,
    max_iter: int,
    output_path: Path,
    filter_maps: List[str] | None,
    metric_column: str,
) -> None:
    if not logs_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {logs_dir}")
    csv_files = sorted(logs_dir.glob("train_log_*_agents_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sweep log CSV files found in {logs_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    normalized_filters = [name.lower() for name in filter_maps] if filter_maps else None
    series_entries: List[Tuple[str, float, np.ndarray, np.ndarray, str]] = []

    for csv_path in csv_files:
        if normalized_filters:
            lower_name = csv_path.name.lower()
            if not any(token in lower_name for token in normalized_filters):
                continue
        iterations, rewards = _read_log(csv_path, max_iter, metric_column)
        if iterations.size == 0 or rewards.size == 0:
            continue
        map_name, agent_count = _parse_map_agents(csv_path)
        if math.isfinite(agent_count):
            label = f"{map_name} (agents={int(agent_count)})"
        else:
            label = csv_path.stem
        series_entries.append((map_name, agent_count, iterations, rewards, label))

    if not series_entries:
        raise ValueError("No valid data within the specified iteration range")

    series_entries.sort(key=lambda x: (x[0], x[1]))

    for _, _, iterations, rewards, label in series_entries:
        ax.plot(iterations, rewards, label=label)

    ax.set_xlabel("Iteration")
    y_label = metric_column
    if metric_column.startswith("best_"):
        y_label = metric_column[len("best_"):]
    y_label = y_label.replace("_", " ")
    title_metric = y_label.title()

    ax.set_ylabel(y_label)
    ax.set_title(f"Sweep {title_metric} curves (iteration <= {max_iter})")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved overview plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward curves from sweep logs")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("policy/train/sweep_results/logs"),
        help="Directory containing train_log_*_agents_*.csv files",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=150,
        help="Maximum iteration to include in the plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("policy/train/sweep_results/plots/sweep_reward_overview.png"),
        help="Path to save the generated plot",
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of map name fragments to filter logs (case-insensitive)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="reward_mean",
        help="CSV column to plot (e.g., reward_mean, best_collision_rate)",
    )
    args = parser.parse_args()
    plot_all_logs(args.logs_dir, args.max_iter, args.output, args.maps, args.metric)


if __name__ == "__main__":
    main()
