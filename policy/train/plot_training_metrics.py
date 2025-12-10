"""Visualization utilities for continue_train_by_gpu logs.

This script reads a training log CSV produced by continue_train_by_gpu.py and
emits several diagnostic plots:

1. Reward curve (mean ± std) with maximum reward overlay.
2. Event rates for goal / collision / timeup when those columns are available.
3. Average steps (and task completion) per iteration if recorded.
4. Parameter heatmap showing how each learned coefficient evolves.

Example usage:

    python policy/train/plot_training_metrics.py \
        --log-csv policy/train/train_log_gpu.csv \
        --output-dir policy/train/plots

The script relies only on NumPy and Matplotlib, both already listed in
requirements.txt.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

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


def _read_log(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        columns: Dict[str, List[float]] = {key: [] for key in reader.fieldnames or []}
        for row in reader:
            for key, storage in columns.items():
                storage.append(_safe_float(row.get(key, "nan")))
    return columns


def _plot_reward_curves(columns: Dict[str, List[float]], out_dir: Path) -> None:
    if "iter" not in columns or "reward_mean" not in columns:
        return
    iterations = np.asarray(columns["iter"])
    reward_mean = np.asarray(columns["reward_mean"])
    reward_std = np.asarray(columns.get("reward_std", []))
    reward_max = np.asarray(columns.get("reward_max", []))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, reward_mean, label="mean", color="tab:blue")
    if reward_std.size == reward_mean.size:
        upper = reward_mean + reward_std
        lower = reward_mean - reward_std
        ax.fill_between(iterations, lower, upper, color="tab:blue", alpha=0.2, label="±1σ")
    if reward_max.size == reward_mean.size:
        ax.plot(iterations, reward_max, label="max", color="tab:orange", linestyle="--")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Reward progression")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    out_path = out_dir / "reward_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_event_rates(columns: Dict[str, List[float]], out_dir: Path) -> None:
    keys = [
        ("best_goal_rate", "Goal"),
        ("best_collision_rate", "Collision"),
        ("best_timeup_rate", "Time-up"),
    ]
    if not all(k in columns for k, _ in keys) or "iter" not in columns:
        return
    iterations = np.asarray(columns["iter"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in keys:
        series = np.asarray(columns[key])
        if series.size != iterations.size:
            continue
        ax.plot(iterations, series, label=label)
    if not ax.lines:
        plt.close(fig)
        return
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Episode outcome rates (best candidate)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    out_path = out_dir / "event_rates.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_steps(columns: Dict[str, List[float]], out_dir: Path) -> None:
    if "best_avg_steps" not in columns or "iter" not in columns:
        return
    iterations = np.asarray(columns["iter"])
    avg_steps = np.asarray(columns["best_avg_steps"])
    avg_tasks = np.asarray(columns.get("best_avg_task_completion", []))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(iterations, avg_steps, color="tab:green", label="Average steps")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Steps", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.grid(True, linestyle=":", linewidth=0.5)

    if avg_tasks.size == iterations.size:
        ax2 = ax1.twinx()
        ax2.plot(iterations, avg_tasks, color="tab:purple", label="Task completion")
        ax2.set_ylabel("Tasks", color="tab:purple")
        ax2.tick_params(axis="y", labelcolor="tab:purple")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title("Average steps / task completion (best candidate)")
    out_path = out_dir / "steps_and_tasks.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_param_heatmap(columns: Dict[str, List[float]], out_dir: Path) -> None:
    param_columns = sorted([c for c in columns.keys() if c.startswith("v")], key=lambda x: int(x[1:]))
    if not param_columns or "iter" not in columns:
        return
    matrix = np.asarray([[columns[c][i] for c in param_columns] for i in range(len(columns[param_columns[0]]))])
    iterations = np.asarray(columns["iter"])

    fig, ax = plt.subplots(figsize=(max(6, len(param_columns) * 0.6), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(param_columns)))
    ax.set_xticklabels(param_columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(iterations)))
    ax.set_yticklabels([f"{int(it)}" for it in iterations])
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Iteration")
    ax.set_title("Parameter evolution (mean vector)")
    fig.colorbar(im, ax=ax, label="Value")

    out_path = out_dir / "parameter_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_plots(log_csv: Path, output_dir: Path) -> None:
    if not log_csv.exists():
        raise FileNotFoundError(f"Log file not found: {log_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = _read_log(log_csv)
    if not columns:
        raise ValueError(f"Log file has no data: {log_csv}")

    _plot_reward_curves(columns, output_dir)
    _plot_event_rates(columns, output_dir)
    _plot_steps(columns, output_dir)
    _plot_param_heatmap(columns, output_dir)
    print(f"Saved plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from continue_train_by_gpu logs.")
    parser.add_argument("--log-csv", type=Path, required=True, help="Path to the training log CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("policy/train/plots"),
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()
    generate_plots(args.log_csv, args.output_dir)


if __name__ == "__main__":
    main()