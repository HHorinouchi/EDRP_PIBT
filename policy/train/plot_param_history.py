#!/usr/bin/env python3
"""Plot parameter trajectories from ES training CSV logs."""

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAM_NAMES = [
    "goal_weight",
    "pick_weight",
    "drop_weight",
    "assign_pick_weight",
    "assign_drop_weight",
    "congestion_weight",
    "step_tolerance",
    "assign_spread_weight",
]


def load_parameter_history(csv_path: Path) -> (List[int], List[List[float]]):
    iterations: List[int] = []
    values: List[List[float]] = [[] for _ in PARAM_NAMES]
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iter_idx = int(float(row["iter"]))
            except (KeyError, ValueError):
                continue
            iterations.append(iter_idx)
            for idx, name in enumerate(PARAM_NAMES):
                key = f"v{idx}"
                try:
                    values[idx].append(float(row[key]))
                except (KeyError, ValueError):
                    values[idx].append(float("nan"))
    return iterations, values


def plot_parameters(iterations: List[int], values: List[List[float]], output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for series, label in zip(values, PARAM_NAMES):
        plt.plot(iterations, series, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Parameter value")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize="small", ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ES parameter trajectories from CSV logs.")
    parser.add_argument("log_csv", type=Path, help="Path to the training log CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (PNG).")
    parser.add_argument("--title", type=str, default=None, help="Plot title override.")
    args = parser.parse_args()

    iterations, values = load_parameter_history(args.log_csv)
    if not iterations:
        raise SystemExit(f"No iteration data found in {args.log_csv}")

    output_path = args.output
    if output_path is None:
        output_path = args.log_csv.with_suffix("_params.png")

    title = args.title or f"Parameter history: {args.log_csv.stem}"
    plot_parameters(iterations, values, output_path, title)
    print(f"Saved parameter plot -> {output_path}")


if __name__ == "__main__":
    main()
