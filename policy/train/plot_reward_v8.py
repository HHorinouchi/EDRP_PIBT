"""Plot reward_mean and v8 trajectories from ES training logs."""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_series(csv_path: Path) -> Tuple[List[float], List[float], List[float]]:
    iterations: List[float] = []
    reward_mean: List[float] = []
    param_v8: List[float] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                iterations.append(float(row.get("iter", len(iterations) + 1)))
                reward_mean.append(float(row["reward_mean"]))
                param_v8.append(float(row["v8"]))
            except (KeyError, TypeError, ValueError):
                iterations.pop() if iterations else None
                reward_mean.pop() if reward_mean else None
                param_v8.pop() if param_v8 else None
                continue
    return iterations, reward_mean, param_v8


def _plot_single(csv_path: Path, output_dir: Path) -> Path:
    iters, rewards, v8_vals = _parse_series(csv_path)
    if not iters:
        raise ValueError(f"No usable data in {csv_path}")

    fig, ax_reward = plt.subplots()
    line_reward, = ax_reward.plot(iters, rewards, color="tab:blue", label="reward_mean")
    ax_reward.set_xlabel("iteration")
    ax_reward.set_ylabel("reward_mean", color="tab:blue")
    ax_reward.tick_params(axis="y", labelcolor="tab:blue")
    ax_reward.grid(True, axis="x", linestyle="--", linewidth=0.5)

    ax_param = ax_reward.twinx()
    line_v8, = ax_param.plot(iters, v8_vals, color="tab:red", label="v8")
    ax_param.set_ylabel("v8", color="tab:red")
    ax_param.tick_params(axis="y", labelcolor="tab:red")

    lines = [line_reward, line_v8]
    labels = [line.get_label() for line in lines]
    ax_reward.legend(lines, labels, loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_reward_v8.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _iter_csv_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
    else:
        for path in sorted(root.glob("*.csv")):
            if path.is_file():
                yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward_mean and v8 time series from training logs.")
    parser.add_argument(
        "logs_path",
        type=Path,
        help="CSV file or directory containing training log CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store plots (defaults to the log directory).",
    )
    args = parser.parse_args()

    target_dir = args.output_dir if args.output_dir is not None else (
        args.logs_path.parent if args.logs_path.is_file() else args.logs_path
    )

    outputs: List[Path] = []
    for csv_path in _iter_csv_files(args.logs_path):
        try:
            outputs.append(_plot_single(csv_path, target_dir))
        except Exception as exc:
            print(f"[WARN] Skip {csv_path}: {exc}")
    if not outputs:
        raise SystemExit("No plots were generated.")
    for out_path in outputs:
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()