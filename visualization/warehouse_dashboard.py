"""Plot recorded warehouse training logs.

The dashboard reads CSV and trajectory JSON. It does not train or infer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _column(path: Path, name: str) -> list[float]:
    import csv

    values = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or name not in reader.fieldnames:
            raise ValueError(f"{path} has no {name} column")
        for row in reader:
            text = row.get(name) or ""
            if text == "":
                continue
            values.append(float(text))
    return values


def plot_series(path: Path, x_name: str, y_names: list[str], output: Path, title: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    figure, axis = plt.subplots(figsize=(8, 4))
    x_values = _column(path, x_name)
    for name in y_names:
        y_values = _column(path, name)
        count = min(len(x_values), len(y_values))
        axis.plot(x_values[:count], y_values[:count], label=name)
    axis.set_xlabel(x_name)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_trajectory(path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(payload.get("points") or [], dtype=float)
    figure, axis = plt.subplots(figsize=(6, 6))
    for obstacle in payload.get("obstacles") or []:
        center = obstacle.get("center")
        radius = obstacle.get("radius")
        if center is not None and radius is not None:
            patch = plt.Circle(center, radius, color="0.6")
            axis.add_patch(patch)
    if len(points):
        axis.plot(points[:, 0], points[:, 1], color="tab:blue", label="path")
        axis.scatter([points[0, 0]], [points[0, 1]], color="tab:green", label="start")
    goal = payload.get("goal")
    if goal is not None:
        axis.scatter([goal[0]], [goal[1]], color="tab:red", label="goal")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(str(payload.get("world_name") or "trajectory"))
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_run(run_directory: Path, output_directory: Path | None = None) -> list[Path]:
    run_directory = Path(run_directory)
    output_directory = Path(output_directory) if output_directory else run_directory / "plots"
    written = []
    progress = run_directory / "progress.csv"
    episodes = run_directory / "episodes.csv"
    if progress.is_file() and progress.stat().st_size > 0:
        target = output_directory / "training_rates.png"
        plot_series(
            progress,
            "timestep",
            ["success_rate", "collision_rate", "timeout_rate"],
            target,
            "Warehouse training rates",
        )
        written.append(target)
        target = output_directory / "training_return.png"
        plot_series(progress, "timestep", ["mean_return"], target, "Mean episode return")
        written.append(target)
    if episodes.is_file() and episodes.stat().st_size > 0:
        target = output_directory / "goal_distance.png"
        plot_series(episodes, "episode_index", ["goal_distance"], target, "Terminal goal distance")
        written.append(target)
        target = output_directory / "speeds.png"
        plot_series(
            episodes,
            "episode_index",
            ["mean_linear_speed_mps", "mean_angular_speed_radps"],
            target,
            "Mean commanded speeds",
        )
        written.append(target)
    trajectories = sorted(run_directory.glob("trajectory_*.json"))
    for index, trajectory in enumerate(trajectories):
        target = output_directory / f"trajectory_{index}.png"
        plot_trajectory(trajectory, target)
        written.append(target)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot a warehouse PPO run directory. Does not train.")
    parser.add_argument("run_directory")
    parser.add_argument("--output-directory")
    args = parser.parse_args(argv)
    written = plot_run(
        Path(args.run_directory),
        None if args.output_directory is None else Path(args.output_directory),
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
