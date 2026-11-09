"""Headless training-curve and learned-route plots."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: list[float], window: int = 100) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return x
    sums = np.cumsum(np.insert(x, 0, 0.0))
    counts = np.minimum(np.arange(1, len(x)+1), window)
    ends = sums[1:]
    starts = np.zeros_like(ends)
    starts[window:] = sums[1:len(x)-window+1]
    return (ends - starts) / counts


def plot_training(metrics: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key, label in zip(axes.flat, ['reward','success','steps','collisions'],
                              ['Episode reward','Success (0/1)','Steps','Collisions']):
        values = [float(m[key]) for m in metrics]
        ax.plot(np.arange(1, len(values)+1), moving_average(values), linewidth=1.5)
        ax.set(title=f'{label} — trailing 100 episode mean', xlabel='Episode', ylabel=label)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_route(obstacles: np.ndarray, route: list[tuple[int,int]], goal: tuple[int,int],
               path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(obstacles, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    if route:
        ys, xs = zip(*route)
        ax.plot(xs, ys, '-o', linewidth=1.5, markersize=3, label='Greedy route')
        ax.scatter([xs[0]], [ys[0]], marker='s', s=110, label='Start')
    ax.scatter([goal[1]], [goal[0]], marker='*', s=180, label='Goal')
    ax.set(xticks=np.arange(obstacles.shape[1]), yticks=np.arange(obstacles.shape[0]), title=title)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
