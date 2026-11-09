"""Headless plots of DQN training history."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_training(metrics, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.asarray([m['global_step'] for m in metrics])
    for ax, key, title in zip(axes.flat, ['reward', 'success', 'steps', 'mean_loss'],
                               ['Reward', 'Success rate', 'Episode length', 'TD loss']):
        values = np.asarray([m[key] for m in metrics], dtype=float)
        # Trailing mean of up to 100 completed episodes; ignore warmup NaNs.
        smoothed = [np.nanmean(values[max(0, i-99):i+1]) if np.isfinite(values[max(0, i-99):i+1]).any()
                    else np.nan for i in range(len(values))]
        ax.plot(x, smoothed)
        ax.set(title=f'{title} — trailing 100 episodes', xlabel='Environment steps')
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def plot_route(obstacles, route, goal, filename, success):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(obstacles, cmap='Greys', vmin=0, vmax=1, origin='upper')
    xs = [int(p[1]) for p in route]
    ys = [int(p[0]) for p in route]
    ax.plot(xs, ys, '-o', markersize=3, label='DQN route')
    ax.scatter([xs[0]], [ys[0]], marker='s', s=110, label='Start')
    ax.scatter([goal[1]], [goal[0]], marker='*', s=170, label='Goal')
    ax.set_xticks(range(obstacles.shape[1]))
    ax.set_yticks(range(obstacles.shape[0]))
    ax.grid(alpha=.2)
    ax.set_title('DQN: goal reached' if success else 'DQN: goal not reached')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
