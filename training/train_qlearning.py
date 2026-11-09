"""Train tabular Q-learning on one fixed obstacle map and fixed goal."""
from __future__ import annotations
import argparse
import csv
import logging
from pathlib import Path
import numpy as np
import yaml
from agents.q_learning import QLearningAgent, QLearningConfig
from environments.fixed_map_env import FixedMapNavigationEnv
from visualization.plot_training import plot_training

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, render_mode: str | None = None) -> FixedMapNavigationEnv:
    section = cfg['map']
    rows = section['rows']
    if len(rows) != section['height'] or any(len(r) != section['width'] or set(r) - {'0','1'} for r in rows):
        raise ValueError('Configured map dimensions or cell values are invalid')
    grid = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    return FixedMapNavigationEnv(grid, tuple(section['goal']), max_steps=cfg['environment']['max_steps'],
                                 cell_size=cfg['environment'].get('cell_size', 48), render_mode=render_mode)


def train(cfg: dict) -> tuple[QLearningAgent, list[dict]]:
    env = build_env(cfg)
    try:
        params = cfg['training']
        if params['episodes'] < 1:
            raise ValueError('Training episodes must be positive')
        agent = QLearningAgent(env.fixed_obstacles, env.fixed_goal,
                               QLearningConfig(**cfg['agent']), seed=params['seed'])
        metrics: list[dict] = []
        for episode in range(params['episodes']):
            obs, _ = env.reset(seed=params['seed'] + episode)
            reward_sum, collisions, steps, success = 0.0, 0, 0, False
            while True:
                state = agent.state(obs)
                action = agent.select_action(state, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = agent.state(next_obs)
                agent.update(state, action, reward, next_state, terminated, truncated)
                reward_sum += reward
                collisions += int(info['collision'])
                steps += 1
                obs = next_obs
                if terminated or truncated:
                    success = bool(terminated)
                    break
            metrics.append({'episode': episode+1, 'reward': reward_sum, 'steps': steps,
                            'collisions': collisions, 'success': int(success), 'epsilon': agent.epsilon})
            agent.decay_epsilon()
            if (episode + 1) % 1000 == 0:
                recent = metrics[-1000:]
                LOGGER.info('episode=%d success_rate=%.3f mean_reward=%.2f epsilon=%.3f',
                            episode+1, np.mean([m['success'] for m in recent]),
                            np.mean([m['reward'] for m in recent]), agent.epsilon)
        return agent, metrics
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/qlearning.yaml')
    parser.add_argument('--episodes', type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg['training']['episodes'] = args.episodes
    agent, metrics = train(cfg)
    agent.save(cfg['training']['checkpoint'])
    csv_path = Path(cfg['training']['metrics'])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    plot_training(metrics, cfg['training']['training_plot'])
    LOGGER.info('Saved checkpoint=%s metrics=%s', cfg['training']['checkpoint'], csv_path)


if __name__ == '__main__':
    main()
