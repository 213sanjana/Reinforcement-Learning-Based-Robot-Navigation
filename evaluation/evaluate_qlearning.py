"""Held-out seeded starts, greedy Q-policy versus random-action baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from agents.q_learning import QLearningAgent, map_signature
from training.train_qlearning import build_env, load_config
from visualization.plot_training import plot_route


def run_episode(env, agent: QLearningAgent | None, start: tuple[int,int],
                rng: np.random.Generator) -> dict:
    obs, _ = env.reset(options={'start': start})
    route = [tuple(map(int, obs['agent']))]
    total, collisions, steps = 0.0, 0, 0
    while True:
        action = int(rng.integers(4)) if agent is None else agent.select_action(agent.state(obs), explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        route.append(tuple(map(int, obs['agent'])))
        total += reward
        collisions += int(info['collision'])
        steps += 1
        if terminated or truncated:
            return {'success': bool(terminated), 'reward': float(total), 'steps': steps,
                    'collisions': collisions, 'route': route}


def summarize(episodes: list[dict]) -> dict:
    return {'success_rate': float(np.mean([e['success'] for e in episodes])),
            'mean_reward': float(np.mean([e['reward'] for e in episodes])),
            'mean_steps': float(np.mean([e['steps'] for e in episodes])),
            'mean_collisions': float(np.mean([e['collisions'] for e in episodes]))}


def evaluate(cfg: dict, agent: QLearningAgent) -> dict:
    env = build_env(cfg)
    try:
        if map_signature(env.fixed_obstacles, env.fixed_goal) != agent.signature:
            raise ValueError('Evaluation map/goal differs from checkpoint')
        n = int(cfg['evaluation']['episodes'])
        if n < 1:
            raise ValueError('Evaluation episodes must be positive')
        seed = int(cfg['evaluation']['seed'])
        starts_rng = np.random.default_rng(seed)
        random_rng = np.random.default_rng(seed + 1)
        starts = [env.starts[int(starts_rng.integers(len(env.starts)))] for _ in range(n)]
        q_results = [run_episode(env, agent, s, random_rng) for s in starts]
        random_results = [run_episode(env, None, s, random_rng) for s in starts]
        # Plot the first rollout, including failures (no fabricated successful path).
        plot_route(env.fixed_obstacles, q_results[0]['route'], env.fixed_goal,
                   cfg['evaluation']['route_plot'],
                   f"Q-Learning: {'goal reached' if q_results[0]['success'] else 'time limit'}")
        return {'evaluation_seed': seed, 'episodes': n, 'map_signature': agent.signature,
                'q_learning': summarize(q_results), 'random': summarize(random_results),
                'first_q_episode': {'start': list(starts[0]), 'success': q_results[0]['success'],
                                    'steps': q_results[0]['steps']}}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/qlearning.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    agent = QLearningAgent.load(args.checkpoint or cfg['training']['checkpoint'])
    report = evaluate(cfg, agent)
    path = Path(cfg['evaluation']['results'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Report: {path}')


if __name__ == '__main__':
    main()
