"""Evaluate on unseen map seeds and paired random-action baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from agents.dqn_agent import DQNAgent
from environments.gridworld_env import RobotNavigationEnv
from training.train_dqn import load_config
from visualization.plot_dqn import plot_route

def evaluate(cfg):
    params = cfg['evaluation']
    agent = DQNAgent.load(cfg['training']['checkpoint'], device=params.get('device', 'auto'))
    env = RobotNavigationEnv(**cfg['environment'])
    if (env.height, env.width) != (agent.height, agent.width):
        raise ValueError('Checkpoint and evaluation grid dimensions differ')
    count, seed = int(params['episodes']), int(params['seed'])
    if count < 1:
        raise ValueError('episodes must be positive')
    results = {'dqn': [], 'random': []}
    try:
        for i in range(count):
            episode_seed = seed + i
            # Reset both policies with the same map and start/goal, but independent action RNG.
            for policy in ('dqn', 'random'):
                obs, _ = env.reset(seed=episode_seed)
                action_rng = np.random.default_rng(seed + 100000 + i)
                start = obs['agent'].tolist()
                route = [start]
                reward_sum, collisions, steps = 0., 0, 0
                while True:
                    action = agent.act(obs, explore=False) if policy == 'dqn' else int(action_rng.integers(4))
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_sum += float(reward)
                    collisions += int(info['collision'])
                    steps += 1
                    route.append(obs['agent'].tolist())
                    if terminated or truncated:
                        break
                results[policy].append({'reward': reward_sum, 'steps': steps,
                                        'collisions': collisions, 'success': int(terminated)})
                if policy == 'dqn' and i == 0:
                    plot_route(obs['obstacles'], route, obs['goal'], params['route_plot'], bool(terminated))
        summary = {'evaluation_seed': seed, 'episodes': count, 'held_out_map_seeds': [seed, seed + count - 1],
                   'checkpoint': str(cfg['training']['checkpoint'])}
        for policy, rows in results.items():
            summary[policy] = {'success_rate': float(np.mean([r['success'] for r in rows])),
                               'mean_reward': float(np.mean([r['reward'] for r in rows])),
                               'mean_steps': float(np.mean([r['steps'] for r in rows])),
                               'mean_collisions': float(np.mean([r['collisions'] for r in rows]))}
        dest = Path(params['results'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
        return summary
    finally:
        env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/dqn.yaml')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg['evaluation']['device'] = args.device
    print(json.dumps(evaluate(cfg), indent=2))

if __name__ == '__main__':
    main()
