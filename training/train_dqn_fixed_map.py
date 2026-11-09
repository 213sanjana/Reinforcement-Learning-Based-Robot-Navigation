"""Run: python -m training.train_dqn --config configs/dqn.yaml"""
import argparse
import csv
import logging
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import yaml
from agents.dqn_agent import DQNAgent, DQNConfig
from environments.gridworld_env import RobotNavigationEnv
from visualization.plot_dqn import plot_training

LOG = logging.getLogger(__name__)

def load_config(path):
    with open(path, encoding='utf8') as f:
        return yaml.safe_load(f)

def make_env(cfg):
    return RobotNavigationEnv(**cfg['environment'])

def train(cfg):
    seed = int(cfg['training']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = make_env(cfg)
    agent = DQNAgent(env.height, env.width, DQNConfig(**cfg['agent']), seed=seed, device=cfg['training']['device'])
    total = int(cfg['training']['total_steps'])
    if total < 1:
        raise ValueError('total_steps must be positive')
    obs, _ = env.reset(seed=seed)
    ep_reward = ep_steps = ep_collisions = 0
    losses = []
    metrics = []
    recent = deque(maxlen=100)
    episode = 0
    try:
        for step in range(1, total + 1):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            loss = agent.observe_step(obs, action, reward, next_obs, terminated, truncated)
            if loss is not None:
                losses.append(loss)
            ep_reward += float(reward)
            ep_steps += 1
            ep_collisions += int(info['collision'])
            obs = next_obs
            if terminated or truncated:
                episode += 1
                row = {'episode': episode, 'global_step': step, 'reward': ep_reward,
                       'steps': ep_steps, 'collisions': ep_collisions, 'success': int(terminated),
                       'epsilon': agent.epsilon,
                       'mean_loss': float(np.mean(losses)) if losses else float('nan')}
                metrics.append(row)
                recent.append(row)
                ep_reward = ep_steps = ep_collisions = 0
                losses = []
                obs, _ = env.reset(seed=seed)
            if step % int(cfg['training']['log_interval']) == 0:
                LOG.info('step=%d/%d episodes=%d success_100=%.3f reward_100=%.2f epsilon=%.3f device=%s',
                         step, total, episode, np.mean([x['success'] for x in recent]) if recent else 0.,
                         np.mean([x['reward'] for x in recent]) if recent else 0., agent.epsilon, agent.device)
        agent.save(cfg['training']['checkpoint'])
        if metrics:
            path = Path(cfg['training']['metrics'])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', newline='', encoding='utf8') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            plot_training(metrics, cfg['training']['figure'])
        LOG.info('Saved checkpoint %s; %d completed episodes', cfg['training']['checkpoint'], len(metrics))
        return agent, metrics
    finally:
        env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/dqn.yaml')
    p.add_argument('--steps', type=int, help='Override training steps for smoke testing')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg['training']['total_steps'] = args.steps
    if args.device is not None:
        cfg['training']['device'] = args.device
    train(cfg)

if __name__ == '__main__':
    main()
