"""Manual keyboard navigation. Usage: python -m scripts.manual_navigation --seed 42"""
import argparse
import logging
from pathlib import Path
import yaml
from environments.gridworld_env import RobotNavigationEnv
from environments.rewards import RewardConfig

LOG = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/gridworld.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    rewards = RewardConfig(**config.pop('rewards'))
    env = RobotNavigationEnv(**config, reward_config=rewards, render_mode='human')
    try:
        import pygame
        obs, info = env.reset(seed=args.seed)
        LOG.info('Start=%s Goal=%s; arrows/WASD to move; R reset; ESC quit', obs['agent'], obs['goal'])
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    keys = {pygame.K_UP:0,pygame.K_w:0,pygame.K_DOWN:1,pygame.K_s:1,
                            pygame.K_LEFT:2,pygame.K_a:2,pygame.K_RIGHT:3,pygame.K_d:3}
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        LOG.info('New map: start=%s goal=%s',obs['agent'],obs['goal'])
                    elif event.key in keys and not env._finished:
                        obs, reward, terminated, truncated, info = env.step(keys[event.key])
                        LOG.info('pos=%s reward=%.1f collision=%s distance=%d',
                                 obs['agent'],reward,info['collision'],info['distance_to_goal'])
                        if terminated or truncated:
                            LOG.info('Episode ended: goal=%s timeout=%s; press R to restart',terminated,truncated)
            pygame.time.wait(15)
    finally:
        env.close()

if __name__ == '__main__':
    main()
