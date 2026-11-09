
"""Greedy validation for a trained DQN agent."""

import numpy as np

from environments.gridworld_env import RobotNavigationEnv


def validate_agent(agent, environment_config, seeds):
    """Evaluate an agent without exploration or learning.

    Args:
        agent: DQNAgent instance.
        environment_config: Environment keyword arguments.
        seeds: Map seeds reserved for validation.

    Returns:
        Dictionary of aggregate validation metrics.
    """
    seeds = list(seeds)

    if not seeds:
        raise ValueError("Validation seeds cannot be empty")

    env = RobotNavigationEnv(**environment_config)

    rewards = []
    successes = []
    episode_lengths = []
    collisions = []

    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))

            total_reward = 0.0
            total_collisions = 0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.act(obs, explore=False)

                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                total_reward += float(reward)
                total_collisions += int(info["collision"])
                steps += 1

            rewards.append(total_reward)
            successes.append(int(terminated))
            episode_lengths.append(steps)
            collisions.append(total_collisions)

    finally:
        env.close()

    return {
        "episodes": len(seeds),
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
        "mean_collisions": float(np.mean(collisions)),
    }
