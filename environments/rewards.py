"""Pure, configurable reward calculation."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RewardConfig:
    goal: float = 100.0
    collision: float = -10.0
    step: float = -1.0
    progress: float = 2.0


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def calculate_reward(previous: tuple[int, int], current: tuple[int, int],
                     goal: tuple[int, int], collision: bool, reached_goal: bool,
                     config: RewardConfig) -> float:
    reward = config.step
    if collision:
        reward += config.collision
    else:
        reward += config.progress * (manhattan(previous, goal) - manhattan(current, goal))
    if reached_goal:
        reward += config.goal
    return float(reward)
