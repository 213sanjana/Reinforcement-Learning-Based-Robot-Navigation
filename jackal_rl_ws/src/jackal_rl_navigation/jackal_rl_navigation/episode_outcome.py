"""Determine RL episode outcomes using goal and collision state."""

import math


def evaluate_episode_outcome(
    goal_distance,
    goal_tolerance,
    collision_detected,
    episode_steps,
    max_episode_steps,
    goal_bonus=20.0,
    collision_penalty=-10.0,
):
    """Return termination flags, reward adjustment and diagnostics."""
    goal_distance = float(goal_distance)
    goal_tolerance = float(goal_tolerance)
    goal_bonus = float(goal_bonus)
    collision_penalty = float(collision_penalty)

    if not all(
        math.isfinite(value)
        for value in (
            goal_distance,
            goal_tolerance,
            goal_bonus,
            collision_penalty,
        )
    ):
        raise ValueError("Outcome parameters must be finite")

    if goal_distance < 0 or goal_tolerance < 0:
        raise ValueError("Distances must be nonnegative")

    if collision_penalty > 0:
        raise ValueError("Collision penalty must not be positive")

    if max_episode_steps < 1 or episode_steps < 0:
        raise ValueError("Invalid episode step limits")

    collision = bool(collision_detected)

    # Collision takes priority over goal success.
    goal_reached = (
        not collision
        and goal_distance <= goal_tolerance
    )

    terminated = collision or goal_reached

    truncated = (
        not terminated
        and episode_steps >= max_episode_steps
    )

    if collision:
        reward_adjustment = collision_penalty
        reason = "collision"
    elif goal_reached:
        reward_adjustment = goal_bonus
        reason = "goal_reached"
    elif truncated:
        reward_adjustment = 0.0
        reason = "time_limit"
    else:
        reward_adjustment = 0.0
        reason = None

    info = {
        "collision_detected": collision,
        "goal_reached": goal_reached,
        "time_limit_reached": truncated,
        "termination_reason": reason,
        "reward_adjustment": reward_adjustment,
    }

    return (
        terminated,
        truncated,
        reward_adjustment,
        info,
    )
