"""Configurable warehouse navigation reward.

The numbers are hyperparameters. They are not evidence that a policy
trained with them can drive the Jackal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Weights for one warehouse step. Penalties are stored as magnitudes."""

    progress_scale: float = 5.0
    goal_reward: float = 100.0
    collision_penalty: float = 100.0
    step_penalty: float = 0.01
    warning_distance_m: float = 0.70
    critical_distance_m: float = 0.30
    warning_penalty: float = 0.2
    critical_penalty: float = 1.0
    critical_slope: float = 0.0
    angular_penalty_scale: float = 0.02
    action_delta_scale: float = 0.0
    lidar_max_range_m: float = 12.0

    def __post_init__(self) -> None:
        magnitudes = (
            self.progress_scale,
            self.goal_reward,
            self.collision_penalty,
            self.step_penalty,
            self.warning_distance_m,
            self.critical_distance_m,
            self.warning_penalty,
            self.critical_penalty,
            self.critical_slope,
            self.angular_penalty_scale,
            self.action_delta_scale,
            self.lidar_max_range_m,
        )
        if not all(math.isfinite(value) for value in magnitudes):
            raise ValueError("Reward parameters must be finite")
        if any(value < 0 for value in magnitudes):
            raise ValueError("Reward magnitudes must be nonnegative")
        if self.critical_distance_m >= self.warning_distance_m:
            raise ValueError("critical_distance_m must be inside the warning distance")
        if self.lidar_max_range_m <= 0:
            raise ValueError("lidar_max_range_m must be positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "RewardConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("Reward configuration must be a mapping")
        known = {field: payload[field] for field in cls.__dataclass_fields__ if field in payload}
        return cls(**known)


def minimum_valid_range(ranges, max_range: float) -> float | None:
    """Minimum positive finite LiDAR return. Missing data stays missing."""

    if ranges is None:
        return None
    values = np.asarray(ranges, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return None
    if not math.isfinite(max_range) or max_range <= 0:
        raise ValueError("max_range must be positive and finite")
    cleaned = np.where(np.isposinf(values), max_range, values)
    valid = np.isfinite(cleaned) & (cleaned > 0.0)
    if not np.any(valid):
        return None
    return float(np.min(cleaned[valid]))


def _proximity(distance: float | None, config: RewardConfig) -> tuple[float, str]:
    if distance is None:
        return 0.0, "unavailable"
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Obstacle distance must be nonnegative and finite when present")
    if distance >= config.warning_distance_m:
        return 0.0, "clear"
    if distance <= config.critical_distance_m:
        extra = config.critical_slope * (config.critical_distance_m - distance)
        return -(config.critical_penalty + extra), "critical"
    span = config.warning_distance_m - config.critical_distance_m
    fraction = (config.warning_distance_m - distance) / span
    return -(config.warning_penalty * fraction), "warning"


def calculate_reward(
    config: RewardConfig,
    *,
    previous_goal_distance: float,
    current_goal_distance: float,
    goal_reached: bool,
    collision: bool,
    min_obstacle_distance: float | None,
    angular_velocity: float,
    previous_action=None,
    action=None,
) -> tuple[float, dict]:
    """Return the scalar reward and the component record for one step."""

    previous_goal_distance = float(previous_goal_distance)
    current_goal_distance = float(current_goal_distance)
    angular_velocity = float(angular_velocity)
    if not all(
        math.isfinite(value)
        for value in (
            previous_goal_distance,
            current_goal_distance,
            angular_velocity,
        )
    ):
        raise ValueError("Goal distances and angular velocity must be finite")
    if previous_goal_distance < 0 or current_goal_distance < 0:
        raise ValueError("Goal distances cannot be negative")

    progress_reward = config.progress_scale * (
        previous_goal_distance - current_goal_distance
    )
    goal_reward = float(config.goal_reward) if goal_reached and not collision else 0.0
    collision_penalty = -float(config.collision_penalty) if collision else 0.0
    step_penalty = -float(config.step_penalty)
    proximity_penalty, proximity_zone = _proximity(min_obstacle_distance, config)
    angular_penalty = -config.angular_penalty_scale * abs(angular_velocity)
    smoothness = 0.0
    if config.action_delta_scale > 0 and previous_action is not None and action is not None:
        before = np.asarray(previous_action, dtype=np.float64)
        after = np.asarray(action, dtype=np.float64)
        if before.shape != (2,) or after.shape != (2,) or not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise ValueError("Action delta requires finite actions of shape (2,)")
        smoothness = -config.action_delta_scale * float(np.abs(after - before).sum())

    total = (
        progress_reward
        + goal_reward
        + collision_penalty
        + step_penalty
        + proximity_penalty
        + angular_penalty
        + smoothness
    )
    if not math.isfinite(total):
        raise ValueError("Reward total must be finite")
    info = {
        "progress_reward": float(progress_reward),
        "goal_reward": float(goal_reward),
        "collision_penalty": float(collision_penalty),
        "step_penalty": float(step_penalty),
        "proximity_penalty": float(proximity_penalty),
        "angular_penalty": float(angular_penalty),
        "action_smoothness_penalty": float(smoothness),
        "total_reward": float(total),
        "goal_distance": float(current_goal_distance),
        "minimum_obstacle_distance": (
            None if min_obstacle_distance is None else float(min_obstacle_distance)
        ),
        "proximity_zone": proximity_zone,
        "collision_state": bool(collision),
    }
    return float(total), info
