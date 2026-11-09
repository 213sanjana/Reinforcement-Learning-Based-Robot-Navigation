"""Units and ranges for the Jackal command path.

Normalized actions are dimensionless values in [-1, 1]. Physical commands
are meters per second and radians per second. Mapping is the only conversion
between them, and a command is mapped once.
"""

from __future__ import annotations

import math

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.safety_monitor import SafetyMonitor

from navigation.jackal_policy import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    OBSERVATION_DIM,
    PolicyError,
    validate_action,
)


def command_contract():
    """Return the limits the current classes actually construct."""

    mapper = ActionMapper()
    safety = SafetyMonitor()
    observation = ObservationBuilder()
    return {
        "observation_dimension": observation.observation_dim,
        "lidar_sectors": observation.lidar_bins,
        "lidar_max_range_m": observation.lidar_max_range,
        "goal_distance_scale_m": observation.goal_max_distance,
        "stored_heading_range": (-1.0, 1.0),
        "stored_heading_units": "wrapped radians divided by pi",
        "normalized_action_shape": (ACTION_DIM,),
        "normalized_action_range": (ACTION_LOW, ACTION_HIGH),
        "normalized_action_units": "dimensionless",
        "max_forward_m_s": mapper.max_forward,
        "max_reverse_m_s": mapper.max_reverse,
        "max_angular_rad_s": mapper.max_angular,
        "linear_acceleration_m_s2": mapper.linear_acceleration,
        "angular_acceleration_rad_s2": mapper.angular_acceleration,
        "min_command_interval_s": mapper.min_command_interval,
        "forward_stop_m": safety.forward_stop_distance,
        "reverse_stop_m": safety.reverse_stop_distance,
        "rotation_clearance_m": safety.rotation_clearance,
        "goal_tolerance_m": 0.30,
        "physical_units": "m/s, rad/s",
    }


def assert_contract_matches_defaults():
    contract = command_contract()
    if contract["observation_dimension"] != OBSERVATION_DIM:
        raise PolicyError("Observation dimension drifted from 40")
    if contract["normalized_action_shape"] != (ACTION_DIM,):
        raise PolicyError("Action dimension drifted from 2")
    if contract["max_forward_m_s"] != 0.30:
        raise PolicyError("Forward speed limit drifted")
    if contract["max_reverse_m_s"] != 0.15:
        raise PolicyError("Reverse speed limit drifted")
    if contract["max_angular_rad_s"] != 0.60:
        raise PolicyError("Angular speed limit drifted")
    return contract


def accept_normalized_action(action):
    """Validate a policy output. This does not convert it to velocity."""

    return validate_action(action)


def assert_physical_command(linear, angular, mapper=None):
    """Reject a velocity outside the mapper limits.

    The values are m/s and rad/s. They are not accepted as a policy action.
    """

    limits = mapper if mapper is not None else ActionMapper()
    linear = float(linear)
    angular = float(angular)
    if not math.isfinite(linear) or not math.isfinite(angular):
        raise PolicyError("Physical command must be finite")
    if linear < -limits.max_reverse or linear > limits.max_forward:
        raise PolicyError("Linear velocity is outside the speed limit")
    if abs(angular) > limits.max_angular:
        raise PolicyError("Angular velocity is outside the speed limit")
    return linear, angular


def goal_loop_enabled(value):
    """Interpret the launch flag. The string 'false' stays disabled."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False
