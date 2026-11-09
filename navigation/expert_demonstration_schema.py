"""Schema for Jackal expert demonstrations used by behavior cloning.

The observation and action layouts match ``JackalGazeboEnv``: a 40-value
float32 observation and a 2-value float32 action in ``[-1, 1]``.
"""

from dataclasses import dataclass
import hashlib
import math
import re

import numpy as np
from gymnasium import spaces

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder


SCHEMA_VERSION = "6e.1.0"
TRAINING_ACTION_KEY = "actions"
REQUESTED_ACTION_KEY = "requested_actions"
APPLIED_VELOCITY_KEY = "applied_velocities"

_SCENARIO_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TERMINATION_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


@dataclass
class DemonstrationEpisode:
    """One aligned expert trajectory."""

    episode_id: int
    scenario_id: str
    termination_reason: str
    goal: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    requested_actions: np.ndarray
    applied_velocities: np.ndarray
    poses: np.ndarray
    simulation_timestamps_ns: np.ndarray

_ARRAY_SPECS = (
    ("observations", np.float32, (40,)),
    ("actions", np.float32, (2,)),
    ("requested_actions", np.float32, (2,)),
    ("applied_velocities", np.float32, (2,)),
    ("poses", np.float64, (3,)),
    ("simulation_timestamps_ns", np.int64, ()),
)


def jackal_policy_spaces():
    """Return the Gymnasium spaces defined by the Jackal environment."""

    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(40,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
    )
    return observation_space, action_space


def inspect_policy_spaces(env):
    """Read the spaces from an environment and require the Jackal contract."""

    observation_space = env.observation_space
    action_space = env.action_space
    validate_jackal_policy_spaces(observation_space, action_space)
    return observation_space, action_space


def validate_jackal_policy_spaces(observation_space, action_space):
    """Reject a space pair that is not the 40D Jackal policy contract."""

    if not isinstance(observation_space, spaces.Box):
        raise TypeError("Observation space must be a Gymnasium Box")

    if observation_space.shape != (40,) or observation_space.dtype != np.float32:
        raise ValueError("Observation space must be float32 with shape (40,)")

    if not (
        np.all(np.isneginf(observation_space.low))
        and np.all(np.isposinf(observation_space.high))
    ):
        raise ValueError("Observation space bounds must be unbounded")

    if not isinstance(action_space, spaces.Box):
        raise TypeError("Action space must be a Gymnasium Box")

    if action_space.shape != (2,) or action_space.dtype != np.float32:
        raise ValueError("Action space must be float32 with shape (2,)")

    if not (
        np.all(np.asarray(action_space.low, dtype=np.float64) == -1.0)
        and np.all(np.asarray(action_space.high, dtype=np.float64) == 1.0)
    ):
        raise ValueError("Action space bounds must be [-1, 1]")


def normalization_metadata(builder=None):
    """Describe how raw sensors become the 40D policy observation."""

    builder = ObservationBuilder() if builder is None else builder

    if int(builder.observation_dim) != 40 or int(builder.lidar_bins) != 36:
        raise ValueError("Observation builder is not the 40D Jackal layout")

    bins = int(builder.lidar_bins)
    return {
        "lidar_bins": bins,
        "lidar_max_range_m": float(builder.lidar_max_range),
        "goal_max_distance_m": float(builder.goal_max_distance),
        "max_linear_velocity_m_s": float(builder.max_linear_velocity),
        "max_angular_velocity_rad_s": float(builder.max_angular_velocity),
        "layout": [
            {
                "name": "lidar",
                "slice": [0, bins],
                "transform": "clip(min_sector_range / lidar_max_range_m, 0, 1)",
            },
            {
                "name": "goal_distance",
                "index": bins,
                "transform": "clip(distance_m / goal_max_distance_m, 0, 1)",
            },
            {
                "name": "goal_heading",
                "index": bins + 1,
                "transform": "wrap(heading_rad) / pi",
            },
            {
                "name": "linear_velocity",
                "index": bins + 2,
                "transform": "clip(linear_m_s / max_linear_velocity_m_s, -1, 1)",
            },
            {
                "name": "angular_velocity",
                "index": bins + 3,
                "transform": "clip(angular_rad_s / max_angular_velocity_rad_s, -1, 1)",
            },
        ],
    }


def robot_configuration_metadata(robot=None, mapper=None):
    """Save the velocity limits and topics used to apply expert commands."""

    if robot is not None:
        mapper = robot.action_mapper
        command_topic = str(robot.command_topic)
        odom_topic = str(robot.odom_topic)
        scan_topic = str(robot.scan_topic)
        namespace = command_topic.strip("/").split("/", 1)[0]
    else:
        mapper = ActionMapper() if mapper is None else mapper
        namespace = "j100_0000"
        command_topic = f"/{namespace}/platform/cmd_vel_unstamped"
        odom_topic = f"/{namespace}/platform/odom"
        scan_topic = f"/{namespace}/sensors/lidar3d_0/scan"

    return {
        "platform": "clearpath_jackal",
        "namespace": namespace,
        "command_topic": command_topic,
        "odom_topic": odom_topic,
        "scan_topic": scan_topic,
        "action_mapper": {
            "max_forward_m_s": float(mapper.max_forward),
            "max_reverse_m_s": float(mapper.max_reverse),
            "max_angular_rad_s": float(mapper.max_angular),
            "linear_acceleration_m_s2": float(mapper.linear_acceleration),
            "angular_acceleration_rad_s2": float(mapper.angular_acceleration),
            "min_command_interval_s": float(mapper.min_command_interval),
        },
    }


def build_schema(observation_space, action_space, storage, normalization, robot_configuration):
    """Return the JSON-serializable dataset schema."""

    validate_jackal_policy_spaces(observation_space, action_space)

    if storage not in ("npz", "hdf5"):
        raise ValueError("storage must be 'npz' or 'hdf5'")

    return {
        "schema_version": SCHEMA_VERSION,
        "storage": storage,
        "training_action_key": TRAINING_ACTION_KEY,
        "requested_action_key": REQUESTED_ACTION_KEY,
        "applied_velocity_key": APPLIED_VELOCITY_KEY,
        "observation_space": _describe_space(observation_space),
        "action_space": _describe_space(action_space),
        "observation_normalization": normalization,
        "robot_configuration": robot_configuration,
        "arrays": {
            "observations": "float32[T, 40]",
            "actions": "float32[T, 2] Gymnasium action of the applied command",
            "requested_actions": "float32[T, 2] Gymnasium action requested by the expert",
            "applied_velocities": "float32[T, 2] linear m/s, angular rad/s",
            "poses": "float64[T, 3] x_m, y_m, yaw_rad",
            "simulation_timestamps_ns": "int64[T]",
            "goal": "float64[2] x_m, y_m",
        },
        "alignment": (
            "Row i of observations, actions, requested_actions, "
            "applied_velocities, poses, and simulation_timestamps_ns "
            "is one transition"
        ),
        "success_rule": "termination_reason == GOAL_REACHED and T >= 1",
        "retains_failed_episodes": True,
    }


def validate_episode(episode, action_space, robot_configuration):
    """Validate one episode and return arrays ready to store."""

    episode_id = int(episode.episode_id)
    if episode_id < 0 or episode_id > 999999:
        raise ValueError("episode_id must be in 0..999999")

    scenario_id = str(episode.scenario_id)
    if _SCENARIO_ID.fullmatch(scenario_id) is None:
        raise ValueError("Invalid scenario_id")

    termination_reason = str(episode.termination_reason)
    if _TERMINATION_REASON.fullmatch(termination_reason) is None:
        raise ValueError("Invalid termination_reason")

    goal = np.array(episode.goal, dtype=np.float64, copy=True)
    if goal.shape != (2,) or not np.all(np.isfinite(goal)):
        raise ValueError("Goal must be a finite x, y pair")

    arrays = {}
    count = None
    for name, dtype, tail in _ARRAY_SPECS:
        value = np.asarray(getattr(episode, name), dtype=dtype)
        if tail == ():
            expected_ndim = 1
        else:
            expected_ndim = 1 + len(tail)

        if value.ndim != expected_ndim:
            raise ValueError(f"Invalid {name} shape")

        if tail != () and value.shape[1:] != tail:
            raise ValueError(f"Invalid {name} shape")

        if count is None:
            count = int(value.shape[0])
        elif int(value.shape[0]) != count:
            raise ValueError("Observation-action rows are not aligned")

        arrays[name] = np.array(value, dtype=dtype, copy=True)

    success = termination_reason == "GOAL_REACHED"
    if success and count < 1:
        raise ValueError("A successful episode needs at least one transition")

    if count > 0:
        _validate_numeric_rows(arrays, action_space, robot_configuration)

    return {
        "episode_id": np.int64(episode_id),
        "scenario_id": scenario_id,
        "termination_reason": termination_reason,
        "success": bool(success),
        "goal": goal,
        **arrays,
    }


def content_digest(payload):
    """Hash the aligned episode contents, excluding the stored digest."""

    hasher = hashlib.sha256()
    hasher.update(SCHEMA_VERSION.encode("utf-8"))

    for key in (
        "episode_id",
        "scenario_id",
        "termination_reason",
        "success",
        "goal",
        "observations",
        "actions",
        "requested_actions",
        "applied_velocities",
        "poses",
        "simulation_timestamps_ns",
    ):
        hasher.update(key.encode("utf-8"))
        value = payload[key]

        if isinstance(value, str):
            hasher.update(value.encode("utf-8"))
            continue

        if isinstance(value, (bool, np.bool_)):
            hasher.update(b"\x01" if bool(value) else b"\x00")
            continue

        if isinstance(value, (int, np.integer)) and not isinstance(value, np.ndarray):
            hasher.update(int(value).to_bytes(8, "little", signed=False))
            continue

        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())

    return hasher.hexdigest()


def episode_from_expert_result(result, episode_id, scenario_id, goal):
    """Convert a live expert result into one recordable episode."""

    transitions = list(result.transitions)
    accepted = int(result.accepted_commands)
    rejected = int(result.rejected_commands)

    if transitions and accepted != len(transitions):
        raise ValueError("Accepted-command count does not match records")

    reason = str(result.status)
    if reason == "GOAL_REACHED" and rejected != 0:
        reason = "REJECTED_COMMANDS"

    count = len(transitions)

    def column(values, dtype, width):
        if count == 0:
            shape = (0, width) if width else (0,)
            return np.zeros(shape, dtype=dtype)
        return np.stack([np.asarray(value, dtype=dtype) for value in values])

    return DemonstrationEpisode(
        episode_id=int(episode_id),
        scenario_id=str(scenario_id),
        termination_reason=reason,
        goal=np.asarray(goal, dtype=np.float64),
        observations=column(
            [item.observation for item in transitions],
            np.float32,
            40,
        ),
        actions=column(
            [item.applied_action for item in transitions],
            np.float32,
            2,
        ),
        requested_actions=column(
            [item.action for item in transitions],
            np.float32,
            2,
        ),
        applied_velocities=column(
            [
                (item.published_linear, item.published_angular)
                for item in transitions
            ],
            np.float32,
            2,
        ),
        poses=column(
            [item.pose for item in transitions],
            np.float64,
            3,
        ),
        simulation_timestamps_ns=column(
            [item.simulation_time_ns for item in transitions],
            np.int64,
            None,
        ),
    )


def _validate_numeric_rows(arrays, action_space, robot_configuration):
    for name, value in arrays.items():
        if name == "simulation_timestamps_ns":
            if np.any(value < 0):
                raise ValueError("Invalid simulation timestamp")
            if np.any(np.diff(value.astype(np.int64)) < 0):
                raise ValueError("Simulation timestamps are not monotonic")
            continue

        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains nonfinite values")

    for name in ("actions", "requested_actions"):
        for row in arrays[name]:
            if not action_space.contains(np.asarray(row, dtype=np.float32)):
                raise ValueError(f"{name} outside the Gymnasium action space")

    limits = robot_configuration["action_mapper"]
    linear = arrays["applied_velocities"][:, 0]
    angular = arrays["applied_velocities"][:, 1]
    tolerance = 1e-5

    if np.any(linear < -float(limits["max_reverse_m_s"]) - tolerance):
        raise ValueError("Applied linear velocity exceeds reverse limit")

    if np.any(linear > float(limits["max_forward_m_s"]) + tolerance):
        raise ValueError("Applied linear velocity exceeds forward limit")

    if np.any(np.abs(angular) > float(limits["max_angular_rad_s"]) + tolerance):
        raise ValueError("Applied angular velocity exceeds limit")

    if not np.all(np.isfinite(arrays["poses"])):
        raise ValueError("poses contains nonfinite values")


def _describe_space(space):
    return {
        "type": "Box",
        "shape": [int(axis) for axis in space.shape],
        "dtype": str(np.dtype(space.dtype)),
        "low": [_bound_token(value) for value in np.asarray(space.low).reshape(-1)],
        "high": [_bound_token(value) for value in np.asarray(space.high).reshape(-1)],
    }


def _bound_token(value):
    number = float(value)
    if math.isinf(number):
        return "-inf" if number < 0 else "inf"
    return number
