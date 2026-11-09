"""Jackal behavior-cloning observation and action contract.

The policy consumes the same 40 float32 values as ``JackalGazeboEnv``.
Indices 0:36 are normalized LiDAR sectors. Indices 36:40 are the robot
state: goal distance, goal heading, linear velocity, and angular velocity.
The Gymnasium action is ``[linear, angular]`` in ``[-1, 1]``.
"""

import os
from pathlib import Path

import numpy as np

from jackal_rl_navigation.action_mapper import ActionMapper
from navigation.expert_demonstration_schema import (
    TRAINING_ACTION_KEY,
    jackal_policy_spaces,
    normalization_metadata,
)


PHASE = "6f.1"
LIDAR_BINS = 36
STATE_DIM = 4
OBSERVATION_DIM = LIDAR_BINS + STATE_DIM
ACTION_DIM = 2
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_NAMES = ("linear", "angular")
STATE_NAMES = (
    "goal_distance",
    "goal_heading",
    "linear_velocity",
    "angular_velocity",
)
GOAL_DISTANCE_INDEX = 36
GOAL_HEADING_INDEX = 37
LINEAR_VELOCITY_INDEX = 38
ANGULAR_VELOCITY_INDEX = 39

if OBSERVATION_DIM != 40 or STATE_DIM != len(STATE_NAMES):
    raise RuntimeError("Jackal observation contract is internally inconsistent")
if (
    GOAL_DISTANCE_INDEX != LIDAR_BINS
    or GOAL_HEADING_INDEX != LIDAR_BINS + 1
    or LINEAR_VELOCITY_INDEX != LIDAR_BINS + 2
    or ANGULAR_VELOCITY_INDEX != LIDAR_BINS + 3
):
    raise RuntimeError("Robot-state indices do not follow the LiDAR block")


def feature_names():
    """Return the 40 observation names in vector order."""

    lidar = tuple(f"lidar_{index:02d}" for index in range(LIDAR_BINS))
    return lidar + STATE_NAMES


def split_observation(observation):
    """Split a 40D observation into LiDAR sectors and robot state."""

    if int(observation.shape[-1]) != OBSERVATION_DIM:
        raise ValueError(
            f"Expected a {OBSERVATION_DIM}-dimensional Jackal observation"
        )
    return observation[..., :LIDAR_BINS], observation[..., LIDAR_BINS:]


def describe_contract():
    """Return the observation layout and Gymnasium action bounds."""

    observation_space, action_space = jackal_policy_spaces()
    mapper = ActionMapper()
    return {
        "phase": PHASE,
        "observation_dim": OBSERVATION_DIM,
        "observation_dtype": "float32",
        "feature_names": list(feature_names()),
        "lidar_slice": [0, LIDAR_BINS],
        "robot_state_slice": [LIDAR_BINS, OBSERVATION_DIM],
        "robot_state_names": list(STATE_NAMES),
        "sensor_normalization": normalization_metadata(),
        "observation_space": {
            "shape": list(observation_space.shape),
            "dtype": observation_space.dtype.name,
        },
        "action_dim": ACTION_DIM,
        "action_dtype": "float32",
        "action_names": list(ACTION_NAMES),
        "action_low": [ACTION_LOW, ACTION_LOW],
        "action_high": [ACTION_HIGH, ACTION_HIGH],
        "action_space": {
            "shape": list(action_space.shape),
            "dtype": action_space.dtype.name,
            "low": [float(value) for value in action_space.low],
            "high": [float(value) for value in action_space.high],
        },
        "training_action_key": TRAINING_ACTION_KEY,
        "velocity_limits": {
            "max_forward_m_s": float(mapper.max_forward),
            "max_reverse_m_s": float(mapper.max_reverse),
            "max_angular_rad_s": float(mapper.max_angular),
        },
    }


def validate_recorded_schema(schema):
    """Raise unless a dataset schema matches the live Jackal contract."""

    if not isinstance(schema, dict):
        raise TypeError("Dataset schema must be a dictionary")

    if schema.get("training_action_key") != TRAINING_ACTION_KEY:
        raise ValueError("Training action key must be 'actions'")

    observation = schema.get("observation_space") or {}
    action = schema.get("action_space") or {}
    if list(observation.get("shape") or []) != [OBSERVATION_DIM]:
        raise ValueError("Observation schema is not 40-dimensional")
    if observation.get("dtype") != "float32" or action.get("dtype") != "float32":
        raise ValueError("Policy arrays must be float32")
    if list(action.get("shape") or []) != [ACTION_DIM]:
        raise ValueError("Action schema is not 2-dimensional")
    if not _bounds_equal(action.get("low"), [ACTION_LOW, ACTION_LOW]):
        raise ValueError("Action low bound must be [-1, -1]")
    if not _bounds_equal(action.get("high"), [ACTION_HIGH, ACTION_HIGH]):
        raise ValueError("Action high bound must be [1, 1]")

    recorded = schema.get("observation_normalization") or {}
    expected = normalization_metadata()
    if int(recorded.get("lidar_bins", -1)) != LIDAR_BINS:
        raise ValueError("LiDAR schema is not 36 bins")
    for key in (
        "lidar_max_range_m",
        "goal_max_distance_m",
        "max_linear_velocity_m_s",
        "max_angular_velocity_rad_s",
    ):
        if not np.isclose(float(recorded[key]), float(expected[key])):
            raise ValueError(f"Sensor normalization mismatch for {key}")
    if recorded.get("layout") != expected["layout"]:
        raise ValueError("Observation layout does not match ObservationBuilder")


def require_env_actions(actions):
    """Raise unless ``actions`` lie in the Gymnasium action box."""

    _, action_space = jackal_policy_spaces()
    array = np.asarray(actions, dtype=np.float32)
    if array.shape == (ACTION_DIM,):
        rows = array.reshape(1, ACTION_DIM)
    elif array.ndim == 2 and array.shape[1] == ACTION_DIM:
        rows = array
    else:
        raise ValueError("Actions must be shaped (2,) or (N, 2)")

    rows = np.ascontiguousarray(rows, dtype=np.float32)
    for row in rows:
        candidate = np.array(row, dtype=np.float32, copy=True)
        if not action_space.contains(candidate):
            raise ValueError("Action is outside the Gymnasium action space")
    return rows


def create_new_file(path, writer):
    """Create ``path`` atomically and refuse to replace an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _bounds_equal(recorded, expected):
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(expected):
        return False
    return bool(np.allclose(
        np.asarray(recorded, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ))
