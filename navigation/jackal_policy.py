"""Selectable Jackal policy interface.

Every backend receives the 40-float observation and returns a normalized
action in [-1, 1]^2. This module validates checkpoint metadata and refuses
to open weight files. It does not call a trainer or start a simulator.
"""

from __future__ import annotations

import json
import math
from typing import Mapping

import numpy as np

OBSERVATION_DIM = 40
ACTION_DIM = 2
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

PROTECTED_MARKERS = (
    "initialized_ppo.zip",
    "jackal_6g1/run_",
)


class PolicyError(ValueError):
    """The policy contract was violated or no weights are available."""


def validate_observation(observation):
    """Return a finite 40-float vector or raise PolicyError."""

    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (OBSERVATION_DIM,):
        raise PolicyError(
            f"Observation must have shape ({OBSERVATION_DIM},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise PolicyError("Observation contains a non-finite value")
    return values


def validate_action(action):
    """Return a finite normalized action or raise PolicyError."""

    values = np.asarray(action, dtype=np.float64)
    if values.shape != (ACTION_DIM,):
        raise PolicyError(
            f"Action must have shape ({ACTION_DIM},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise PolicyError("Action contains a non-finite value")
    if not np.all((values >= ACTION_LOW) & (values <= ACTION_HIGH)):
        raise PolicyError("Action values must be within [-1, 1]")
    return values.astype(np.float64, copy=False)


def bind_device(device):
    """Accept cpu, or cuda when the installed torch build can see a GPU."""

    if device == "cpu":
        return "cpu"
    if device != "cuda":
        raise PolicyError(f"Unsupported device: {device}")
    try:
        import torch
    except ImportError as exc:
        raise PolicyError("CUDA was requested but torch is not installed") from exc
    if not torch.cuda.is_available():
        raise PolicyError("CUDA was requested but is not available")
    return "cuda"


def validate_checkpoint_metadata(metadata):
    """Check a sidecar description.

    A path, zip name, or protected Phase 6G.1 marker is refused. The
    returned record always has ``weights_loaded`` false.
    """

    if not isinstance(metadata, Mapping):
        raise PolicyError("Checkpoint metadata must be a mapping")

    try:
        blob = json.dumps(metadata, sort_keys=True, default=str)
    except (TypeError, ValueError) as exc:
        raise PolicyError("Checkpoint metadata is not serializable") from exc

    for marker in PROTECTED_MARKERS:
        if marker in blob:
            raise PolicyError("Refusing to load a protected checkpoint")

    for key in ("weights_path", "checkpoint_path", "zip_path"):
        if metadata.get(key):
            raise PolicyError("Refusing to open checkpoint weights")

    if metadata.get("observation_dim") != OBSERVATION_DIM:
        raise PolicyError("Checkpoint metadata does not match the 40-D observation")
    if metadata.get("action_dim") != ACTION_DIM:
        raise PolicyError("Checkpoint metadata does not match the 2-D action")

    low = metadata.get("action_low", ACTION_LOW)
    high = metadata.get("action_high", ACTION_HIGH)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high)):
        raise PolicyError("Action bounds must be numeric")
    if float(low) != ACTION_LOW or float(high) != ACTION_HIGH:
        raise PolicyError("Action bounds must be [-1, 1]")
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise PolicyError("Action bounds must be finite")

    algorithm = metadata.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise PolicyError("Checkpoint metadata requires an algorithm name")

    return {
        "algorithm": algorithm.strip(),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_low": ACTION_LOW,
        "action_high": ACTION_HIGH,
        "weights_loaded": False,
    }


class MetadataOnlyPolicy:
    """Policy slot that has metadata and no loaded weights."""

    role = "rl_policy"

    def __init__(self, metadata, device="cpu"):
        self.metadata = validate_checkpoint_metadata(metadata)
        self.device = bind_device(device)
        self._reset_count = 0

    def reset(self):
        self._reset_count += 1

    def predict(self, observation, deterministic=True):
        validate_observation(observation)
        if not isinstance(deterministic, bool):
            raise PolicyError("deterministic must be a boolean")
        raise PolicyError(
            f"No weights loaded for {self.metadata['algorithm']}"
        )


class ScriptedPolicy:
    """Test double that repeats one validated action."""

    role = "rl_policy"

    def __init__(self, action, device="cpu"):
        self.action = validate_action(action)
        self.device = bind_device(device)
        self._reset_count = 0

    def reset(self):
        self._reset_count += 1

    def predict(self, observation, deterministic=True):
        validate_observation(observation)
        if not isinstance(deterministic, bool):
            raise PolicyError("deterministic must be a boolean")
        return self.action.copy()
