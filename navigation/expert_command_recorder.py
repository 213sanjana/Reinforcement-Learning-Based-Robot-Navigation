
"""Capture commands at the existing robot publishing boundary."""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class PublishedCommand:
    monotonic_time: float
    linear: float
    angular: float
    action: tuple
    observation: np.ndarray


class ExpertCommandRecorder:
    """Record successfully published velocity commands."""

    def __init__(self):
        self.records = []

    def record(self, observation, action, linear, angular):
        observation = np.asarray(observation, dtype=np.float32).copy()
        action = np.asarray(action, dtype=np.float64)

        if observation.shape != (40,):
            raise ValueError("Expected a 40D observation")

        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation contains non-finite values")

        if action.shape != (2,):
            raise ValueError("Expected a 2D action")

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values")

        if not np.all((-1 <= action) & (action <= 1)):
            raise ValueError("Action outside normalized bounds")

        if not all(math.isfinite(float(v)) for v in (linear, angular)):
            raise ValueError("Invalid velocity command")

        record = PublishedCommand(
            monotonic_time=time.monotonic(),
            linear=float(linear),
            angular=float(angular),
            action=tuple(float(v) for v in action),
            observation=observation,
        )

        self.records.append(record)
        return record
