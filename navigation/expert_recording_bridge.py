
"""Capture accepted expert actions and submitted ROS velocity commands."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ExpertTransition:
    observation: np.ndarray
    action: np.ndarray
    published_linear: float
    published_angular: float
    simulation_time_ns: int
    applied_action: np.ndarray
    pose: tuple


class ExpertRecordingBridge:
    """Observe the existing robot publisher without creating a new one."""

    def __init__(self, robot):
        self.robot = robot
        self.records = []
        self._pending = None
        self._published = None

        if getattr(robot, "expert_publish_callback", None) is not None:
            raise RuntimeError("An expert publish callback is already installed")

        robot.expert_publish_callback = self._on_publish

    def _on_publish(self, linear, angular, simulation_time_ns):
        if self._pending is None:
            return

        self._published = (
            float(linear),
            float(angular),
            int(simulation_time_ns),
        )

    def execute(self, action, pose):
        """Execute an action and record it only after successful submission."""

        if self._pending is not None:
            raise RuntimeError("Another expert command is pending")

        observation = self.robot.get_observation()

        if observation is None:
            return False

        observation = np.asarray(observation, dtype=np.float32).copy()
        action = np.asarray(action, dtype=np.float32).copy()

        if observation.shape != (40,) or not np.all(np.isfinite(observation)):
            return False

        if action.shape != (2,) or not np.all(np.isfinite(action)):
            return False

        if not np.all((-1.0 <= action) & (action <= 1.0)):
            return False

        if len(pose) < 3 or not all(
            math.isfinite(float(v)) for v in pose[:3]
        ):
            return False

        self._pending = True
        self._published = None

        try:
            accepted = self.robot.command_action(action)

            if not accepted or self._published is None:
                return False

            linear, angular, sim_time = self._published

            mapper = self.robot.action_mapper

            if linear >= 0.0:
                normalized_linear = linear / mapper.max_forward
            else:
                normalized_linear = linear / mapper.max_reverse

            normalized_angular = angular / mapper.max_angular

            applied_action = np.asarray(
                [
                    np.clip(normalized_linear, -1.0, 1.0),
                    np.clip(normalized_angular, -1.0, 1.0),
                ],
                dtype=np.float32,
            )

            if not np.all(np.isfinite(applied_action)):
                return False

            self.records.append(
                ExpertTransition(
                    observation=observation,
                    action=action,
                    published_linear=linear,
                    published_angular=angular,
                    simulation_time_ns=sim_time,
                    applied_action=applied_action,
                    pose=tuple(float(v) for v in pose[:3]),
                )
            )

            return True

        finally:
            self._pending = None
            self._published = None

    def close(self):
        """Detach the recorder; do not change robot ownership."""
        if getattr(self.robot, "expert_publish_callback", None) == self._on_publish:
            self.robot.expert_publish_callback = None
