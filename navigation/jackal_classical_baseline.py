"""Goal-direction baseline that consumes the 40-D Jackal observation.

The output is a normalized action, not a wheel command. Callers still pass
it through ActionMapper and SafetyMonitor. A successful kinematic check of
this baseline is not a reinforcement-learning result and is not a Gazebo
result.
"""

from __future__ import annotations

import math

import numpy as np

from navigation.jackal_policy import (
    PolicyError,
    bind_device,
    validate_action,
    validate_observation,
)

GOAL_DISTANCE_SCALE = 20.0
HEADING_SCALE = math.pi
LIDAR_RANGE_SCALE = 12.0
GOAL_TOLERANCE_M = 0.30
FORWARD_STOP_M = 0.80
SIDE_CLEARANCE_M = 0.70
SIDE_RELEASE_M = 0.90
FORWARD_HALF_ANGLE = math.radians(25.0)
SIDE_SECTOR_ANGLE = math.radians(120.0)
HEADING_SLOW_RAD = 0.50
ANGULAR_COMMAND_SCALE = 0.60
AVOID_LINEAR = 0.35


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class LidarLayout:
    """Equal-angle reading of the 36 stored sectors.

    Sector 0 is the first observation entry, which is the ``angle_min`` side
    of the scan. It is the forward beam only when the scan is centered so
    that a middle sector has angle zero. The default layout is a full circle
    starting at -pi, used by the offline tests. A live scan must pass its
    own ``angle_min`` and ``angle_max``.
    """

    def __init__(self, angle_min=-math.pi, angle_max=math.pi, bins=36):
        if bins != 36:
            raise PolicyError("The Jackal observation contract uses 36 LiDAR sectors")
        if not all(math.isfinite(v) for v in (angle_min, angle_max)):
            raise PolicyError("LiDAR angles must be finite")
        if angle_max <= angle_min:
            raise PolicyError("angle_max must be greater than angle_min")
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.bins = int(bins)

    def centers(self):
        width = (self.angle_max - self.angle_min) / self.bins
        return self.angle_min + (np.arange(self.bins) + 0.5) * width


class ClassicalGoalBaseline:
    """Turn toward the goal, slow down to steer, and refuse a blocked front.

    ``role`` is classical so a later report cannot present the command as
    the learned policy. Safety intervention is applied by the caller.
    """

    role = "classical_baseline"
    algorithm_family = "classical"

    def __init__(self, layout=None, device="cpu"):
        self.layout = layout if layout is not None else LidarLayout()
        self.device = bind_device(device)
        self.goal_tolerance_m = GOAL_TOLERANCE_M
        self.forward_stop_m = FORWARD_STOP_M
        self.side_clearance_m = SIDE_CLEARANCE_M
        self.side_release_m = SIDE_RELEASE_M
        self.forward_half_angle = FORWARD_HALF_ANGLE
        self._reset_count = 0
        self._avoid_sign = 0.0

    def reset(self):
        self._reset_count += 1
        self._avoid_sign = 0.0

    def _sector_stats(self, ranges, mask):
        if not np.any(mask):
            return float("inf"), float("inf")
        selected = ranges[mask]
        return float(np.min(selected)), float(np.mean(selected))

    def predict(self, observation, deterministic=True):
        values = validate_observation(observation)
        if not isinstance(deterministic, bool):
            raise PolicyError("deterministic must be a boolean")

        ranges = values[:36] * LIDAR_RANGE_SCALE
        distance = float(values[36] * GOAL_DISTANCE_SCALE)
        heading = float(values[37] * HEADING_SCALE)
        centers = self.layout.centers()
        wrapped = np.array([_wrap(float(c)) for c in centers])

        forward = np.abs(wrapped) <= self.forward_half_angle
        if not np.any(forward):
            raise PolicyError("LiDAR layout does not cover the forward sector")
        left = (wrapped > self.forward_half_angle) & (wrapped < SIDE_SECTOR_ANGLE)
        right = (wrapped < -self.forward_half_angle) & (wrapped > -SIDE_SECTOR_ANGLE)
        forward_range = float(np.min(ranges[forward]))
        left_min, left_mean = self._sector_stats(ranges, left)
        right_min, right_mean = self._sector_stats(ranges, right)

        if distance <= self.goal_tolerance_m:
            self._avoid_sign = 0.0
            return validate_action([0.0, 0.0])

        if forward_range < self.forward_stop_m:
            turn = 1.0 if left_mean >= right_mean else -1.0
            if self._avoid_sign != 0.0 and abs(left_mean - right_mean) < 0.25:
                turn = self._avoid_sign
            self._avoid_sign = turn
            return validate_action([0.0, turn])

        if (
            self._avoid_sign != 0.0
            and min(left_min, right_min) <= self.side_release_m
        ):
            linear = self._goal_linear(distance, heading, scale=AVOID_LINEAR)
            return validate_action([linear, self._avoid_sign])

        if left_min < self.side_clearance_m or right_min < self.side_clearance_m:
            if left_min < right_min:
                turn = -1.0
            elif right_min < left_min:
                turn = 1.0
            else:
                turn = 1.0 if left_mean >= right_mean else -1.0
            self._avoid_sign = turn
            linear = self._goal_linear(distance, heading, scale=AVOID_LINEAR)
            return validate_action([linear, turn])

        self._avoid_sign = 0.0
        angular = float(np.clip(heading / ANGULAR_COMMAND_SCALE, -1.0, 1.0))
        return validate_action([self._goal_linear(distance, heading, scale=1.0), angular])

    @staticmethod
    def _goal_linear(distance, heading, scale):
        if abs(heading) > HEADING_SLOW_RAD:
            return 0.0
        return float(np.clip(distance / 1.5, 0.0, 1.0)) * float(scale)
