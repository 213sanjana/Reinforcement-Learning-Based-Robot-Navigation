"""Forward obstacle checks using Jackal LiDAR observations."""

import math
import numpy as np


class JackalLidarSafety:
    """Check whether the robot's forward sector is obstacle-free."""

    def __init__(
        self,
        stop_distance=1.2,
        forward_half_angle=math.radians(25),
    ):
        if not math.isfinite(stop_distance) or stop_distance <= 0:
            raise ValueError("Stop distance must be positive and finite")

        if not 0 < forward_half_angle <= math.pi:
            raise ValueError("Invalid forward sector angle")

        self.stop_distance = float(stop_distance)
        self.forward_half_angle = float(forward_half_angle)

    def forward_clearance(self, ranges, angles):
        """Return minimum valid distance in the forward sector."""

        ranges = np.asarray(ranges, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)

        if ranges.ndim != 1 or angles.ndim != 1:
            raise ValueError("LiDAR inputs must be one-dimensional")

        if len(ranges) != len(angles):
            raise ValueError("LiDAR ranges and angles must match")

        if len(ranges) == 0:
            return None

        wrapped_angles = np.arctan2(
            np.sin(angles),
            np.cos(angles),
        )

        forward = np.abs(wrapped_angles) <= self.forward_half_angle

        # Positive infinity is treated as no detected return.
        # NaN, negative infinity, zero and negative ranges are invalid.
        selected = ranges[forward]
        valid = selected[
            (np.isfinite(selected) & (selected > 0))
            | np.isposinf(selected)
        ]

        if len(valid) == 0:
            return None

        return float(np.min(valid))

    def is_forward_safe(self, ranges, angles):
        """Fail closed if clearance is unavailable or too small."""

        clearance = self.forward_clearance(ranges, angles)

        return (
            clearance is not None
            and clearance >= self.stop_distance
        )
