"""Conservative LiDAR clearance checks for in-place Jackal turning."""

import math

import numpy as np


class JackalTurnSafety:
    """Require valid, sufficiently clear LiDAR coverage before turning."""

    def __init__(
        self,
        clearance=1.2,
        sectors=36,
        max_invalid_fraction=0.05,
    ):
        if not math.isfinite(clearance) or clearance <= 0:
            raise ValueError("Clearance must be positive and finite")

        if not isinstance(sectors, int) or sectors < 8:
            raise ValueError("At least eight sectors are required")

        if not 0 <= max_invalid_fraction < 1:
            raise ValueError("Invalid fraction must be in [0, 1)")

        self.clearance = float(clearance)
        self.sectors = sectors
        self.max_invalid_fraction = float(max_invalid_fraction)

    def check(self, ranges, angles):
        """Return (safe, reason) for an in-place turn."""

        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)

        if (
            ranges.ndim != 1
            or angles.ndim != 1
            or len(ranges) != len(angles)
            or len(ranges) == 0
        ):
            return False, "INVALID_SCAN"

        if not np.isfinite(angles).all():
            return False, "INVALID_ANGLES"

        wrapped = np.mod(angles + math.pi, 2 * math.pi)

        # Divide the full 360-degree scan into equal angular sectors.
        indices = np.floor(
            wrapped * self.sectors / (2 * math.pi)
        ).astype(int)

        indices = np.clip(indices, 0, self.sectors - 1)

        for sector in range(self.sectors):
            selected = ranges[indices == sector]

            if len(selected) == 0:
                return False, "MISSING_COVERAGE"

            # Positive infinity means no detected return.
            valid = (
                (np.isfinite(selected) & (selected > 0))
                | np.isposinf(selected)
            )

            invalid_fraction = 1.0 - np.mean(valid)

            if invalid_fraction > self.max_invalid_fraction:
                return False, "INVALID_RETURNS"

            if not np.any(valid):
                return False, "INVALID_RETURNS"

            nearest = float(np.min(selected[valid]))

            if nearest < self.clearance:
                return False, "INSUFFICIENT_CLEARANCE"

        return True, "CLEAR"
