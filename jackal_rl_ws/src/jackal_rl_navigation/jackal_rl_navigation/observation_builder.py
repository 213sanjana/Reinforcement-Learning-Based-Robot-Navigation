"""Build normalized neural-network observations from Jackal sensor data."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ObservationResult:
    """Normalized observation and its validity information."""

    vector: np.ndarray
    valid: bool
    reason: str = ""


class ObservationBuilder:
    """Convert LiDAR, goal and velocity data into a fixed-size vector."""

    def __init__(
        self,
        lidar_bins=36,
        lidar_max_range=12.0,
        goal_max_distance=20.0,
        max_linear_velocity=0.3,
        max_angular_velocity=0.6,
    ):
        if not isinstance(lidar_bins, int) or isinstance(lidar_bins, bool):
            raise ValueError("lidar_bins must be an integer")

        if lidar_bins <= 0:
            raise ValueError("lidar_bins must be positive")

        limits = (
            lidar_max_range,
            goal_max_distance,
            max_linear_velocity,
            max_angular_velocity,
        )

        if not all(math.isfinite(v) and v > 0 for v in limits):
            raise ValueError("Normalization limits must be positive and finite")

        self.lidar_bins = lidar_bins
        self.lidar_max_range = float(lidar_max_range)
        self.goal_max_distance = float(goal_max_distance)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)

        self.observation_dim = lidar_bins + 4

    @staticmethod
    def normalize_angle(angle):
        """Wrap an angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def expand_partial_scan(
        ranges,
        angle_min,
        angle_increment,
        angle_max,
        samples=720,
        blind_range=12.0,
    ):
        """Place a forward scan into the full-circle bins the policy learned.

        A 180 degree sensor leaves the rear unmeasured. Those bins stay at
        the maximum range. A short fill looks like a wall behind the robot
        and the saved policy then refuses to drive. A scan that already
        covers the circle is returned unchanged.
        """

        values = np.asarray(ranges, dtype=np.float32).reshape(-1)
        span = abs(float(angle_max) - float(angle_min))
        if len(values) == 0 or span >= (2.0 * math.pi - 0.2):
            return values
        full = np.full(int(samples), float(blind_range), dtype=np.float32)
        angles = (
            float(angle_min)
            + np.arange(len(values), dtype=np.float64) * float(angle_increment)
        )
        wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
        indices = np.rint(
            (wrapped + math.pi) / (2.0 * math.pi) * int(samples)
        ).astype(int)
        full[np.clip(indices, 0, int(samples) - 1)] = values
        return full

    def downsample_lidar(self, ranges):
        """Return normalized distance bins and a validity flag."""
        values = np.asarray(ranges, dtype=np.float32)

        if values.ndim != 1 or len(values) < self.lidar_bins:
            return None, False

        output = np.zeros(self.lidar_bins, dtype=np.float32)

        for index, indices in enumerate(
            np.array_split(np.arange(len(values)), self.lidar_bins)
        ):
            sector = values[indices]

            # Positive infinity represents no obstacle return.
            cleaned = np.where(
                np.isposinf(sector),
                self.lidar_max_range,
                sector,
            )

            valid = np.isfinite(cleaned) & (cleaned > 0)

            if not np.any(valid):
                return None, False

            minimum = float(np.min(cleaned[valid]))

            output[index] = np.clip(
                minimum / self.lidar_max_range,
                0.0,
                1.0,
            )

        return output, True

    def build(
        self,
        lidar_ranges,
        goal_distance,
        goal_heading,
        linear_velocity,
        angular_velocity,
    ):
        """Build a normalized observation with explicit validity status."""

        empty = np.zeros(self.observation_dim, dtype=np.float32)

        features = (
            goal_distance,
            goal_heading,
            linear_velocity,
            angular_velocity,
        )

        if not all(math.isfinite(float(v)) for v in features):
            return ObservationResult(empty, False, "Non-finite input feature")

        if goal_distance < 0:
            return ObservationResult(empty, False, "Negative goal distance")

        lidar, lidar_valid = self.downsample_lidar(lidar_ranges)

        if not lidar_valid:
            return ObservationResult(empty, False, "Invalid LiDAR data")

        goal_features = np.array(
            [
                np.clip(
                    goal_distance / self.goal_max_distance,
                    0.0,
                    1.0,
                ),
                self.normalize_angle(goal_heading) / math.pi,
            ],
            dtype=np.float32,
        )

        velocity_features = np.array(
            [
                np.clip(
                    linear_velocity / self.max_linear_velocity,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    angular_velocity / self.max_angular_velocity,
                    -1.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [lidar, goal_features, velocity_features]
        ).astype(np.float32)

        if (
            observation.shape != (self.observation_dim,)
            or not np.all(np.isfinite(observation))
        ):
            return ObservationResult(empty, False, "Invalid observation")

        return ObservationResult(observation, True)


if __name__ == "__main__":
    builder = ObservationBuilder()

    sample_ranges = np.full(900, 5.0, dtype=np.float32)

    result = builder.build(
        lidar_ranges=sample_ranges,
        goal_distance=4.0,
        goal_heading=0.5,
        linear_velocity=0.1,
        angular_velocity=0.0,
    )

    print("Observation valid:", result.valid)
    print("Observation shape:", result.vector.shape)
    print("Observation dtype:", result.vector.dtype)
    print("Observation minimum:", result.vector.min())
    print("Observation maximum:", result.vector.max())
    print("Observation vector:", result.vector)
