"""Safety checks for Jackal reinforcement-learning navigation."""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str
    linear: float
    angular: float


class SafetyMonitor:
    """Validate sensor freshness, obstacle clearance and episode state."""

    def __init__(
        self,
        forward_stop_distance=0.8,
        reverse_stop_distance=0.5,
        forward_half_angle=math.radians(25),
        reverse_half_angle=math.radians(25),
        sensor_timeout=0.5,
        rotation_clearance=0.65,
    ):
        parameters = (
            forward_stop_distance,
            reverse_stop_distance,
            forward_half_angle,
            reverse_half_angle,
            sensor_timeout,
            rotation_clearance,
        )

        if not all(math.isfinite(v) and v > 0 for v in parameters):
            raise ValueError("Safety parameters must be positive and finite")

        if forward_half_angle > math.pi / 2:
            raise ValueError("Forward sector must not exceed 90 degrees")

        if reverse_half_angle > math.pi / 2:
            raise ValueError("Reverse sector must not exceed 90 degrees")

        self.forward_stop_distance = float(forward_stop_distance)
        self.reverse_stop_distance = float(reverse_stop_distance)
        self.forward_half_angle = float(forward_half_angle)
        self.reverse_half_angle = float(reverse_half_angle)
        self.sensor_timeout = float(sensor_timeout)
        self.rotation_clearance = float(rotation_clearance)

        self.emergency_active = False
        self.episode_terminated = False
        self.intervention_count = 0
        self.last_intervention = ""

    def emergency_stop(self):
        """Latch emergency-stop state."""
        self.emergency_active = True

    def reset_emergency(self):
        """Explicitly clear emergency-stop state."""
        self.emergency_active = False

    def terminate_episode(self):
        """Prevent further movement until an explicit episode reset."""
        self.episode_terminated = True

    def reset_episode(self):
        """Reset episode termination without clearing emergency stop."""
        self.episode_terminated = False

    def _record_intervention(self, reason):
        self.intervention_count += 1
        self.last_intervention = reason

    def _stop(self, reason):
        self._record_intervention(reason)
        return SafetyResult(False, reason, 0.0, 0.0)

    @staticmethod
    def _clearance(ranges, angles, center, half_angle):
        """Return minimum valid range within a monitored angular sector."""

        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)

        if ranges.ndim != 1 or angles.ndim != 1:
            return None

        if ranges.size == 0 or ranges.shape != angles.shape:
            return None

        difference = np.arctan2(
            np.sin(angles - center),
            np.cos(angles - center),
        )

        sector = np.abs(difference) <= half_angle

        if not np.any(sector):
            return None

        selected = ranges[sector]

        # Fail closed when measurements are invalid.
        if np.any(np.isnan(selected)):
            return None

        if np.any(np.isneginf(selected)):
            return None

        if np.any(selected <= 0):
            return None

        # Positive infinity represents no obstacle return.
        valid = selected[
            (np.isfinite(selected) & (selected > 0))
            | np.isposinf(selected)
        ]

        if valid.size == 0:
            return None

        return float(np.min(valid))

    def evaluate(
        self,
        linear,
        angular,
        lidar_ranges,
        lidar_angles,
        scan_age,
        odom_age,
    ):
        """Return an allowed command or a zero-velocity stop.

        Forward, reverse, and rotation are checked separately. A blocked
        axis is removed. The other axis is kept when it still has clearance.
        A request that loses both axes is a full stop. A pure turn with
        insufficient clearance is still a full stop.
        """

        if self.emergency_active:
            return self._stop("Emergency stop active")

        if self.episode_terminated:
            return self._stop("Episode terminated")

        values = (linear, angular, scan_age, odom_age)

        if not all(math.isfinite(float(v)) for v in values):
            return self._stop("Non-finite command or sensor age")

        if scan_age < 0 or odom_age < 0:
            return self._stop("Invalid sensor timestamp")

        if scan_age > self.sensor_timeout:
            return self._stop("LiDAR data stale")

        if odom_age > self.sensor_timeout:
            return self._stop("Odometry data stale")

        requested_linear = float(linear)
        requested_angular = float(angular)
        linear_out = requested_linear
        angular_out = requested_angular
        notes = []

        # Turning is checked on the side being entered. An obstacle on the
        # opposite side must not cancel a turn that leaves it.
        if abs(angular_out) > 1e-6:
            turn_center = math.pi / 2.0 if angular_out > 0.0 else -math.pi / 2.0
            turn_clearance = self._clearance(
                lidar_ranges,
                lidar_angles,
                center=turn_center,
                half_angle=math.pi / 2.0,
            )

            if turn_clearance is None:
                angular_out = 0.0
                notes.append("Rotation LiDAR unavailable")
            elif turn_clearance < self.rotation_clearance:
                angular_out = 0.0
                notes.append("Rotation clearance insufficient")

        if linear_out > 0.0:
            forward_clearance = self._clearance(
                lidar_ranges,
                lidar_angles,
                center=0.0,
                half_angle=self.forward_half_angle,
            )

            if forward_clearance is None:
                linear_out = 0.0
                notes.append("Forward LiDAR unavailable")
            elif forward_clearance < self.forward_stop_distance:
                linear_out = 0.0
                notes.append("Forward obstacle too close")

        if linear_out < 0.0:
            reverse_clearance = self._clearance(
                lidar_ranges,
                lidar_angles,
                center=math.pi,
                half_angle=self.reverse_half_angle,
            )

            if reverse_clearance is None:
                linear_out = 0.0
                notes.append("Rear LiDAR unavailable")
            elif reverse_clearance < self.reverse_stop_distance:
                linear_out = 0.0
                notes.append("Rear obstacle too close")

        stopped = abs(linear_out) <= 1e-9 and abs(angular_out) <= 1e-9
        requested = (
            abs(requested_linear) > 1e-9 or abs(requested_angular) > 1e-9
        )

        if notes and stopped and requested:
            return self._stop(notes[0])

        if notes:
            reason = "Partial: " + "; ".join(notes)
            self._record_intervention(reason)
            return SafetyResult(True, reason, linear_out, angular_out)

        return SafetyResult(True, "Command permitted", linear_out, angular_out)


if __name__ == "__main__":
    monitor = SafetyMonitor()

    angles = np.linspace(-math.pi, math.pi, 900)
    ranges = np.full(900, 5.0)

    print("========== SAFETY MONITOR TEST ==========")

    def check(label, linear, angular, scan_age=0.1, odom_age=0.1):
        result = monitor.evaluate(
            linear=linear,
            angular=angular,
            lidar_ranges=ranges,
            lidar_angles=angles,
            scan_age=scan_age,
            odom_age=odom_age,
        )

        print(
            f"{label}: safe={result.safe}, "
            f"reason={result.reason}, "
            f"command=({result.linear:.2f}, {result.angular:.2f})"
        )

    check("Normal forward", 0.2, 0.0)
    check("Normal reverse", -0.1, 0.0)

    ranges[np.abs(angles) < 0.1] = 0.4
    check("Front obstacle", 0.2, 0.0)

    ranges[:] = 5.0
    check("Stale LiDAR", 0.2, 0.0, scan_age=1.0)

    monitor.emergency_stop()
    check("Emergency stop", 0.2, 0.0)

    monitor.reset_emergency()
    monitor.terminate_episode()
    check("Episode terminated", 0.2, 0.0)

    print("========== TEST COMPLETE ==========")
