"""Pure-Python differential-drive waypoint controller."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerCommand:
    linear: float
    angular: float
    finished: bool
    waypoint_index: int


class WaypointController:
    """Convert robot pose and waypoints into bounded velocities."""

    def __init__(
        self,
        waypoints,
        max_linear=0.30,
        max_angular=0.60,
        linear_gain=0.8,
        angular_gain=1.5,
        waypoint_tolerance=0.12,
        goal_tolerance=0.15,
        rotate_threshold=0.35,
    ):
        if not waypoints:
            raise ValueError("Waypoints cannot be empty")

        self.waypoints = [
            tuple(map(float, point))
            for point in waypoints
        ]

        if any(
            len(point) != 2
            or not all(math.isfinite(v) for v in point)
            for point in self.waypoints
        ):
            raise ValueError("Invalid waypoint coordinates")

        parameters = (
            max_linear,
            max_angular,
            linear_gain,
            angular_gain,
            waypoint_tolerance,
            goal_tolerance,
            rotate_threshold,
        )

        if not all(
            math.isfinite(v) and v > 0
            for v in parameters
        ):
            raise ValueError("Controller parameters must be positive")

        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.linear_gain = float(linear_gain)
        self.angular_gain = float(angular_gain)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.goal_tolerance = float(goal_tolerance)
        self.rotate_threshold = float(rotate_threshold)

        self.index = 0
        self.finished = False

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    def reset(self):
        self.index = 0
        self.finished = False


    def compute(self, x, y, yaw):
        """Compute bounded velocity toward the next waypoint."""

        x, y, yaw = map(float, (x, y, yaw))

        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError("Robot pose must be finite")

        if self.finished:
            return ControllerCommand(
                0.0, 0.0, True, self.index
            )

        # Check final goal before intermediate waypoints.
        gx, gy = self.waypoints[-1]

        goal_distance = math.hypot(gx - x, gy - y)

        if goal_distance <= self.goal_tolerance:
            self.index = len(self.waypoints) - 1
            self.finished = True

            return ControllerCommand(
                0.0, 0.0, True, self.index
            )

        # Advance past waypoints already reached.
        # Never move the waypoint index backwards.
        for i in range(
            len(self.waypoints) - 1,
            self.index,
            -1,
        ):
            wx, wy = self.waypoints[i]

            if math.hypot(wx - x, wy - y) <= self.waypoint_tolerance:
                self.index = min(
                    i + 1,
                    len(self.waypoints) - 1,
                )
                break

        while self.index < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.index]

            if math.hypot(wx - x, wy - y) > self.waypoint_tolerance:
                break

            self.index += 1

        wx, wy = self.waypoints[self.index]

        dx = wx - x
        dy = wy - y

        distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)

        heading_error = self.normalize_angle(
            target_heading - yaw
        )

        angular = max(
            -self.max_angular,
            min(
                self.max_angular,
                self.angular_gain * heading_error,
            ),
        )

        if abs(heading_error) > self.rotate_threshold:
            linear = 0.0
        else:
            linear = min(
                self.max_linear,
                self.linear_gain * distance,
            )

            linear *= max(
                0.0,
                math.cos(heading_error),
            )

        return ControllerCommand(
            float(linear),
            float(angular),
            False,
            self.index,
        )
