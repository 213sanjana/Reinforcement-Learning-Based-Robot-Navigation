import math
from dataclasses import dataclass


@dataclass
class PurePursuitCommand:
    linear: float
    angular: float
    reached_goal: bool
    waypoint_index: int


class PurePursuitController:
    """Lookahead-based path follower for differential-drive robots."""

    def __init__(
        self,
        waypoints,
        lookahead_distance=0.45,
        max_linear=0.25,
        max_angular=0.60,
        goal_tolerance=0.15,
    ):
        if not waypoints:
            raise ValueError("Waypoints cannot be empty")

        self.waypoints = [
            tuple(map(float, p)) for p in waypoints
        ]

        if any(
            len(p) != 2 or not all(math.isfinite(v) for v in p)
            for p in self.waypoints
        ):
            raise ValueError("Invalid waypoint coordinates")

        if not all(
            math.isfinite(v) and v > 0
            for v in (
                lookahead_distance,
                max_linear,
                max_angular,
                goal_tolerance,
            )
        ):
            raise ValueError("Controller parameters must be positive")

        self.lookahead_distance = lookahead_distance
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.goal_tolerance = goal_tolerance
        self.index = 0
        self.finished = False

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute(self, x, y, yaw):
        x, y, yaw = map(float, (x, y, yaw))

        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError("Robot pose must be finite")

        gx, gy = self.waypoints[-1]
        goal_distance = math.hypot(gx - x, gy - y)

        if goal_distance <= self.goal_tolerance:
            self.finished = True

        if self.finished:
            return PurePursuitCommand(
                0.0, 0.0, True, self.index
            )

        # Find the nearest waypoint without going backwards.
        nearest = min(
            range(self.index, len(self.waypoints)),
            key=lambda i: math.hypot(
                self.waypoints[i][0] - x,
                self.waypoints[i][1] - y,
            ),
        )

        self.index = nearest

        # Find a lookahead target farther along the route.
        target_index = self.index

        for i in range(self.index, len(self.waypoints)):
            wx, wy = self.waypoints[i]

            if math.hypot(wx - x, wy - y) >= self.lookahead_distance:
                target_index = i
                break
        else:
            target_index = len(self.waypoints) - 1

        tx, ty = self.waypoints[target_index]

        dx = tx - x
        dy = ty - y

        distance = max(math.hypot(dx, dy), 1e-6)

        heading = math.atan2(dy, dx)

        alpha = self.normalize_angle(heading - yaw)

        curvature = 2.0 * math.sin(alpha) / distance

        # Slow down on tight curves, but keep a cruise speed until the
        # final approach. Otherwise a sharp corner crawls until the
        # episode budget expires before the goal.
        linear = self.max_linear / (1.0 + abs(curvature))

        if goal_distance > 0.5:
            linear = max(linear, min(0.15, self.max_linear))

        # Reduce speed near final goal.
        linear = min(linear, 0.6 * goal_distance)

        # Do not drive forward toward a target behind the robot.
        if abs(alpha) > math.pi / 2:
            linear = 0.0
            angular = max(
                -self.max_angular,
                min(self.max_angular, 1.5 * alpha),
            )
        else:
            angular = curvature * linear

            if abs(angular) > self.max_angular:
                angular = math.copysign(self.max_angular, angular)

        return PurePursuitCommand(
            float(linear),
            float(angular),
            False,
            target_index,
        )
