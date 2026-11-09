"""Convert discrete GridWorld directions to Jackal velocity commands."""

import math

from environments.gridworld_env import RobotNavigationEnv


class JackalActionAdapter:
    """Heading-aware controller for four discrete grid directions."""

    def __init__(
        self,
        linear_speed=0.20,
        angular_speed=0.45,
        heading_tolerance=0.15,
    ):
        if not all(
            math.isfinite(v) and v > 0
            for v in (linear_speed, angular_speed, heading_tolerance)
        ):
            raise ValueError("Adapter parameters must be positive and finite")

        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        self.heading_tolerance = float(heading_tolerance)

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def target_heading(self, action):
        """Map row/column action to world-frame heading."""

        if not isinstance(action, int) or isinstance(action, bool):
            raise ValueError("Action must be an integer from 0 to 3")

        if not 0 <= action < len(RobotNavigationEnv.ACTIONS):
            raise ValueError("Action must be between 0 and 3")

        row_delta, col_delta = RobotNavigationEnv.ACTIONS[action]

        world_dx = col_delta
        world_dy = -row_delta

        return math.atan2(world_dy, world_dx)

    def velocity_command(self, action, current_heading):
        """Turn toward target heading, then move forward."""

        if not math.isfinite(current_heading):
            raise ValueError("Heading must be finite")

        target = self.target_heading(action)
        error = self.normalize_angle(target - current_heading)

        if abs(error) > self.heading_tolerance:
            angular = self.angular_speed if error > 0 else -self.angular_speed
            return 0.0, angular

        return self.linear_speed, 0.0
