"""Combine discrete navigation actions with basic LiDAR safety checks."""

from dataclasses import dataclass
import math

from ros_integration.jackal_action_adapter import JackalActionAdapter
from ros_integration.jackal_lidar_safety import JackalLidarSafety
from ros_integration.jackal_turn_safety import JackalTurnSafety


@dataclass(frozen=True)
class NavigationDecision:
    linear: float
    angular: float
    status: str


class JackalNavigationAdapter:
    """Generate preliminary velocity decisions from discrete actions."""

    def __init__(self):
        self.actions = JackalActionAdapter()
        self.safety = JackalLidarSafety(stop_distance=1.5)
        self.turn_safety = JackalTurnSafety(clearance=1.5)

    def decide(self, action, observation):
        """Return a velocity decision without publishing ROS commands."""

        if observation is None:
            return NavigationDecision(0.0, 0.0, "NO_OBSERVATION")

        if not math.isfinite(observation.heading):
            return NavigationDecision(0.0, 0.0, "INVALID_HEADING")

        if (
            observation.scan_age > 1.0
            or observation.odom_age > 1.0
            or observation.scan_age < 0
            or observation.odom_age < 0
        ):
            return NavigationDecision(0.0, 0.0, "STALE_SENSOR")

        # Validate the action even if an obstacle is present.
        linear, angular = self.actions.velocity_command(
            action, observation.heading
        )

        clearance = self.safety.forward_clearance(
            observation.lidar_ranges,
            observation.lidar_angles,
        )

        if clearance is None:
            return NavigationDecision(0.0, 0.0, "INVALID_SCAN")

        if angular != 0.0:
            turn_safe, reason = self.turn_safety.check(
                observation.lidar_ranges,
                observation.lidar_angles,
            )
            if not turn_safe:
                return NavigationDecision(
                    0.0, 0.0, f"TURN_BLOCKED_{reason}"
                )
            return NavigationDecision(0.0, angular, "TURN_CLEAR")

        if clearance < self.safety.stop_distance:
            return NavigationDecision(0.0, 0.0, "OBSTACLE_AHEAD")

        return NavigationDecision(linear, 0.0, "FORWARD_CLEAR")
