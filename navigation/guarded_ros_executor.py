"""Guarded A* ROS 2 executor.

Read-only integration test. Never publishes movement commands.
"""

import math
import time

import rclpy

from navigation.astar_ros_executor import AStarROSExecutor
from navigation.guarded_navigation import (
    GuardedNavigation,
    velocity_to_action,
)


class GuardedROSExecutor(AStarROSExecutor):
    """Integrate A*, waypoint tracking, and navigation safety."""

    def __init__(self, goal=(3.0, 0.0), inflation_radius=0.45):
        super().__init__(goal=goal, inflation_radius=inflation_radius)

        self.guard = None
        self.last_log = 0.0

    def step(self):
        pose = self.get_pose()

        if pose is None or self.node.scan is None:
            return "WAITING_FOR_SENSORS"

        if self.controller is None:
            if not self.initialize_plan():
                return "WAITING_FOR_PLAN"

        if self.guard is None:
            self.guard = GuardedNavigation(
                path=self.path,
                controller=self.controller,
                stall_timeout=12.0,
                navigation_timeout=120.0,
            )

        decision = self.guard.evaluate(
            pose=pose,
            sensor_ages=self.node._sensor_ages(),
            collision_detected=(
                self.collision_monitor.collision_detected
            ),
            now=time.monotonic(),
        )

        if decision.status != "RUNNING":
            print(
                f"NAVIGATION_STATUS={decision.status} "
                f"REASON={decision.reason}"
            )
            return decision.status

        action = velocity_to_action(
            decision.linear,
            decision.angular,
            self.node.action_mapper,
        )

        now = time.monotonic()

        if now - self.last_log >= 1.0:
            distance = math.hypot(
                pose[0] - self.goal[0],
                pose[1] - self.goal[1],
            )

            print(
                f"STATUS=GUARDED_DRY_RUN "
                f"POSE=({pose[0]:.3f},{pose[1]:.3f}) "
                f"GOAL_DISTANCE={distance:.3f} "
                f"ROUTE_PROGRESS={decision.progress:.3f} "
                f"LINEAR={decision.linear:.3f} "
                f"ANGULAR={decision.angular:.3f} "
                f"ACTION=({action[0]:.3f},{action[1]:.3f}) "
                f"COLLISION={self.collision_monitor.collision_detected}"
            )

            self.last_log = now

        # DRY RUN: do not call command_action().
        return "RUNNING"

    def run(self, duration=8.0):
        print("========== GUARDED ROS DRY RUN ==========")
        print("Velocity publication: DISABLED")

        deadline = time.monotonic() + duration
        final_status = "TIME_LIMIT"

        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(
                    self.node,
                    timeout_sec=0.05,
                )

                status = self.step()

                if status in (
                    "SAFETY_STOP",
                    "TIMEOUT",
                    "STALLED",
                    "GOAL_REACHED",
                ):
                    final_status = status
                    break

            print("FINAL_STATUS:", final_status)

        finally:
            # Dry-run: do not publish zero velocity.
            self.node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


def main():
    rclpy.init()

    executor = GuardedROSExecutor()
    executor.run(duration=8.0)


if __name__ == "__main__":
    main()
