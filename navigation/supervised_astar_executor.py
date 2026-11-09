"""Supervised A* navigation executor.

Dry-run only: never publishes velocity commands.
"""

import math
import time

import rclpy

from navigation.astar_ros_executor import AStarROSExecutor
from navigation.navigation_safety import evaluate_navigation_safety
from navigation.progress_watchdog import NavigationProgressWatchdog


class SupervisedAStarExecutor(AStarROSExecutor):
    """A* waypoint controller with additional navigation checks."""

    def __init__(self, goal=(3.0, 0.0)):
        super().__init__(goal=goal)

        self.progress_watchdog = NavigationProgressWatchdog(
            timeout=120.0,
            stall_timeout=10.0,
            min_progress=0.10,
        )

        self.started = False
        self.last_log = 0.0

    def step(self):
        pose = self.get_pose()

        if pose is None:
            return "WAITING_FOR_ODOMETRY"

        if self.node.scan is None:
            return "WAITING_FOR_LIDAR"

        if self.controller is None:
            if not self.initialize_plan():
                return "WAITING_FOR_PLAN"

        ages = self.node._sensor_ages()

        safety = evaluate_navigation_safety(
            pose=pose,
            path=self.path,
            sensor_ages=ages,
            collision_detected=(
                self.collision_monitor.collision_detected
            ),
        )

        if not safety.safe:
            print("SAFETY REJECTION:", safety.reason)
            return "SAFETY_STOP"

        command = self.controller.compute(*pose)

        if command.finished:
            print("GOAL REACHED")
            return "GOAL_REACHED"

        if not self.started:
            self.progress_watchdog.reset(pose)
            self.started = True

        # The progress watchdog is intentionally not enforced
        # during dry-run: the robot is expected to remain still.
        # It will be enabled only when movement is authorized.

        now = time.monotonic()

        if now - self.last_log >= 1.0:
            goal_distance = math.hypot(
                pose[0] - self.goal[0],
                pose[1] - self.goal[1],
            )

            print(
                f"STATUS=DRY_RUN "
                f"POSE=({pose[0]:.3f},{pose[1]:.3f}) "
                f"GOAL_DISTANCE={goal_distance:.3f} "
                f"WAYPOINT={command.waypoint_index} "
                f"LINEAR={command.linear:.3f} "
                f"ANGULAR={command.angular:.3f} "
                f"PATH_DEVIATION={safety.path_distance:.3f} "
                f"COLLISION={self.collision_monitor.collision_detected}"
            )

            self.last_log = now

        return "DRY_RUN"

    def run(self, duration=15.0):
        print("========== SUPERVISED A* DRY RUN ==========")
        print("Movement publishing: DISABLED")

        deadline = time.monotonic() + duration

        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(
                    self.node,
                    timeout_sec=0.05,
                )

                status = self.step()

                if status in ("SAFETY_STOP", "GOAL_REACHED"):
                    print("Executor finished:", status)
                    break

        finally:
            # Dry-run: do not publish even a zero command.
            self.node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


def main():
    rclpy.init()

    executor = SupervisedAStarExecutor()
    executor.run(duration=15.0)


if __name__ == "__main__":
    main()
