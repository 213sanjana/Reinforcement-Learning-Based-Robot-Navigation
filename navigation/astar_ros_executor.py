"""ROS 2 A* waypoint executor — read-only dry-run mode."""

import math
import time
from pathlib import Path

import rclpy

from jackal_rl_navigation.robot_interface import JackalRLInterface
from jackal_rl_navigation.collision_monitor import CollisionMonitor

from planners.gazebo_astar import GazeboAStarPlanner
from planners.waypoint_controller import WaypointController


class AStarROSExecutor:
    """Compute A* navigation commands without publishing movement."""

    def __init__(self, goal=(3.0, 0.0), inflation_radius=0.45):
        self.node = JackalRLInterface()

        self.collision_monitor = CollisionMonitor(self.node)

        self.goal = tuple(map(float, goal))

        world = Path(
            "jackal_rl_ws/worlds/"
            "jackal_obstacle_test_collision.sdf"
        )

        self.planner = GazeboAStarPlanner(world, inflation_radius=inflation_radius)

        self.controller = None
        self.path = None

        self.last_log = 0.0

    def get_pose(self):
        """Read planar pose from live odometry."""

        odom = self.node.odom

        if odom is None:
            return None

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        return (
            float(p.x),
            float(p.y),
            float(yaw),
        )

    def initialize_plan(self):
        """Generate A* path from current odometry."""

        pose = self.get_pose()

        if pose is None:
            return False

        start = (pose[0], pose[1])

        path = self.planner.plan(
            start,
            self.goal,
        )

        if not path:
            raise RuntimeError(
                "A* could not find a valid path"
            )

        self.path = path

        self.controller = WaypointController(path)

        print(
            f"A* initialized: "
            f"start={start}, "
            f"goal={self.goal}, "
            f"waypoints={len(path)}"
        )

        return True

    def step(self):
        """Calculate commands without sending them to the robot."""

        pose = self.get_pose()

        if pose is None:
            return

        if self.controller is None:
            if not self.initialize_plan():
                return

        command = self.controller.compute(*pose)

        collision = self.collision_monitor.collision_detected

        ages = self.node._sensor_ages()

        now = time.monotonic()

        if now - self.last_log >= 1.0:
            print(
                f"POSE=({pose[0]:.2f}, "
                f"{pose[1]:.2f}, "
                f"{pose[2]:.2f}) "
                f"WAYPOINT={command.waypoint_index} "
                f"LINEAR={command.linear:.3f} "
                f"ANGULAR={command.angular:.3f} "
                f"FINISHED={command.finished} "
                f"COLLISION={collision} "
                f"SENSOR_AGES={ages}"
            )

            self.last_log = now

        # DRY RUN: intentionally no velocity publishing.

    def run(self, duration=15.0):
        """Run the executor in observation-only mode."""

        print("========== A* ROS DRY RUN ==========")
        print("Movement publishing: DISABLED")

        end = time.monotonic() + duration

        try:
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(
                    self.node,
                    timeout_sec=0.05,
                )

                self.step()

        finally:
            # Do not call stop() in dry-run mode:
            # it publishes a zero-velocity command.
            self.node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


def main():
    rclpy.init()

    executor = AStarROSExecutor()

    executor.run(duration=15.0)


if __name__ == "__main__":
    main()
