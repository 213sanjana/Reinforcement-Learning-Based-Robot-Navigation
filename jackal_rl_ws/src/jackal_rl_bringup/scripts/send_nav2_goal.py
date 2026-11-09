#!/usr/bin/env python3
"""Send one Nav2 goal stamped with Gazebo time.

The RViz Nav2 Goal button stamps wall-clock time. The planner then
rejects the goal because the map transform is on simulation time.
"""

import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter


class Sender(Node):
    def __init__(self, x, y):
        super().__init__(
            "send_nav2_goal",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True),
            ],
        )
        self.x = x
        self.y = y
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def send(self):
        print("Waiting for Nav2...", flush=True)
        if not self.client.wait_for_server(timeout_sec=15.0):
            print("Nav2 is not ready. Start the Nav2 terminal first.", flush=True)
            return 1
        started = time.monotonic()
        while self.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.monotonic() - started > 15.0:
                print("Gazebo clock is stopped. Start the cafe terminal first.", flush=True)
                return 1
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.orientation.w = 1.0
        goal.pose = pose
        print(
            f"Goal sent to ({self.x:.2f}, {self.y:.2f}). The robot should start moving.",
            flush=True,
        )
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            print("Nav2 rejected the goal.", flush=True)
            return 1
        print("Goal accepted.", flush=True)
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        print("Navigation finished.", flush=True)
        return 0


def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -3.0
    rclpy.init()
    node = Sender(x, y)
    try:
        code = node.send()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
