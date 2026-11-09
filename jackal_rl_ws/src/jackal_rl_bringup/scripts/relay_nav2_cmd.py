#!/usr/bin/env python3
"""Forward a Nav2 command onto the Jackal velocity topic.

Run this only after the learned policy has been stopped. Both publishers
on the same velocity topic will fight.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter


class Relay(Node):
    def __init__(self):
        super().__init__(
            "relay_nav2_cmd",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True),
            ],
        )
        self.publisher = self.create_publisher(
            Twist, "/j100_0000/platform/cmd_vel_unstamped", 10
        )
        self.create_subscription(
            Twist, "/j100_0000/nav2/cmd_vel", self.publisher.publish, 10
        )


def main():
    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
