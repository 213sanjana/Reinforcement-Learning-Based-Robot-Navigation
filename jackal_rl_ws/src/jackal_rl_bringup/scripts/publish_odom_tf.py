#!/usr/bin/env python3
"""Publish odom -> base_link from the Jackal odometry topic.

RViz needs odom -> base_link to draw the robot model and the laser.
This demo does not run robot_localization, so the transform is taken
from the platform odometry topic.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


class OdomTf(Node):
    def __init__(self):
        super().__init__(
            "jackal_odom_tf",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True),
            ],
        )
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Odometry,
            "/j100_0000/platform/odom",
            self._publish,
            qos_profile_sensor_data,
        )

    def _publish(self, message):
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init()
    node = OdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
