"""Read-only ROS 2 observation interface for the simulated Jackal J100."""

from dataclasses import dataclass
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


@dataclass(frozen=True)
class JackalObservation:
    """One snapshot of the latest available robot observations."""

    lidar_ranges: np.ndarray
    lidar_angles: np.ndarray
    position_x: float
    position_y: float
    heading: float
    linear_velocity: float
    angular_velocity: float
    scan_stamp: float
    odom_stamp: float
    scan_age: float
    odom_age: float


class JackalObserver(Node):
    """Subscribe to Jackal LiDAR and odometry without sending commands."""

    def __init__(self, namespace="j100_0000"):
        super().__init__("jackal_rl_observer")

        namespace = namespace.strip("/")

        self.scan_topic = f"/{namespace}/sensors/lidar3d_0/scan"
        self.odom_topic = f"/{namespace}/platform/odom"

        self._scan = None
        self._odom = None
        self._scan_received_at = None
        self._odom_received_at = None

        self.scan_subscription = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._on_scan,
            10,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._on_odom,
            10,
        )

    def _on_scan(self, message):
        self._scan = message
        self._scan_received_at = time.monotonic()

    def _on_odom(self, message):
        self._odom = message
        self._odom_received_at = time.monotonic()

    @staticmethod
    def _stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def get_observation(self, max_age=1.0):
        """Return a snapshot, or None if sensors are missing or stale."""

        if self._scan is None or self._odom is None:
            return None

        now = time.monotonic()
        scan_age = now - self._scan_received_at
        odom_age = now - self._odom_received_at

        if scan_age > max_age or odom_age > max_age:
            return None

        scan = self._scan
        odom = self._odom

        ranges = np.asarray(scan.ranges, dtype=np.float32).copy()

        angles = (
            float(scan.angle_min)
            + np.arange(len(ranges), dtype=np.float32)
            * float(scan.angle_increment)
        )

        position = odom.pose.pose.position
        quaternion = odom.pose.pose.orientation

        heading = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2),
        )

        return JackalObservation(
            lidar_ranges=ranges,
            lidar_angles=angles,
            position_x=float(position.x),
            position_y=float(position.y),
            heading=heading,
            linear_velocity=float(odom.twist.twist.linear.x),
            angular_velocity=float(odom.twist.twist.angular.z),
            scan_stamp=self._stamp_seconds(scan.header.stamp),
            odom_stamp=self._stamp_seconds(odom.header.stamp),
            scan_age=scan_age,
            odom_age=odom_age,
        )


def main():
    rclpy.init()
    observer = JackalObserver()

    try:
        observer.get_logger().info(
            f"Listening to {observer.scan_topic} and {observer.odom_topic}"
        )

        while rclpy.ok():
            rclpy.spin_once(observer, timeout_sec=0.1)

            observation = observer.get_observation()

            if observation is not None:
                print(
                    f"Position=({observation.position_x:.2f}, "
                    f"{observation.position_y:.2f}) "
                    f"Heading={observation.heading:.2f} "
                    f"LiDAR beams={len(observation.lidar_ranges)}"
                )
                break

    except KeyboardInterrupt:
        pass

    finally:
        observer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
