"""Passive runtime checks for live Jackal sensor streams."""

import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


LIDAR_TOPIC = "/j100_0000/sensors/lidar3d_0/scan"
ODOM_TOPIC = "/j100_0000/platform/odom"


def stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stream_is_fresh(stamps, received_times, now, max_age=0.5):
    if len(stamps) < 2 or len(received_times) < 2:
        return False

    if stamps[-1] <= stamps[0]:
        return False

    if stamps[-1] <= 0:
        return False

    if now - received_times[-1] > max_age:
        return False

    return True


@dataclass(frozen=True)
class RuntimePreflightResult:
    passed: bool
    checks: dict


class SensorObserver(Node):
    def __init__(self):
        super().__init__("expert_runtime_preflight")

        self.lidar_stamps = []
        self.odom_stamps = []
        self.lidar_received = []
        self.odom_received = []

        self.create_subscription(
            LaserScan,
            LIDAR_TOPIC,
            self.on_lidar,
            10,
        )

        self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.on_odom,
            10,
        )

    def on_lidar(self, message):
        self.lidar_stamps.append(stamp_ns(message))
        self.lidar_received.append(time.monotonic())

    def on_odom(self, message):
        self.odom_stamps.append(stamp_ns(message))
        self.odom_received.append(time.monotonic())


def run_runtime_preflight(timeout=5.0):
    """Observe sensor streams without issuing robot commands."""

    if timeout <= 0:
        raise ValueError("timeout must be positive")

    owns_context = not rclpy.ok()

    if owns_context:
        rclpy.init()

    node = None

    try:
        node = SensorObserver()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

            now = time.monotonic()

            lidar_ok = stream_is_fresh(
                node.lidar_stamps,
                node.lidar_received,
                now,
            )

            odom_ok = stream_is_fresh(
                node.odom_stamps,
                node.odom_received,
                now,
            )

            if lidar_ok and odom_ok:
                break

        now = time.monotonic()

        checks = {
            "lidar_messages_received":
                len(node.lidar_stamps) >= 2,
            "odom_messages_received":
                len(node.odom_stamps) >= 2,
            "lidar_timestamps_advancing":
                len(node.lidar_stamps) >= 2
                and node.lidar_stamps[-1] > node.lidar_stamps[0],
            "odom_timestamps_advancing":
                len(node.odom_stamps) >= 2
                and node.odom_stamps[-1] > node.odom_stamps[0],
            "lidar_fresh":
                stream_is_fresh(
                    node.lidar_stamps,
                    node.lidar_received,
                    now,
                ),
            "odom_fresh":
                stream_is_fresh(
                    node.odom_stamps,
                    node.odom_received,
                    now,
                ),
        }

        return RuntimePreflightResult(
            passed=all(checks.values()),
            checks=checks,
        )

    finally:
        if node is not None:
            node.destroy_node()

        if owns_context and rclpy.ok():
            rclpy.shutdown()


def main():
    result = run_runtime_preflight()

    for name, passed in result.checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(
        "RUNTIME PREFLIGHT:",
        "PASS" if result.passed else "FAIL",
    )

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
