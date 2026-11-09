"""Bounded ROS 2 velocity command interface for the Jackal J100."""

import math
import time

from geometry_msgs.msg import Twist
from rclpy.node import Node


class JackalController(Node):
    """Publish bounded velocity commands with a timeout watchdog."""

    def __init__(
        self,
        namespace="j100_0000",
        max_linear=0.3,
        max_angular=0.6,
        command_timeout=0.4,
    ):
        super().__init__("jackal_rl_controller")

        if not all(
            math.isfinite(value) and value > 0
            for value in (max_linear, max_angular, command_timeout)
        ):
            raise ValueError("Velocity limits and timeout must be positive and finite")

        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.command_timeout = float(command_timeout)

        namespace = namespace.strip("/")
        self.command_topic = f"/{namespace}/platform/cmd_vel_unstamped"

        self.publisher = self.create_publisher(
            Twist,
            self.command_topic,
            10,
        )

        self._last_command_time = None
        self._moving = False

        self.watchdog = self.create_timer(
            0.05,
            self._check_timeout,
        )

    def command(self, linear, angular):
        """Publish a velocity command within configured limits."""

        linear = float(linear)
        angular = float(angular)

        if not math.isfinite(linear) or not math.isfinite(angular):
            raise ValueError("Velocity commands must be finite")

        linear = max(-self.max_linear, min(self.max_linear, linear))
        angular = max(-self.max_angular, min(self.max_angular, angular))

        message = Twist()
        message.linear.x = linear
        message.angular.z = angular

        self.publisher.publish(message)

        self._last_command_time = time.monotonic()
        self._moving = linear != 0.0 or angular != 0.0

        return linear, angular

    def stop(self):
        """Publish an explicit zero-velocity command."""

        self.publisher.publish(Twist())
        self._moving = False
        self._last_command_time = None

    def _check_timeout(self):
        """Stop when active commands are no longer refreshed."""

        if not self._moving or self._last_command_time is None:
            return

        elapsed = time.monotonic() - self._last_command_time

        if elapsed >= self.command_timeout:
            self.stop()
            self.get_logger().warning(
                "Command timeout: published stop command"
            )


def main():
    import rclpy

    rclpy.init()
    controller = JackalController()

    try:
        controller.get_logger().info(
            f"Controller ready on {controller.command_topic}"
        )

        # Intentionally send no movement commands.
        controller.stop()
        rclpy.spin_once(controller, timeout_sec=0.2)

    finally:
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
