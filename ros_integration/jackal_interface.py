"""Combined observation and control interface for simulated Jackal J100."""

import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from ros_integration.jackal_observer import JackalObserver
from ros_integration.jackal_controller import JackalController


class JackalInterface:
    """Manage Jackal observation and velocity control together."""

    def __init__(self, namespace="j100_0000"):
        if not rclpy.ok():
            raise RuntimeError("Call rclpy.init() before creating JackalInterface")

        self.observer = JackalObserver(namespace)
        self.controller = JackalController(namespace)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.observer)
        self.executor.add_node(self.controller)

        self._closed = False

    def observe(self, timeout=5.0, max_age=1.0):
        """Wait for fresh LiDAR and odometry observations."""

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)

            observation = self.observer.get_observation(max_age=max_age)

            if observation is not None:
                return observation

        raise TimeoutError("Fresh Jackal observations were not received")

    def command(self, linear, angular):
        """Send a bounded velocity command."""
        return self.controller.command(linear, angular)

    def stop(self):
        """Publish zero velocity."""
        self.controller.stop()

    def spin_for(self, duration):
        """Process ROS callbacks, including the controller watchdog."""

        deadline = time.monotonic() + duration

        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)

    def close(self):
        """Stop the robot and release ROS resources."""

        if self._closed:
            return

        try:
            self.controller.stop()
            self.spin_for(0.2)
        finally:
            self.executor.remove_node(self.observer)
            self.executor.remove_node(self.controller)
            self.observer.destroy_node()
            self.controller.destroy_node()
            self.executor.shutdown()
            self._closed = True


def main():
    rclpy.init()
    interface = None

    try:
        interface = JackalInterface()

        print("========== LIVE INTEGRATION TEST ==========", flush=True)

        observation = interface.observe(timeout=15.0)

        print(
            f"Position: ({observation.position_x:.3f}, "
            f"{observation.position_y:.3f})"
        )
        print(f"Heading: {observation.heading:.3f} rad")
        print(f"LiDAR beams: {len(observation.lidar_ranges)}")
        print(f"Scan age: {observation.scan_age:.3f} s")
        print(f"Odometry age: {observation.odom_age:.3f} s")
        print(f"Controller topic: {interface.controller.command_topic}")

        assert len(observation.lidar_ranges) > 0
        assert observation.scan_age <= 1.0
        assert observation.odom_age <= 1.0

        print("PASS: Combined interface receives fresh observations")
        print("PASS: Controller available without movement")

    finally:
        if interface is not None:
            interface.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
