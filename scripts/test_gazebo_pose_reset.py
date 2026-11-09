"""Verify Jackal pose resetting with bounded sensor synchronization."""

import math
import time

from jackal_rl_navigation.gazebo_env import JackalGazeboEnv


def odom_stamp(env):
    msg = env.robot.odom

    if msg is None:
        return None

    return (
        msg.header.stamp.sec
        + msg.header.stamp.nanosec * 1e-9
    )


def pose_xy(env):
    p = env.robot.odom.pose.pose.position
    return float(p.x), float(p.y)


def wait_for_odom(env, previous_stamp=None, timeout=10.0):
    """Advance simulation and wait for newer odometry."""

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        # Process ROS discovery and pending callbacks.
        env.executor.spin_once(timeout_sec=0.1)

        stamp = odom_stamp(env)

        if stamp is not None:
            if previous_stamp is None or stamp > previous_stamp:
                return stamp

        # Advance paused Gazebo to generate new sensor data.
        env.controller.step(100)

        # Allow callbacks to process the new simulation data.
        for _ in range(5):
            env.executor.spin_once(timeout_sec=0.05)

    raise TimeoutError(
        "No fresh Jackal odometry within timeout"
    )


def main():
    env = None
    initial = None
    moved = None
    restored = None

    try:
        env = JackalGazeboEnv(sensor_timeout=8.0)

        print("\n=== STEP 21: INITIAL SENSOR SYNCHRONIZATION ===")

        env.controller.pause()
        env.robot.stop()

        initial_stamp = wait_for_odom(env)

        initial = pose_xy(env)

        print("Initial odometry:", initial)
        print("Initial timestamp:", initial_stamp)

        print("\n=== STEP 22: ROBOT REPOSITIONING ===")

        print("Setting Gazebo pose to (-1.0, 0.0)")

        assert env.controller.set_pose(
            -1.0, 0.0, 0.0
        )

        moved_stamp = wait_for_odom(
            env,
            previous_stamp=initial_stamp,
        )

        moved = pose_xy(env)

        print("Updated odometry:", moved)
        print("Updated timestamp:", moved_stamp)

        displacement = math.dist(initial, moved)

        print("Odometry displacement:", displacement)

        if abs(moved[0] + 1.0) < 0.2:
            print("PASS: Odometry reflects Gazebo repositioning")
        else:
            print("WARNING: Odometry does not reflect world pose directly")

        print("\n=== STEP 23: RESTORE ORIGINAL POSITION ===")

        print("Restoring Gazebo pose to (0.0, 0.0)")

        assert env.controller.set_pose(
            0.0, 0.0, 0.0
        )

        restored_stamp = wait_for_odom(
            env,
            previous_stamp=moved_stamp,
        )

        restored = pose_xy(env)

        print("Restored odometry:", restored)
        print("Restored timestamp:", restored_stamp)

        velocity = env.robot.odom.twist.twist

        linear = float(velocity.linear.x)
        angular = float(velocity.angular.z)

        print("Linear velocity:", linear)
        print("Angular velocity:", angular)

        if math.dist(restored, initial) < 0.2:
            print("PASS: Odometry returned near original position")
        else:
            print("WARNING: Odometry differs from initial position")

        if abs(linear) < 0.02 and abs(angular) < 0.02:
            print("PASS: Robot velocity near zero")
        else:
            print("WARNING: Robot velocity is not near zero")

        print("\n========== TEST COMPLETED ==========")

    finally:
        if env is not None:
            try:
                # Attempt restoration even if an earlier check failed.
                env.robot.stop()
                env.controller.pause()
                env.controller.set_pose(0.0, 0.0, 0.0)
                env.controller.step(100)
            except Exception as exc:
                print("WARNING: Cleanup restoration failed:", exc)
            finally:
                env.close()

        print("Environment cleanup completed")


if __name__ == "__main__":
    main()
