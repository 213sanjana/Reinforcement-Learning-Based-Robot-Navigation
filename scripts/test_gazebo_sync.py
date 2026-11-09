"""Live Gazebo/Gymnasium observation synchronization test."""

import numpy as np
import rclpy

from jackal_rl_navigation.gazebo_env import JackalGazeboEnv


def stamp(message):
    value = message.header.stamp
    return value.sec + value.nanosec * 1e-9


def main():
    env = None

    try:
        print("\n=== STEP 19: Initialize Gymnasium environment ===")

        env = JackalGazeboEnv(sensor_timeout=8.0)

        print("Environment initialized")
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)

        print("\n=== STEP 20: Test fresh observations ===")

        previous_scan = None
        previous_odom = None

        for index in range(1, 4):
            print(f"\n--- Synchronization cycle {index}/3 ---")

            # Ensure the robot is not being commanded to move.
            env.robot.stop()

            # Advance simulation by 100 physics iterations.
            env.controller.step(100)

            # Process ROS callbacks and require fresh sensor messages.
            # A few simulation batches may be needed for LiDAR updates.
            for attempt in range(10):
                try:
                    observation, info = env._wait_for_observation(
                        newer_than=previous_scan
                        if previous_scan is not None
                        else None
                    )
                except TimeoutError:
                    env.controller.step(100)
                    continue

                if previous_odom is not None:
                    if info["odom_stamp"] <= previous_odom:
                        env.controller.step(100)
                        continue

                break
            else:
                raise RuntimeError(
                    "Failed to obtain synchronized fresh sensor data"
                )

            assert observation.shape == (40,)
            assert observation.dtype == np.float32
            assert np.all(np.isfinite(observation))
            assert env.observation_space.contains(observation)

            previous_scan = info["scan_stamp"]
            previous_odom = info["odom_stamp"]

            print("Observation shape:", observation.shape)
            print("Observation dtype:", observation.dtype)
            print("LiDAR timestamp:", previous_scan)
            print("Odometry timestamp:", previous_odom)
            print("Finite values:", bool(np.all(np.isfinite(observation))))
            print("PASS: Fresh observation received")

        print("\nPASS: Three synchronization cycles completed")

    finally:
        if env is not None:
            env.close()

        print("Environment cleanup completed")


if __name__ == "__main__":
    main()
