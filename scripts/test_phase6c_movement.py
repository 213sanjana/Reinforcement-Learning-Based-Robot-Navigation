"""Diagnostic test for Phase 6C Gymnasium movement."""

import math
import traceback

import numpy as np

from jackal_rl_navigation.gazebo_env import JackalGazeboEnv


def main():
    env = None

    try:
        print("Creating Gymnasium environment...", flush=True)

        env = JackalGazeboEnv(sensor_timeout=8.0)

        print("Resetting Jackal...", flush=True)

        observation, info = env.reset(
            seed=42,
            options={
                "start": (0.0, 0.0, 0.0),
                "goal": (2.0, 0.0),
            },
        )

        start = env.robot.odom.pose.pose.position
        start_xy = (float(start.x), float(start.y))

        print("Starting odometry:", start_xy, flush=True)

        action = np.array([0.2, 0.0], dtype=np.float32)

        accepted_count = 0

        for index in range(5):
            print(
                f"Executing movement step {index + 1}...",
                flush=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            accepted = info["action_accepted"]
            accepted_count += int(accepted)

            print(
                f"Step {index + 1}: "
                f"accepted={accepted}, "
                f"distance={observation[36] * 20.0:.3f} m, "
                f"scan={info['scan_stamp']:.3f}, "
                f"odom={info['odom_stamp']:.3f}",
                flush=True,
            )

            assert observation.shape == (40,)
            assert np.all(np.isfinite(observation))

        env.robot.stop()

        end = env.robot.odom.pose.pose.position
        end_xy = (float(end.x), float(end.y))

        displacement = math.hypot(
            end_xy[0] - start_xy[0],
            end_xy[1] - start_xy[1],
        )

        print("\n=== RESULTS ===", flush=True)
        print("Accepted actions:", accepted_count, "/ 5")
        print("Displacement:", round(displacement, 4), "m")

        if displacement > 0.005:
            print("PASS: Movement verified", flush=True)
        else:
            print(
                "DIAGNOSTIC: Movement below verification threshold",
                flush=True,
            )

    except Exception:
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            print("Closing environment...", flush=True)
            env.close()
            print("Environment closed", flush=True)


if __name__ == "__main__":
    main()
