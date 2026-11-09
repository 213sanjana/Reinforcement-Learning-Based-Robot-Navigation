"""Read the Jackal's actual world pose from Gazebo Fortress."""

import math
import re
import subprocess
import time

from jackal_rl_navigation.gazebo_controller import gazebo_process_env


class WorldPoseError(RuntimeError):
    """Raised when a valid Gazebo world pose cannot be obtained."""


def parse_model_pose(output, robot="j100_0000"):
    """Extract the world-frame model pose from Pose_V text."""

    # Match the model pose, not wheel or base_link poses.
    blocks = re.split(r"(?=^pose \{)", output, flags=re.MULTILINE)

    for block in blocks:
        if not block.startswith("pose {"):
            continue

        name = re.search(
            r'^\s*name:\s*"([^"]+)"',
            block,
            re.MULTILINE,
        )

        if name is None or name.group(1) != robot:
            continue

        position = re.search(
            r"position\s*\{([^}]*)\}",
            block,
            re.DOTALL,
        )

        if position is None:
            continue

        def value(field, text, default=0.0):
            match = re.search(
                rf"\b{field}:\s*"
                r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
                text,
            )
            return float(match.group(1)) if match else default

        x = value("x", position.group(1))
        y = value("y", position.group(1))
        z = value("z", position.group(1))

        orientation = re.search(
            r"orientation\s*\{([^}]*)\}",
            block,
            re.DOTALL,
        )

        if orientation is None:
            raise WorldPoseError("Model orientation missing")

        q = orientation.group(1)

        qx = value("x", q)
        qy = value("y", q)
        qz = value("z", q)
        qw = value("w", q)

        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        result = (x, y, z, yaw)

        if not all(math.isfinite(v) for v in result):
            raise WorldPoseError("Non-finite Gazebo pose")

        return result

    raise WorldPoseError(f"Model pose not found: {robot}")


class GazeboWorldPose:
    """Read model world pose using the verified Gazebo topic."""

    def __init__(self, robot="j100_0000", timeout=5.0, world="jackal_obstacle_test", partition=None):
        self.robot = robot
        self.timeout = float(timeout)
        self.partition = partition
        self.topic = f"/world/{world}/dynamic_pose/info"

    def read(self, controller=None):
        """Read a pose after generating fresh simulation data."""

        if controller is None:
            raise WorldPoseError(
                "A GazeboController is required for paused simulation"
            )

        process = subprocess.Popen(
            [
                "ign", "topic",
                "-e",
                "-t", self.topic,
                "-n", "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=gazebo_process_env(self.partition),
        )

        try:
            # Give Gazebo Transport time to establish subscription.
            time.sleep(1.0)

            # Generate a fresh pose message.
            controller.step(100)

            output, errors = process.communicate(
                timeout=self.timeout
            )

            if process.returncode != 0:
                raise WorldPoseError(
                    f"Pose subscriber failed: {errors}"
                )

            return parse_model_pose(output, self.robot)

        except subprocess.TimeoutExpired as exc:
            raise WorldPoseError(
                "Timed out waiting for fresh Gazebo pose"
            ) from exc

        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    def read_latest(self):
        """Read one world pose while simulation is already running.

        Unlike ``read``, this does not step the physics. A paused world
        will not publish a fresh pose, and the call times out.
        """

        completed = subprocess.run(
            ["ign", "topic", "-e", "-t", self.topic, "-n", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout,
            env=gazebo_process_env(self.partition),
            check=False,
        )
        if completed.returncode != 0:
            raise WorldPoseError(f"Pose subscriber failed: {completed.stderr}")
        return parse_model_pose(completed.stdout, self.robot)

    def wait_for_position(
        self,
        x,
        y,
        tolerance=0.15,
        timeout=8.0,
        controller=None,
    ):
        """Wait until the model reaches a requested world position."""

        deadline = time.monotonic() + timeout
        last_pose = None

        while time.monotonic() < deadline:
            try:
                last_pose = self.read(controller=controller)
            except WorldPoseError:
                continue

            error = math.hypot(
                last_pose[0] - x,
                last_pose[1] - y,
            )

            if error <= tolerance:
                return last_pose

        raise WorldPoseError(
            f"Pose verification failed. Last pose: {last_pose}"
        )
