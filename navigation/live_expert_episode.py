
"""Single-episode expert navigation using existing ROS/Gazebo infrastructure."""

from dataclasses import dataclass
import math
import time

import rclpy

from jackal_rl_navigation.gazebo_env import JackalGazeboEnv
from navigation.expert_navigation import ExpertNavigator
from navigation.frame_alignment import (
    WorldAlignedPlanner,
    alignment_from_poses,
    wait_for_fresh_sensors,
)
from navigation.expert_execution_coordinator import ExpertExecutionCoordinator
from navigation.expert_recording_bridge import ExpertRecordingBridge
from navigation.live_astar_executor import velocity_to_action
from planners.gazebo_astar import GazeboAStarPlanner


@dataclass
class ExpertEpisodeResult:
    status: str
    planned_path: list
    executed_poses: list
    transitions: list
    accepted_commands: int
    rejected_commands: int
    frame_alignment: dict = None


class LiveExpertEpisode:
    """Run a guarded expert episode with one existing ROS publisher."""

    def __init__(
        self,
        env,
        planner,
        controller_type="pure-pursuit",
        enable_motion=False,
        command_interval=0.1,
    ):
        if not math.isfinite(command_interval) or command_interval < 0.05:
            raise ValueError("Invalid command interval")

        self.env = env
        self.robot = env.robot
        self.planner = planner
        self.enable_motion = bool(enable_motion)

        if self.enable_motion:
            collision_config = getattr(env, "collision_config", None)
            collision_monitor = getattr(env, "collision_monitor", None)

            if (
                collision_config is None
                or not collision_config.enabled
                or collision_monitor is None
            ):
                raise RuntimeError(
                    "Expert motion requires enabled collision monitoring"
                )

        self.command_interval = float(command_interval)

        self.navigator = ExpertNavigator(
            planner=planner,
            controller_type=controller_type,
        )

        self.bridge = ExpertRecordingBridge(self.robot)
        self.coordinator = ExpertExecutionCoordinator(
            self.robot, self.navigator, self.bridge
        )

        self.executed_poses = []
        self.last_command_time = None
        self._episode_prepared = False
        self.frame_alignment = None

    def _fresh_episode_state(self):
        """Recreate navigation and recording state for a new episode."""

        self.robot.stop()

        self.bridge.close()

        self.navigator = ExpertNavigator(
            planner=self.planner,
            controller_type=self.navigator.controller_type,
            navigation_timeout=self.navigator.navigation_timeout,
        )

        self.bridge = ExpertRecordingBridge(self.robot)

        self.coordinator = ExpertExecutionCoordinator(
            self.robot,
            self.navigator,
            self.bridge,
        )

        self.executed_poses = []
        self.last_command_time = None
        self._episode_prepared = False
        self.frame_alignment = None

    @staticmethod
    def pose_from_odom(odom):
        q = odom.pose.pose.orientation
        p = odom.pose.pose.position

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        return float(p.x), float(p.y), float(yaw)

    def prepare(self, start=(0.0, 0.0, 0.0), goal=(3.0, 0.0)):
        """Reset simulation and initialize a fresh expert plan."""

        try:
            # Normal runners have a fully initialized planner.
            # Preserve reset-failure cleanup for partially initialized
            # runners used by existing safety regression tests.
            if hasattr(self, "planner"):
                self._fresh_episode_state()

            observation, info = self.env.reset(
                options={"start": start, "goal": goal}
            )

            if self.robot.odom is None:
                raise RuntimeError("Odometry unavailable after reset")

            pose = self.pose_from_odom(self.robot.odom)
            goal_odom = tuple(info["goal_odom"])
            world = info["robot_world_pose"]
            world_pose = (float(world[0]), float(world[1]), float(world[3]))
            self.frame_alignment = alignment_from_poses(world_pose, pose)
            self.navigator.planner = WorldAlignedPlanner(
                self.planner,
                world_pose,
                pose,
            )

            path = self.navigator.plan(
                (pose[0], pose[1]),
                goal_odom,
            )

            if not path:
                raise RuntimeError(
                    f"Expert planning failed: {self.navigator.terminal_status}"
                )

            self.executed_poses = [pose]
            self.last_command_time = None
            self._episode_prepared = True

            return observation, info

        except BaseException:
            # Fail closed if reset or planning is interrupted.
            try:
                self.robot.stop()
            finally:
                self.bridge.close()
            raise

    def run(self, duration=45.0):
        """Run an episode; motion requires explicit enable_motion=True."""

        if self.navigator.guard is None:
            raise RuntimeError("Call prepare() before run()")

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")

        # The episode budget is the navigation limit. A shorter guard
        # would abort a route that is still inside its scenario duration.
        self.navigator.guard.navigation_timeout = float(duration)

        if self.enable_motion:
            resumed = self.env.controller.resume()

            if not resumed:
                self.robot.stop()
                raise RuntimeError(
                    "Gazebo failed to resume; motion aborted"
                )

            if not wait_for_fresh_sensors(self.env, self.robot, timeout=2.0):
                self.robot.stop()
                return self._result("SENSOR_LOSS")

        deadline = time.monotonic() + duration
        status = "TIME_LIMIT"

        try:
            while rclpy.ok() and time.monotonic() < deadline:
                if not self.enable_motion:
                    # reset() leaves Gazebo paused. Advance physics
                    # without enabling autonomous velocity commands.
                    stepped = self.env.controller.step(20)

                    if not stepped:
                        status = "SIMULATION_STEP_FAILED"
                        break

                self.env.executor.spin_once(timeout_sec=0.05)

                if self.robot.odom is None or self.robot.scan is None:
                    status = "SENSOR_LOSS"
                    break

                pose = self.pose_from_odom(self.robot.odom)
                self.executed_poses.append(pose)

                ages = self.robot._sensor_ages()

                if ages is None:
                    status = "SENSOR_LOSS"
                    break

                if self.env.collision_monitor is not None:
                    collision = self.env.collision_monitor.collision_detected
                else:
                    collision = False

                # Fail closed before evaluating or publishing commands.
                if collision:
                    status = "SAFETY_STOP"
                    self.navigator.terminate("COLLISION")
                    self.robot.stop()
                    break

                sensor_timeout = self.robot.safety_monitor.sensor_timeout

                if (
                    not all(math.isfinite(float(age)) for age in ages)
                    or any(age > sensor_timeout for age in ages)
                ):
                    # One late sample is not a loss if the stream resumes.
                    if wait_for_fresh_sensors(
                        self.env, self.robot, timeout=1.0
                    ):
                        continue

                    status = "SENSOR_LOSS"
                    self.navigator.terminate("STALE_SENSORS")
                    self.robot.stop()
                    break

                if not self.enable_motion:
                    # Dry-run mode must not advance the navigation
                    # watchdog or publish expert movement commands.
                    status = "DRY_RUN"
                    continue

                now = time.monotonic()

                if (
                    self.last_command_time is not None
                    and now - self.last_command_time < self.command_interval
                ):
                    continue

                self.last_command_time = now

                status = self.coordinator.step(
                    pose=pose,
                    sensor_ages=ages,
                    collision_detected=collision,
                    now=now,
                )

                if status != "RUNNING":
                    break

            if status == "RUNNING" and time.monotonic() >= deadline:
                status = "TIME_LIMIT"

        finally:
            self.robot.stop()

        return self._result(status)

    def _result(self, status):
        return ExpertEpisodeResult(
            status=status,
            planned_path=list(self.navigator.path or []),
            executed_poses=list(self.executed_poses),
            transitions=list(self.bridge.records),
            accepted_commands=self.coordinator.accepted,
            rejected_commands=self.coordinator.rejected,
            frame_alignment=self.frame_alignment,
        )

    def close(self):
        """Stop, detach recorder, and release the environment."""
        try:
            self.robot.stop()
        finally:
            try:
                self.bridge.close()
            finally:
                self.env.close()
