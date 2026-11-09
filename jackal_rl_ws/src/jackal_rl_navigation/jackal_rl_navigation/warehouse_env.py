"""Gymnasium environment for Jackal warehouse navigation.

``execution="offline"`` is a kinematic stand-in for unit tests. It does not
start ROS or Gazebo. ``execution="gazebo"`` drives the existing
``JackalGazeboEnv`` and its single velocity publisher, and it refuses to
construct that object unless live control is explicitly authorized.
``execution="stationary_observation"`` is a separate read-only path. It does
not construct ``JackalGazeboEnv`` and it does not call that class's reset.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_observation import StationaryObservationSession
from jackal_rl_navigation.warehouse_episode import (
    FATAL_REASONS,
    EpisodeTracker,
    decide_episode,
)
from jackal_rl_navigation.warehouse_reward import calculate_reward, minimum_valid_range
from jackal_rl_navigation.warehouse_scenarios import pair_key, rejection_reason, select_scenario


LOGGER = logging.getLogger("warehouse_rl")

PUBLISHER_OWNER = "jackal_rl_interface"


def prepare_action(action) -> np.ndarray:
    """Reject non-finite actions and clamp the rest to [-1, 1]."""

    values = np.asarray(action, dtype=np.float32)
    if values.shape != (2,):
        raise ValueError("Action must have shape (2,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action must contain finite values")
    return np.clip(values, -1.0, 1.0).astype(np.float32)


def _ray_circle(origin, direction, center, radius) -> float | None:
    offset = origin - center
    b = 2.0 * float(np.dot(direction, offset))
    c = float(np.dot(offset, offset) - radius * radius)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    hits = [(-b - root) / 2.0, (-b + root) / 2.0]
    positive = [hit for hit in hits if hit > 1e-6]
    return min(positive) if positive else None


class _OfflinePlant:
    """One command owner, acceleration-limited kinematics, and synthetic LiDAR."""

    def __init__(self, config: Mapping[str, object]):
        self.config = config
        self.limits = config["limits"]
        self.reward_config = config["reward_config"]
        self.mapper = ActionMapper()
        self.builder = ObservationBuilder()
        self.publishers = [PUBLISHER_OWNER]
        self.controller_ready = True
        self.lidar_fresh = True
        self.odom_fresh = True
        self.clock_advancing = True
        self.simulator_alive = True
        self.emergency = False
        self.collision_latched = False
        self.episode_terminated = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear = 0.0
        self.angular = 0.0
        self.goal = (1.0, 0.0)
        self.sim_time = 0.0
        self.applied_commands = []
        self.owner = PUBLISHER_OWNER

    def emergency_stop(self) -> None:
        self.emergency = True
        self._zero_motion()

    def ingest_contact(self, collision: bool) -> bool:
        """Latch an obstacle contact. A later benign report does not clear it."""

        if collision:
            self.collision_latched = True
            self._zero_motion()
        return self.collision_latched

    def clear_emergency(self) -> None:
        """Operator action. Episode reset does not call this."""

        self.emergency = False

    def _zero_motion(self) -> None:
        stopped = self.mapper.emergency_stop()
        self.linear = stopped.linear
        self.angular = stopped.angular
        self.applied_commands.append((0.0, 0.0, "stop"))

    def _circles(self):
        circles = []
        for obstacle in self.config.get("obstacles") or []:
            center = obstacle.get("center")
            radius = obstacle.get("radius")
            if center is not None and radius is not None:
                circles.append((np.asarray(center, dtype=np.float64), float(radius)))
        return circles

    def ranges(self) -> np.ndarray | None:
        if not self.lidar_fresh:
            return None
        bins = self.builder.lidar_bins
        output = np.full(bins, self.builder.lidar_max_range, dtype=np.float32)
        origin = np.array([self.x, self.y], dtype=np.float64)
        circles = self._circles()
        for index in range(bins):
            angle = -math.pi + (2.0 * math.pi * index / bins) + self.yaw
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            nearest = self.builder.lidar_max_range
            for center, radius in circles:
                hit = _ray_circle(origin, direction, center, radius)
                if hit is not None:
                    nearest = min(nearest, hit)
            output[index] = nearest
        return output

    def goal_geometry(self) -> tuple[float, float]:
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) - self.yaw
        return distance, heading

    def observation(self):
        if not self.lidar_fresh or not self.odom_fresh:
            return None
        ranges = self.ranges()
        distance, heading = self.goal_geometry()
        result = self.builder.build(
            lidar_ranges=ranges,
            goal_distance=distance,
            goal_heading=heading,
            linear_velocity=self.linear,
            angular_velocity=self.angular,
        )
        if not result.valid:
            return None
        return result.vector

    def overlapping(self) -> bool:
        margin = float(self.config.get("robot_radius_m", 0.25))
        for center, radius in self._circles():
            if math.hypot(self.x - center[0], self.y - center[1]) <= radius + margin:
                return True
        ranges = self.ranges()
        if ranges is not None and float(np.min(ranges)) <= margin:
            return True
        return False

    def apply(self, action: np.ndarray) -> tuple[bool, str]:
        if len(self.publishers) != 1 or self.publishers[0] != PUBLISHER_OWNER:
            return False, "ownership lost: duplicate command publisher"
        if self.owner != PUBLISHER_OWNER:
            return False, "ownership lost"
        if not self.controller_ready:
            self._zero_motion()
            return False, "controller inactive"
        if not self.simulator_alive or not self.clock_advancing:
            self._zero_motion()
            return False, "simulator timeout"
        if self.emergency:
            self._zero_motion()
            return False, "Emergency stop active"
        if self.collision_latched:
            self._zero_motion()
            return False, "Collision latch"
        if self.episode_terminated:
            self._zero_motion()
            return False, "Episode terminated"
        if not self.lidar_fresh or not self.odom_fresh:
            self._zero_motion()
            return False, "LiDAR data stale" if not self.lidar_fresh else "Odometry data stale"
        if bool(np.allclose(action, 0.0, atol=1e-7)):
            self._zero_motion()
            return True, "stationary"
        try:
            command = self.mapper.map_action(action, self.limits.control_dt_s)
        except ValueError as exc:
            self._zero_motion()
            return False, str(exc)
        self.linear = command.linear
        self.angular = command.angular
        self.applied_commands.append((command.linear, command.angular, "command"))
        return True, "accepted"


class JackalWarehouseRLEnv(gym.Env):
    """40-D Jackal navigation environment with an explicit safety gate."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        config=None,
        *,
        execution: str | None = None,
        authorize_live: bool = False,
        render_mode: str | None = None,
        session=None,
        transport=None,
        connect: bool = False,
    ):
        super().__init__()
        self.config = config if isinstance(config, dict) and "limits" in config else load_warehouse_config(config)
        self.execution = execution or self.config.get("execution", "offline")
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._abort_reason = None
        self._closed = False
        self._episode_open = False
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._previous_distance = None
        self._tracker = EpisodeTracker()
        self._last_observation = np.zeros(40, dtype=np.float32)
        self._scenario = None
        self.session = session
        self.plant = None
        if isinstance(self.session, StationaryObservationSession):
            if authorize_live or connect:
                raise RuntimeError(
                    "authorize_live does not enable motion on the observation-only path"
                )
            self.execution = "stationary_observation"
        elif self.session is not None:
            self.execution = "gazebo"
        elif self.execution == "stationary_observation":
            if authorize_live:
                raise RuntimeError(
                    "authorize_live does not enable motion on the observation-only path"
                )
            self.session = StationaryObservationSession(
                self.config,
                transport=transport,
                connect=connect,
            )
        elif self.execution == "offline":
            self.plant = _OfflinePlant(self.config)
        elif self.execution == "gazebo":
            if not authorize_live:
                raise RuntimeError(
                    "Gazebo execution creates the robot command interface. "
                    "Pass authorize_live=True only after a separate motion authorization."
                )
            self.session = GazeboWarehouseSession(self.config)
        else:
            raise ValueError(
                "execution must be 'offline', 'gazebo', or 'stationary_observation'"
            )

    def _constraints(self) -> dict:
        return {
            "min_separation_m": self.config.get("min_separation_m", 0.5),
            "robot_radius_m": self.config.get("robot_radius_m", 0.25),
            "goal_tolerance_m": self.config["limits"].goal_tolerance_m,
            "bounds_xy": self.config.get("bounds_xy"),
            "obstacles": self.config.get("obstacles") or [],
        }

    def _choose_scenario(self, options: Mapping[str, object]) -> dict:
        if "start" in options or "goal" in options:
            scenario = {
                "id": options.get("scenario_id", "explicit"),
                "split": self.config.get("split", "train"),
                "start": options.get("start"),
                "goal": options.get("goal"),
            }
            reason = rejection_reason(scenario, self._constraints())
            if reason is not None:
                raise ValueError(reason)
            if self.config.get("require_listed_pair", False):
                key = pair_key(scenario["start"], scenario["goal"])
                listed = {
                    pair_key(item["start"], item["goal"])
                    for item in self.config.get("scenarios") or []
                    if str(item.get("split", "train")) == self.config.get("split", "train")
                }
                if key not in listed:
                    raise ValueError("start-goal pair is not in the configured split")
            scenario["start"] = tuple(float(value) for value in scenario["start"])
            if len(scenario["start"]) == 2:
                scenario["start"] = (*scenario["start"], 0.0)
            scenario["goal"] = tuple(float(value) for value in scenario["goal"])
            scenario["goal_frame"] = str(options.get("goal_frame", "world"))
            return scenario
        return select_scenario(
            self.config.get("scenarios") or [],
            self.np_random,
            str(self.config.get("split", "train")),
            self._constraints(),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment")
        if self._abort_reason is not None:
            self._halt()
            raise RuntimeError(self._abort_reason)
        options = {} if options is None else dict(options)
        scenario = self._choose_scenario(options)
        if self.plant is not None:
            observation, info = self._reset_offline(scenario)
        else:
            observation, info = self.session.reset(seed=seed, scenario=scenario)
        self._scenario = scenario
        self._episode_open = True
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._previous_distance = float(info["goal_distance"])
        self._last_observation = np.asarray(observation, dtype=np.float32)
        self._tracker.begin(*info["pose_xy"])
        info.update(
            {
                "scenario_id": scenario.get("id"),
                "split": self.config.get("split", "train"),
                "world_name": self.config.get("world_name"),
                "execution": self.execution,
                "observation_valid": True,
                "episode_length": 0,
            }
        )
        return self._last_observation.copy(), info

    def retarget(self, goal, *, reset_steps: bool = True) -> np.ndarray:
        """Aim the current episode at another destination without moving the robot."""
        if self.execution == "stationary_observation":
            raise RuntimeError("Cannot retarget a stationary observation")
        goal = (float(goal[0]), float(goal[1]))
        if self.execution == "gazebo":
            if getattr(self.session, "route_frame", "odom") == "world":
                from jackal_rl_navigation.goal_frames import world_goal_to_odom

                goal = world_goal_to_odom(
                    goal,
                    self.session.world_robot_pose,
                    self.session.odom_robot_pose,
                )
            observation = self.session._assign_odom_goal(goal)
            inner = self.session.env
            inner.episode_done = False
            monitor = inner.robot.safety_monitor
            if getattr(monitor, "emergency_active", False):
                raise RuntimeError("Emergency stop is latched")
            monitor.reset_episode()
            if reset_steps:
                inner.episode_steps = 0
            self._previous_distance = float(inner.previous_goal_distance)
        else:
            if self.plant.emergency or self.plant.collision_latched:
                raise RuntimeError("Emergency stop is latched" if self.plant.emergency else "Collision latch")
            self.plant.goal = goal
            self.plant.episode_terminated = False
            distance, _heading = self.plant.goal_geometry()
            self._previous_distance = distance
            observation = self.plant.observation()
        self._episode_open = True
        self._last_observation = np.asarray(observation, dtype=np.float32)
        return self._last_observation.copy()

    def _reset_offline(self, scenario):
        plant = self.plant
        if plant.emergency:
            plant._zero_motion()
            raise RuntimeError("Emergency stop is latched; reset cannot clear it")
        if not plant.controller_ready:
            raise RuntimeError("Controller is not ready")
        if not plant.lidar_fresh or not plant.odom_fresh:
            raise RuntimeError("LiDAR and odometry are not fresh")
        if not plant.simulator_alive or not plant.clock_advancing:
            raise RuntimeError("Simulation clock is not advancing")
        linear_limit = float(self.config.get("stationary_linear_limit_mps", 0.02))
        angular_limit = float(self.config.get("stationary_angular_limit_radps", 0.05))
        if abs(plant.linear) > linear_limit or abs(plant.angular) > angular_limit:
            plant._zero_motion()
            raise RuntimeError("Robot is not stationary")
        plant.collision_latched = False
        plant.episode_terminated = False
        plant.mapper.reset()
        plant.x, plant.y, plant.yaw = (float(value) for value in scenario["start"])
        plant.goal = (float(scenario["goal"][0]), float(scenario["goal"][1]))
        plant.linear = 0.0
        plant.angular = 0.0
        if plant.overlapping():
            raise RuntimeError("Start pose intersects a configured obstacle")
        observation = plant.observation()
        if observation is None:
            raise RuntimeError("Reset did not produce a valid observation")
        distance, _heading = plant.goal_geometry()
        return observation, {
            "start_world": scenario["start"],
            "goal_world": scenario["goal"],
            "pose_xy": (plant.x, plant.y),
            "goal_distance": distance,
            "collision_state": False,
            "minimum_obstacle_distance": minimum_valid_range(
                plant.ranges(), plant.builder.lidar_max_range,
            ),
        }

    def step(self, action):
        if self.execution == "stationary_observation":
            raise RuntimeError(
                "step() is blocked on the stationary observation path. "
                "It does not publish velocity or step the simulator."
            )
        if self._closed:
            raise RuntimeError("Cannot step a closed environment")
        if not self._episode_open:
            raise RuntimeError("Call reset() before step()")
        if self._abort_reason is not None:
            self._halt()
            raise RuntimeError(self._abort_reason)
        try:
            prepared = prepare_action(action)
        except ValueError:
            self._halt()
            raise
        if self.plant is not None:
            transition = self._step_offline(prepared)
        else:
            transition = self.session.step(
                prepared,
                previous_distance=self._previous_distance,
                previous_action=self._previous_action,
            )
        observation, reward, terminated, truncated, info = transition
        self._previous_action = prepared
        if info.get("goal_distance") is not None:
            self._previous_distance = float(info["goal_distance"])
        self._last_observation = np.asarray(observation, dtype=np.float32)
        if info.get("fatal") or info.get("termination_reason") in FATAL_REASONS:
            self._abort_reason = (
                "Training aborted: " + str(info.get("termination_reason"))
            )
            info["fatal"] = True
        info["scenario_id"] = None if self._scenario is None else self._scenario.get("id")
        info["split"] = self.config.get("split", "train")
        if terminated or truncated:
            self._episode_open = False
            LOGGER.info(
                "episode_end reason=%s reward=%.4f goal_distance=%s",
                info.get("termination_reason"),
                reward,
                info.get("goal_distance"),
            )
        return observation, reward, terminated, truncated, info

    def _step_offline(self, action):
        plant = self.plant
        before = (plant.x, plant.y, plant.yaw)
        accepted, reason = plant.apply(action)
        if not accepted:
            distance, _heading = plant.goal_geometry()
            return self._finish(
                plant,
                before,
                action,
                accepted=False,
                command_reason=reason,
                distance=distance,
                collided=plant.collision_latched,
            )
        dt = plant.limits.control_dt_s
        plant.x += plant.linear * math.cos(plant.yaw) * dt
        plant.y += plant.linear * math.sin(plant.yaw) * dt
        yaw_delta = plant.angular * dt
        plant.yaw = math.atan2(math.sin(plant.yaw + yaw_delta), math.cos(plant.yaw + yaw_delta))
        plant.sim_time += dt
        collided = plant.overlapping()
        if collided:
            plant.x, plant.y, plant.yaw = before
            plant.collision_latched = True
            plant._zero_motion()
            yaw_delta = 0.0
        distance, _heading = plant.goal_geometry()
        return self._finish(
            plant,
            before,
            action,
            accepted=True,
            command_reason=None,
            distance=distance,
            collided=collided,
            yaw_delta=yaw_delta,
        )

    def _finish(
        self,
        plant,
        before,
        action,
        *,
        accepted,
        command_reason,
        distance,
        collided,
        yaw_delta=0.0,
    ):
        observation = plant.observation()
        valid = observation is not None and bool(np.all(np.isfinite(observation)))
        if not valid:
            observation = self._last_observation.copy()
        clearance = minimum_valid_range(plant.ranges(), plant.builder.lidar_max_range)
        terminated, truncated, reason, fatal = decide_episode(
            goal_distance=distance,
            limits=plant.limits,
            collision=collided,
            episode_steps=self._tracker.steps + 1,
            command_reason=None if accepted else command_reason,
            sensor_failure=not plant.lidar_fresh or not plant.odom_fresh,
            simulator_failure=not plant.simulator_alive or not plant.clock_advancing,
            invalid_observation=not valid,
        )
        goal_reached = reason == "goal_reached"
        reward, reward_info = calculate_reward(
            plant.reward_config,
            previous_goal_distance=self._previous_distance,
            current_goal_distance=distance,
            goal_reached=goal_reached,
            collision=collided,
            min_obstacle_distance=clearance,
            angular_velocity=plant.angular,
            previous_action=self._previous_action,
            action=action,
        )
        self._tracker.record(
            x=plant.x,
            y=plant.y,
            yaw_delta=yaw_delta,
            linear=plant.linear,
            angular=plant.angular,
            clearance=clearance,
            reward=reward,
            dt=plant.limits.control_dt_s,
            reward_info=reward_info,
        )
        if terminated or truncated:
            plant.episode_terminated = True
            plant._zero_motion()
        info = dict(reward_info)
        info.update(self._tracker.metrics(reason=reason, goal_distance=distance))
        info.update(
            {
                "episode_length": self._tracker.steps,
                "action_accepted": accepted,
                "command_reason": command_reason,
                "observation_valid": valid,
                "fatal": fatal,
                "collision_state": collided,
                "pose_xy": (plant.x, plant.y),
                "start_pose": before,
            }
        )
        return observation, reward, terminated, truncated, info

    def render(self):
        points = np.asarray(self._tracker.positions or [(0.0, 0.0)], dtype=np.float32)
        image = _trajectory_image(
            points,
            self._scenario.get("goal") if self._scenario else (1.0, 0.0),
            self.config.get("obstacles") or [],
        )
        if self.render_mode == "human":
            LOGGER.info("trajectory_points=%s", len(points))
        return image

    def export_trajectory(self) -> dict:
        scenario = self._scenario or {}
        return {
            "points": [list(point) for point in self._tracker.positions],
            "start": list(scenario.get("start", (0.0, 0.0, 0.0)))[:2],
            "goal": list(scenario.get("goal", (0.0, 0.0))),
            "obstacles": list(self.config.get("obstacles") or []),
            "world_name": self.config.get("world_name"),
        }

    def _halt(self) -> None:
        if self.execution == "stationary_observation":
            return
        if self.plant is not None:
            self.plant._zero_motion()
            self.plant.episode_terminated = True
        elif self.session is not None:
            self.session.halt()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._episode_open = False
        if self.execution == "stationary_observation":
            if self.session is not None:
                self.session.close()
            return
        try:
            self._halt()
        finally:
            if self.session is not None:
                self.session.close()


def _trajectory_image(points: np.ndarray, goal, obstacles) -> np.ndarray:
    image = np.full((240, 240, 3), 245, dtype=np.uint8)
    finite = np.asarray(list(points) + [goal], dtype=np.float64)
    minimum = finite.min(axis=0) - 0.5
    span = np.maximum(finite.max(axis=0) - minimum, 1.0)

    def pixel(point):
        xy = (np.asarray(point[:2], dtype=np.float64) - minimum) / span
        col = int(np.clip(xy[0], 0.0, 0.999) * 239)
        row = int(np.clip(1.0 - xy[1], 0.0, 0.999) * 239)
        return row, col

    for obstacle in obstacles:
        center = obstacle.get("center")
        if center is None:
            continue
        row, col = pixel(center)
        image[max(0, row - 3):row + 4, max(0, col - 3):col + 4] = (160, 160, 160)
    for point in points:
        row, col = pixel(point)
        image[row, col] = (30, 90, 200)
    start_row, start_col = pixel(points[0])
    goal_row, goal_col = pixel(goal)
    image[max(0, start_row - 2):start_row + 3, max(0, start_col - 2):start_col + 3] = (20, 160, 60)
    image[max(0, goal_row - 2):goal_row + 3, max(0, goal_col - 2):goal_col + 3] = (200, 40, 40)
    return image


class GazeboWarehouseSession:
    """Adapt the existing Gazebo environment without a second publisher."""

    def __init__(self, config: Mapping[str, object], env=None):
        self.config = config
        self.reward_config = config["reward_config"]
        self.limits = config["limits"]
        self._owns_env = env is None
        self._controller_query = None
        if env is None:
            from jackal_rl_navigation.collision_config import CollisionConfig
            from jackal_rl_navigation.gazebo_env import JackalGazeboEnv
            from jackal_rl_navigation.world_catalog import require_navigation_world

            require_navigation_world(self.config)
            self.env = JackalGazeboEnv(
                sensor_timeout=self.limits.sensor_timeout_s,
                collision_config=CollisionConfig(enabled=True, collision_penalty=0.0),
                world=self.config["world_name"],
                partition=self.config["gazebo_partition"],
            )
            self._controller_query = _controller_query(self.env, self.limits.sensor_timeout_s)
        else:
            self.env = env
        self.env.reward_calculator = _ZeroInnerReward()
        self.env.goal_bonus = 0.0
        self.env.goal_tolerance = self.limits.goal_tolerance_m
        self.env.max_episode_steps = self.limits.max_episode_steps
        if getattr(self.env, "collision_config", None) is not None:
            penalty = getattr(self.env.collision_config, "collision_penalty", None)
            if penalty not in (None, 0.0):
                raise RuntimeError("Inner collision penalty must stay zero; the warehouse reward owns it")
        self._tracker = EpisodeTracker()
        self._last_observation = np.zeros(40, dtype=np.float32)

    def reset(self, *, seed, scenario):
        robot = self.env.robot
        if bool(robot.safety_monitor.emergency_active):
            robot.stop()
            raise RuntimeError("Emergency stop is latched; reset cannot clear it")
        self._require_controllers(robot)
        start = tuple(scenario["start"])
        goal = tuple(scenario["goal"])
        observation, info = self.env.reset(
            seed=seed,
            options={"start": start, "goal": goal[:2]},
        )
        self._assert_ready(robot)
        if scenario.get("goal_frame") == "odom":
            observation = self._assign_odom_goal(goal)
        elif getattr(self, "route_frame", None) == "world":
            from jackal_rl_navigation.goal_frames import odom_pose_from_message

            world_pose = info["robot_world_pose"]
            self.world_robot_pose = (float(world_pose[0]), float(world_pose[1]), float(world_pose[3]))
            self.odom_robot_pose = odom_pose_from_message(robot.odom)
        distance = float(self.env.previous_goal_distance)
        self._tracker = EpisodeTracker()
        pose = _pose_xy(robot)
        self._previous_yaw = _yaw(robot)
        self._tracker.begin(*pose)
        self._last_observation = np.asarray(observation, dtype=np.float32)
        info = dict(info)
        info.update(
            {
                "pose_xy": pose,
                "goal_distance": distance,
                "minimum_obstacle_distance": _scan_minimum(robot, self.reward_config),
                "collision_state": False,
            }
        )
        return observation, info

    def step(self, action, *, previous_distance, previous_action):
        from jackal_rl_navigation.gazebo_controller import (
            AmbiguousSimulationStep,
            GazeboControlError,
        )

        robot = self.env.robot
        before = _pose_xy(robot)
        try:
            observation, _ignored, terminated, truncated, info = self.env.step(action)
        except TimeoutError as exc:
            return self._failure(str(exc), "stale_sensors", previous_distance, previous_action, action)
        except (GazeboControlError, AmbiguousSimulationStep) as exc:
            return self._failure(str(exc), "simulator_timeout", previous_distance, previous_action, action)
        info = dict(info)
        current = float(info.get("current_distance", previous_distance))
        collided = bool(info.get("collision_detected", False))
        accepted = bool(info.get("action_accepted", False)) or bool(info.get("stationary", False))
        command_reason = None if accepted else str(getattr(robot, "last_safety_reason", "") or "")
        clearance = _scan_minimum(robot, self.reward_config)
        angular = _angular(robot)
        if not accepted and not collided and not terminated:
            rejected_terminated, rejected_truncated, reason, fatal = decide_episode(
                goal_distance=current,
                limits=self.limits,
                collision=False,
                episode_steps=int(info.get("episode_steps", 1)),
                command_reason=command_reason,
            )
            terminated = rejected_terminated
            truncated = rejected_truncated
            fatal_reason = reason
        else:
            reason = info.get("termination_reason")
            if collided:
                reason = "collision"
                terminated, truncated = True, False
            elif info.get("goal_reached"):
                reason = "goal_reached"
                terminated, truncated = True, False
            elif truncated or info.get("time_limit_reached"):
                reason = "time_limit"
                terminated, truncated = False, True
            fatal_reason = reason
        observation = np.asarray(observation, dtype=np.float32)
        valid = bool(np.all(np.isfinite(observation))) and bool(
            self.env.observation_space.contains(observation)
        )
        if not valid:
            observation = self._last_observation.copy()
        reward, reward_info = calculate_reward(
            self.reward_config,
            previous_goal_distance=float(previous_distance),
            current_goal_distance=current,
            goal_reached=reason == "goal_reached",
            collision=reason == "collision",
            min_obstacle_distance=clearance,
            angular_velocity=angular,
            previous_action=previous_action,
            action=action,
        )
        pose = _pose_xy(robot)
        yaw = _yaw(robot)
        previous_yaw = getattr(self, "_previous_yaw", yaw)
        yaw_delta = math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
        self._previous_yaw = yaw
        self._tracker.record(
            x=pose[0],
            y=pose[1],
            yaw_delta=yaw_delta,
            linear=_linear(robot),
            angular=angular,
            clearance=clearance,
            reward=reward,
            dt=self.limits.control_dt_s,
            reward_info=reward_info,
        )
        fatal = fatal_reason in FATAL_REASONS
        if (terminated or truncated) and not self.env.episode_done:
            self.env.episode_done = True
            robot.terminate_episode()
        merged = dict(reward_info)
        merged.update(self._tracker.metrics(reason=reason, goal_distance=current))
        merged.update(
            {
                "episode_length": self._tracker.steps,
                "action_accepted": accepted,
                "command_reason": command_reason,
                "observation_valid": valid,
                "fatal": fatal,
                "collision_state": reason == "collision",
                "collision_pairs": list(info.get("collision_pairs") or []),
                "pose_xy": pose,
                "start_pose": before,
                "termination_reason": reason,
            }
        )
        self._last_observation = observation
        return observation, reward, terminated, truncated, merged

    def _failure(self, message, reason, previous_distance, previous_action, action):
        self.halt()
        reward, reward_info = calculate_reward(
            self.reward_config,
            previous_goal_distance=float(previous_distance),
            current_goal_distance=float(previous_distance),
            goal_reached=False,
            collision=False,
            min_obstacle_distance=None,
            angular_velocity=0.0,
            previous_action=previous_action,
            action=action,
        )
        info = dict(reward_info)
        info.update(
            {
                "episode_length": self._tracker.steps,
                "termination_reason": reason,
                "success": False,
                "collision": False,
                "timeout": False,
                "fatal": True,
                "observation_valid": False,
                "action_accepted": False,
                "command_reason": message,
                "goal_distance": float(previous_distance),
                "collision_state": False,
                "minimum_obstacle_distance": None,
                "path_length_m": self._tracker.path_length_m,
                "duration_s": self._tracker.duration_s,
                "failure_message": message,
            }
        )
        return self._last_observation.copy(), reward, False, True, info

    def _assign_odom_goal(self, goal) -> np.ndarray:
        """Keep a requested destination in the odometry frame.

        The Gazebo reset converts a world goal by the odometry offset. A
        destination typed by the operator is already compared with pose_xy,
        so that conversion must not move the target.
        """

        robot = self.env.robot
        robot.goal_x = float(goal[0])
        robot.goal_y = float(goal[1])
        frame = getattr(getattr(robot.odom, "header", None), "frame_id", None) or "odom"
        robot.goal_frame = frame
        relative = robot._goal_relative()
        if relative is None:
            raise RuntimeError("Goal distance unavailable for the odom destination")
        self.env.previous_goal_distance = float(relative[0])
        observation = np.asarray(robot.get_observation(), dtype=np.float32)
        if observation.shape != (40,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("Odom destination did not produce a valid observation")
        return observation

    def _assert_ready(self, robot) -> None:
        linear_limit = float(self.config.get("stationary_linear_limit_mps", 0.02))
        angular_limit = float(self.config.get("stationary_angular_limit_radps", 0.05))
        deadline = time.monotonic() + float(self.limits.sensor_timeout_s)

        def measurement():
            if robot.scan is None or robot.odom is None:
                return None
            twist = robot.odom.twist.twist
            linear = math.hypot(float(twist.linear.x), float(getattr(twist.linear, "y", 0.0)))
            angular = abs(float(twist.angular.z))
            return linear, angular

        while True:
            sample = measurement()
            if sample is None:
                robot.stop()
                raise RuntimeError("LiDAR and odometry are not fresh")
            linear, angular = sample
            if linear <= linear_limit and angular <= angular_limit:
                self._require_controllers(robot)
                return
            robot.stop()
            sample = measurement()
            if sample is not None and sample[0] <= linear_limit and sample[1] <= angular_limit:
                self._require_controllers(robot)
                return
            executor = getattr(self.env, "executor", None)
            if executor is None or time.monotonic() >= deadline:
                raise RuntimeError("Robot is not stationary after reset")
            executor.spin_once(timeout_sec=0.05)

    def _require_controllers(self, robot) -> None:
        from jackal_rl_navigation.warehouse_observation import require_active_controllers

        query = getattr(robot, "list_controllers", None)
        if not callable(query):
            query = self._controller_query
        if not callable(query):
            raise RuntimeError("controller_manager response is missing")
        require_active_controllers(query())

    def halt(self) -> None:
        robot = getattr(self.env, "robot", None)
        if robot is not None and hasattr(robot, "terminate_episode"):
            robot.terminate_episode()
        elif robot is not None and hasattr(robot, "stop"):
            robot.stop()

    def close(self) -> None:
        if self._owns_env:
            self.env.close()


class _ZeroInnerReward:
    """Keep the existing environment's reward contract without double counting."""

    def calculate(self, previous_distance, current_distance):
        previous_distance = float(previous_distance)
        current_distance = float(current_distance)
        if not math.isfinite(previous_distance) or not math.isfinite(current_distance):
            raise ValueError("Goal distances must be finite")
        if previous_distance < 0 or current_distance < 0:
            raise ValueError("Goal distances cannot be negative")
        return 0.0, {
            "previous_distance": previous_distance,
            "current_distance": current_distance,
            "progress": previous_distance - current_distance,
            "progress_reward": 0.0,
            "step_penalty": 0.0,
        }


def _controller_query(env, timeout_s: float):
    """Read-only controller list on the node that already owns the command topic."""

    from controller_manager_msgs.srv import ListControllers

    from jackal_rl_navigation.warehouse_observation import controllers_dict

    node = env.robot
    client = node.create_client(
        ListControllers,
        "/j100_0000/controller_manager/list_controllers",
    )

    def query():
        if not client.wait_for_service(timeout_sec=float(timeout_s)):
            raise RuntimeError("controller_manager list_controllers is not available")
        future = client.call_async(ListControllers.Request())
        deadline = time.monotonic() + float(timeout_s)
        while not future.done():
            if time.monotonic() >= deadline:
                raise RuntimeError("controller_manager timed out")
            env.executor.spin_once(timeout_sec=0.05)
        return controllers_dict(future.result())

    return query


def _pose_xy(robot) -> tuple[float, float]:
    position = robot.odom.pose.pose.position
    return float(position.x), float(position.y)


def _linear(robot) -> float:
    return float(robot.odom.twist.twist.linear.x)


def _angular(robot) -> float:
    return float(robot.odom.twist.twist.angular.z)


def _yaw(robot) -> float:
    pose = getattr(getattr(robot.odom, "pose", None), "pose", None)
    orientation = getattr(pose, "orientation", None)
    if orientation is None or not hasattr(orientation, "w"):
        return 0.0
    sin_yaw = 2.0 * (
        float(orientation.w) * float(orientation.z)
        + float(getattr(orientation, "x", 0.0)) * float(getattr(orientation, "y", 0.0))
    )
    cos_yaw = 1.0 - 2.0 * (
        float(getattr(orientation, "y", 0.0)) ** 2
        + float(orientation.z) ** 2
    )
    return math.atan2(sin_yaw, cos_yaw)


def _scan_minimum(robot, reward_config):
    scan = getattr(robot, "scan", None)
    ranges = None if scan is None else getattr(scan, "ranges", None)
    return minimum_valid_range(ranges, reward_config.lidar_max_range_m)
