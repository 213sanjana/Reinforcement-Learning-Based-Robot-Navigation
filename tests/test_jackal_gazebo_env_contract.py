"""Unit tests for the real JackalGazeboEnv transition contract.

ROS objects are replaced with fakes. These tests do not claim that Gazebo,
LiDAR, or odometry were live.
"""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

try:
    from jackal_rl_navigation.collision_config import CollisionConfig
    from jackal_rl_navigation.gazebo_controller import GazeboControlError, AmbiguousSimulationStep
    from jackal_rl_navigation.gazebo_env import JackalGazeboEnv
    from jackal_rl_navigation.reward_calculator import NavigationReward
except Exception as exc:
    pytest.skip(
        f"ROS Jackal environment is unavailable: {exc}",
        allow_module_level=True,
    )


def _header(sec):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=int(sec), nanosec=0),
        frame_id="odom",
        child_frame_id="base_link",
    )


def _odom(sec, x=0.0, y=0.0):
    return SimpleNamespace(
        header=_header(sec),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0),
                angular=SimpleNamespace(z=0.0),
            )
        ),
    )


def _scan(sec):
    return SimpleNamespace(header=_header(sec), ranges=[1.0] * 36)


class _Robot:
    def __init__(self):
        self.scan = _scan(1)
        self.odom = _odom(1)
        self.stops = 0
        self.commands = []
        self.observation = np.zeros(40, dtype=np.float32)
        self.distance = 1.5
        self.goal_x = None
        self.goal_y = None
        self.goal_frame = None
        self.terminated = False
        self.safety_monitor = SimpleNamespace(
            reset_episode=lambda: None,
            reset_emergency=lambda: None,
        )
        self.action_mapper = SimpleNamespace(reset=lambda: None)

    def stop(self):
        self.stops += 1

    def command_action(self, action):
        self.commands.append(np.asarray(action, dtype=np.float32).copy())
        return True

    def get_observation(self):
        if self.observation is None:
            return None
        return np.array(self.observation, dtype=np.float32)

    def _goal_relative(self):
        if self.distance is None:
            return None
        return float(self.distance), 0.0

    def terminate_episode(self):
        self.terminated = True
        self.stop()


class _Executor:
    def spin_once(self, timeout_sec=0.0):
        return None


class _Controller:
    def __init__(self, robot):
        self.robot = robot
        self.steps = []
        self.poses = []
        self.paused = False
        self.error = None
        self.refresh = True

    def pause(self):
        self.paused = True

    def set_pose(self, x, y, yaw):
        self.poses.append((x, y, yaw))

    def step(self, iterations):
        self.steps.append(iterations)
        if self.error is not None:
            raise self.error
        if self.refresh:
            stamp = 1 if self.robot.scan is None else self.robot.scan.header.stamp.sec + 1
            self.robot.scan = _scan(stamp)
            self.robot.odom = _odom(stamp)


class _World:
    def wait_for_position(self, x, y, controller=None):
        return (float(x), float(y), 0.3, 0.0)


def _env(sensor_timeout=0.05):
    env = JackalGazeboEnv.__new__(JackalGazeboEnv)
    gym.Env.__init__(env)
    env.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32,
    )
    env.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
    )
    env.robot = _Robot()
    env.executor = _Executor()
    env.controller = _Controller(env.robot)
    env.world_pose = _World()
    env.sensor_timeout = sensor_timeout
    env.max_episode_steps = 300
    env.goal_tolerance = 0.30
    env.goal_bonus = 20.0
    env.reward_calculator = NavigationReward()
    env.collision_config = CollisionConfig(enabled=False)
    env.collision_monitor = None
    env.previous_goal_distance = 2.0
    env.episode_steps = 0
    env.episode_done = False
    env._has_reset = False
    env.closed = False
    env.goal_world = (2.0, 0.0)
    return env


def _ready(env):
    env._has_reset = True
    env.episode_done = False
    return env


def test_reset_and_step_return_finite_contract_values():
    env = _env()
    observation, info = env.reset(seed=7, options={"start": (0.0, -1.5, 0.0), "goal": (2.5, -1.5)})
    assert observation.shape == (40,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert info["start_world"] == (0.0, -1.5, 0.0)
    assert info["goal_world"] == (2.5, -1.5)
    assert info["episode_steps"] == 0
    assert "goal_odom" in info
    assert env.controller.paused is True
    assert env.robot.stops >= 1

    env.robot.distance = 1.5
    next_observation, reward, terminated, truncated, step_info = env.step(
        np.array([0.2, -0.3], dtype=np.float32)
    )
    assert next_observation.shape == (40,)
    assert np.all(np.isfinite(next_observation))
    assert math.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert terminated is False and truncated is False
    assert step_info["termination_reason"] is None
    assert env.robot.commands and env.robot.commands[0].shape == (2,)
    assert np.allclose(env.robot.commands[0], [0.2, -0.3])


def test_invalid_actions_do_not_advance_physics():
    env = _ready(_env())
    for action in (
        np.array([2.0, 0.0], dtype=np.float32),
        np.array([np.nan, 0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            env.step(action)
    assert env.controller.steps == []
    assert env.robot.stops >= 1


def test_goal_collision_and_timeout_are_distinct():
    goal_env = _ready(_env())
    goal_env.robot.distance = 0.2
    _observation, reward, terminated, truncated, info = goal_env.step(
        np.array([0.1, 0.0], dtype=np.float32)
    )
    assert terminated is True and truncated is False
    assert info["termination_reason"] == "goal_reached"
    assert info["goal_reached"] is True and info["collision_detected"] is False
    assert reward > 10.0
    with pytest.raises(RuntimeError, match="reset"):
        goal_env.step(np.zeros(2, dtype=np.float32))

    collision_env = _ready(_env())
    collision_env.collision_monitor = SimpleNamespace(
        collision_detected=True,
        collision_pairs=["robot|box"],
    )
    collision_env.collision_config = CollisionConfig(enabled=True)
    collision_env.robot.distance = 0.1
    _observation, reward, terminated, truncated, info = collision_env.step(
        np.array([0.1, 0.0], dtype=np.float32)
    )
    assert terminated is True and truncated is False
    assert info["termination_reason"] == "collision"
    assert info["goal_reached"] is False
    assert info["reward_adjustment"] == -10.0
    assert math.isfinite(reward)

    timeout_env = _ready(_env())
    timeout_env.max_episode_steps = 1
    timeout_env.robot.distance = 2.0
    _observation, _reward, terminated, truncated, info = timeout_env.step(
        np.zeros(2, dtype=np.float32)
    )
    assert terminated is False and truncated is True
    assert info["termination_reason"] == "time_limit"
    assert info["time_limit_reached"] is True


def test_reset_after_termination_starts_a_new_episode():
    env = _env()
    env.reset(options={"start": (0.0, 0.0, 0.0), "goal": (2.0, 0.0)})
    env.robot.distance = 0.1
    _observation, _reward, terminated, _truncated, _info = env.step(
        np.array([0.2, 0.0], dtype=np.float32)
    )
    assert terminated is True
    observation, info = env.reset(options={"start": (1.0, 0.0, 0.0), "goal": (2.0, 0.0)})
    assert env.episode_done is False
    assert env.episode_steps == 0
    assert info["episode_steps"] == 0
    assert observation.shape == (40,)
    assert np.all(np.isfinite(observation))


def test_missing_or_nonfinite_observations_are_not_returned():
    stale = _ready(_env())
    stale.controller.refresh = False
    with pytest.raises(TimeoutError, match="fresh LiDAR"):
        stale.step(np.array([0.2, 0.0], dtype=np.float32))
    assert stale.robot.stops >= 1

    missing = _ready(_env())
    missing.robot.observation = None
    with pytest.raises(TimeoutError, match="fresh LiDAR"):
        missing.step(np.array([0.2, 0.0], dtype=np.float32))

    invalid = _ready(_env())
    invalid.robot.observation = np.full(40, np.nan, dtype=np.float32)
    with pytest.raises(TimeoutError, match="fresh LiDAR"):
        invalid.step(np.array([0.2, 0.0], dtype=np.float32))


def test_unavailable_goal_and_sensors_fail_before_a_false_observation():
    env = _ready(_env())
    env.robot.scan = None
    with pytest.raises(RuntimeError, match="Sensors unavailable"):
        env.step(np.array([0.1, 0.0], dtype=np.float32))

    reset_env = _env()
    reset_env.robot.observation = None
    with pytest.raises(RuntimeError, match="invalid or stale"):
        reset_env.reset(options={"start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})

    nan_reset = _env()
    nan_reset.robot.observation = np.full(40, np.nan, dtype=np.float32)
    with pytest.raises(RuntimeError, match="invalid observation"):
        nan_reset.reset(options={"start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})


@pytest.mark.parametrize("error_class", [GazeboControlError, AmbiguousSimulationStep])
def test_gazebo_control_failure_stops_the_robot_and_propagates(error_class):
    env = _ready(_env())
    env.controller.error = error_class("Gazebo service timed out")
    with pytest.raises(error_class, match="timed out"):
        env.step(np.array([0.4, 0.0], dtype=np.float32))
    assert env.robot.stops >= 1
    assert env.controller.steps == [100]
    assert env.episode_steps == 0


def test_reset_timeout_and_closed_environment():
    env = _env()
    env.controller.refresh = False
    with pytest.raises(TimeoutError, match="episode reset"):
        env.reset(options={"start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    with pytest.raises(ValueError, match="finite"):
        env.reset(options={"start": (math.nan, 0.0, 0.0), "goal": (1.0, 0.0)})
    env.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros(2, dtype=np.float32))
