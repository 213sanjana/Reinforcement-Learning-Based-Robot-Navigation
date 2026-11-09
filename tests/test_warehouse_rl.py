"""Offline tests for the warehouse PPO environment. No Gazebo and no training."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.reward_calculator import NavigationReward
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_env import (
    GazeboWarehouseSession,
    JackalWarehouseRLEnv,
    prepare_action,
)
from jackal_rl_navigation.warehouse_episode import EpisodeLimits, classify_rejection, decide_episode
from jackal_rl_navigation.warehouse_estop import main as estop_main
from jackal_rl_navigation.warehouse_reward import RewardConfig, calculate_reward, minimum_valid_range
from jackal_rl_navigation.warehouse_scenarios import rejection_reason, select_scenario
from training.warehouse_curriculum import advancement_decision


ROOT = Path(__file__).resolve().parents[1]


def _offline(**overrides):
    config = load_warehouse_config(None)
    config.update(overrides)
    return config


def _limits(**overrides):
    config = _offline()
    values = {
        "goal_tolerance_m": config["limits"].goal_tolerance_m,
        "max_episode_steps": config["limits"].max_episode_steps,
        "sensor_timeout_s": config["limits"].sensor_timeout_s,
        "control_frequency_hz": config["limits"].control_frequency_hz,
    }
    values.update(overrides)
    config["limits"] = EpisodeLimits(**values)
    return config


def test_existing_navigation_reward_defaults_are_unchanged():
    reward = NavigationReward()
    assert reward.progress_scale == 10.0
    assert reward.step_penalty == 0.01


def test_observation_contract_and_goal_geometry():
    builder = ObservationBuilder()
    ranges = np.full(900, 5.0, dtype=np.float32)
    result = builder.build(ranges, 4.0, 0.5, 0.1, -0.2)
    assert result.valid
    assert result.vector.shape == (40,)
    assert result.vector.dtype == np.float32
    assert np.all(np.isfinite(result.vector))
    left = builder.build(ranges, 2.0, np.pi / 2, 0.0, 0.0)
    assert left.vector[36] == pytest.approx(2.0 / 20.0)
    assert left.vector[37] == pytest.approx(0.5)
    stale = builder.build(np.full(36, np.nan), 1.0, 0.0, 0.0, 0.0)
    assert not stale.valid
    assert minimum_valid_range([np.nan, np.inf, 0.4, -1.0], 12.0) == pytest.approx(0.4)
    assert minimum_valid_range([np.nan, 0.0, -2.0], 12.0) is None


def test_action_conversion_clamps_and_keeps_physical_limits():
    with pytest.raises(ValueError):
        prepare_action([np.nan, 0.0])
    with pytest.raises(ValueError):
        prepare_action([np.inf, 0.0])
    assert np.allclose(prepare_action([1.4, -2.0]), [1.0, -1.0])
    mapper = ActionMapper()
    forward = mapper.map_action([1.0, 0.0], dt=2.0)
    assert forward.linear == pytest.approx(0.30)
    assert forward.angular == pytest.approx(0.0)
    mapper.reset()
    reverse = mapper.map_action([-1.0, 1.0], dt=2.0)
    assert reverse.linear == pytest.approx(-0.15)
    assert reverse.angular == pytest.approx(0.60)
    assert -0.15 <= reverse.linear <= 0.30
    assert abs(reverse.angular) <= 0.60


def test_reward_components_and_goal_collision_exclusion():
    config = RewardConfig()
    reward, info = calculate_reward(
        config,
        previous_goal_distance=2.0,
        current_goal_distance=1.5,
        goal_reached=False,
        collision=False,
        min_obstacle_distance=2.0,
        angular_velocity=0.5,
    )
    assert info["progress_reward"] == pytest.approx(2.5)
    assert info["step_penalty"] == pytest.approx(-0.01)
    assert info["proximity_penalty"] == 0.0
    assert info["angular_penalty"] == pytest.approx(-0.01)
    assert info["total_reward"] == pytest.approx(sum(
        info[key] for key in (
            "progress_reward", "goal_reward", "collision_penalty",
            "step_penalty", "proximity_penalty", "angular_penalty",
            "action_smoothness_penalty",
        )
    ))
    away, away_info = calculate_reward(
        config, previous_goal_distance=1.0, current_goal_distance=1.4,
        goal_reached=False, collision=False, min_obstacle_distance=1.0, angular_velocity=0.0,
    )
    assert away_info["progress_reward"] < 0
    assert away < 0
    goal, goal_info = calculate_reward(
        config, previous_goal_distance=0.4, current_goal_distance=0.2,
        goal_reached=True, collision=False, min_obstacle_distance=1.0, angular_velocity=0.0,
    )
    assert goal_info["goal_reward"] == 100.0
    both, both_info = calculate_reward(
        config, previous_goal_distance=0.4, current_goal_distance=0.2,
        goal_reached=True, collision=True, min_obstacle_distance=0.1, angular_velocity=0.0,
    )
    assert both_info["goal_reward"] == 0.0
    assert both_info["collision_penalty"] == -100.0
    assert both_info["proximity_zone"] == "critical"
    assert both < goal
    near, near_info = calculate_reward(
        config, previous_goal_distance=1.0, current_goal_distance=1.0,
        goal_reached=False, collision=False, min_obstacle_distance=0.5, angular_velocity=0.0,
    )
    assert near_info["proximity_penalty"] < 0
    assert near_info["collision_penalty"] == 0.0


def test_episode_outcomes_do_not_treat_timeout_as_success():
    limits = EpisodeLimits(max_episode_steps=4)
    terminated, truncated, reason, fatal = decide_episode(
        goal_distance=0.2, limits=limits, collision=False, episode_steps=2,
    )
    assert (terminated, truncated, reason, fatal) == (True, False, "goal_reached", False)
    terminated, truncated, reason, fatal = decide_episode(
        goal_distance=0.1, limits=limits, collision=True, episode_steps=2,
    )
    assert reason == "collision" and terminated and not truncated and not fatal
    terminated, truncated, reason, fatal = decide_episode(
        goal_distance=2.0, limits=limits, collision=False, episode_steps=4,
    )
    assert (terminated, truncated, reason) == (False, True, "time_limit")
    assert classify_rejection("LiDAR data stale")[2:] == ("stale_sensors", True)
    assert classify_rejection("Emergency stop active")[2:] == ("safety_fault", True)
    assert classify_rejection("Forward obstacle too close")[2] == "safety_rejected"


def test_sampler_rejects_blocked_and_overlapping_pairs():
    constraints = {
        "min_separation_m": 0.5,
        "robot_radius_m": 0.25,
        "goal_tolerance_m": 0.3,
        "obstacles": [{"center": [0.5, 0.0], "radius": 0.2}],
    }
    blocked = {"start": [0.0, 0.0, 0.0], "goal": [1.0, 0.0]}
    assert rejection_reason(blocked, constraints) is not None
    clear = {"id": "clear", "split": "train", "start": [0.0, 1.0, 0.0], "goal": [1.0, 1.0]}
    selected = select_scenario([blocked | {"id": "bad", "split": "train"}, clear], np.random.default_rng(1), "train", constraints)
    assert selected["id"] == "clear"


def test_configured_splits_are_disjoint_and_gazebo_training_is_refused():
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    assert config["execution"] == "gazebo"
    assert config["reward_config"].progress_scale == 5.0
    assert config["reward_config"].goal_reward == 100.0
    assert config["limits"].max_episode_steps == 600
    assert config["limits"].goal_tolerance_m == 0.3
    with pytest.raises(SystemExit):
        from training.warehouse_ppo_train import train
        train(config=config, output_dir=ROOT / "artifacts" / "warehouse_ppo" / "runs" / "should_not_exist")


def test_offline_episode_lifecycle():
    config = _limits(control_frequency_hz=1.0, max_episode_steps=5, goal_tolerance_m=0.30)
    config["min_separation_m"] = 0.31
    config["scenarios"] = [
        {"id": "near", "split": "train", "start": [0.0, 0.0, 0.0], "goal": [0.35, 0.0]},
    ]
    env = JackalWarehouseRLEnv(config)
    observation, info = env.reset(seed=3)
    assert observation.shape == (40,) and observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    observation, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert terminated and not truncated
    assert info["termination_reason"] == "goal_reached"
    assert info["goal_reward"] == 100.0
    assert info["success"] is True and info["timeout"] is False
    assert reward > 50.0
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(2, dtype=np.float32))
    commands = len(env.plant.applied_commands)
    with pytest.raises(RuntimeError):
        env.step(np.ones(2, dtype=np.float32))
    assert len(env.plant.applied_commands) == commands

    collision = _limits(control_frequency_hz=1.0, max_episode_steps=5)
    env = JackalWarehouseRLEnv(collision)
    env.reset(seed=1)
    env.config["obstacles"] = [{"center": [0.2, 0.0], "radius": 0.1}]
    _observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated and info["termination_reason"] == "collision"
    assert info["collision_penalty"] == -100.0
    assert info["goal_reward"] == 0.0
    assert env.plant.ingest_contact(False) is True
    env.close()

    limited = _limits(max_episode_steps=1)
    env = JackalWarehouseRLEnv(limited)
    env.reset(seed=2)
    _observation, _reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert truncated and not terminated
    assert info["termination_reason"] == "time_limit"
    assert info["success"] is False
    env.close()


def test_safety_faults_stop_commands_and_do_not_clear_emergency():
    env = JackalWarehouseRLEnv(_offline())
    env.reset(seed=1)
    start = env.plant.x
    env.plant.lidar_fresh = False
    _observation, _reward, terminated, truncated, info = env.step([1.0, 0.0])
    assert truncated and info["fatal"] and info["termination_reason"] == "stale_sensors"
    assert env.plant.x == start
    with pytest.raises(RuntimeError, match="Training aborted"):
        env.reset()
    env.close()

    env = JackalWarehouseRLEnv(_offline())
    env.reset(seed=1)
    env.plant.emergency_stop()
    _observation, _reward, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and info["termination_reason"] == "safety_fault"
    with pytest.raises(RuntimeError, match="Emergency"):
        env.plant.emergency = True
        env._abort_reason = None
        env._episode_open = False
        env.reset()
    env.plant.clear_emergency()
    env._abort_reason = None
    observation, _info = env.reset(seed=2)
    assert observation.shape == (40,)
    env.plant.controller_ready = False
    with pytest.raises(RuntimeError, match="Controller"):
        env.reset()
    env.close()

    env = JackalWarehouseRLEnv(_offline())
    env.reset(seed=4)
    env.plant.publishers.append("second_owner")
    x = env.plant.x
    _observation, _reward, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated and info["fatal"]
    assert info["termination_reason"] == "ownership_lost"
    assert env.plant.x == x
    assert len(env.plant.publishers) == 2
    env.close()


def test_new_modules_do_not_create_a_publisher():
    paths = [
        ROOT / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/warehouse_env.py",
        ROOT / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/warehouse_estop.py",
        ROOT / "training/warehouse_ppo_train.py",
        ROOT / "evaluation/warehouse_evaluate.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        assert "create_publisher" not in names
    assert estop_main([]) == 2


def test_advancement_uses_measured_rates_only():
    rules = {"min_episodes": 20, "min_success_rate": 0.8, "max_collision_rate": 0.1, "max_safety_failures": 0}
    refused = advancement_decision({"episodes": 20, "success_rate": 0.5, "collision_rate": 0.0, "safety_failures": 0}, rules)
    assert refused["eligible"] is False and refused["changed_stage"] is False
    allowed = advancement_decision({"episodes": 20, "success_rate": 0.9, "collision_rate": 0.05, "safety_failures": 0}, rules)
    assert allowed["eligible"] is True


def test_gazebo_session_uses_warehouse_reward_on_a_fake_environment():
    pytest.importorskip("rclpy")
    from types import SimpleNamespace
    import gymnasium as gym
    from jackal_rl_navigation.collision_config import CollisionConfig

    class Robot:
        def __init__(self):
            self.scan = SimpleNamespace(ranges=[1.0] * 36)
            self.odom = SimpleNamespace(
                pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0))),
                twist=SimpleNamespace(twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.0, y=0.0),
                    angular=SimpleNamespace(z=0.0),
                )),
            )
            self.safety_monitor = SimpleNamespace(emergency_active=False)
            self.last_safety_reason = ""
            self.stops = 0
            self.terminated = False

        def list_controllers(self):
            return {
                "controller": [
                    {"name": "joint_state_broadcaster", "state": "active"},
                    {"name": "platform_velocity_controller", "state": "active"},
                ]
            }

        def stop(self):
            self.stops += 1

        def terminate_episode(self):
            self.terminated = True
            self.stop()

    class Inner:
        def __init__(self):
            self.robot = Robot()
            self.collision_config = CollisionConfig(enabled=True, collision_penalty=0.0)
            self.episode_done = False
            self.previous_goal_distance = None
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
            self.distance = 1.0
            self.accept = True
            self.collide = False

        def reset(self, *, seed=None, options=None):
            self.episode_done = False
            self.previous_goal_distance = 1.0
            return np.zeros(40, dtype=np.float32), {}

        def step(self, action):
            info = {
                "current_distance": self.distance,
                "previous_distance": self.previous_goal_distance,
                "action_accepted": self.accept,
                "stationary": False,
                "collision_detected": self.collide,
                "goal_reached": self.distance <= 0.3 and not self.collide,
                "termination_reason": None,
                "episode_steps": 1,
                "time_limit_reached": False,
            }
            if info["goal_reached"]:
                info["termination_reason"] = "goal_reached"
            if self.collide:
                info["termination_reason"] = "collision"
                info["goal_reached"] = False
            terminated = bool(info["termination_reason"])
            self.previous_goal_distance = self.distance
            return np.zeros(40, dtype=np.float32), 99.0, terminated, False, info

    config = _offline()
    inner = Inner()
    session = GazeboWarehouseSession(config, env=inner)
    observation, info = session.reset(seed=1, scenario={"id": "fake", "start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    assert observation.shape == (40,)
    assert info["goal_distance"] == pytest.approx(1.0)
    inner.distance = 0.2
    _observation, reward, terminated, truncated, info = session.step(
        np.array([0.2, 0.0], dtype=np.float32),
        previous_distance=1.0,
        previous_action=np.zeros(2, dtype=np.float32),
    )
    assert terminated and not truncated
    assert info["goal_reward"] == 100.0
    assert reward != 99.0
    assert info["total_reward"] == pytest.approx(reward)

    inner = Inner()
    inner.distance = 0.1
    inner.collide = True
    session = GazeboWarehouseSession(config, env=inner)
    session.reset(seed=1, scenario={"id": "fake", "start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    _observation, reward, terminated, _truncated, info = session.step(
        np.ones(2, dtype=np.float32), previous_distance=1.0, previous_action=np.zeros(2),
    )
    assert terminated and info["termination_reason"] == "collision"
    assert info["goal_reward"] == 0.0
    assert info["collision_penalty"] == -100.0
    assert inner.robot.terminated is True

    inner = Inner()
    inner.accept = False
    inner.robot.last_safety_reason = "LiDAR data stale"
    session = GazeboWarehouseSession(config, env=inner)
    session.reset(seed=1, scenario={"id": "fake", "start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    _observation, _reward, terminated, truncated, info = session.step(
        np.ones(2, dtype=np.float32), previous_distance=1.0, previous_action=np.zeros(2),
    )
    assert truncated and not terminated
    assert info["fatal"] and info["termination_reason"] == "stale_sensors"


def test_odom_destination_is_not_shifted_by_the_world_offset():
    from types import SimpleNamespace

    config = _offline()

    class Robot:
        def __init__(self):
            self.scan = SimpleNamespace(ranges=[1.0] * 36)
            self.odom = SimpleNamespace(
                header=SimpleNamespace(frame_id="odom"),
                pose=SimpleNamespace(pose=SimpleNamespace(
                    position=SimpleNamespace(x=2.5, y=0.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )),
                twist=SimpleNamespace(twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.0, y=0.0),
                    angular=SimpleNamespace(z=0.0),
                )),
            )
            self.safety_monitor = SimpleNamespace(emergency_active=False)
            self.goal_x = 99.0
            self.goal_y = 99.0
            self.goal_frame = ""

        def list_controllers(self):
            return {
                "controller": [
                    {"name": "joint_state_broadcaster", "state": "active"},
                    {"name": "platform_velocity_controller", "state": "active"},
                ]
            }

        def stop(self):
            return None

        def _goal_relative(self):
            position = self.odom.pose.pose.position
            return math.hypot(self.goal_x - position.x, self.goal_y - position.y), 0.0

        def get_observation(self):
            return np.zeros(40, dtype=np.float32)

    class Inner:
        def __init__(self):
            self.robot = Robot()
            self.collision_config = None
            self.previous_goal_distance = 9.0

        def reset(self, *, seed=None, options=None):
            self.previous_goal_distance = 9.0
            return np.zeros(40, dtype=np.float32), {}

    session = GazeboWarehouseSession(config, env=Inner())
    _observation, info = session.reset(
        seed=1,
        scenario={
            "id": "destination",
            "start": (0.0, 0.0, 0.0),
            "goal": (1.5, 0.0),
            "goal_frame": "odom",
        },
    )
    assert session.env.robot.goal_x == pytest.approx(1.5)
    assert session.env.robot.goal_y == pytest.approx(0.0)
    assert info["goal_distance"] == pytest.approx(1.0)


def test_reset_accepts_a_robot_that_stops_when_commanded():
    from types import SimpleNamespace

    config = _offline()

    class Robot:
        def __init__(self):
            self.scan = SimpleNamespace(ranges=[1.0] * 36)
            self.odom = SimpleNamespace(
                pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0))),
                twist=SimpleNamespace(twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.2, y=0.0),
                    angular=SimpleNamespace(z=0.2),
                )),
            )
            self.safety_monitor = SimpleNamespace(emergency_active=False)

        def list_controllers(self):
            return {
                "controller": [
                    {"name": "joint_state_broadcaster", "state": "active"},
                    {"name": "platform_velocity_controller", "state": "active"},
                ]
            }

        def stop(self):
            self.odom.twist.twist.linear.x = 0.0
            self.odom.twist.twist.angular.z = 0.0

    class Inner:
        def __init__(self):
            self.robot = Robot()
            self.collision_config = None
            self.previous_goal_distance = 1.0

        def reset(self, *, seed=None, options=None):
            return np.zeros(40, dtype=np.float32), {}

    session = GazeboWarehouseSession(config, env=Inner())
    _observation, info = session.reset(
        seed=1, scenario={"id": "settle", "start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)},
    )
    assert info["goal_distance"] == pytest.approx(1.0)
    assert session.env.robot.odom.twist.twist.linear.x == 0.0


def test_reset_rejects_motion_that_stop_does_not_clear():
    from types import SimpleNamespace

    config = _offline()

    class Robot:
        def __init__(self):
            self.scan = SimpleNamespace(ranges=[1.0] * 36)
            self.odom = SimpleNamespace(
                pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0))),
                twist=SimpleNamespace(twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.2, y=0.0),
                    angular=SimpleNamespace(z=0.0),
                )),
            )
            self.safety_monitor = SimpleNamespace(emergency_active=False)
            self.stops = 0

        def list_controllers(self):
            return {
                "controller": [
                    {"name": "joint_state_broadcaster", "state": "active"},
                    {"name": "platform_velocity_controller", "state": "active"},
                ]
            }

        def stop(self):
            self.stops += 1

    class Inner:
        def __init__(self):
            self.robot = Robot()
            self.collision_config = None
            self.previous_goal_distance = 1.0

        def reset(self, *, seed=None, options=None):
            return np.zeros(40, dtype=np.float32), {}

    session = GazeboWarehouseSession(config, env=Inner())
    with pytest.raises(RuntimeError, match="not stationary after reset"):
        session.reset(seed=1, scenario={"start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    assert session.env.robot.stops == 1


def test_missing_controller_list_blocks_the_active_reset():
    config = _offline()

    class Robot:
        controller_ready = True
        safety_monitor = type("Safety", (), {"emergency_active": False})()
        stops = 0

        def stop(self):
            self.stops += 1

    class Inner:
        def __init__(self):
            self.robot = Robot()
            self.resets = 0

        def reset(self, *, seed=None, options=None):
            self.resets += 1
            return np.zeros(40, dtype=np.float32), {}

    inner = Inner()
    session = GazeboWarehouseSession(config, env=inner)
    with pytest.raises(RuntimeError, match="missing"):
        session.reset(seed=0, scenario={"start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    assert inner.resets == 0
    assert inner.robot.stops == 0


def test_live_warehouse_session_targets_the_external_warehouse(monkeypatch):
    pytest.importorskip("rclpy")
    import jackal_rl_navigation.gazebo_env as gazebo_env

    captured = {}

    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before the command node")

    monkeypatch.setattr(gazebo_env, "JackalGazeboEnv", FakeEnv)
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    with pytest.raises(RuntimeError, match="stop before the command node"):
        GazeboWarehouseSession(config)
    assert captured["world"] == "industrial-warehouse"
    assert captured["partition"] == "jackal_rl_external"

    config["world_name"] = "jackal_obstacle_test"
    config["gazebo_partition"] = "jackal_rl"
    with pytest.raises(RuntimeError, match="industrial-warehouse"):
        GazeboWarehouseSession(config)
    assert captured["world"] == "industrial-warehouse"


def test_default_gazebo_target_stays_on_the_obstacle_world():
    from jackal_rl_navigation.gazebo_controller import (
        JACKAL_GAZEBO_PARTITION,
        GazeboController,
        gazebo_process_env,
    )
    from jackal_rl_navigation.world_pose import GazeboWorldPose

    controller = GazeboController()
    assert controller.world == "jackal_obstacle_test"
    assert controller.partition is None
    assert gazebo_process_env()["IGN_PARTITION"] == JACKAL_GAZEBO_PARTITION == "jackal_rl"
    external = gazebo_process_env("jackal_rl_external")
    assert external["IGN_PARTITION"] == external["GZ_PARTITION"] == "jackal_rl_external"
    aimed = GazeboController(world="industrial-warehouse", partition="jackal_rl_external")
    assert aimed.prefix == "/world/industrial-warehouse"
    assert aimed.partition == "jackal_rl_external"
    assert GazeboWorldPose().topic == "/world/jackal_obstacle_test/dynamic_pose/info"
    pose = GazeboWorldPose(world="industrial-warehouse", partition="jackal_rl_external")
    assert pose.topic == "/world/industrial-warehouse/dynamic_pose/info"
    assert pose.partition == "jackal_rl_external"


def test_retarget_keeps_the_pose_and_changes_the_goal():
    env = JackalWarehouseRLEnv(_offline())
    env.reset(options={"start": (0.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    pose = (env.plant.x, env.plant.y)
    observation = env.retarget((0.0, -1.0))
    assert (env.plant.x, env.plant.y) == pose
    assert env.plant.goal == (0.0, -1.0)
    assert env._episode_open is True
    assert observation.shape == (40,)


def test_gazebo_retarget_clears_the_stop_without_a_new_reset():
    config = _offline()

    class Monitor:
        emergency_active = False

        def reset_episode(self):
            self.cleared = True

    class Session:
        def __init__(self):
            self.env = type("Inner", (), {})()
            self.env.episode_done = True
            self.env.episode_steps = 40
            self.env.previous_goal_distance = 4.0
            self.env.robot = type("Robot", (), {"safety_monitor": Monitor()})()
            self.assigned = None

        def _assign_odom_goal(self, goal):
            self.assigned = tuple(goal)
            self.env.previous_goal_distance = 1.25
            return np.zeros(40, dtype=np.float32)

    session = Session()
    env = JackalWarehouseRLEnv(config, session=session)
    env._episode_open = False
    observation = env.retarget((2.3, -1.2))
    assert session.assigned == (2.3, -1.2)
    assert session.env.episode_done is False
    assert session.env.episode_steps == 0
    session.env.episode_steps = 40
    env.retarget((2.3, -1.2), reset_steps=False)
    assert session.env.episode_steps == 40
    assert session.env.robot.safety_monitor.cleared is True
    assert env._episode_open is True
    assert observation.shape == (40,)
    assert env.plant is None
    session.route_frame = "world"
    session.world_robot_pose = (0.0, 0.0, 0.0)
    session.odom_robot_pose = (1.3, 6.3, 0.0)
    env.retarget((1.2, 6.6))
    assert session.assigned == pytest.approx((2.5, 12.9))
