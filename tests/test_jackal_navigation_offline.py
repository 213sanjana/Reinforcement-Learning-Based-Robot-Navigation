"""Offline Jackal navigation contracts. No ROS launch, Gazebo, or training."""

import math

import numpy as np
import pytest

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.goal_frames import world_goal_to_odom
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.safety_monitor import SafetyMonitor

from navigation.jackal_classical_baseline import ClassicalGoalBaseline, LidarLayout
from navigation.jackal_episode_metrics import (
    JackalEpisodeRecord,
    summarize,
    validate_episode,
    result_label,
)
from navigation.jackal_navigation_manager import GoalSpec, NavigationManager
from navigation.jackal_navigation_state import TERMINAL_STATES, NavState
from navigation.jackal_policy import (
    MetadataOnlyPolicy,
    PolicyError,
    ScriptedPolicy,
    bind_device,
    validate_checkpoint_metadata,
)


def _scan(fill=5.0, n=720):
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    ranges = np.full(n, fill, dtype=np.float64)
    return angles, ranges


def _goal(**overrides):
    values = dict(
        x=4.0, y=0.0, yaw=None, tolerance_m=0.30, timeout_s=30.0,
        max_steps=20, scenario_id="world1_open", map_id="jackal_demo_open",
        split="fixed", seed=7,
    )
    values.update(overrides)
    return GoalSpec(**values)


def _record(**overrides):
    values = dict(
        scenario_id="open",
        start_pose=(0.0, 0.0, 0.0),
        goal_pose=(4.0, 0.0, 0.0),
        map_id="world1_open",
        policy_id="ScriptedPolicy",
        algorithm_family="reinforcement_learning",
        seed=1,
        duration_s=12.0,
        steps=40,
        goal_completed=False,
        collision_count=0,
        timeout=True,
        final_goal_distance=2.0,
        path_length=5.0,
        straight_line_distance=4.0,
        path_efficiency=0.8,
        average_linear_velocity=0.1,
        average_angular_velocity=0.0,
        safety_intervention_count=1,
        invalid_observation_count=0,
        invalid_action_count=0,
        split="fixed",
        termination="TIMEOUT",
    )
    values.update(overrides)
    return JackalEpisodeRecord(**values)


def _observation(distance, heading, ranges=None):
    if ranges is None:
        ranges = np.full(360, 8.0)
    return ObservationBuilder().build(ranges, distance, heading, 0.0, 0.0)


def test_observation_contract_orders_sectors_and_rejects_bad_values():
    builder = ObservationBuilder()
    samples = np.concatenate([
        np.full(180, 1.0),
        np.full(180, 10.0),
    ])
    result = builder.build(samples, 4.0, 0.0, 0.0, 0.0)
    assert result.valid
    assert result.vector.shape == (40,)
    assert result.vector[0] == pytest.approx(1.0 / 12.0)
    assert result.vector[35] == pytest.approx(10.0 / 12.0)

    infinite = builder.build(np.full(360, np.inf), 1.0, 0.0, 0.0, 0.0)
    assert infinite.valid
    assert np.all(infinite.vector[:36] == pytest.approx(1.0))

    sparse = np.full(360, 4.0)
    sparse[0] = np.nan
    assert builder.build(sparse, 1.0, 0.0, 0.1, -0.1).valid

    one_bad_sector = np.full(36, 4.0)
    one_bad_sector[3] = np.nan
    assert not builder.build(one_bad_sector, 1.0, 0.0, 0.0, 0.0).valid

    wrapped = builder.build(np.full(36, 4.0), 1.0, 3.0 * math.pi / 2.0, 0.0, 0.0)
    assert wrapped.vector[37] == pytest.approx(-0.5)
    assert not builder.build(np.full(36, 4.0), -0.1, 0.0, 0.0, 0.0).valid
    clipped = builder.build(np.full(36, 4.0), 1.0, 0.0, 1.0, 0.0)
    assert clipped.vector[38] == pytest.approx(1.0)
    assert not builder.build(np.full(36, 4.0), 1.0, 0.0, np.nan, 0.0).valid


def test_matching_frames_preserve_the_world_goal():
    goal = world_goal_to_odom((3.0, -1.0), (1.0, 2.0, 0.4), (1.0, 2.0, 0.4))
    assert goal == pytest.approx((3.0, -1.0))


def test_action_mapper_respects_asymmetric_velocity_limits():
    mapper = ActionMapper()
    for _ in range(20):
        command = mapper.map_action([1.0, 1.0], dt=0.1)
    assert command.linear == pytest.approx(0.30)
    assert command.angular == pytest.approx(0.60)

    mapper.reset()
    for _ in range(20):
        command = mapper.map_action([-1.0, -1.0], dt=0.1)
    assert command.linear == pytest.approx(-0.15)
    assert command.angular == pytest.approx(-0.60)

    with pytest.raises(ValueError):
        mapper.map_action([1.2, 0.0], dt=0.1)
    with pytest.raises(ValueError):
        mapper.map_action([0.0, 0.0], dt=0.04)


def test_safety_keeps_forward_speed_when_only_the_turn_is_blocked():
    monitor = SafetyMonitor()
    angles, ranges = _scan()
    side = np.abs(np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))) < 0.2
    ranges[side] = 0.40

    reduced = monitor.evaluate(0.20, 0.20, ranges, angles, 0.1, 0.1)
    assert reduced.safe
    assert reduced.linear == pytest.approx(0.20)
    assert reduced.angular == 0.0
    assert "Rotation clearance insufficient" in reduced.reason
    assert monitor.intervention_count == 1

    blocked = monitor.evaluate(0.0, 0.20, ranges, angles, 0.1, 0.1)
    assert not blocked.safe
    assert blocked.linear == 0.0
    assert blocked.angular == 0.0


def test_checkpoint_metadata_refuses_weights_and_dimension_drift():
    accepted = validate_checkpoint_metadata({
        "algorithm": "bc_initialized_ppo",
        "observation_dim": 40,
        "action_dim": 2,
    })
    assert accepted["weights_loaded"] is False
    with pytest.raises(PolicyError):
        validate_checkpoint_metadata({
            "algorithm": "ppo",
            "observation_dim": 40,
            "action_dim": 2,
            "weights_path": "artifacts/ppo/jackal_6g1/run_20261022_230534/initialized_ppo.zip",
        })
    with pytest.raises(PolicyError):
        validate_checkpoint_metadata({
            "algorithm": "ppo",
            "observation_dim": 39,
            "action_dim": 2,
        })
    policy = MetadataOnlyPolicy(accepted)
    with pytest.raises(PolicyError):
        policy.predict(np.zeros(40))


def test_device_binding_stays_explicit():
    assert bind_device("cpu") == "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if available:
        assert bind_device("cuda") == "cuda"
    else:
        with pytest.raises(PolicyError):
            bind_device("cuda")


def test_classical_baseline_turns_toward_the_goal_and_stops_at_tolerance():
    baseline = ClassicalGoalBaseline()
    ahead = _observation(1.0, 0.0)
    action = baseline.predict(ahead.vector)
    assert action[0] > 0.0
    assert action[1] == pytest.approx(0.0)

    left = _observation(math.hypot(1.0, 1.0), math.pi / 4)
    turning = baseline.predict(left.vector)
    assert turning[0] == pytest.approx(0.0)
    assert turning[1] > 0.0

    layout = LidarLayout()
    wrapped = np.array([
        math.atan2(math.sin(float(c)), math.cos(float(c))) for c in layout.centers()
    ])
    ranges = np.ones(36)
    ranges[np.abs(wrapped) <= math.radians(25)] = 0.3 / 12.0
    blocked = np.zeros(40, dtype=np.float64)
    blocked[:36] = ranges
    blocked[36] = 2.0 / 20.0
    avoid = baseline.predict(blocked)
    assert avoid[0] == pytest.approx(0.0)
    assert avoid[1] == pytest.approx(1.0)

    arrived = _observation(0.2, 0.4)
    assert baseline.predict(arrived.vector).tolist() == [0.0, 0.0]
    assert baseline.role == "classical_baseline"


def test_classical_baseline_reduces_distance_in_a_kinematic_toy():
    """Unicycle integration. This is not a Gazebo result."""

    baseline = ClassicalGoalBaseline()
    mapper = ActionMapper()
    goal = (1.2, 0.0)
    x = y = yaw = 0.0
    initial = math.hypot(goal[0], goal[1])
    for _ in range(120):
        distance = math.hypot(goal[0] - x, goal[1] - y)
        heading = math.atan2(goal[1] - y, goal[0] - x) - yaw
        observation = _observation(distance, heading)
        action = baseline.predict(observation.vector)
        command = mapper.map_action(action, dt=0.1)
        x += command.linear * math.cos(yaw) * 0.1
        y += command.linear * math.sin(yaw) * 0.1
        yaw += command.angular * 0.1
    final = math.hypot(goal[0] - x, goal[1] - y)
    assert final < initial
    assert final <= 0.30


def test_navigation_state_machine_stops_after_every_terminal_state():
    policy = ScriptedPolicy([1.0, 0.0])
    manager = NavigationManager(policy, start_pose=(0.0, 0.0, 0.0))
    assert manager.step(
        None, *_scan(), 0.1, 0.1, 1.0, False, 0.1, (0.0, 0.0), 0.0
    ).commanding is False

    manager.set_goal(_goal(max_steps=1))
    assert manager.state is NavState.INITIALIZING
    angles, ranges = _scan()
    observation = _observation(2.0, 0.0, np.full(360, 8.0))
    first = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.1, 0.0), 0.1
    )
    assert first.state is NavState.NAVIGATING
    assert first.commanding is True
    assert first.source == "rl_policy"
    assert first.linear > 0.0

    timeout = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.2, 0.0), 0.2
    )
    assert timeout.state is NavState.TIMEOUT
    assert timeout.commanding is False
    assert timeout.linear == 0.0
    held = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.2, 0.0), 0.3
    )
    assert held.state is NavState.TIMEOUT
    assert held.commanding is False
    record = manager.finish(1.5)
    assert record.timeout and not record.goal_completed
    assert record.termination == "TIMEOUT"

    manager.reset()
    manager.set_goal(_goal())
    reached = manager.step(
        observation, ranges, angles, 0.1, 0.1, 0.2, False, 0.1, (3.8, 0.0), 0.2
    )
    assert reached.state is NavState.GOAL_REACHED
    assert manager.step(
        observation, ranges, angles, 0.1, 0.1, 0.2, False, 0.1, (3.8, 0.0), 0.3
    ).commanding is False

    manager.reset()
    manager.set_goal(_goal())
    collision = manager.step(
        observation, ranges, angles, 0.1, 0.1, 0.1, True, 0.1, (3.9, 0.0), 0.2
    )
    assert collision.state is NavState.COLLISION
    assert not manager.record.goal_completed

    manager.reset()
    manager.set_goal(_goal())
    stale = manager.step(
        observation, ranges, angles, 1.0, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.1
    )
    assert stale.state is NavState.SENSOR_ERROR

    manager.reset()
    manager.set_goal(_goal())
    invalid = _observation(2.0, 0.0, np.full(36, np.nan))
    assert manager.step(
        invalid, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.1
    ).state is NavState.SENSOR_ERROR

    class Boom:
        role = "rl_policy"
        algorithm_family = "reinforcement_learning"

        def reset(self):
            return None

        def predict(self, observation, deterministic=True):
            raise PolicyError("policy failed")

    manager = NavigationManager(Boom(), start_pose=(0.0, 0.0, 0.0))
    manager.set_goal(_goal())
    failed = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.1
    )
    assert failed.state is NavState.POLICY_ERROR
    assert manager.record.invalid_action_count == 1

    manager = NavigationManager(policy, start_pose=(0.0, 0.0, 0.0))
    manager.set_goal(_goal())
    stopped = manager.emergency_stop()
    assert stopped.state is NavState.EMERGENCY_STOP
    assert stopped.commanding is False
    with pytest.raises(Exception):
        manager.set_goal(_goal())
    assert NavState.GOAL_REACHED in TERMINAL_STATES


def test_safety_intervention_is_not_the_policy_source():
    policy = ScriptedPolicy([1.0, 1.0])
    manager = NavigationManager(policy, start_pose=(0.0, 0.0, 0.0))
    manager.set_goal(_goal(max_steps=5))
    angles, ranges = _scan()
    side = np.abs(np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))) < 0.2
    ranges[side] = 0.40
    observation = _observation(2.0, 0.0)
    decision = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.1
    )
    assert decision.state is NavState.NAVIGATING
    assert decision.source == "safety_intervention"
    assert decision.commanding is True
    assert decision.angular == 0.0
    assert decision.linear > 0.0
    assert manager.mapper.previous.angular == pytest.approx(0.0)


def test_timeouts_stay_in_the_success_denominator():
    rows = [_record(scenario_id=f"fixed_{index}", seed=index) for index in range(4)]
    summary = summarize(rows, split="fixed")
    assert summary["episodes"] == 4
    assert summary["successes"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["collision_rate"] == 0.0
    assert summary["timeouts"] == 4
    assert "not success" in summary["note"]

    success = _record(
        goal_completed=True,
        timeout=False,
        collision_count=0,
        termination="GOAL_REACHED",
        final_goal_distance=0.2,
        path_length=4.5,
        straight_line_distance=4.0,
        path_efficiency=4.0 / 4.5,
        split="unseen",
        scenario_id="unseen_1",
    )
    mixed = summarize(rows + [success])
    assert mixed["success_rate"] == pytest.approx(1 / 5)
    assert mixed["mean_time_to_goal_s"] == pytest.approx(12.0)
    unseen = summarize([success], split="unseen")
    assert unseen["success_rate"] == 1.0
    with pytest.raises(Exception):
        validate_episode(_record(goal_completed=True, timeout=True, termination="TIMEOUT"))


def test_classical_success_is_not_labeled_as_reinforcement_learning():
    record = _record(
        policy_id="ClassicalGoalBaseline",
        algorithm_family="classical",
        goal_completed=True,
        timeout=False,
        termination="GOAL_REACHED",
        final_goal_distance=0.2,
        path_length=4.2,
        straight_line_distance=4.0,
        path_efficiency=4.0 / 4.2,
    )
    assert "not a reinforcement-learning result" in result_label(record)
