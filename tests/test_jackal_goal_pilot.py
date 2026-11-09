"""Mocked classical pilot. No ROS node, Gazebo, training, or checkpoint load."""

import math
from pathlib import Path

import numpy as np
import pytest

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.safety_monitor import SafetyMonitor

from navigation.jackal_command_contract import (
    assert_contract_matches_defaults,
    goal_loop_enabled,
)
from navigation.jackal_episode_metrics import result_label, summarize
from navigation.jackal_goal_pilot import GoalPilot, PilotError, refuse_checkpoint_controller
from navigation.jackal_navigation_manager import GoalSpec, NavigationManager
from navigation.jackal_navigation_state import NavState
from navigation.jackal_policy import PolicyError, ScriptedPolicy


def _scan(fill=8.0):
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    ranges = np.full(720, fill, dtype=np.float64)
    return angles, ranges


def _paint(ranges, angles, center, half, distance):
    delta = np.arctan2(np.sin(angles - center), np.cos(angles - center))
    ranges[np.abs(delta) <= half] = distance
    return ranges


def _observation(ranges, distance, heading):
    return ObservationBuilder().build(ranges, distance, heading, 0.0, 0.0)


def _pilot(goal=(4.0, 0.0), max_steps=20, timeout_s=30.0, split="fixed"):
    mapper = ActionMapper()
    safety = SafetyMonitor()
    published = []

    def publish(linear, angular):
        published.append((float(linear), float(angular)))
        return True

    pilot = GoalPilot.classical(
        mapper,
        safety,
        publish,
        start_pose=(0.0, 0.0, 0.0),
    )
    pilot.manager.set_goal(GoalSpec(
        x=goal[0], y=goal[1], yaw=None, tolerance_m=0.30, timeout_s=timeout_s,
        max_steps=max_steps, scenario_id="world1_open", map_id="jackal_demo_open",
        split=split, seed=1,
    ))
    return pilot, published, mapper


def _cycle(pilot, ranges, angles, distance, heading, elapsed, collision=False):
    observation = _observation(ranges, distance, heading)
    return pilot.cycle(
        observation, ranges, angles, 0.1, 0.1, distance, collision,
        0.1, (0.0, 0.0), elapsed,
    )


def test_command_contract_locks_units_and_the_goal_loop_flag():
    contract = assert_contract_matches_defaults()
    assert contract["normalized_action_range"] == (-1.0, 1.0)
    assert contract["physical_units"] == "m/s, rad/s"
    assert contract["max_forward_m_s"] == 0.30
    assert contract["max_reverse_m_s"] == 0.15
    assert contract["max_angular_rad_s"] == 0.60
    assert goal_loop_enabled("false") is False
    assert goal_loop_enabled(False) is False
    assert goal_loop_enabled("true") is True


def test_checkpoint_names_are_refused_and_the_pilot_source_does_not_train():
    assert refuse_checkpoint_controller("classical_baseline") == "classical_baseline"
    with pytest.raises(PolicyError):
        refuse_checkpoint_controller(
            "artifacts/ppo/jackal_6g1/run_20261022_230534/initialized_ppo.zip"
        )
    with pytest.raises(PolicyError):
        refuse_checkpoint_controller("ppo")
    source = Path("navigation/jackal_goal_pilot.py").read_text(encoding="utf-8")
    assert "model.learn" not in source
    assert "ign gazebo" not in source


def test_open_straight_route_maps_once_and_publishes_physical_speed():
    pilot, published, mapper = _pilot()
    angles, ranges = _scan()
    decision, _ = _cycle(pilot, ranges, angles, 2.0, 0.0, 0.1)
    assert decision.commanding is True
    assert decision.source == "classical_baseline"
    assert decision.normalized_action[0] == pytest.approx(1.0)
    assert decision.normalized_action[1] == pytest.approx(0.0)
    assert published == [(pytest.approx(0.03), pytest.approx(0.0))]
    assert decision.linear == pytest.approx(0.03)
    assert pilot.last_map_delta == 1
    assert pilot.manager.map_invocations == 1
    fresh = ActionMapper()
    once = fresh.map_action([1.0, 0.0], 0.1)
    twice = fresh.map_action([once.linear, once.angular], 0.1)
    assert published[0][0] == pytest.approx(once.linear)
    assert published[0][0] != pytest.approx(twice.linear)
    assert pilot.manager.policy.algorithm_family == "classical"


def test_goal_behind_the_robot_turns_in_place():
    pilot, published, _mapper = _pilot()
    angles, ranges = _scan()
    decision, _ = _cycle(pilot, ranges, angles, 2.0, math.pi, 0.1)
    assert decision.normalized_action[0] == pytest.approx(0.0)
    assert decision.normalized_action[1] == pytest.approx(1.0)
    assert published[0][0] == pytest.approx(0.0)
    assert published[0][1] == pytest.approx(0.06)
    assert pilot.last_map_delta == 1
    assert pilot.manager.policy.algorithm_family == "classical"


def test_obstacle_ahead_requests_a_turn_inside_the_speed_limit():
    pilot, published, _mapper = _pilot()
    angles, ranges = _scan()
    _paint(ranges, angles, 0.0, math.radians(25), 0.70)
    decision, _ = _cycle(pilot, ranges, angles, 2.0, 0.0, 0.1)
    assert decision.normalized_action[0] == pytest.approx(0.0)
    assert decision.normalized_action[1] == pytest.approx(1.0)
    assert published[0][0] == pytest.approx(0.0)
    assert published[0][1] == pytest.approx(0.06)
    assert abs(published[0][1]) <= 0.60


def test_side_obstacle_turns_away_and_keeps_forward_speed():
    pilot, published, mapper = _pilot()
    angles, ranges = _scan()
    _paint(ranges, angles, math.pi / 2, 0.2, 0.40)
    decision, _ = _cycle(pilot, ranges, angles, 2.0, 0.2, 0.1)
    assert decision.normalized_action[0] > 0.0
    assert decision.normalized_action[1] < 0.0
    assert decision.source == "classical_baseline"
    assert published[0][0] > 0.0
    assert published[0][1] < 0.0
    assert abs(published[0][1]) <= 0.60
    assert mapper.previous.angular == pytest.approx(published[0][1])
    assert pilot.last_map_delta == 1
    assert pilot.manager.policy.algorithm_family == "classical"


def test_narrow_passage_keeps_forward_speed_and_drops_the_heading_correction():
    pilot, published, mapper = _pilot()
    angles, ranges = _scan()
    _paint(ranges, angles, math.pi / 2, 0.3, 0.55)
    _paint(ranges, angles, -math.pi / 2, 0.3, 0.55)
    decision, _ = _cycle(pilot, ranges, angles, 2.0, 0.2, 0.1)
    assert decision.commanding is True
    assert published[0][0] > 0.0
    assert published[0][1] == pytest.approx(0.0)
    assert mapper.previous.linear == pytest.approx(published[0][0])
    assert mapper.previous.angular == pytest.approx(0.0)
    assert pilot.publish_count == 1


def test_goal_reached_publishes_zero_without_mapping_and_is_labeled_classical():
    pilot, published, _mapper = _pilot(goal=(0.2, 0.0))
    angles, ranges = _scan()
    decision, _ = _cycle(pilot, ranges, angles, 0.2, 0.0, 0.4)
    assert decision.state is NavState.GOAL_REACHED
    assert decision.commanding is False
    assert pilot.last_map_delta == 0
    assert published == [(0.0, 0.0)]
    record = pilot.record
    assert record.goal_completed is True
    assert record.timeout is False
    assert record.algorithm_family == "classical"
    assert record.policy_id == "ClassicalGoalBaseline"
    assert "not a reinforcement-learning result" in result_label(record)
    held, _ = _cycle(pilot, ranges, angles, 0.2, 0.0, 0.5)
    assert held.commanding is False
    assert held.linear == 0.0 and held.angular == 0.0
    assert pilot.manager.map_invocations == 0


def test_invalid_sensor_and_timeout_and_bad_action_stay_classical_failures():
    pilot, published, _mapper = _pilot()
    angles, ranges = _scan()
    invalid = ObservationBuilder().build(np.full(36, np.nan), 2.0, 0.0, 0.0, 0.0)
    decision, _ = pilot.cycle(
        invalid, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.2,
    )
    assert decision.state is NavState.SENSOR_ERROR
    assert published[-1] == (0.0, 0.0)
    assert pilot.record.algorithm_family == "classical"
    assert pilot.record.goal_completed is False

    timed, timed_messages, _mapper = _pilot(max_steps=1)
    first, _ = _cycle(timed, ranges, angles, 2.0, 0.0, 0.1)
    assert first.commanding is True
    second, _ = _cycle(timed, ranges, angles, 2.0, 0.0, 0.3)
    assert second.state is NavState.TIMEOUT
    assert timed_messages[-1] == (0.0, 0.0)
    assert timed.manager.map_invocations == 1
    assert timed.record.timeout is True
    assert timed.record.goal_completed is False
    assert timed.record.algorithm_family == "classical"
    summary = summarize([timed.record], split="fixed")
    assert summary["success_rate"] == 0.0
    assert summary["timeouts"] == 1

    class OutOfRange:
        role = "classical_baseline"
        algorithm_family = "classical"

        def reset(self):
            return None

        def predict(self, observation, deterministic=True):
            return [2.0, 0.0]

    mapper = ActionMapper()
    safety = SafetyMonitor()
    messages = []
    manager = NavigationManager(
        OutOfRange(), mapper=mapper, safety=safety, start_pose=(0.0, 0.0, 0.0),
    )
    pilot = GoalPilot(manager, lambda linear, angular: messages.append((linear, angular)) or True)
    manager.set_goal(GoalSpec(
        x=4.0, y=0.0, yaw=None, tolerance_m=0.30, timeout_s=30.0, max_steps=5,
        scenario_id="bad", map_id="jackal_demo_open", split="fixed", seed=3,
    ))
    failed, _ = _cycle(pilot, ranges, angles, 2.0, 0.0, 0.2)
    assert failed.state is NavState.POLICY_ERROR
    assert messages == [(0.0, 0.0)]
    assert manager.map_invocations == 0
    assert pilot.record.invalid_action_count == 1
    assert pilot.record.algorithm_family == "classical"


def test_emergency_stop_and_a_second_publisher_are_rejected():
    pilot, published, _mapper = _pilot()
    angles, ranges = _scan()
    _cycle(pilot, ranges, angles, 2.0, 0.0, 0.1)
    stopped = pilot.manager.emergency_stop()
    assert stopped.state is NavState.EMERGENCY_STOP
    assert len(published) == 1
    held, _ = _cycle(pilot, ranges, angles, 2.0, 0.0, 0.4)
    assert held.commanding is False
    assert published[-1] == (0.0, 0.0)
    assert pilot.manager.map_invocations == 1
    rl = NavigationManager(
        ScriptedPolicy([0.0, 0.0]),
        mapper=ActionMapper(),
        safety=SafetyMonitor(),
    )
    with pytest.raises(PilotError):
        GoalPilot(rl, lambda linear, angular: True)


def test_one_cycle_uses_one_publisher_callback():
    pilot, published, _mapper = _pilot()
    angles, ranges = _scan()
    _cycle(pilot, ranges, angles, 2.0, 0.0, 0.1)
    assert pilot.publish_count == 1
    assert len(published) == 1
    assert pilot._publish_velocity is not None
