"""Offline classical avoidance cases. These are not Gazebo results."""

import math

import numpy as np
import pytest

from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.safety_monitor import SafetyMonitor
from navigation.jackal_classical_baseline import (
    FORWARD_STOP_M,
    SIDE_CLEARANCE_M,
    ClassicalGoalBaseline,
    LidarLayout,
)
from navigation.jackal_navigation_manager import GoalSpec, NavigationManager
from navigation.jackal_navigation_state import NavState
from navigation.jackal_policy import ScriptedPolicy


def _ranges(fill=8.0):
    return np.full(360, fill, dtype=np.float64)


def _paint(ranges, center, half, distance):
    angles = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
    delta = np.arctan2(np.sin(angles - center), np.cos(angles - center))
    ranges[np.abs(delta) <= half] = distance
    return ranges


def _observe(distance, heading, ranges):
    return ObservationBuilder().build(ranges, distance, heading, 0.0, 0.0)


def _goal():
    return GoalSpec(
        x=4.0, y=0.0, yaw=None, tolerance_m=0.30, timeout_s=30.0,
        max_steps=20, scenario_id="avoidance", map_id="offline",
        split="fixed", seed=1,
    )


def test_clear_path_drives_toward_the_goal():
    action = ClassicalGoalBaseline().predict(_observe(4.0, 0.0, _ranges()).vector)
    assert action[0] > 0.0
    assert action[1] == pytest.approx(0.0)
    assert FORWARD_STOP_M == pytest.approx(0.80)


def test_obstacle_directly_ahead_turns_toward_the_clearer_side():
    ranges = _paint(_ranges(), 0.0, math.radians(20), 0.40)
    action = ClassicalGoalBaseline().predict(_observe(3.0, 0.0, ranges).vector)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(1.0)


def test_obstacle_on_the_left_turns_right():
    ranges = _paint(_ranges(), math.radians(70), math.radians(15), 0.55)
    action = ClassicalGoalBaseline().predict(_observe(3.0, 0.0, ranges).vector)
    assert action[0] > 0.0
    assert action[1] < 0.0
    assert 0.55 < SIDE_CLEARANCE_M


def test_obstacle_on_the_right_turns_left():
    ranges = _paint(_ranges(), math.radians(-70), math.radians(15), 0.55)
    action = ClassicalGoalBaseline().predict(_observe(3.0, 0.0, ranges).vector)
    assert action[0] > 0.0
    assert action[1] > 0.0


def test_obstacles_on_both_sides_choose_the_clearer_side_once():
    ranges = _paint(_ranges(), math.radians(70), math.radians(15), 0.50)
    ranges = _paint(ranges, math.radians(-70), math.radians(15), 0.62)
    baseline = ClassicalGoalBaseline()
    first = baseline.predict(_observe(3.0, 0.0, ranges).vector)
    swapped = _paint(_ranges(), math.radians(70), math.radians(15), 0.61)
    swapped = _paint(swapped, math.radians(-70), math.radians(15), 0.60)
    second = baseline.predict(_observe(3.0, 0.0, swapped).vector)
    assert first[1] < 0.0
    assert second[1] == pytest.approx(first[1])


def test_goal_reached_commands_zero():
    action = ClassicalGoalBaseline().predict(_observe(0.20, 0.4, _ranges()).vector)
    assert action.tolist() == [0.0, 0.0]


def test_goal_behind_the_robot_turns_before_driving():
    action = ClassicalGoalBaseline().predict(_observe(2.0, math.pi, _ranges()).vector)
    assert action[0] == pytest.approx(0.0)
    assert action[1] > 0.0


def test_narrow_passage_keeps_the_committed_direction():
    tight = _paint(_ranges(), math.radians(80), math.radians(20), 0.50)
    tight = _paint(tight, math.radians(-80), math.radians(20), 0.58)
    baseline = ClassicalGoalBaseline()
    first = baseline.predict(_observe(2.0, 0.0, tight).vector)
    flipped = _paint(_ranges(), math.radians(80), math.radians(20), 0.57)
    flipped = _paint(flipped, math.radians(-80), math.radians(20), 0.56)
    second = baseline.predict(_observe(2.0, 0.0, flipped).vector)
    assert first[1] != 0.0
    assert second[1] == pytest.approx(first[1])


def test_invalid_lidar_and_stale_sensors_stop_in_the_manager():
    policy = ScriptedPolicy([1.0, 0.0])
    manager = NavigationManager(policy, start_pose=(0.0, 0.0, 0.0))
    manager.set_goal(_goal())
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = np.full(360, 5.0)
    invalid = ObservationBuilder().build(np.full(36, np.nan), 2.0, 0.0, 0.0, 0.0)
    assert invalid.valid is False
    decision = manager.step(
        invalid, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.1
    )
    assert decision.state is NavState.SENSOR_ERROR
    assert decision.linear == 0.0

    manager.reset()
    manager.set_goal(_goal())
    fresh = ObservationBuilder().build(np.full(360, 5.0), 2.0, 0.0, 0.0, 0.0)
    stale = manager.step(
        fresh, ranges, angles, 1.0, 0.1, 2.0, False, 0.1, (0.0, 0.0), 0.1
    )
    assert stale.state is NavState.SENSOR_ERROR


def test_turning_recovers_toward_the_goal_after_the_side_is_clear():
    baseline = ClassicalGoalBaseline()
    blocked = _paint(_ranges(), math.radians(70), math.radians(15), 0.50)
    avoiding = baseline.predict(_observe(3.0, 0.0, blocked).vector)
    assert avoiding[1] < 0.0
    recovered = baseline.predict(_observe(3.0, 0.2, _ranges()).vector)
    assert recovered[0] > 0.0
    assert recovered[1] == pytest.approx(0.2 / 0.60)


def test_turn_away_from_a_side_obstacle_is_allowed_by_safety():
    monitor = SafetyMonitor()
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = np.full(360, 5.0)
    ranges[np.abs(np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))) < 0.2] = 0.40
    away = monitor.evaluate(0.20, -0.40, ranges, angles, 0.1, 0.1)
    assert away.linear > 0.0
    assert away.angular < 0.0
    into = monitor.evaluate(0.20, 0.40, ranges, angles, 0.1, 0.1)
    assert into.angular == pytest.approx(0.0)
    assert "Rotation clearance" in into.reason


def test_layout_still_uses_thirty_six_sectors():
    assert len(LidarLayout().centers()) == 36


def test_collision_flag_stops_the_manager_and_holds_zero():
    policy = ScriptedPolicy([1.0, 0.0])
    manager = NavigationManager(policy, start_pose=(0.0, 0.0, 0.0))
    manager.set_goal(_goal())
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = np.full(360, 5.0)
    observation = ObservationBuilder().build(ranges, 2.0, 0.0, 0.0, 0.0)
    collision = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, True, 0.1, (1.0, 0.0), 0.2
    )
    assert collision.state is NavState.COLLISION
    assert collision.linear == 0.0 and collision.angular == 0.0
    held = manager.step(
        observation, ranges, angles, 0.1, 0.1, 2.0, False, 0.1, (1.0, 0.0), 0.4
    )
    assert held.state is NavState.COLLISION
    assert held.commanding is False
    assert manager.record.collision_count == 1
    assert manager.record.goal_completed is False
