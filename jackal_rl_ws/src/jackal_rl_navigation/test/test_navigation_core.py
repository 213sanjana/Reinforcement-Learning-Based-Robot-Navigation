"""Functional tests for Jackal RL navigation components."""

import math

import numpy as np
import pytest

from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.safety_monitor import SafetyMonitor


def test_observation_shape_and_finiteness():
    builder = ObservationBuilder()

    result = builder.build(
        lidar_ranges=np.full(360, 5.0),
        goal_distance=4.0,
        goal_heading=0.3,
        linear_velocity=0.1,
        angular_velocity=0.2,
    )

    assert result.valid
    assert result.vector.shape == (40,)
    assert result.vector.dtype == np.float32
    assert np.all(np.isfinite(result.vector))


def test_invalid_observation():
    builder = ObservationBuilder()

    result = builder.build(
        lidar_ranges=np.full(360, np.nan),
        goal_distance=4.0,
        goal_heading=0.0,
        linear_velocity=0.0,
        angular_velocity=0.0,
    )

    assert not result.valid


def test_action_acceleration_limits():
    mapper = ActionMapper()

    command = mapper.map_action([1.0, 1.0], dt=0.1)

    assert command.linear == pytest.approx(0.03)
    assert command.angular == pytest.approx(0.06)


def test_action_rejects_invalid_values():
    mapper = ActionMapper()

    with pytest.raises(ValueError):
        mapper.map_action([np.nan, 0.0], dt=0.1)


def test_action_emergency_stop():
    mapper = ActionMapper()

    mapper.map_action([1.0, 1.0], dt=0.1)
    command = mapper.emergency_stop()

    assert command.linear == 0.0
    assert command.angular == 0.0


def make_scan():
    angles = np.linspace(-math.pi, math.pi, 720)
    ranges = np.full(720, 5.0)
    return angles, ranges


def evaluate(monitor, angles, ranges, linear=0.2, angular=0.0,
             scan_age=0.1, odom_age=0.1):
    return monitor.evaluate(
        linear=linear,
        angular=angular,
        lidar_ranges=ranges,
        lidar_angles=angles,
        scan_age=scan_age,
        odom_age=odom_age,
    )


def test_safe_forward():
    monitor = SafetyMonitor()
    angles, ranges = make_scan()

    result = evaluate(monitor, angles, ranges)

    assert result.safe


def test_forward_obstacle_stop():
    monitor = SafetyMonitor()
    angles, ranges = make_scan()

    ranges[np.abs(angles) < 0.1] = 0.3

    result = evaluate(monitor, angles, ranges)

    assert not result.safe
    assert result.linear == 0.0


def test_stale_sensor_stop():
    monitor = SafetyMonitor()
    angles, ranges = make_scan()

    result = evaluate(
        monitor, angles, ranges, scan_age=1.0
    )

    assert not result.safe


def test_emergency_stop_latch():
    monitor = SafetyMonitor()
    angles, ranges = make_scan()

    monitor.emergency_stop()

    result = evaluate(monitor, angles, ranges)

    assert not result.safe

    monitor.reset_emergency()

    result = evaluate(monitor, angles, ranges)

    assert result.safe


def test_episode_termination():
    monitor = SafetyMonitor()
    angles, ranges = make_scan()

    monitor.terminate_episode()

    assert not evaluate(monitor, angles, ranges).safe

    monitor.reset_episode()

    assert evaluate(monitor, angles, ranges).safe


def test_reverse_blocked_without_rear_lidar():
    monitor = SafetyMonitor()

    angles = np.linspace(
        -math.pi / 2,
        math.pi / 2,
        360,
    )

    ranges = np.full(360, 5.0)

    result = evaluate(
        monitor,
        angles,
        ranges,
        linear=-0.1,
    )

    assert not result.safe
    assert result.linear == 0.0
    assert result.reason == "Rear LiDAR unavailable"


def test_invalid_forward_lidar():
    monitor = SafetyMonitor()
    angles, ranges = make_scan()

    ranges[np.abs(angles) < 0.1] = np.nan

    result = evaluate(monitor, angles, ranges)

    assert not result.safe
