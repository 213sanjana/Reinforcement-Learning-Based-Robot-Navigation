"""Unit tests for the differential-drive waypoint controller."""

import math

import pytest

from planners.waypoint_controller import WaypointController


def test_straight_movement():
    controller = WaypointController([(0, 0), (2, 0)])
    command = controller.compute(0, 0, 0)

    assert command.linear > 0
    assert abs(command.angular) < 1e-9


def test_rotate_toward_goal():
    controller = WaypointController([(0, 0), (0, 2)])
    command = controller.compute(0, 0, 0)

    assert command.linear == 0
    assert command.angular > 0


def test_negative_rotation():
    controller = WaypointController([(0, 0), (0, -2)])
    command = controller.compute(0, 0, 0)

    assert command.linear == 0
    assert command.angular < 0


def test_velocity_limits():
    controller = WaypointController([(0, 0), (10, 10)])

    for yaw in [-math.pi, -1, 0, 1, math.pi]:
        command = controller.compute(0, 0, yaw)

        assert 0 <= command.linear <= 0.30
        assert abs(command.angular) <= 0.60


def test_waypoint_transition():
    controller = WaypointController([
        (0, 0),
        (1, 0),
        (2, 0),
    ])

    command = controller.compute(1, 0, 0)

    assert command.waypoint_index == 2


def test_goal_completion():
    controller = WaypointController([(0, 0), (2, 0)])

    command = controller.compute(2, 0, 0)

    assert command.finished
    assert command.linear == 0
    assert command.angular == 0


def test_angle_wrapping():
    result = WaypointController.normalize_angle(
        3 * math.pi
    )

    assert abs(abs(result) - math.pi) < 1e-9


def test_reset():
    controller = WaypointController([(0, 0), (1, 0)])

    controller.compute(1, 0, 0)
    assert controller.finished

    controller.reset()
    assert not controller.finished


def test_invalid_waypoints():
    with pytest.raises(ValueError):
        WaypointController([])

    with pytest.raises(ValueError):
        WaypointController([(float("nan"), 0)])


def test_invalid_pose():
    controller = WaypointController([(0, 0), (1, 0)])

    with pytest.raises(ValueError):
        controller.compute(float("nan"), 0, 0)
