"""Tests for guarded A* navigation decisions."""

from types import SimpleNamespace

import pytest

from navigation.guarded_navigation import (
    GuardedNavigation,
    velocity_to_action,
)


PATH = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


class FakeController:
    def compute(self, x, y, yaw):
        return SimpleNamespace(
            linear=0.2,
            angular=0.3,
            finished=(x >= 1.95),
        )


def make_guard(**kwargs):
    return GuardedNavigation(
        PATH,
        FakeController(),
        **kwargs,
    )


def test_normal_navigation():
    guard = make_guard()

    result = guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=0.0,
    )

    assert result.status == "RUNNING"
    assert result.linear == 0.2


def test_collision_stop():
    guard = make_guard()

    result = guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        True,
        now=0.0,
    )

    assert result.status == "SAFETY_STOP"


def test_stale_sensor_stop():
    guard = make_guard()

    result = guard.evaluate(
        (0.0, 0.0, 0.0),
        (1.0, 0.01),
        False,
        now=0.0,
    )

    assert result.status == "SAFETY_STOP"


def test_path_deviation_stop():
    guard = make_guard()

    result = guard.evaluate(
        (0.0, 0.6, 0.0),
        (0.01, 0.01),
        False,
        now=0.0,
    )

    assert result.status == "SAFETY_STOP"


def test_route_stall():
    guard = make_guard(stall_timeout=5.0)

    guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=0.0,
    )

    result = guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=6.0,
    )

    assert result.status == "STALLED"


def test_navigation_timeout():
    guard = make_guard(navigation_timeout=5.0)

    guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=0.0,
    )

    result = guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=6.0,
    )

    assert result.status == "TIMEOUT"


def test_goal_reached():
    guard = make_guard()

    result = guard.evaluate(
        (2.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=0.0,
    )

    assert result.status == "GOAL_REACHED"


def test_terminal_state_is_latched():
    guard = make_guard()

    guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        True,
        now=0.0,
    )

    result = guard.evaluate(
        (0.0, 0.0, 0.0),
        (0.01, 0.01),
        False,
        now=1.0,
    )

    assert result.status == "SAFETY_STOP"


def test_velocity_conversion():
    mapper = SimpleNamespace(
        max_forward=0.30,
        max_reverse=0.15,
        max_angular=0.60,
    )

    linear, angular = velocity_to_action(
        0.15,
        0.30,
        mapper,
    )

    assert linear == pytest.approx(0.5)
    assert angular == pytest.approx(0.5)


def test_velocity_clamping():
    mapper = SimpleNamespace(
        max_forward=0.30,
        max_reverse=0.15,
        max_angular=0.60,
    )

    assert velocity_to_action(1.0, 2.0, mapper) == (1.0, 1.0)


def test_invalid_velocity():
    mapper = SimpleNamespace(
        max_forward=0.30,
        max_reverse=0.15,
        max_angular=0.60,
    )

    with pytest.raises(ValueError):
        velocity_to_action(float("nan"), 0.0, mapper)
