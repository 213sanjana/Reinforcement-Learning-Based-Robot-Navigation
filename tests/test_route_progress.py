"""Tests for A* route progress monitoring."""

from navigation.route_progress import RouteProgressWatchdog


PATH = [
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
]


def test_initialization():
    watchdog = RouteProgressWatchdog(PATH)
    watchdog.reset((0.0, 0.0), now=0.0)
    assert watchdog.best_progress == 0.0


def test_forward_progress():
    watchdog = RouteProgressWatchdog(
        PATH,
        stall_timeout=5.0,
    )

    watchdog.reset((0.0, 0.0), now=0.0)

    assert watchdog.evaluate((0.2, 0.0), now=3.0)[0]
    assert watchdog.evaluate((0.4, 0.0), now=6.0)[0]


def test_stationary_stall():
    watchdog = RouteProgressWatchdog(
        PATH,
        stall_timeout=5.0,
    )

    watchdog.reset((0.0, 0.0), now=0.0)

    safe, reason, _ = watchdog.evaluate(
        (0.0, 0.0),
        now=6.0,
    )

    assert not safe
    assert reason == "No forward progress along A* route"


def test_sideways_motion_does_not_count():
    watchdog = RouteProgressWatchdog(
        PATH,
        stall_timeout=5.0,
    )

    watchdog.reset((0.0, 0.0), now=0.0)

    safe, _, progress = watchdog.evaluate(
        (0.0, 0.2),
        now=6.0,
    )

    assert not safe
    assert progress == 0.0


def test_backward_motion_does_not_count():
    watchdog = RouteProgressWatchdog(
        PATH,
        stall_timeout=5.0,
    )

    watchdog.reset((0.5, 0.0), now=0.0)

    safe, _, _ = watchdog.evaluate(
        (0.2, 0.0),
        now=6.0,
    )

    assert not safe


def test_invalid_path():
    try:
        RouteProgressWatchdog([(0.0, 0.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_progress_projection():
    watchdog = RouteProgressWatchdog(PATH)

    distance, progress = watchdog.project(1.5, 0.1)

    assert abs(distance - 0.1) < 1e-6
    assert abs(progress - 1.5) < 1e-6
