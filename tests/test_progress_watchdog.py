"""Unit tests for navigation progress monitoring."""

import pytest

from navigation.progress_watchdog import NavigationProgressWatchdog


def test_initialization():
    watchdog = NavigationProgressWatchdog()
    assert watchdog.evaluate((0.0, 0.0), now=0.0)[0]


def test_normal_progress():
    watchdog = NavigationProgressWatchdog(
        stall_timeout=5.0,
        min_progress=0.1,
    )

    watchdog.reset((0.0, 0.0), now=0.0)

    assert watchdog.evaluate((0.15, 0.0), now=3.0)[0]
    assert watchdog.evaluate((0.30, 0.0), now=6.0)[0]


def test_stall_detection():
    watchdog = NavigationProgressWatchdog(
        stall_timeout=5.0,
    )

    watchdog.reset((0.0, 0.0), now=0.0)

    safe, reason = watchdog.evaluate(
        (0.0, 0.0),
        now=6.0,
    )

    assert not safe
    assert reason == "Navigation stalled"


def test_navigation_timeout():
    watchdog = NavigationProgressWatchdog(
        timeout=10.0,
        stall_timeout=20.0,
    )

    watchdog.reset((0.0, 0.0), now=0.0)

    safe, reason = watchdog.evaluate(
        (0.0, 0.0),
        now=11.0,
    )

    assert not safe
    assert reason == "Navigation timeout"


def test_invalid_position():
    watchdog = NavigationProgressWatchdog()

    watchdog.reset((0.0, 0.0), now=0.0)

    safe, reason = watchdog.evaluate(
        (float("nan"), 0.0),
        now=1.0,
    )

    assert not safe
    assert reason == "Invalid robot position"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        NavigationProgressWatchdog(stall_timeout=0.0)


def test_reset():
    watchdog = NavigationProgressWatchdog(
        stall_timeout=5.0,
    )

    watchdog.reset((0.0, 0.0), now=0.0)
    watchdog.reset((1.0, 1.0), now=10.0)

    assert watchdog.evaluate((1.0, 1.0), now=12.0)[0]
