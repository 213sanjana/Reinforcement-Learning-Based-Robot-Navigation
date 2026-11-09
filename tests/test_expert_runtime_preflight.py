
from types import SimpleNamespace

import pytest

from navigation import expert_runtime_preflight as runtime


def make_message(seconds, nanoseconds=0):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=seconds,
                nanosec=nanoseconds,
            )
        )
    )


def test_stamp_conversion():
    message = make_message(12, 345)

    assert runtime.stamp_ns(message) == 12_000_000_345


def test_healthy_stream():
    assert runtime.stream_is_fresh(
        stamps=[1_000_000_000, 1_100_000_000],
        received_times=[10.0, 10.1],
        now=10.2,
    )


def test_missing_messages():
    assert not runtime.stream_is_fresh(
        stamps=[],
        received_times=[],
        now=10.0,
    )


def test_single_message_rejected():
    assert not runtime.stream_is_fresh(
        stamps=[1_000_000_000],
        received_times=[10.0],
        now=10.1,
    )


def test_frozen_timestamps_rejected():
    assert not runtime.stream_is_fresh(
        stamps=[1_000_000_000, 1_000_000_000],
        received_times=[10.0, 10.1],
        now=10.2,
    )


def test_decreasing_timestamps_rejected():
    assert not runtime.stream_is_fresh(
        stamps=[2_000_000_000, 1_000_000_000],
        received_times=[10.0, 10.1],
        now=10.2,
    )


def test_zero_timestamps_rejected():
    assert not runtime.stream_is_fresh(
        stamps=[0, 0],
        received_times=[10.0, 10.1],
        now=10.2,
    )


def test_stale_stream_rejected():
    assert not runtime.stream_is_fresh(
        stamps=[1_000_000_000, 1_100_000_000],
        received_times=[10.0, 10.1],
        now=11.0,
    )


def test_freshness_boundary():
    assert runtime.stream_is_fresh(
        stamps=[1_000_000_000, 1_100_000_000],
        received_times=[10.0, 10.1],
        now=10.6,
        max_age=0.5,
    )


def test_custom_freshness_limit():
    assert not runtime.stream_is_fresh(
        stamps=[1_000_000_000, 1_100_000_000],
        received_times=[10.0, 10.1],
        now=10.4,
        max_age=0.2,
    )


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError):
        runtime.run_runtime_preflight(timeout=0)


def test_negative_timeout_rejected():
    with pytest.raises(ValueError):
        runtime.run_runtime_preflight(timeout=-1)


def test_result_structure():
    result = runtime.RuntimePreflightResult(
        passed=True,
        checks={"lidar_fresh": True},
    )

    assert result.passed is True
    assert result.checks["lidar_fresh"] is True
