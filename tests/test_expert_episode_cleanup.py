
from types import SimpleNamespace

import pytest

from navigation.live_expert_episode import LiveExpertEpisode


def make_runner():
    events = []

    robot = SimpleNamespace(
        odom=None,
        stop=lambda: events.append("stop"),
    )

    env = SimpleNamespace(
        robot=robot,
        reset=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("Reset failed")
        ),
        close=lambda: events.append("env_close"),
    )

    runner = LiveExpertEpisode.__new__(LiveExpertEpisode)
    runner.env = env
    runner.robot = robot
    runner.bridge = SimpleNamespace(
        close=lambda: events.append("bridge_close")
    )

    return runner, events


def test_prepare_failure_stops_and_detaches():
    runner, events = make_runner()

    with pytest.raises(RuntimeError, match="Reset failed"):
        runner.prepare()

    assert events == ["stop", "bridge_close"]


def test_close_stops_and_releases_resources():
    runner, events = make_runner()

    runner.close()

    assert events == [
        "stop",
        "bridge_close",
        "env_close",
    ]


def test_close_releases_resources_even_if_stop_fails():
    runner, events = make_runner()

    def failing_stop():
        events.append("stop")
        raise RuntimeError("Stop failed")

    runner.robot.stop = failing_stop

    with pytest.raises(RuntimeError, match="Stop failed"):
        runner.close()

    assert events == [
        "stop",
        "bridge_close",
        "env_close",
    ]
