
from types import SimpleNamespace

import pytest

from navigation.live_expert_episode import LiveExpertEpisode


class FakeRobot:
    def __init__(self):
        self.expert_publish_callback = None
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def make_episode():
    robot = FakeRobot()
    env = SimpleNamespace(robot=robot)

    episode = LiveExpertEpisode(
        env=env,
        planner=object(),
        enable_motion=False,
    )

    return episode, robot


def test_repeated_fresh_state_is_independent():
    episode, robot = make_episode()

    try:
        episode._fresh_episode_state()

        first_bridge = episode.bridge
        first_navigator = episode.navigator
        first_coordinator = episode.coordinator

        first_bridge.records.append("first_episode")
        first_coordinator.accepted = 7
        first_coordinator.rejected = 1

        episode._fresh_episode_state()

        assert episode.bridge is not first_bridge
        assert episode.navigator is not first_navigator
        assert episode.coordinator is not first_coordinator

        assert episode.bridge.records == []
        assert episode.coordinator.accepted == 0
        assert episode.coordinator.rejected == 0

        assert first_bridge.records == ["first_episode"]
        assert robot.expert_publish_callback is not None

    finally:
        episode.bridge.close()


def test_repeated_initialization_detaches_old_callback():
    episode, robot = make_episode()

    try:
        previous_callback = robot.expert_publish_callback

        for _ in range(3):
            episode._fresh_episode_state()

            current_callback = robot.expert_publish_callback

            assert current_callback is not None
            assert current_callback != previous_callback

            previous_callback = current_callback

    finally:
        episode.bridge.close()

    assert robot.expert_publish_callback is None


def test_interrupted_state_initialization_stops_robot(monkeypatch):
    episode, robot = make_episode()

    try:
        def fail_bridge_close():
            raise RuntimeError("Simulated cleanup failure")

        monkeypatch.setattr(
            episode.bridge,
            "close",
            fail_bridge_close,
        )

        with pytest.raises(
            RuntimeError,
            match="Simulated cleanup failure",
        ):
            episode._fresh_episode_state()

        assert robot.stop_calls >= 1

    finally:
        robot.expert_publish_callback = None


def test_fresh_state_resets_prepared_flag():
    episode, robot = make_episode()

    try:
        episode._episode_prepared = True
        episode._fresh_episode_state()

        assert episode._episode_prepared is False
        assert episode.last_command_time is None
        assert episode.executed_poses == []

    finally:
        episode.bridge.close()
