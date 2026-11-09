
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from navigation.live_expert_episode import LiveExpertEpisode


class FakeRobot:
    def __init__(self):
        self.expert_publish_callback = None
        self.stop_calls = 0
        self.odom = object()

    def stop(self):
        self.stop_calls += 1


class FakeEnvironment:
    def __init__(self):
        self.robot = FakeRobot()
        self.reset_calls = 0
        self.fail_next_reset = False

    def reset(self, *, options=None):
        self.reset_calls += 1

        if self.fail_next_reset:
            raise RuntimeError("Simulated reset failure")

        return (
            np.zeros(40, dtype=np.float32),
            {"goal_odom": (3.0, 0.0)},
        )


class FakePlanner:
    def plan(self, start, goal):
        return [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]

    def validate_path(self, path):
        return True


def make_episode():
    env = FakeEnvironment()
    episode = LiveExpertEpisode(
        env=env,
        planner=FakePlanner(),
        enable_motion=False,
    )
    return episode, env


def test_repeated_prepare_creates_fresh_state():
    episode, env = make_episode()

    with patch.object(
        LiveExpertEpisode,
        "pose_from_odom",
        return_value=(0.0, 0.0, 0.0),
    ):
        try:
            episode.prepare()

            first_navigator = episode.navigator
            first_bridge = episode.bridge

            episode.coordinator.accepted = 5
            episode.bridge.records.append("previous_episode")

            episode.prepare()

            assert env.reset_calls == 2
            assert episode.navigator is not first_navigator
            assert episode.bridge is not first_bridge
            assert episode.coordinator.accepted == 0
            assert episode.coordinator.rejected == 0
            assert episode.bridge.records == []
            assert episode._episode_prepared is True
            assert episode.executed_poses == [(0.0, 0.0, 0.0)]

        finally:
            episode.bridge.close()


def test_prepare_failure_stops_and_detaches_recorder():
    episode, env = make_episode()
    env.fail_next_reset = True

    with pytest.raises(RuntimeError, match="Simulated reset failure"):
        episode.prepare()

    assert env.robot.stop_calls >= 1
    assert env.robot.expert_publish_callback is None


def test_prepare_failure_does_not_mark_episode_ready():
    episode, env = make_episode()
    env.fail_next_reset = True

    with pytest.raises(RuntimeError, match="Simulated reset failure"):
        episode.prepare()

    assert episode._episode_prepared is False


def test_repeated_prepare_does_not_create_velocity_publisher():
    episode, env = make_episode()

    with patch.object(
        LiveExpertEpisode,
        "pose_from_odom",
        return_value=(0.0, 0.0, 0.0),
    ):
        try:
            episode.prepare()
            episode.prepare()

            assert not hasattr(env.robot, "create_publisher")
            assert env.robot.expert_publish_callback is not None

        finally:
            episode.bridge.close()
