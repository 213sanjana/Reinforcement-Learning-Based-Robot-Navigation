from types import SimpleNamespace

from navigation.live_expert_episode import LiveExpertEpisode


class FakeRobot:
    def __init__(self):
        self.expert_publish_callback = None
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_fresh_episode_state_resets_records_and_counters():
    robot = FakeRobot()
    env = SimpleNamespace(robot=robot)

    episode = LiveExpertEpisode(
        env,
        planner=object(),
        enable_motion=False,
    )

    old_bridge = episode.bridge

    episode.bridge.records.append(object())
    episode.coordinator.accepted = 4
    episode.coordinator.rejected = 2

    episode._fresh_episode_state()

    assert episode.bridge is not old_bridge
    assert episode.bridge.records == []
    assert episode.coordinator.accepted == 0
    assert episode.coordinator.rejected == 0
    assert episode.navigator.guard is None
    assert robot.expert_publish_callback is not None
    assert robot.stop_calls >= 1

    episode.bridge.close()


def test_fresh_episode_state_detaches_previous_callback():
    robot = FakeRobot()
    env = SimpleNamespace(robot=robot)

    episode = LiveExpertEpisode(
        env,
        planner=object(),
        enable_motion=False,
    )

    old_callback = robot.expert_publish_callback

    episode._fresh_episode_state()

    assert robot.expert_publish_callback is not None
    assert robot.expert_publish_callback != old_callback

    episode.bridge.close()

    assert robot.expert_publish_callback is None
