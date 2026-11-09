import pytest
from types import SimpleNamespace

from navigation.live_expert_episode import LiveExpertEpisode


def make_environment(enabled, monitor):
    return SimpleNamespace(
        robot=object(),
        collision_config=SimpleNamespace(enabled=enabled),
        collision_monitor=monitor,
    )


def test_motion_rejects_disabled_collision_monitor():
    env = make_environment(False, None)

    with pytest.raises(RuntimeError, match="collision monitoring"):
        LiveExpertEpisode(env, planner=object(), enable_motion=True)


def test_motion_rejects_missing_collision_monitor():
    env = make_environment(True, None)

    with pytest.raises(RuntimeError, match="collision monitoring"):
        LiveExpertEpisode(env, planner=object(), enable_motion=True)


def test_motion_rejects_missing_collision_config():
    env = SimpleNamespace(robot=object())

    with pytest.raises(RuntimeError, match="collision monitoring"):
        LiveExpertEpisode(env, planner=object(), enable_motion=True)
