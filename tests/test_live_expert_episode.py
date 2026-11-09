
from types import SimpleNamespace
import math

import pytest

from navigation.live_expert_episode import LiveExpertEpisode


def test_pose_from_odom():
    odom = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=0.0, w=1.0
                ),
            )
        )
    )

    pose = LiveExpertEpisode.pose_from_odom(odom)

    assert pose == (1.0, 2.0, 0.0)


def test_invalid_interval():
    env = SimpleNamespace(robot=SimpleNamespace())

    with pytest.raises(ValueError):
        LiveExpertEpisode(
            env=env,
            planner=None,
            command_interval=0.01,
        )


def test_result_structure():
    from navigation.live_expert_episode import ExpertEpisodeResult

    result = ExpertEpisodeResult(
        status="GOAL_REACHED",
        planned_path=[(0, 0), (1, 0)],
        executed_poses=[(0, 0, 0)],
        transitions=[],
        accepted_commands=10,
        rejected_commands=0,
    )

    assert result.status == "GOAL_REACHED"
    assert result.accepted_commands == 10
