
import inspect

from jackal_rl_navigation.gazebo_controller import GazeboController
from navigation.live_expert_episode import LiveExpertEpisode


def test_resume_uses_existing_world_control():
    controller = GazeboController.__new__(GazeboController)
    controller.prefix = "/world/test_world"

    calls = []

    controller._request = lambda *args: (
        calls.append(args) or True
    )

    assert controller.resume() is True

    assert len(calls) == 1
    assert calls[0][0] == "/world/test_world/control"
    assert calls[0][3] == "pause: false"


def test_resume_failure_aborts_live_run():
    source = inspect.getsource(LiveExpertEpisode.run)

    assert "self.env.controller.resume()" in source
    assert "Gazebo failed to resume" in source
    assert "self.robot.stop()" in source
