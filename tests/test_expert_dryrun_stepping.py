
import inspect

from navigation.live_expert_episode import LiveExpertEpisode


def test_dryrun_advances_paused_simulation():
    source = inspect.getsource(LiveExpertEpisode.run)

    assert "self.env.controller.step(20)" in source
    assert "SIMULATION_STEP_FAILED" in source


def test_dryrun_does_not_resume_continuous_physics():
    source = inspect.getsource(LiveExpertEpisode.run)

    assert "if self.enable_motion:" in source
    assert "self.env.controller.resume()" in source


def test_prepare_required_before_run():
    source = inspect.getsource(LiveExpertEpisode.run)

    assert 'if self.navigator.guard is None:' in source
    assert 'Call prepare() before run()' in source
