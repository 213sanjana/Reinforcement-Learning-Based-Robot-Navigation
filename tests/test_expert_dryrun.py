
import inspect

from navigation.live_expert_episode import LiveExpertEpisode


def test_motion_disabled_by_default():
    signature = inspect.signature(LiveExpertEpisode.__init__)
    assert signature.parameters["enable_motion"].default is False


def test_dryrun_does_not_evaluate_watchdog():
    source = inspect.getsource(LiveExpertEpisode.run)

    dryrun = source.split(
        "if not self.enable_motion:", 1
    )[1].split("now = time.monotonic()", 1)[0]

    assert 'status = "DRY_RUN"' in dryrun
    assert "self.navigator.evaluate(" not in dryrun
    assert "self.coordinator.step(" not in dryrun


def test_live_execution_still_uses_coordinator():
    source = inspect.getsource(LiveExpertEpisode.run)

    assert "self.coordinator.step(" in source
    assert "self.robot.stop()" in source
