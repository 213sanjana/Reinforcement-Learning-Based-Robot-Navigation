import inspect

from navigation.live_astar_executor import LiveAStarExecutor


def test_command_interval_is_configured():
    source = inspect.getsource(LiveAStarExecutor.__init__)
    assert "self.command_interval = 0.1" in source


def test_command_throttling_exists():
    source = inspect.getsource(LiveAStarExecutor.step)
    assert "now - self.last_action_attempt" in source


def test_rejection_remains_terminal():
    source = inspect.getsource(LiveAStarExecutor.step)
    assert 'return "COMMAND_REJECTED"' in source


def test_stop_on_exit():
    source = inspect.getsource(LiveAStarExecutor.run)
    assert "self.node.stop()" in source
