"""Tests for live A* executor safety defaults."""

import inspect

from navigation.live_astar_executor import LiveAStarExecutor


def test_motion_disabled_by_default():
    signature = inspect.signature(
        LiveAStarExecutor.__init__
    )

    assert (
        signature.parameters["enable_motion"].default
        is False
    )


def test_rejected_command_is_terminal():
    source = inspect.getsource(
        LiveAStarExecutor.step
    )

    assert 'return "COMMAND_REJECTED"' in source


def test_stop_called_on_live_exit():
    source = inspect.getsource(
        LiveAStarExecutor.run
    )

    assert "finally:" in source
    assert "self.node.stop()" in source


def test_live_duration_bounded():
    source = inspect.getsource(
        __import__(
            "navigation.live_astar_executor",
            fromlist=["main"],
        ).main
    )

    assert "args.duration <= 120" in source
