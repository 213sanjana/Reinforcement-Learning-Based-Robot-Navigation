
from types import SimpleNamespace

import pytest

from navigation.expert_navigation import ExpertNavigator


class FakePlanner:
    def __init__(self, path=None, valid=True):
        self.path = path
        self.valid = valid

    def plan(self, start, goal):
        return self.path

    def validate_path(self, path):
        return self.valid


def test_invalid_controller():
    with pytest.raises(ValueError):
        ExpertNavigator(FakePlanner(), controller_type="unknown")


def test_invalid_coordinates():
    navigator = ExpertNavigator(FakePlanner())
    with pytest.raises(ValueError):
        navigator.plan((float("nan"), 0), (1, 1))


def test_unreachable_goal():
    navigator = ExpertNavigator(FakePlanner(path=None))
    assert navigator.plan((0, 0), (2, 2)) is None
    assert navigator.terminal_status == "PLANNING_FAILED"


def test_invalid_path():
    navigator = ExpertNavigator(
        FakePlanner(path=[(0, 0), (1, 1)], valid=False)
    )
    assert navigator.plan((0, 0), (1, 1)) is None
    assert navigator.terminal_status == "INVALID_PATH"


def test_valid_plan():
    path = [(0, 0), (0.5, 0), (1, 0)]
    navigator = ExpertNavigator(FakePlanner(path=path))
    assert navigator.plan((0, 0), (1, 0)) == path
    assert navigator.guard is not None


def test_not_initialized():
    navigator = ExpertNavigator(FakePlanner())
    decision = navigator.evaluate((0, 0, 0), (0, 0))
    assert decision.status == "NOT_INITIALIZED"


def test_termination():
    navigator = ExpertNavigator(FakePlanner())
    navigator.terminate("INTERRUPTED")
    decision = navigator.evaluate((0, 0, 0), (0, 0))
    assert decision.status == "INTERRUPTED"
    assert decision.linear == 0.0
    assert decision.angular == 0.0
