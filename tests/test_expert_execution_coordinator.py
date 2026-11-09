
from types import SimpleNamespace

from navigation.expert_execution_coordinator import (
    ExpertExecutionCoordinator,
)


class FakeRobot:
    def __init__(self):
        self.action_mapper = SimpleNamespace(
            max_forward=0.3,
            max_reverse=0.15,
            max_angular=0.6,
        )
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1


class FakeNavigator:
    def __init__(self, status="RUNNING", linear=0.15, angular=0.3):
        self.status = status
        self.linear = linear
        self.angular = angular
        self.terminated = None

    def evaluate(self, **kwargs):
        return SimpleNamespace(
            status=self.status,
            linear=self.linear,
            angular=self.angular,
        )

    def terminate(self, reason):
        self.terminated = reason


class FakeBridge:
    def __init__(self, accept=True):
        self.accept = accept
        self.actions = []

    def execute(self, action, pose):
        self.actions.append((action, pose))
        return self.accept


def test_successful_command():
    robot = FakeRobot()
    navigator = FakeNavigator()
    bridge = FakeBridge()

    coordinator = ExpertExecutionCoordinator(
        robot, navigator, bridge
    )

    assert coordinator.step((0, 0, 0), (0, 0)) == "RUNNING"
    assert coordinator.accepted == 1
    assert coordinator.rejected == 0
    assert len(bridge.actions) == 1
    assert robot.stop_count == 0


def test_goal_reached_stops_robot():
    robot = FakeRobot()
    navigator = FakeNavigator(status="GOAL_REACHED")
    bridge = FakeBridge()

    coordinator = ExpertExecutionCoordinator(
        robot, navigator, bridge
    )

    assert coordinator.step((1, 0, 0), (0, 0)) == "GOAL_REACHED"
    assert robot.stop_count == 1
    assert bridge.actions == []


def test_collision_stops_robot():
    robot = FakeRobot()
    navigator = FakeNavigator(status="SAFETY_STOP")
    bridge = FakeBridge()

    coordinator = ExpertExecutionCoordinator(
        robot, navigator, bridge
    )

    assert coordinator.step(
        (1, 0, 0), (0, 0), collision_detected=True
    ) == "SAFETY_STOP"

    assert robot.stop_count == 1
    assert bridge.actions == []


def test_rejected_command_stops_robot():
    robot = FakeRobot()
    navigator = FakeNavigator()
    bridge = FakeBridge(accept=False)

    coordinator = ExpertExecutionCoordinator(
        robot, navigator, bridge
    )

    assert coordinator.step((0, 0, 0), (0, 0)) == "COMMAND_REJECTED"
    assert coordinator.rejected == 1
    assert robot.stop_count == 1
    assert navigator.terminated == "COMMAND_REJECTED"


def test_invalid_command_stops_robot():
    robot = FakeRobot()
    navigator = FakeNavigator(linear=float("nan"))
    bridge = FakeBridge()

    coordinator = ExpertExecutionCoordinator(
        robot, navigator, bridge
    )

    assert coordinator.step((0, 0, 0), (0, 0)) == "INVALID_COMMAND"
    assert robot.stop_count == 1
    assert bridge.actions == []
