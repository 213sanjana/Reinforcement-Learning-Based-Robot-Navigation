
from types import SimpleNamespace
import numpy as np

from navigation.expert_recording_bridge import ExpertRecordingBridge


class FakeRobot:
    def __init__(self):
        self.expert_publish_callback = None
        self.action_mapper = SimpleNamespace(
            max_forward=0.3,
            max_reverse=0.15,
            max_angular=0.6,
        )
        self.accept = True
        self.observation = np.zeros(40, dtype=np.float32)

    def get_observation(self):
        return self.observation

    def command_action(self, action):
        if not self.accept:
            return False

        if self.expert_publish_callback:
            self.expert_publish_callback(0.05, 0.02, 123456)

        return True


def test_records_published_command():
    robot = FakeRobot()
    bridge = ExpertRecordingBridge(robot)

    assert bridge.execute([0.5, 0.1], (1.0, 2.0, 0.3))
    assert len(bridge.records) == 1

    record = bridge.records[0]
    assert record.observation.shape == (40,)
    assert record.published_linear == 0.05
    assert record.published_angular == 0.02
    assert record.simulation_time_ns == 123456
    assert record.pose == (1.0, 2.0, 0.3)

    bridge.close()
    assert robot.expert_publish_callback is None


def test_rejected_action_not_recorded():
    robot = FakeRobot()
    robot.accept = False
    bridge = ExpertRecordingBridge(robot)

    assert not bridge.execute([0.5, 0.1], (0, 0, 0))
    assert len(bridge.records) == 0


def test_invalid_observation_not_recorded():
    robot = FakeRobot()
    robot.observation = np.zeros(39, dtype=np.float32)
    bridge = ExpertRecordingBridge(robot)

    assert not bridge.execute([0, 0], (0, 0, 0))
    assert len(bridge.records) == 0


def test_invalid_action_not_recorded():
    robot = FakeRobot()
    bridge = ExpertRecordingBridge(robot)

    assert not bridge.execute([2, 0], (0, 0, 0))
    assert len(bridge.records) == 0


def test_stop_without_pending_action_not_recorded():
    robot = FakeRobot()
    bridge = ExpertRecordingBridge(robot)

    robot.expert_publish_callback(0.0, 0.0, 123456)

    assert len(bridge.records) == 0
