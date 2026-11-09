
import numpy as np

from navigation.expert_recording_bridge import ExpertRecordingBridge


class FakeMapper:
    max_forward = 0.3
    max_reverse = 0.15
    max_angular = 0.6


class FakeRobot:
    def __init__(self):
        self.action_mapper = FakeMapper()
        self.expert_publish_callback = None
        self.accept = True
        self.publish_stop_on_reject = False

    def get_observation(self):
        return np.zeros(40, dtype=np.float32)

    def command_action(self, action):
        if not self.accept:
            if self.publish_stop_on_reject:
                self.expert_publish_callback(0.0, 0.0, 123)
            return False

        self.expert_publish_callback(0.06, -0.12, 123)
        return True


def test_applied_action_matches_published_velocity():
    robot = FakeRobot()
    bridge = ExpertRecordingBridge(robot)

    assert bridge.execute([1.0, 1.0], (0, 0, 0))

    record = bridge.records[0]

    np.testing.assert_allclose(
        record.applied_action,
        [0.2, -0.2],
        atol=1e-6,
    )

    np.testing.assert_allclose(
        record.action,
        [1.0, 1.0],
        atol=1e-6,
    )


def test_rejected_stop_is_not_training_action():
    robot = FakeRobot()
    robot.accept = False
    robot.publish_stop_on_reject = True

    bridge = ExpertRecordingBridge(robot)

    assert bridge.execute([0.5, 0.0], (0, 0, 0)) is False
    assert bridge.records == []


def test_recorded_observation_is_independent_copy():
    robot = FakeRobot()
    bridge = ExpertRecordingBridge(robot)

    assert bridge.execute([0.5, 0], (0, 0, 0))

    record = bridge.records[0]
    assert record.observation.shape == (40,)
    assert record.observation.dtype == np.float32
