
import numpy as np
import pytest

from navigation.expert_command_recorder import ExpertCommandRecorder


def test_valid_record():
    recorder = ExpertCommandRecorder()
    record = recorder.record(
        np.zeros(40),
        [0.5, -0.25],
        0.1,
        -0.05,
    )
    assert record.observation.shape == (40,)
    assert record.action == (0.5, -0.25)
    assert record.linear == 0.1
    assert len(recorder.records) == 1


def test_invalid_observation():
    recorder = ExpertCommandRecorder()
    with pytest.raises(ValueError):
        recorder.record(np.zeros(39), [0, 0], 0, 0)


def test_invalid_action():
    recorder = ExpertCommandRecorder()
    with pytest.raises(ValueError):
        recorder.record(np.zeros(40), [2, 0], 0, 0)


def test_invalid_velocity():
    recorder = ExpertCommandRecorder()
    with pytest.raises(ValueError):
        recorder.record(np.zeros(40), [0, 0], float("nan"), 0)


def test_observation_is_copied():
    recorder = ExpertCommandRecorder()
    observation = np.zeros(40, dtype=np.float32)
    record = recorder.record(observation, [0, 0], 0, 0)
    observation[0] = 99
    assert record.observation[0] == 0
