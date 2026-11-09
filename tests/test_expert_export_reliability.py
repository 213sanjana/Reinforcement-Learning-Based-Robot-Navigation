
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from navigation.expert_dataset_export import export_expert_episode


def make_result():
    transition = SimpleNamespace(
        observation=np.zeros(40, dtype=np.float32),
        action=np.array([0.5, 0.0], dtype=np.float32),
        applied_action=np.array([0.4, 0.0], dtype=np.float32),
        published_linear=0.1,
        published_angular=0.0,
        simulation_time_ns=1000,
        pose=(0.0, 0.0, 0.0),
    )

    return SimpleNamespace(
        status="GOAL_REACHED",
        transitions=[transition],
        accepted_commands=1,
        rejected_commands=0,
        planned_path=[(0.0, 0.0), (1.0, 0.0)],
        executed_poses=[(0.0, 0.0, 0.0)],
    )


def test_successful_export_creates_both_files(tmp_path):
    npz_path, json_path = export_expert_episode(
        make_result(), tmp_path, 0
    )

    assert npz_path.exists()
    assert json_path.exists()


def test_existing_outputs_are_never_overwritten(tmp_path):
    result = make_result()

    paths = export_expert_episode(result, tmp_path, 1)
    original_npz = paths[0].read_bytes()
    original_json = paths[1].read_bytes()

    with pytest.raises(FileExistsError):
        export_expert_episode(result, tmp_path, 1)

    assert paths[0].read_bytes() == original_npz
    assert paths[1].read_bytes() == original_json


def test_json_write_failure_removes_partial_pair(tmp_path):
    result = make_result()

    with patch(
        "navigation.expert_dataset_export.json.dump",
        side_effect=OSError("Simulated JSON write failure"),
    ):
        with pytest.raises(OSError):
            export_expert_episode(result, tmp_path, 2)

    assert not (tmp_path / "expert_episode_0002.npz").exists()
    assert not (tmp_path / "expert_episode_0002.json").exists()


def test_invalid_episode_creates_no_files(tmp_path):
    result = make_result()
    result.status = "SAFETY_STOP"

    with pytest.raises(ValueError):
        export_expert_episode(result, tmp_path, 3)

    assert list(tmp_path.iterdir()) == []
