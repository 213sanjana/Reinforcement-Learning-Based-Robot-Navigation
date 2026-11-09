from types import SimpleNamespace

import numpy as np
import pytest

from navigation.expert_dataset_export import export_expert_episode


def make_result(status="GOAL_REACHED", accepted=2, rejected=0):
    transitions = []

    for index in range(2):
        transitions.append(
            SimpleNamespace(
                observation=np.zeros(40, dtype=np.float32),
                action=np.array([0.5, 0.1], dtype=np.float32),
                applied_action=np.array([0.3, 0.1], dtype=np.float32),
                published_linear=0.1,
                published_angular=0.05,
                simulation_time_ns=1000 + index,
                pose=(float(index), 0.0, 0.0),
            )
        )

    return SimpleNamespace(
        status=status,
        planned_path=[(0.0, 0.0), (1.0, 0.0)],
        executed_poses=[(0.0, 0.0, 0.0)],
        transitions=transitions,
        accepted_commands=accepted,
        rejected_commands=rejected,
    )


def test_successful_export(tmp_path):
    npz_path, json_path = export_expert_episode(
        make_result(), tmp_path, 0
    )

    assert npz_path.exists()
    assert json_path.exists()

    with np.load(npz_path) as data:
        assert data["observations"].shape == (2, 40)
        assert data["applied_actions"].shape == (2, 2)
        np.testing.assert_allclose(
            data["applied_actions"][0], [0.3, 0.1]
        )


def test_no_overwrite(tmp_path):
    result = make_result()

    export_expert_episode(result, tmp_path, 1)

    with pytest.raises(FileExistsError):
        export_expert_episode(result, tmp_path, 1)


def test_reject_dry_run(tmp_path):
    with pytest.raises(ValueError):
        export_expert_episode(
            make_result(status="DRY_RUN"), tmp_path, 2
        )


def test_reject_command_failure(tmp_path):
    with pytest.raises(ValueError):
        export_expert_episode(
            make_result(accepted=1, rejected=1), tmp_path, 3
        )


def test_reject_invalid_observations(tmp_path):
    result = make_result()
    result.transitions[0].observation = np.zeros(39)

    with pytest.raises(ValueError):
        export_expert_episode(result, tmp_path, 4)
