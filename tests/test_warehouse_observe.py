"""Offline tests for the stationary observation recorder. No ROS graph is attached."""

import ast
from pathlib import Path

import numpy as np
import pytest

from evaluation.warehouse_observe import OBSERVE_ENV, OBSERVE_VALUE, observe
from jackal_rl_navigation.warehouse_estop import AUTHORIZATION_ENV, AUTHORIZATION_VALUE
from jackal_rl_navigation.warehouse_observation import StationaryObservationSession
from tests.test_warehouse_stationary_observation import RecordingProbe, _config


ROOT = Path(__file__).resolve().parents[1]


def _authorized(monkeypatch):
    monkeypatch.setenv(OBSERVE_ENV, OBSERVE_VALUE)


def test_motion_authorization_does_not_attach(monkeypatch, tmp_path):
    monkeypatch.setenv(AUTHORIZATION_ENV, AUTHORIZATION_VALUE)
    monkeypatch.delenv(OBSERVE_ENV, raising=False)

    def connect(_config):
        raise AssertionError("ROS attach")

    output = tmp_path / "observation"
    with pytest.raises(SystemExit, match="Motion authorization"):
        observe(_config(), output, authorize=True, connect_session=connect)
    assert not output.exists()


def test_injected_observation_is_not_labeled_live(monkeypatch, tmp_path):
    _authorized(monkeypatch)
    probe = RecordingProbe()
    session = StationaryObservationSession(_config(), transport=probe)
    output = tmp_path / "observation"
    record = observe(_config(), output, authorize=True, session=session)
    assert record["status"] == "observation_collected"
    assert record["evidence_class"] == "injected_transport"
    assert record["observation_shape"] == [40]
    assert record["observation_dtype"] == "float32"
    assert record["observation_finite"] is True
    assert len(record["observation"]) == 40
    assert all(np.isfinite(record["observation"]))
    assert record["gazebo_launched_by_this_process"] is False
    assert record["robot_commanded_by_this_process"] is False
    assert record["training"] is False
    assert record["checkpoint_loaded"] is False
    assert record["navigation_result"] is False
    assert record["real_world"] is False
    assert record["scenario_id"] == "ext_train_east"
    saved = (output / "observation.json").read_text(encoding="utf-8")
    assert "observation_collected" in saved
    probe_calls = probe.gazebo_calls
    assert probe_calls == []
    assert probe.publications == []


def test_failure_is_recorded_without_a_launch(monkeypatch, tmp_path):
    _authorized(monkeypatch)

    def connect(_config):
        raise RuntimeError("partition refused")

    output = tmp_path / "failed"
    record = observe(_config(), output, authorize=True, connect_session=connect)
    assert record["status"] == "failed"
    assert "partition refused" in record["error"]
    assert record["gazebo_launched_by_this_process"] is False
    assert record["navigation_result"] is False
    assert record["observation"] is None


def test_historical_and_training_trees_are_refused(monkeypatch, tmp_path):
    _authorized(monkeypatch)
    session = StationaryObservationSession(_config(), transport=RecordingProbe())
    with pytest.raises(RuntimeError, match="protected"):
        observe(
            _config(),
            ROOT / "artifacts" / "project_audit" / "new_observation",
            authorize=True,
            session=session,
        )
    with pytest.raises(RuntimeError, match="observations"):
        observe(
            _config(),
            ROOT / "artifacts" / "warehouse_ppo" / "not_an_observation",
            authorize=True,
            session=session,
        )
    existing = tmp_path / "already"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        observe(_config(), existing, authorize=True, session=session)


def test_recorder_source_has_no_launch_or_command_path():
    source = (ROOT / "evaluation" / "warehouse_observe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "create_publisher",
        "set_pose",
        "pause",
        "stop",
        "PPO",
        "learn",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            assert name not in forbidden
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {
                "jackal_rl_navigation.gazebo_controller",
                "jackal_rl_navigation.gazebo_env",
                "jackal_rl_navigation.robot_interface",
                "stable_baselines3",
            }
    assert "ros2" not in source
    assert "subprocess" not in source
