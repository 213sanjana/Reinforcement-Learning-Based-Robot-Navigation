"""Offline checks for the recorded industrial-route report.

These tests read historical JSON. They do not launch Gazebo, load a policy
zip, or write into artifacts/warehouse_ppo.
"""

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.navigation_report import (
    NOT_RECORDED,
    RECORDED,
    TASK_FAMILIES,
    NavigationReportError,
    build_industrial_route_report,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
SPAWN = ROOT / "artifacts" / "warehouse_ppo" / "repeatability" / "spawn_route"
CHECKPOINT = ROOT / "artifacts" / "warehouse_ppo" / "runs" / "run_20261103T054820Z_bf036311"
METADATA = CHECKPOINT / "metadata.json"
MODEL = CHECKPOINT / "ppo_model.zip"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_digests() -> dict[str, str]:
    paths = sorted(SPAWN.glob("*.json"))
    paths.append(METADATA)
    return {str(path.relative_to(ROOT)): _digest(path) for path in paths}


def _copy_trials(tmp_path: Path) -> Path:
    dest = tmp_path / "spawn_route"
    dest.mkdir()
    for path in SPAWN.glob("*.json"):
        (dest / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def _family(report: dict) -> dict:
    return report["task_families"]["industrial_route_14"]


def test_spawn_route_report_matches_historical_summary():
    before = _historical_digests()
    model_mtime = MODEL.stat().st_mtime_ns
    report = build_industrial_route_report(SPAWN, repo_root=ROOT)
    family = _family(report)
    summary = json.loads((SPAWN / "summary.json").read_text(encoding="utf-8"))
    metrics = family["metrics"]

    assert report["live_execution"] is False
    assert report["active_family"] == "industrial_route_14"
    assert family["status"] == RECORDED
    assert family["validation"]["passed"] is True
    assert metrics["episodes"] == 5
    assert metrics["successes"] == 1
    assert metrics["success_rate"] == summary["success_rate"] == 0.2
    assert metrics["collisions"] == 0
    assert metrics["collision_rate"] == 0.0
    assert metrics["timeouts"] == 0
    assert metrics["timeout_rate"] == 0.0
    assert metrics["mean_path_length_m"] == summary["mean_path_length_m"]
    assert metrics["mean_duration_s"] == summary["mean_duration_s"]
    assert metrics["mean_min_clearance_m"] == summary["mean_min_clearance_m"]
    assert metrics["mean_terminal_goal_distance_m"] == summary["mean_terminal_goal_distance_m"]
    assert metrics["checkpoints_expected"] == 14
    assert metrics["checkpoints_reached_per_trial"] == [14, 13, 14, 14, 14]
    assert metrics["trials_with_all_checkpoint_names"] == 4
    assert metrics["safety_rejected_episodes"] == 1

    trials = family["trials"]
    assert [trial["seed"] for trial in trials] == [1, 2, 3, 4, 5]
    assert trials[0]["success"] is True
    assert trials[0]["checkpoints_reached_count"] == 14
    assert trials[1]["success"] is False
    assert trials[1]["termination_reason"] == "safety_rejected"
    assert trials[1]["checkpoints_reached_count"] == 13
    assert [trial["trial"] for trial in trials if trial["success"] is False and trial["checkpoints_reached_count"] == 14] == [3, 4, 5]
    assert sum(trial["success"] for trial in trials) / len(trials) == metrics["success_rate"]

    identity = family["identity"]
    assert identity["world"]["value"] == "industrial-warehouse"
    assert identity["controller"]["value"] == "ppo"
    assert identity["controller"]["note"] == "Checkpoint policy class is MlpPolicy."
    assert identity["checkpoint"]["value"] == "artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311"
    assert identity["seed"]["value"] == [1, 2, 3, 4, 5]
    assert identity["seed"]["checkpoint_training_seed"]["value"] == 42
    assert identity["route"]["id"] == "spawn_route"
    assert identity["route"]["spawn"] == [0.0, 0.0, 0.0]
    assert len(identity["route"]["checkpoints"]) == 14
    assert identity["checkpoint_record"]["weights_loaded"] is False
    assert identity["checkpoint_record"]["training_execution"] == "offline"
    assert identity["checkpoint_record"]["git_head"] == "176cbcbef8de60aef4df58ddc39f8d14d1f1311d"
    assert _historical_digests() == before
    assert MODEL.stat().st_mtime_ns == model_mtime


def test_unavailable_fields_stay_null_and_families_stay_separate():
    report = build_industrial_route_report(SPAWN, repo_root=ROOT)
    family = _family(report)
    metrics = family["metrics"]
    identity = family["identity"]

    for key in (
        "mean_time_to_goal_s",
        "mean_linear_speed_mps",
        "mean_angular_speed_radps",
        "mean_rotations",
        "mean_return",
        "safety_rejection_count",
        "collision_event_count",
        "path_efficiency",
    ):
        assert metrics[key] is None
        assert family["unavailable"][key]
    assert identity["configuration"]["value"] is None
    assert identity["goal_tolerance_m"]["value"] is None
    assert identity["execution_mode"]["value"] is None
    assert "evaluation execution mode" in identity["execution_mode"]["reason"]

    assert list(report["task_families"]) == list(TASK_FAMILIES)
    for name in TASK_FAMILIES:
        if name == "industrial_route_14":
            continue
        entry = report["task_families"][name]
        assert entry["status"] == NOT_RECORDED
        assert entry["metrics"] is None
        assert entry["reason"]
    assert report["comparison"]["methods"] == {
        "ppo": RECORDED,
        "nav2": NOT_RECORDED,
        "astar": NOT_RECORDED,
    }
    assert report["comparison"]["ranking"] is None


def test_changed_trial_fails_validation(tmp_path):
    copied = _copy_trials(tmp_path)
    trial = json.loads((copied / "trial_01.json").read_text(encoding="utf-8"))
    trial["success"] = False
    (copied / "trial_01.json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(NavigationReportError, match="successes"):
        build_industrial_route_report(copied, repo_root=ROOT)


def test_reordered_checkpoints_fail_validation(tmp_path):
    copied = _copy_trials(tmp_path)
    trial = json.loads((copied / "trial_02.json").read_text(encoding="utf-8"))
    trial["checkpoints_reached"] = list(reversed(trial["checkpoints_reached"]))
    (copied / "trial_02.json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(NavigationReportError, match="ordered prefix"):
        build_industrial_route_report(copied, repo_root=ROOT)


def test_missing_trial_file_fails_validation(tmp_path):
    copied = _copy_trials(tmp_path)
    (copied / "trial_05.json").unlink()
    with pytest.raises(NavigationReportError, match="does not match"):
        build_industrial_route_report(copied, repo_root=ROOT)


def test_write_report_uses_a_new_directory_and_refuses_history(tmp_path):
    before = _historical_digests()
    report = build_industrial_route_report(SPAWN, repo_root=ROOT)
    destination = tmp_path / "industrial_route_14_report"
    path = write_report(report, destination, source_dir=SPAWN, repo_root=ROOT)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["task_families"]["industrial_route_14"]["metrics"]["successes"] == 1
    with pytest.raises(FileExistsError):
        write_report(report, destination, source_dir=SPAWN, repo_root=ROOT)

    blocked = ROOT / "artifacts" / "warehouse_ppo" / "repeatability" / "offline_report_forbidden"
    assert not blocked.exists()
    with pytest.raises(NavigationReportError, match="Refusing to write"):
        write_report(report, blocked, source_dir=SPAWN, repo_root=ROOT)
    assert not blocked.exists()
    with pytest.raises(NavigationReportError, match="recorded trial directory"):
        write_report(report, SPAWN / "nested_report", source_dir=SPAWN, repo_root=ROOT)
    assert _historical_digests() == before
    assert not (SPAWN / "nested_report").exists()
