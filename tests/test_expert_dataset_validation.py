"""Phase 6E.3 checks for collected expert demonstrations."""

import json
from pathlib import Path

import numpy as np

from navigation.expert_dataset_validation import (
    _episode_checks,
    validate_collection,
)
from navigation.expert_demonstration_recorder import ExpertDemonstrationRecorder
from navigation.expert_demonstration_schema import (
    DemonstrationEpisode,
    jackal_policy_spaces,
)


def episode(episode_id, scenario_id, goal_feature=0.008, success_steps=4):
    observations = np.full((success_steps, 40), 0.4, dtype=np.float32)
    observations[:, 36] = goal_feature
    actions = np.zeros((success_steps, 2), dtype=np.float32)
    actions[:, 0] = 0.4
    velocities = np.zeros((success_steps, 2), dtype=np.float32)
    velocities[:, 0] = 0.12
    poses = np.zeros((success_steps, 3), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, 0.2, success_steps)
    return DemonstrationEpisode(
        episode_id=episode_id,
        scenario_id=scenario_id,
        termination_reason="GOAL_REACHED",
        goal=np.array([1.0, 1.5], dtype=np.float64),
        observations=observations,
        actions=actions,
        requested_actions=np.array(actions, copy=True),
        applied_velocities=velocities,
        poses=poses,
        simulation_timestamps_ns=np.arange(success_steps, dtype=np.int64) * 100_000_000,
    )


def scenario(scenario_id, fingerprint, duration=70.0):
    return {
        "scenario_id": scenario_id,
        "fingerprint": fingerprint,
        "duration_s": duration,
        "start": [0.0, 1.5, 0.0],
        "goal": [1.0, 1.5],
        "area_class": "open",
        "length_class": "short",
        "turn_class": "straight",
        "planned_path": [[0.0, 1.5], [1.0, 1.5]],
    }


def write_collection(root, episodes_by_split, manifests):
    root.mkdir()
    (root / "manifests").mkdir()
    for split, rows in manifests.items():
        (root / "manifests" / f"{split}.json").write_text(json.dumps({
            "split": split,
            "scenarios": rows,
        }))
    spaces = jackal_policy_spaces()
    for split, episodes in episodes_by_split.items():
        recorder = ExpertDemonstrationRecorder(root / split, spaces[0], spaces[1])
        try:
            for item in episodes:
                recorder.record(item)
        finally:
            recorder.close()
        log = root / split / "attempts.jsonl"
        with log.open("w", encoding="utf-8") as handle:
            for item in episodes:
                handle.write(json.dumps({
                    "scenario_id": item.scenario_id,
                    "navigation_time_s": 10.0,
                }) + "\n")


def test_valid_pilot_passes_and_writes_plots(tmp_path):
    root = tmp_path / "dataset"
    write_collection(
        root,
        {"pilot": [episode(0, "pilot_0000"), episode(1, "pilot_0001")]},
        {"pilot": [scenario("pilot_0000", "a"), scenario("pilot_0001", "b")]},
    )
    # The second episode must not be an exact copy of the first.
    report = validate_collection(root, tmp_path / "report")
    assert report["validation_passed"] is False
    assert report["findings"]["duplicates"]

    root = tmp_path / "clean"
    second = episode(1, "pilot_0001")
    second.observations = np.array(second.observations, copy=True)
    second.observations[:, 0] = 0.2
    write_collection(
        root,
        {"pilot": [episode(0, "pilot_0000"), second]},
        {
            "pilot": [scenario("pilot_0000", "a"), scenario("pilot_0001", "b")],
            "train": [scenario("train_0000", "c")],
        },
    )
    report = validate_collection(root, tmp_path / "clean_report")
    assert report["validation_passed"] is True
    assert report["episodes"]["successes"] == 2
    assert report["episodes"]["failures"] == 0
    assert report["datasets_loaded"] == ["pilot"]
    assert "train" in report["datasets_without_episodes"]
    for name in (
        "action_distribution.png",
        "lidar_distribution.png",
        "goal_distance_distribution.png",
        "sample_trajectories.png",
        "validation_report.json",
    ):
        assert (tmp_path / "clean_report" / name).is_file()


def test_split_leakage_rejects_the_dataset(tmp_path):
    root = tmp_path / "dataset"
    write_collection(
        root,
        {"pilot": [episode(0, "pilot_0000")]},
        {
            "pilot": [scenario("pilot_0000", "same")],
            "test": [scenario("test_0000", "same")],
        },
    )
    report = validate_collection(root, tmp_path / "report")
    assert report["validation_passed"] is False
    assert any(
        not item["passed"] and item["name"] == "scenario_split_separation"
        for item in report["checks"]
    )


def test_incomplete_success_and_backward_time_are_rejected(tmp_path):
    root = tmp_path / "dataset"
    bad = episode(0, "pilot_0000", goal_feature=0.2)
    write_collection(
        root,
        {"pilot": [bad]},
        {"pilot": [scenario("pilot_0000", "a")]},
    )
    report = validate_collection(root, tmp_path / "report")
    assert report["validation_passed"] is False
    assert report["findings"]["incomplete"]

    timed = episode(0, "pilot_0000")
    root = tmp_path / "long"
    write_collection(
        root,
        {"pilot": [timed]},
        {"pilot": [scenario("pilot_0000", "a", duration=1.0)]},
    )
    log = root / "pilot" / "attempts.jsonl"
    log.write_text(json.dumps({
        "scenario_id": "pilot_0000",
        "navigation_time_s": 50.0,
    }) + "\n")
    report = validate_collection(root, tmp_path / "long_report")
    assert report["validation_passed"] is False
    assert report["findings"]["unusually_long"]


def test_backward_timestamps_fail_the_monotonic_check():
    item = episode(0, "pilot_0000")
    item.simulation_timestamps_ns = np.array([3, 2], dtype=np.int64)
    item.observations = item.observations[:2]
    item.actions = item.actions[:2]
    item.requested_actions = item.requested_actions[:2]
    item.applied_velocities = item.applied_velocities[:2]
    item.poses = item.poses[:2]
    payload = {
        "episode_id": 0,
        "scenario_id": "pilot_0000",
        "termination_reason": "GOAL_REACHED",
        "success": True,
        "goal": item.goal,
        "observations": item.observations,
        "actions": item.actions,
        "requested_actions": item.requested_actions,
        "applied_velocities": item.applied_velocities,
        "poses": item.poses,
        "simulation_timestamps_ns": item.simulation_timestamps_ns,
    }
    findings = {"unusually_long": [], "incomplete": [], "duplicates": []}
    checks = _episode_checks(
        [("pilot", payload)],
        {"pilot": {"scenarios": [scenario("pilot_0000", "a")]}},
        -1.0,
        1.0,
        20.0,
        findings,
    )
    monotonic = next(item for item in checks if item["name"] == "monotonic_timestamps")
    assert monotonic["passed"] is False
