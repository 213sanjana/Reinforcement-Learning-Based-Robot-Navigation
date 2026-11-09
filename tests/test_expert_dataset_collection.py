"""Phase 6E.2 scenario manifests and collection accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from navigation.expert_dataset_collection import (
    build_report,
    collect_split,
    load_config,
    preview,
    run_stage,
    validate_pilot,
)
from navigation.expert_demonstration_recorder import (
    ExpertDemonstrationRecorder,
    load_dataset,
)
from navigation.expert_demonstration_schema import jackal_policy_spaces
from navigation.expert_evaluation_plan import diagnostic_scenarios
from navigation.expert_scenario_generator import (
    SCHEMA_VERSION,
    assert_scenario_feasible,
    generate_manifests,
    planner_from_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "expert_dataset_6e2.yaml"


def tiny_config(tmp_path):
    source = yaml.safe_load(CONFIG_PATH.read_text())
    source["output_root"] = str(tmp_path / "v6e.2.0")
    source["stages"] = {
        "pilot": {"attempts": 4, "success_target": None},
        "train": {"attempts": 4, "success_target": 2},
        "validation": {"attempts": 4, "success_target": 1},
        "test": {"attempts": 4, "success_target": 1},
    }
    source["quotas"] = {
        "pilot": {
            "open_short_straight": 2,
            "obstacle_medium_turning": 2,
        },
        "train": {
            "open_medium_straight": 2,
            "obstacle_long_turning": 2,
        },
        "validation": {
            "open_long_straight": 2,
            "obstacle_short_turning": 2,
        },
        "test": {
            "open_short_straight": 2,
            "obstacle_medium_turning": 2,
        },
    }
    path = tmp_path / "collection.yaml"
    path.write_text(yaml.safe_dump(source))
    return load_config(path)


def transition(index=0):
    return SimpleNamespace(
        observation=np.full(40, 0.2, dtype=np.float32),
        action=np.array([0.5, 0.0], dtype=np.float32),
        applied_action=np.array([0.4, 0.0], dtype=np.float32),
        published_linear=0.12,
        published_angular=0.0,
        simulation_time_ns=1_000 + index,
        pose=(float(index) * 0.1, 0.0, 0.0),
    )


def result(status, steps=1, rejected=0):
    transitions = [transition(index) for index in range(steps)]
    return SimpleNamespace(
        status=status,
        transitions=transitions,
        accepted_commands=len(transitions),
        rejected_commands=rejected,
    )


class FakeRunner:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.prepared = 0
        self.closed = False

    def prepare(self, start, goal):
        self.prepared += 1
        if len(start) != 3 or len(goal) != 2:
            raise ValueError("invalid reset pose")

    def run(self, duration):
        status = self.statuses.pop(0)
        if status == "RAISE":
            raise TimeoutError("simulator timed out")
        if status == "PLANNING_FAILED":
            return result(status, steps=0)
        return result(status, steps=2)

    def close(self):
        self.closed = True


def record(tmp_path, scenarios, statuses, success_target=None):
    config = tiny_config(tmp_path)
    runner = FakeRunner(statuses)
    spaces = jackal_policy_spaces()
    recorder = ExpertDemonstrationRecorder(
        tmp_path / "episodes",
        spaces[0],
        spaces[1],
    )
    try:
        collect_split(
            scenarios,
            runner,
            recorder,
            tmp_path / "episodes" / "attempts.jsonl",
            command_interval_s=0.1,
            success_target=success_target,
        )
    finally:
        recorder.close()
    return runner


def test_phase_6d_expert_scenarios_remain_available():
    identifiers = [scenario.scenario_id for scenario in diagnostic_scenarios()]
    assert identifiers == [
        "low_speed_single",
        "reachable_north",
        "reachable_around",
        "unreachable_goal",
        "invalid_start",
        "timeout_short",
        "collision_probe",
    ]


def test_shipped_targets_and_versioned_directory(tmp_path):
    config = load_config(CONFIG_PATH)
    assert config["stages"]["pilot"]["attempts"] == 20
    assert config["stages"]["train"]["success_target"] == 500
    assert config["stages"]["validation"]["success_target"] == 100
    assert config["stages"]["test"]["success_target"] == 100
    assert config["output_root"].name == "v6e.2.0"
    assert config["seed"] == 642


def test_config_rejects_unversioned_output(tmp_path):
    source = yaml.safe_load(CONFIG_PATH.read_text())
    source["output_root"] = str(tmp_path / "dataset")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="versioned"):
        load_config(path)


def test_generated_splits_are_feasible_and_disjoint(tmp_path):
    config = tiny_config(tmp_path)
    manifests = generate_manifests(config)
    planner = planner_from_config(config)
    fingerprints = []
    areas = set()
    lengths = set()
    turns = set()

    for manifest in manifests.values():
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert isinstance(manifest["split_seed"], int)
        for scenario in manifest["scenarios"]:
            assert_scenario_feasible(planner, scenario)
            fingerprints.append(scenario["fingerprint"])
            areas.add(scenario["area_class"])
            lengths.add(scenario["length_class"])
            turns.add(scenario["turn_class"])

    assert len(fingerprints) == len(set(fingerprints))
    assert areas == {"open", "obstacle_rich"}
    assert {"short", "medium", "long"} <= lengths
    assert turns == {"straight", "turning"}

    again = generate_manifests(config)
    assert (
        manifests["pilot"]["scenarios"][0]["fingerprint"]
        == again["pilot"]["scenarios"][0]["fingerprint"]
    )


def test_preview_writes_manifests_without_episodes(tmp_path):
    config = tiny_config(tmp_path)
    preview(config)
    root = config["output_root"]
    assert (root / "manifests" / "pilot.json").is_file()
    assert (root / "dataset_version.json").is_file()
    assert not (root / "pilot").exists()
    first = (root / "manifests" / "train.json").read_bytes()
    preview(config)
    assert (root / "manifests" / "train.json").read_bytes() == first


def test_preview_refuses_a_different_schema_version(tmp_path):
    config = tiny_config(tmp_path)
    root = config["output_root"]
    root.mkdir()
    (root / "dataset_version.json").write_text(
        json.dumps({"schema_version": "6e.1.0"})
    )
    with pytest.raises(RuntimeError, match="schema"):
        preview(config)


def _copy_dataset(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            (destination / path.name).write_bytes(path.read_bytes())


def test_failures_are_recorded_and_not_counted_as_success(tmp_path):
    config = tiny_config(tmp_path)
    scenarios = preview(config)["pilot"]["scenarios"]
    runner = record(
        tmp_path,
        scenarios,
        ["GOAL_REACHED", "TIME_LIMIT", "RAISE", "PLANNING_FAILED"],
    )

    episodes = load_dataset(tmp_path / "episodes")
    assert [episode["termination_reason"] for episode in episodes] == [
        "GOAL_REACHED",
        "TIME_LIMIT",
        "TIMEOUT",
        "PLANNING_FAILED",
    ]
    assert sum(episode["success"] for episode in episodes) == 1
    assert runner.prepared == 4

    _copy_dataset(tmp_path / "episodes", config["output_root"] / "pilot")
    report = build_report(config, "pilot", session_time_s=1.5)
    assert report["attempts"] == 4
    assert report["successes"] == 1
    assert report["failures"] == 3
    assert report["accepted_commands"] == 4
    assert report["expert_command_seconds"] == pytest.approx(0.4)
    assert report["collection_time_s"] == pytest.approx(1.5)


def test_rejected_record_does_not_stop_the_split(tmp_path):
    config = tiny_config(tmp_path)
    scenarios = preview(config)["pilot"]["scenarios"][:2]
    spaces = jackal_policy_spaces()
    recorder = ExpertDemonstrationRecorder(
        tmp_path / "episodes",
        spaces[0],
        spaces[1],
    )
    original = recorder.record
    calls = {"n": 0}

    def reject_once(episode):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("Simulation timestamps are not monotonic")
        return original(episode)

    recorder.record = reject_once
    try:
        session = collect_split(
            scenarios,
            FakeRunner(["GOAL_REACHED", "GOAL_REACHED"]),
            recorder,
            tmp_path / "episodes" / "attempts.jsonl",
            command_interval_s=0.1,
        )
    finally:
        recorder.close()

    stored = load_dataset(tmp_path / "episodes")
    assert [episode["scenario_id"] for episode in stored] == [
        scenarios[1]["scenario_id"]
    ]
    assert session["successes_so_far"] == 1
    attempts = [
        json.loads(line)
        for line in (tmp_path / "episodes" / "attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[0]["status"] == "RECORD_REJECTED"
    assert attempts[0]["success"] is False
    assert attempts[1]["success"] is True


def test_resume_does_not_duplicate_episodes(tmp_path):
    config = tiny_config(tmp_path)
    scenarios = generate_manifests(config)["pilot"]["scenarios"]
    runner = FakeRunner(["GOAL_REACHED", "GOAL_REACHED", "GOAL_REACHED", "GOAL_REACHED"])
    spaces = jackal_policy_spaces()
    output = tmp_path / "episodes"
    recorder = ExpertDemonstrationRecorder(output, spaces[0], spaces[1])
    try:
        collect_split(
            scenarios[:2],
            runner,
            recorder,
            output / "attempts.jsonl",
            command_interval_s=0.1,
        )
    finally:
        recorder.close()

    resumed_runner = FakeRunner(["GOAL_REACHED", "GOAL_REACHED"])
    recorder = ExpertDemonstrationRecorder(output, spaces[0], spaces[1])
    try:
        collect_split(
            scenarios,
            resumed_runner,
            recorder,
            output / "attempts.jsonl",
            command_interval_s=0.1,
        )
    finally:
        recorder.close()

    episodes = load_dataset(output)
    assert [episode["episode_id"] for episode in episodes] == [0, 1, 2, 3]
    assert resumed_runner.prepared == 2


def test_collection_stops_at_the_success_target(tmp_path):
    config = tiny_config(tmp_path)
    scenarios = generate_manifests(config)["train"]["scenarios"]
    runner = record(tmp_path, scenarios, ["GOAL_REACHED"] * 4, success_target=2)
    episodes = load_dataset(tmp_path / "episodes")
    assert len(episodes) == 2
    assert runner.prepared == 2


def test_later_stage_waits_for_pilot_quality(tmp_path):
    config = tiny_config(tmp_path)
    preview(config)
    created = []

    def factory(scenario):
        created.append(scenario["scenario_id"])
        return FakeRunner(["GOAL_REACHED"])

    with pytest.raises(RuntimeError, match="Pilot"):
        run_stage(
            config,
            "train",
            allow_execution=True,
            runner_factory=factory,
            preflight=lambda: True,
            publisher_count=lambda: 0,
        )

    assert created == []


def test_disabled_execution_does_not_create_a_runner(tmp_path):
    config = tiny_config(tmp_path)
    with pytest.raises(RuntimeError, match="disabled"):
        run_stage(
            config,
            "pilot",
            allow_execution=False,
            runner_factory=lambda scenario: (_ for _ in ()).throw(
                AssertionError("runner created")
            ),
        )


def test_pilot_quality_requires_both_areas(tmp_path):
    config = tiny_config(tmp_path)
    scenarios = preview(config)["pilot"]["scenarios"]
    record(tmp_path, scenarios, ["GOAL_REACHED"] * 4)
    _copy_dataset(tmp_path / "episodes", config["output_root"] / "pilot")

    quality = validate_pilot(config)
    assert quality["attempts"] == 4
    assert quality["successes"] == 4
    assert quality["failures"] == 0
    assert quality["quality_passed"] is True
