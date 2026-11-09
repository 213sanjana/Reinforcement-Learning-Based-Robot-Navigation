"""Offline qualification tests for the Jackal interface harness."""

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from jackal_rl_navigation.action_mapper import ActionMapper, VelocityCommand
from navigation.jackal_interface_qualification import (
    COMPARISON_ATOL,
    CapturedCommandSink,
    ConstantDummyPredictor,
    OfflineHarness,
    QualificationError,
    SensorSnapshot,
    assert_case,
    build_snapshot_input,
    decode_fixture_token,
    execute_case,
    load_manifest,
    main,
    qualify_snapshot,
    verify_source_pins,
)


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "jackal_interface_qualification_v1.json"
HARNESS = REPO / "navigation" / "jackal_interface_qualification.py"
DATA = load_manifest(FIXTURE)
CASES = DATA["cases"]
BASE = DATA["base"]
BANNED_IMPORTS = (
    "rclpy",
    "geometry_msgs",
    "sensor_msgs",
    "nav_msgs",
    "torch",
    "stable_baselines3",
    "gymnasium",
    "jackal_rl_navigation.gazebo_env",
    "jackal_rl_navigation.gazebo_controller",
    "jackal_rl_navigation.gazebo_transport",
    "jackal_rl_navigation.robot_interface",
    "jackal_rl_navigation.collision_monitor",
    "jackal_rl_navigation.safety_monitor",
    "jackal_rl_navigation.world_pose",
    "navigation.jackal_ppo_train",
    "navigation.jackal_bc_policy",
    "navigation.jackal_bc_inference",
    "navigation.jackal_bc_evaluation",
    "navigation.live_astar_executor",
    "navigation.guarded_ros_executor",
)


def _close(expected, actual):
    return actual is not None and len(expected) == len(actual) and all(
        abs(float(left) - float(right)) <= COMPARISON_ATOL for left, right in zip(expected, actual)
    )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_qualification_case(case):
    actual = execute_case(case, BASE)
    assert_case(case, actual)
    for step in actual["steps"]:
        if step.get("accepted") is False and step.get("captured"):
            assert step["command"] is not None
            assert _close([0.0, 0.0], step["command"])


def test_fixture_ids_groups_and_source_pins():
    identifiers = [case["id"] for case in CASES]
    assert identifiers == [f"F{index:03d}" for index in range(1, 101)]
    assert len(set(identifiers)) == 100
    groups = {}
    for case in CASES:
        groups[case["group"]] = groups.get(case["group"], 0) + 1
    assert groups == {
        "sector_order": 36,
        "scan_boundaries": 12,
        "state_features": 12,
        "snapshot_guards": 16,
        "action_mapping": 16,
        "lifecycle_and_pure_frames": 8,
    }
    verify_source_pins(DATA["source_hashes"])
    text = FIXTURE.read_text(encoding="utf-8")
    for token in ("NaN", "+Inf", "-Inf", "pi", "-pi", "3*pi/2"):
        assert f'"{token}"' in text
    with pytest.raises(ValueError):
        json.loads(text.replace('"NaN"', "NaN"), parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    assert decode_fixture_token("__import__('os')") == "__import__('os')"


def test_harness_imports_are_limited():
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    allowed = {
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "sys",
        "numpy",
        "jackal_rl_navigation.action_mapper",
        "jackal_rl_navigation.episode_outcome",
        "jackal_rl_navigation.goal_frames",
        "jackal_rl_navigation.observation_builder",
    }
    assert set(modules) == allowed
    for module_name in modules:
        for banned in BANNED_IMPORTS:
            assert module_name != banned and not module_name.startswith(banned + ".")


def test_import_blocks_live_and_model_modules():
    script = """
import importlib.abc
import sys
banned = %r
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        for prefix in banned:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError("banned import " + fullname)
        if fullname.startswith("ros_gz"):
            raise ImportError("banned import " + fullname)
        return None
sys.meta_path.insert(0, Guard())
import navigation.jackal_interface_qualification
for name in list(sys.modules):
    for prefix in banned:
        if name == prefix or name.startswith(prefix + "."):
            raise SystemExit("loaded " + name)
    if name.startswith("ros_gz"):
        raise SystemExit("loaded " + name)
print("import-ok")
""" % (BANNED_IMPORTS,)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "import-ok" in completed.stdout


def _fresh():
    harness = OfflineHarness()
    harness.seed_accepted_stamps(100, 100)
    return harness


def _dispatch(harness, dispatch_s=10.0, **overrides):
    raw = build_snapshot_input(BASE, overrides, "structural", 0)
    predictor = ConstantDummyPredictor(
        action=overrides.get("action", BASE["action"]),
        dt=overrides.get("dt", BASE["dt"]),
    )
    return harness.dispatch(raw, dispatch_s, predictor=predictor)


def test_repeatable_observation_bytes_and_commands():
    first = execute_case(CASES[0], BASE)
    second = execute_case(CASES[0], BASE)
    assert first["steps"][0]["observation_sha256"] == second["steps"][0]["observation_sha256"]
    assert first["steps"][0]["observation_dtype"] == "float32"
    assert first["steps"][0]["observation_shape"] == [40]
    assert _close(first["steps"][0]["command"], second["steps"][0]["command"])
    assert _close(first["sink_commands"][0], second["sink_commands"][0])


def test_input_copy_and_frozen_records():
    ranges = [6.0] * 72
    raw = build_snapshot_input(BASE, {"ranges": ranges}, "copy", 0)
    snapshot = SensorSnapshot.from_fixture(raw)
    raw["ranges"][0] = -9.0
    assert snapshot.ranges[0] == 6.0
    with pytest.raises(AttributeError):
        snapshot.distance_m = 1.0
    harness = _fresh()
    result = harness.dispatch(snapshot, 10.0)
    with pytest.raises(ValueError):
        result.observation_view()[0] = 0.0
    with pytest.raises(AttributeError):
        harness.sink.records[-1].linear = 4.0
    assert result.observation_sha256
    assert _close(result.command, (0.03, -0.06))


def test_sink_fault_latches_without_claiming_delivery():
    sink = CapturedCommandSink(fault=RuntimeError("sink down"))
    harness = OfflineHarness(sink=sink)
    harness.seed_accepted_stamps(100, 100)
    first = _dispatch(harness)
    assert first.code == "SINK_FAILED"
    assert first.captured is False
    assert first.command is None
    assert sink.records == ()
    second = _dispatch(
        harness,
        10.1,
        scan_stamp_ns=102,
        odom_stamp_ns=102,
        capture_mono_s=10.1,
        scan_received_mono_s=10.0,
        odom_received_mono_s=10.0,
    )
    assert second.code == "SINK_FAILED"
    assert second.captured is False
    assert sink.records == ()


def test_external_sink_is_rejected():
    class ExternalSink:
        def capture(self, record):
            raise AssertionError("external sink called")

    with pytest.raises(TypeError):
        OfflineHarness(sink=ExternalSink())


def test_wrong_geometry_and_missing_frames_fail_closed():
    harness = _fresh()
    geometry = _dispatch(harness, angle_increment_rad=0.01)
    assert geometry.code == "GEOMETRY_MISMATCH"
    assert geometry.predictor_calls == 0
    assert _close(geometry.command, (0.0, 0.0))
    missing_child = _dispatch(_fresh(), omit=["child_frame"])
    missing_odom = _dispatch(_fresh(), omit=["odom_frame"])
    assert missing_child.code == "MISSING_INPUT"
    assert missing_odom.code == "MISSING_INPUT"
    assert missing_child.predictor_calls == 0
    assert missing_odom.predictor_calls == 0


def test_repeated_and_rewound_stamps_reject_without_replacing_history():
    repeated = _fresh()
    first = _dispatch(repeated)
    assert first.accepted
    second = _dispatch(
        repeated,
        10.1,
        scan_stamp_ns=102,
        odom_stamp_ns=101,
        capture_mono_s=10.1,
        scan_received_mono_s=10.0,
        odom_received_mono_s=10.0,
    )
    assert second.code == "NONADVANCING_STAMP"
    assert second.predictor_calls == 0
    assert abs(repeated.sink.records[0].linear - first.command[0]) <= COMPARISON_ATOL
    assert repeated.sink.records[1].linear == 0.0
    rewound = _fresh()
    accepted = _dispatch(rewound)
    assert accepted.accepted
    later = _dispatch(
        rewound,
        10.1,
        scan_stamp_ns=100,
        odom_stamp_ns=103,
        capture_mono_s=10.1,
        scan_received_mono_s=10.0,
        odom_received_mono_s=10.0,
    )
    assert later.code == "NONADVANCING_STAMP"
    assert rewound.sink.records[0].linear == accepted.command[0]


def test_invalid_dt_does_not_change_mapper_state_or_capture_motion():
    for bad_dt in (-1.0, float("inf")):
        mapper = ActionMapper()
        mapper.previous = VelocityCommand(0.3, 0.6)
        before = (mapper.previous.linear, mapper.previous.angular)
        with pytest.raises(ValueError, match="positive and finite"):
            mapper.map_action([0.0, 0.0], bad_dt)
        assert mapper.previous.linear == before[0]
        assert mapper.previous.angular == before[1]
        harness = _fresh()
        result = _dispatch(harness, action=[0.0, 0.0], dt=bad_dt)
        assert result.code == "INVALID_ACTION"
        assert result.predictor_calls == 1
        assert _close(result.command, (0.0, 0.0))
        assert harness.sink.records[-1].linear == 0.0
        assert harness.sink.records[-1].angular == 0.0


def test_malformed_scalar_types_are_not_repaired():
    boolean = _dispatch(_fresh(), distance_m=True)
    text = _dispatch(_fresh(), distance_m="10")
    assert boolean.code == "MALFORMED_INPUT"
    assert text.code == "MALFORMED_INPUT"
    assert boolean.predictor_calls == 0
    assert text.predictor_calls == 0
    assert _close(boolean.command, (0.0, 0.0))
    assert _close(text.command, (0.0, 0.0))


def test_stop_is_idempotent_in_mapper_state():
    harness = _fresh()
    first = harness.stop()
    second = harness.stop()
    assert first.code == "STOPPED"
    assert second.code == "STOPPED"
    assert _close(first.mapper_previous, (0.0, 0.0))
    assert _close(second.mapper_previous, (0.0, 0.0))
    assert len(harness.sink.records) == 2
    assert harness.sink.records[0].linear == 0.0
    assert harness.sink.records[1].angular == 0.0


def test_integrity_failure_is_distinct_from_episode_termination():
    by_id = {case["id"]: case for case in CASES}
    stale = execute_case(by_id["F064"], BASE)
    timeout = execute_case(by_id["F095"], BASE)
    assert stale["steps"][0]["code"] == "STALE_INPUT"
    assert timeout["steps"][0]["code"] == "TIME_LIMIT"
    assert timeout["steps"][0]["termination_reason"] == "time_limit"
    assert stale["steps"][0]["code"] != timeout["steps"][0]["code"]
    assert timeout["steps"][1]["predictor_calls"] == 0
    assert _close(timeout["steps"][1]["command"], [0.0, 0.0])


def test_cli_rejects_unknown_mode_and_existing_output(tmp_path):
    missing = tmp_path / "missing-output"
    assert main(["evaluate", "--manifest", str(FIXTURE), "--output", str(missing)]) == 2
    assert not missing.exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    assert main(["fixtures", "--manifest", str(FIXTURE), "--output", str(existing)]) == 2


def test_cli_fixtures_mode_writes_only_requested_results(tmp_path):
    output = tmp_path / "fresh-cli"
    code = main(["fixtures", "--manifest", str(FIXTURE), "--output", str(output)])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] == 100
    assert summary["failed"] == 0
    assert summary["skipped"] == 0
    assert summary["case_count"] == 100
    results = json.loads((output / "case_results.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in results] == [f"F{index:03d}" for index in range(1, 101)]
    assert all(item["passed"] for item in results)


def _harness_state(harness):
    return (
        harness._episode_id, harness._episode_identity_unknown, harness._latch,
        frozenset(harness._failed_episode_ids),
        harness._last_scan_stamp, harness._last_odom_stamp,
        harness._last_dispatch_mono_s, harness._last_snapshot,
        harness._mapper.previous, harness.sink.records,
    )


@pytest.mark.parametrize("overrides,code", [
    ({"scan_received_mono_s": 9.4}, "STALE_INPUT"),
    ({"ranges": {"count": 72, "fill": 0}}, "INVALID_OBSERVATION"),
])
def test_first_failure_requires_different_episode_and_preserves_history(overrides, code):
    harness = _fresh()
    first = _dispatch(harness, **overrides)
    assert first.code == code and first.latched and first.predictor_calls == 0
    assert harness._episode_id == BASE["episode_id"]
    assert (harness._last_scan_stamp, harness._last_odom_stamp) == (100, 100)
    before_reset = _harness_state(harness)
    reset = harness.reset_episode(BASE["episode_id"])
    assert not reset.accepted and reset.code == "EPISODE_REARM_REJECTED"
    assert _harness_state(harness) == before_reset
    blocked = _dispatch(harness)
    assert not blocked.accepted and blocked.code == code and blocked.predictor_calls == 0
    history = harness.sink.records
    assert len(history) == 2
    assert [(r.linear, r.angular) for r in history] == [(0.0, 0.0)] * 2
    assert [r.sequence for r in history] == [1, 2]
    assert all(r.episode_id == BASE["episode_id"] for r in history)
    assert harness.reset_episode("different-episode").accepted
    assert harness.sink.records == history
    assert harness._latch is None
    assert harness._last_scan_stamp is None and harness._last_odom_stamp is None
    recovered = _dispatch(harness, episode_id="different-episode")
    assert recovered.accepted and _close((0.03, -0.06), recovered.command)
    assert harness.sink.records[:2] == history
    last = harness.sink.records[-1]
    assert last.sequence == 3 and last.episode_id == "different-episode"
    assert last.run_id == BASE["run_id"] and last.snapshot_id == "structural-s0"
    assert last.observation_hash == recovered.observation_sha256
    assert last.dummy_id == "constant-dummy-v1" and last.action == (0.5, -0.5)
    state = _harness_state(harness)
    assert harness.reset_episode(BASE["episode_id"]).code == "EPISODE_REARM_REJECTED"
    assert _harness_state(harness) == state


@pytest.mark.parametrize("overrides", [None, {"omit": ["episode_id"]}, {"episode_id": ""}, {"episode_id": True}])
def test_unidentified_first_failure_cannot_be_reset_or_relabelled(overrides):
    harness = _fresh()
    raw = None if overrides is None else build_snapshot_input(BASE, overrides, "unknown", 0)
    first = harness.dispatch(raw, 10.0)
    assert first.latched and not first.accepted and first.predictor_calls == 0
    assert harness._episode_identity_unknown and harness._episode_id is None
    for name in (BASE["episode_id"], "different-episode"):
        before = _harness_state(harness)
        reset = harness.reset_episode(name)
        assert not reset.accepted and reset.code == "EPISODE_IDENTITY_UNKNOWN"
        assert _harness_state(harness) == before
    blocked = _dispatch(harness)
    assert blocked.code == first.code and blocked.predictor_calls == 0
    assert harness._episode_id is None
    assert len(harness.sink.records) == 2
    assert all((r.linear, r.angular, r.episode_id) == (0.0, 0.0, "") for r in harness.sink.records)


@pytest.mark.parametrize("field,value,code", [
    ("scan_stamp_ns", 0, "INVALID_STAMP"),
    ("scan_stamp_ns", True, "INVALID_STAMP"),
    ("odom_stamp_ns", 0, "INVALID_STAMP"),
    ("odom_stamp_ns", False, "INVALID_STAMP"),
    ("distance_m", True, "MALFORMED_INPUT"),
    ("bearing_rad", "0", "MALFORMED_INPUT"),
    ("linear_mps", False, "MALFORMED_INPUT"),
    ("angular_rps", "0.1", "MALFORMED_INPUT"),
    ("scan_received_mono_s", "9.9", "MALFORMED_INPUT"),
    ("odom_received_mono_s", None, "MALFORMED_INPUT"),
    ("capture_mono_s", [10.0], "MALFORMED_INPUT"),
    ("capture_mono_s", True, "MALFORMED_INPUT"),
])
@pytest.mark.parametrize("construction", ["direct", "replace"])
def test_typed_snapshot_validation_rejects_without_prediction(field, value, code, construction):
    raw = build_snapshot_input(BASE, {}, "typed", 0)
    if construction == "direct":
        raw[field] = value
        snapshot = SensorSnapshot(**raw)
    else:
        snapshot = replace(SensorSnapshot.from_fixture(raw), **{field: value})
    with pytest.raises(QualificationError) as error:
        qualify_snapshot(snapshot, 10.0)
    assert error.value.code == code
    harness = OfflineHarness()  # Zero stamps must reject even without prior stamps.
    predictor = ConstantDummyPredictor()
    rejected = harness.dispatch(snapshot, 10.0, predictor=predictor)
    assert rejected.code == code and rejected.latched and rejected.captured
    assert rejected.predictor_calls == predictor.calls == 0
    assert not rejected.accepted and rejected.command == (0.0, 0.0)
    assert len(harness.sink.records) == 1
    record = harness.sink.records[0]
    assert (record.sequence, record.episode_id, record.linear, record.angular) == (1, BASE["episode_id"], 0.0, 0.0)
    assert record.decision == "REJECT" and record.reason == code
    assert record.dummy_id == "" and record.action == ()
    state = _harness_state(harness)
    assert harness.reset_episode(BASE["episode_id"]).code == "EPISODE_REARM_REJECTED"
    assert _harness_state(harness) == state
    again = _dispatch(harness)
    assert again.code == code and again.predictor_calls == 0
    assert harness.sink.records[0] == record
    assert len(harness.sink.records) == 2
    assert all((r.linear, r.angular) == (0.0, 0.0) for r in harness.sink.records)


@pytest.mark.parametrize("construction", ["direct", "replace"])
def test_typed_snapshot_detaches_ranges_at_construction_and_entry(construction):
    raw = build_snapshot_input(BASE, {}, "detached", 0)
    ranges = raw["ranges"]
    snapshot = SensorSnapshot(**raw) if construction == "direct" else replace(
        SensorSnapshot.from_fixture(raw), ranges=ranges
    )
    ranges[0] = -100.0
    assert snapshot.ranges == (6.0,) * 72
    prepared = OfflineHarness().prepare(snapshot, 10.0)
    assert prepared.snapshot is not snapshot
    assert prepared.snapshot.ranges == (6.0,) * 72
    assert prepared.valid and _close([0.5] * 36, prepared.readonly_vector()[:36])
    with pytest.raises(ValueError):
        prepared.readonly_vector()[0] = 0.0


def test_association_hash_mismatch_records_actual_zero_and_latches():
    harness = _fresh()
    raw = build_snapshot_input(BASE, {}, "hash-mismatch", 0)
    predictor = ConstantDummyPredictor(observation_hash="0" * 64)
    result = harness.dispatch(raw, 10.0, predictor=predictor)
    assert result.code == "ASSOCIATION_MISMATCH" and result.latched
    assert not result.accepted and result.captured and predictor.calls == 1
    assert len(harness.sink.records) == 1
    record = harness.sink.records[0]
    assert (record.linear, record.angular) == (0.0, 0.0)
    assert record.snapshot_id == raw["snapshot_id"] and record.run_id == raw["run_id"]
    assert record.episode_id == raw["episode_id"] and record.sequence == 1
    assert record.observation_hash == result.observation_sha256 != predictor.observation_hash
    assert record.dummy_id == "" and record.action == ()
    assert record.decision == "REJECT" and record.reason == result.code
    blocked = harness.dispatch(raw, 10.1, predictor=predictor)
    assert blocked.predictor_calls == 0 and predictor.calls == 1
    assert harness.sink.records[0] == record
    assert len(harness.sink.records) == 2
    assert harness.sink.records[1].sequence == 2
    assert (harness.sink.records[1].linear, harness.sink.records[1].angular) == (0.0, 0.0)


@pytest.mark.parametrize("case_id", ["F001", "F064"])
@pytest.mark.parametrize("fault", ["missing", "extra_motion", "wrong_command", "wrong_identity", "wrong_sequence"])
def test_fixture_assertions_detect_actual_sink_faults(monkeypatch, case_id, fault):
    capture = CapturedCommandSink.capture

    def faulty_capture(sink, record):
        if fault == "missing":
            return
        if fault == "extra_motion":
            capture(sink, replace(record, linear=0.2))
        if fault == "wrong_command":
            record = replace(record, linear=0.2)
        if fault == "wrong_identity":
            record = replace(record, snapshot_id="unrelated-snapshot")
        if fault == "wrong_sequence":
            record = replace(record, sequence=99)
        capture(sink, record)

    monkeypatch.setattr(CapturedCommandSink, "capture", faulty_capture)
    case = next(case for case in CASES if case["id"] == case_id)
    actual = execute_case(case, BASE)
    if fault == "missing":
        assert actual["sink_commands"] == []
        assert actual["steps"][0]["captured"] is True
    with pytest.raises(AssertionError, match="sink_"):
        assert_case(case, actual)


def test_lifecycle_checks_complete_history_and_reset_metadata(monkeypatch):
    case = next(case for case in CASES if case["id"] == "F097")
    actual = execute_case(case, BASE)
    assert_case(case, actual)
    for previous, current in zip(actual["steps"], actual["steps"][1:]):
        assert current["sink_before"] == previous["sink_after"]
    for index in (3, 4):
        assert actual["steps"][index]["sink_before"] == actual["steps"][index]["sink_after"]
    final = actual["steps"][-1]
    assert final["sink_after"][:-1] == actual["steps"][2]["sink_after"]
    assert [r["sequence"] for r in final["sink_after"]] == [1, 2, 3, 4]
    assert [r["episode_id"] for r in final["sink_after"]] == [BASE["episode_id"]] * 3 + ["episode-new"]
    assert all(set(r) == {
        "sequence", "run_id", "episode_id", "snapshot_id", "observation_hash", "dummy_id", "action",
        "linear", "angular", "monotonic_s", "decision", "reason",
    } for r in final["sink_after"])
    capture = CapturedCommandSink.capture

    def overwrite_history(sink, record):
        if sink._records:
            sink._records[0] = replace(sink._records[0], run_id="rewritten")
        capture(sink, record)

    monkeypatch.setattr(CapturedCommandSink, "capture", overwrite_history)
    with pytest.raises(AssertionError, match="sink_history"):
        assert_case(case, execute_case(case, BASE))


def test_sink_failure_preserves_existing_records_and_rejected_reset_state():
    harness = _fresh()
    first = _dispatch(harness)
    assert first.accepted
    history = harness.sink.records
    harness.sink._fault = RuntimeError("memory capture failure")
    second = _dispatch(
        harness, 10.1, scan_stamp_ns=102, odom_stamp_ns=102,
        capture_mono_s=10.1, scan_received_mono_s=10.0, odom_received_mono_s=10.0,
    )
    assert second.code == "SINK_FAILED" and not second.captured and second.command is None
    assert second.mapper_previous == (0.0, 0.0)
    assert harness.sink.records == history
    state = _harness_state(harness)
    assert harness.reset_episode(BASE["episode_id"]).code == "EPISODE_REARM_REJECTED"
    assert _harness_state(harness) == state
    blocked = _dispatch(harness)
    assert blocked.code == "SINK_FAILED" and blocked.predictor_calls == 0
    assert not blocked.captured and blocked.command is None
    assert harness.sink.records == history
