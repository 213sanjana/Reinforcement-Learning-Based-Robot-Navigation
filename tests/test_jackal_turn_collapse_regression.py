"""Offline checks for the saved Phase 6H.2 right-turn regression."""

import json
from pathlib import Path

import pytest

from navigation.jackal_turn_collapse_regression import (
    RegressionError,
    default_run_directory,
    load_matrix,
    run_regression,
    stream_telemetry,
    validate_inventory,
)


@pytest.fixture(scope="module")
def result():
    return run_regression(default_run_directory())


def test_matrix_inventory_is_complete(result):
    root = default_run_directory()
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    analysis = json.loads((root / "analysis.json").read_text())
    expected_ids = tuple(item["episode_id"] for item in manifest["episode_order"])
    inventory = result.inventory
    assert inventory.episode_ids == expected_ids
    assert inventory.episode_count == 52
    assert len(set(expected_ids)) == 52
    assert inventory.initial_episodes == 26
    assert inventory.final_episodes == 26
    assert summary["training_steps"] == 0
    assert inventory.training_steps == 0
    assert analysis["instrumentation_gate_passed"] is True
    assert inventory.instrumentation_gate_passed is True


def test_final_policy_angular_command_is_saturated_right(result):
    counts, means, not_below = _independent_angular(default_run_directory() / "telemetry.jsonl")
    saturation = result.saturation
    assert counts == {"initial": 3461, "final": 5878}
    assert saturation.telemetry_rows_initial == 3461
    assert saturation.telemetry_rows_final == 5878
    assert saturation.recomputed_angular_mean_initial == pytest.approx(means["initial"])
    assert saturation.recomputed_angular_mean_final == pytest.approx(means["final"])
    assert saturation.recomputed_angular_mean_final <= -0.84
    assert saturation.sign_threshold == 0.01
    assert saturation.sample_saturation_threshold == -0.8
    assert saturation.episode_mean_saturation_threshold == -0.84
    assert saturation.every_final_right_action_fraction_is_one
    assert saturation.every_final_episode_mean_within_saturation
    assert saturation.every_final_sample_below_sign_threshold
    assert saturation.final_rows_not_below_sample_saturation == not_below
    assert not_below > 0
    assert saturation.every_final_row_below_sample_saturation is False
    assert saturation.sample_saturation_threshold <= saturation.least_negative_final_angular < -saturation.sign_threshold


def test_final_policy_spends_more_path_than_progress(result):
    progress = result.progress
    assert progress.progress_path_ratio_is_spl is False
    assert progress.paths_differ_on_at_least_one_episode is True
    assert progress.final_episodes_subject_to_ratio_cap == 24
    assert progress.heading_exception_episodes == 2
    assert progress.initial_open_episodes == 10
    assert progress.translation_north_final_ratios == (0.0, 0.0)
    assert all(value < 0 for value in progress.translation_north_final_net_progress_m)


def test_obstacle_episodes_time_out_for_both_policies(result):
    obstacles = result.obstacles
    assert obstacles.episode_count == 12
    assert obstacles.final_ratio_max <= 0.08
    assert obstacles.initial_ratio_min >= 0.69


def test_final_policy_does_not_exceed_initial_on_matched_seeds(result):
    comparison = result.comparison
    assert comparison.scenario_count == 13
    assert comparison.pairs == 26
    assert comparison.final_does_not_exceed_initial is True
    assert comparison.both_seed_loss_scenarios == (
        "distance_1p5",
        "distance_2p5",
        "translation_north",
        "open_south",
    )


def test_regression_does_not_write_evidence():
    root = default_run_directory()
    supervisor = root.with_name(root.name + "_supervisor")
    before = _fingerprint(root)
    supervisor_before = _fingerprint(supervisor)
    run_regression(root)
    assert _fingerprint(root) == before
    assert _fingerprint(supervisor) == supervisor_before


def test_missing_manifest_is_rejected(tmp_path):
    with pytest.raises(RegressionError, match="manifest.json"):
        load_matrix(tmp_path)


def test_duplicate_episode_id_is_rejected(tmp_path):
    _write_matrix(tmp_path, [_episode("same"), _episode("same")])
    with pytest.raises(RegressionError, match="duplicate episode ID same"):
        load_matrix(tmp_path)


def test_analysis_order_mismatch_is_rejected(tmp_path):
    episodes = [_episode("anchor_initial_642"), _episode("anchor_final_642", policy="final")]
    _write_matrix(tmp_path, episodes)
    analysis_path = tmp_path / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["episodes"].reverse()
    analysis_path.write_text(json.dumps(analysis))
    with pytest.raises(RegressionError, match="analysis.json episodes episode order"):
        load_matrix(tmp_path)


def test_false_instrumentation_gate_is_rejected(tmp_path):
    _write_matrix(tmp_path, [_episode("anchor_initial_642")], gate=False)
    loaded = load_matrix(tmp_path)
    with pytest.raises(RegressionError, match="instrumentation_gate_passed"):
        validate_inventory(loaded)


def test_nonzero_training_steps_are_rejected(tmp_path):
    _write_matrix(tmp_path, [_episode("anchor_initial_642")], training_steps=3)
    loaded = load_matrix(tmp_path)
    with pytest.raises(RegressionError, match="training_steps"):
        validate_inventory(loaded)


def test_blank_telemetry_lines_are_ignored(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "\n"
        '{"episode_id":"anchor_initial_642","policy":"initial","actor_mean":[0.1,-0.2],'
        '"step":1,"net_progress_m":0.4,"cumulative_path_length_m":0.5}\n'
        "\n"
    )
    telemetry = stream_telemetry(path, {"anchor_initial_642": "initial"})
    assert telemetry.rows_by_policy["initial"] == 1
    assert telemetry.rows_by_policy["final"] == 0


def test_malformed_telemetry_is_not_skipped(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        '{"episode_id":"anchor_initial_642","policy":"initial","actor_mean":[0.1,-0.2],'
        '"step":1,"net_progress_m":0.4,"cumulative_path_length_m":0.5}\n'
        "{bad\n"
    )
    with pytest.raises(RegressionError, match=r"telemetry.jsonl line 2: invalid JSON"):
        stream_telemetry(path, {"anchor_initial_642": "initial"})


def test_telemetry_missing_actor_mean_is_rejected(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        '{"episode_id":"anchor_initial_642","policy":"initial","step":1,'
        '"net_progress_m":0.4,"cumulative_path_length_m":0.5}\n'
    )
    with pytest.raises(RegressionError, match="actor_mean"):
        stream_telemetry(path, {"anchor_initial_642": "initial"})


def test_missing_telemetry_episode_is_rejected(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text("\n")
    with pytest.raises(RegressionError, match="anchor_initial_642: no telemetry rows"):
        stream_telemetry(path, {"anchor_initial_642": "initial"})


def test_unknown_telemetry_episode_is_rejected(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        '{"episode_id":"missing","policy":"initial","actor_mean":[0.0,-0.2],'
        '"step":1,"net_progress_m":0.1,"cumulative_path_length_m":0.2}\n'
    )
    with pytest.raises(RegressionError, match="unknown episode_id"):
        stream_telemetry(path, {"anchor_initial_642": "initial"})


def _independent_angular(path):
    counts = {"initial": 0, "final": 0}
    totals = {"initial": 0.0, "final": 0.0}
    not_below = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            policy = row["policy"]
            angular = row["actor_mean"][1]
            counts[policy] += 1
            totals[policy] += angular
            if policy == "final" and not angular < -0.8:
                not_below += 1
    means = {policy: totals[policy] / counts[policy] for policy in counts}
    return counts, means, not_below


def _fingerprint(root):
    fingerprint = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        fingerprint.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return tuple(fingerprint)


def _episode(episode_id, policy="initial", scenario="anchor", seed=642):
    return {
        "episode_id": episode_id,
        "policy": policy,
        "scenario": scenario,
        "seed": seed,
    }


def _write_matrix(directory, episodes, training_steps=0, gate=True):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    summary_rows = []
    analysis_rows = []
    jsonl_rows = []
    for episode in episodes:
        manifest_rows.append(dict(episode))
        outcome = {
            "success": True,
            "collision": False,
            "timeout": False,
            "termination_reason": "goal_reached",
            "control_steps": 1,
            "path_length_m": 0.55,
        }
        summary_rows.append({**episode, "net_progress_m": 0.4, "path_length_m": 0.5, "path_efficiency": 0.8, "evaluator": outcome})
        analysis_rows.append({
            **episode,
            "valid": True,
            "success": True,
            "collision": False,
            "timeout": False,
            "termination_reason": "goal_reached",
            "control_steps": 1,
            "net_progress_m": 0.4,
            "aligned_snapshot_path_m": 0.5,
            "evaluator_pose_path_m": 0.55,
            "progress_path_ratio": 0.8,
            "steering": {"actor_angular": {"mean": -0.2}, "right_action_fraction": 1.0},
        })
        jsonl_rows.append({**episode, "net_progress_m": 0.4, "path_length_m": 0.5, "path_efficiency": 0.8, "evaluator": outcome})
    (directory / "manifest.json").write_text(json.dumps({"episode_order": manifest_rows}))
    (directory / "summary.json").write_text(json.dumps({"training_steps": training_steps, "episodes": summary_rows}))
    (directory / "analysis.json").write_text(json.dumps({"instrumentation_gate_passed": gate, "episodes": analysis_rows}))
    (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in jsonl_rows))
    (directory / "telemetry.jsonl").write_text("")
