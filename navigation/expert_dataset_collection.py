"""Collect Jackal A* demonstrations into versioned dataset directories.

Preview writes scenario manifests and does not initialize ROS.
Pilot and later stages reuse one existing expert runner. Training,
validation, and test collection refuse to start until the pilot
dataset passes quality checks.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from navigation.expert_demonstration_recorder import (
    ExpertDemonstrationRecorder,
    load_dataset,
)
from navigation.expert_demonstration_schema import (
    episode_from_expert_result,
    jackal_policy_spaces,
)
from navigation.expert_scenario_generator import (
    SCHEMA_VERSION,
    assert_disjoint,
    generate_manifests,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NO_MOTION_MESSAGE = "No robot commands or simulation resets were issued."
SPLITS = ("pilot", "train", "validation", "test")


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Config schema must be {SCHEMA_VERSION}")

    world = Path(config["world"])
    if not world.is_absolute():
        world = REPO_ROOT / world
    if not world.is_file():
        raise FileNotFoundError(f"World not found: {world}")
    config["world"] = world

    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if output_root.name != f"v{SCHEMA_VERSION}":
        raise ValueError(
            f"Output directory must be versioned as v{SCHEMA_VERSION}"
        )
    config["output_root"] = output_root

    for split in SPLITS:
        attempts = int(config["stages"][split]["attempts"])
        quota = sum(int(value) for value in config["quotas"][split].values())
        if attempts != quota:
            raise ValueError(f"{split} attempts and quotas differ")
        target = config["stages"][split]["success_target"]
        if target is not None and int(target) > attempts:
            raise ValueError(f"{split} success target exceeds attempts")

    if config["storage"] not in ("npz", "hdf5"):
        raise ValueError("storage must be npz or hdf5")

    return config


def preview(config):
    """Write manifests if this versioned directory has no episodes yet."""

    output_root = config["output_root"]
    version_path = output_root / "dataset_version.json"
    _refuse_foreign_version(version_path)

    if _any_recorded_episodes(output_root):
        manifests = load_manifests(output_root)
        assert_disjoint(manifests)
        return manifests

    manifests = generate_manifests(config)
    _write_manifests(output_root, manifests, config)
    return manifests


def load_manifests(output_root):
    manifests = {}
    for split in SPLITS:
        path = Path(output_root) / "manifests" / f"{split}.json"
        with path.open("r", encoding="utf-8") as handle:
            manifests[split] = json.load(handle)
    assert_disjoint(manifests)
    return manifests


def collect_split(
    scenarios,
    runner,
    recorder,
    attempts_path,
    *,
    command_interval_s,
    success_target=None,
):
    """Run remaining scenarios on one expert runner and record every outcome."""

    recorded = _recorded_scenario_ids(recorder.output_dir)
    successes = _success_count(recorder.output_dir)
    started = time.monotonic()
    new_outcomes = []

    for scenario in scenarios:
        if scenario["scenario_id"] in recorded:
            continue

        if success_target is not None and successes >= int(success_target):
            break

        outcome = _attempt(scenario, runner, recorder)
        _append_jsonl(attempts_path, outcome)
        new_outcomes.append(outcome)
        recorded.add(scenario["scenario_id"])
        if outcome["success"]:
            successes += 1

    return {
        "new_outcomes": new_outcomes,
        "session_time_s": time.monotonic() - started,
        "successes_so_far": successes,
    }


def build_report(config, split, session_time_s):
    """Count attempts from stored episodes, not from the candidate manifest."""

    output_dir = config["output_root"] / split
    manifest = load_manifests(config["output_root"])[split]
    episodes = load_dataset(output_dir) if (output_dir / "manifest.json").exists() else []
    attempts = _read_jsonl(output_dir / "attempts.jsonl")
    by_scenario = {}
    for item in attempts:
        by_scenario[item["scenario_id"]] = item

    status_counts = {}
    navigation_time = 0.0
    accepted = 0
    for episode in episodes:
        reason = episode["termination_reason"]
        status_counts[reason] = status_counts.get(reason, 0) + 1
        logged = by_scenario.get(episode["scenario_id"], {})
        navigation_time += float(logged.get("navigation_time_s") or 0.0)
        accepted += int(episode["observations"].shape[0])

    successes = sum(1 for episode in episodes if episode["success"])
    failures = len(episodes) - successes
    command_seconds = accepted * float(config["command_interval_s"])
    report_path = output_dir / "collection_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        session_time_s += float(previous.get("collection_time_s") or 0.0)

    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "seed": manifest["seed"],
        "split_seed": manifest["split_seed"],
        "manifest_scenarios": len(manifest["scenarios"]),
        "success_target": manifest["success_target"],
        "attempts": len(episodes),
        "successes": successes,
        "failures": failures,
        "status_counts": status_counts,
        "collection_time_s": round(float(session_time_s), 4),
        "navigation_time_s": round(navigation_time, 4),
        "accepted_commands": accepted,
        "expert_command_seconds": round(command_seconds, 4),
        "successful_demonstrations": successes,
    }


def validate_pilot(config):
    """Check the pilot records. Attempts are not counted as successes."""

    output_root = config["output_root"]
    manifest = load_manifests(output_root)["pilot"]
    output_dir = output_root / "pilot"
    reasons = []
    episodes = []

    try:
        episodes = load_dataset(output_dir)
    except Exception as exc:
        reasons.append(f"pilot dataset failed integrity checks: {exc}")

    expected_ids = [item["scenario_id"] for item in manifest["scenarios"]]
    actual_ids = [episode["scenario_id"] for episode in episodes]

    if actual_ids != expected_ids and sorted(actual_ids) != sorted(expected_ids):
        if len(episodes) != len(expected_ids):
            reasons.append(
                f"pilot recorded {len(episodes)} of {len(expected_ids)} attempts"
            )
        else:
            reasons.append("pilot scenario ids do not match the manifest")

    if len(actual_ids) != len(set(actual_ids)):
        reasons.append("pilot contains duplicate scenarios")

    successes = [episode for episode in episodes if episode["success"]]
    failures = [episode for episode in episodes if not episode["success"]]

    if any(episode["termination_reason"] != "GOAL_REACHED" for episode in successes):
        reasons.append("a non-goal episode was counted as a success")

    if any(episode["observations"].shape[0] < 1 for episode in successes):
        reasons.append("a successful episode has no transitions")

    areas = {item["area_class"] for item in manifest["scenarios"]}
    lengths = {item["length_class"] for item in manifest["scenarios"]}
    turns = {item["turn_class"] for item in manifest["scenarios"]}

    if not {"open", "obstacle_rich"} <= areas:
        reasons.append("pilot manifest is missing an open or obstacle-rich area")
    if len(lengths) < 2:
        reasons.append("pilot manifest does not vary route length")
    if not {"straight", "turning"} <= turns:
        reasons.append("pilot manifest does not vary turning")

    by_id = {item["scenario_id"]: item for item in manifest["scenarios"]}
    success_areas = {
        by_id[episode["scenario_id"]]["area_class"]
        for episode in successes
        if episode["scenario_id"] in by_id
    }
    if not {"open", "obstacle_rich"} <= success_areas:
        reasons.append(
            "pilot has no successful demonstration in both open and obstacle-rich areas"
        )

    try:
        assert_disjoint(load_manifests(output_root))
    except ValueError as exc:
        reasons.append(str(exc))

    return {
        "schema_version": SCHEMA_VERSION,
        "quality_passed": not reasons,
        "reasons": reasons,
        "attempts": len(episodes),
        "successes": len(successes),
        "failures": len(failures),
    }


def run_stage(
    config,
    stage,
    *,
    allow_execution=False,
    runner_factory=None,
    preflight=None,
    publisher_count=None,
):
    """Collect one split. Later splits require a validated pilot."""

    if stage not in SPLITS:
        raise ValueError(f"Unknown stage {stage}")

    manifests = preview(config)

    if stage != "pilot":
        quality_path = config["output_root"] / "pilot" / "quality.json"
        if not quality_path.is_file():
            raise RuntimeError(
                "Pilot quality has not been validated. "
                "Collection was not started."
            )
        quality = json.loads(quality_path.read_text())
        if not quality.get("quality_passed"):
            raise RuntimeError(
                "Pilot quality checks failed. Collection was not started."
            )

    if not allow_execution:
        raise RuntimeError(
            "Collection execution is disabled. " + NO_MOTION_MESSAGE
        )

    if preflight is None:
        from navigation.collect_expert_demonstrations import (
            run_collection_preflight,
        )
        preflight = run_collection_preflight

    if not preflight():
        raise RuntimeError("Preflight failed. " + NO_MOTION_MESSAGE)

    if publisher_count is None:
        from navigation.evaluate_expert_navigation import _publisher_count
        publisher_count = _publisher_count

    before = publisher_count()
    if before != 0:
        raise RuntimeError(
            "Velocity topic does not have exactly zero publishers "
            f"(observed {before})."
        )

    if runner_factory is None:
        from navigation.expert_live_runner_factory import create_live_runner

        def runner_factory(scenario):
            return create_live_runner(
                scenario,
                world_path=config["world"],
                inflation_radius=float(config["inflation_radius"]),
                allow_live=True,
            )

    runner = runner_factory(manifests[stage]["scenarios"][0])
    observation_space, action_space = jackal_policy_spaces()
    output_dir = config["output_root"] / stage
    recorder = ExpertDemonstrationRecorder(
        output_dir,
        observation_space,
        action_space,
        storage=config["storage"],
    )

    try:
        session = collect_split(
            manifests[stage]["scenarios"],
            runner,
            recorder,
            output_dir / "attempts.jsonl",
            command_interval_s=float(config["command_interval_s"]),
            success_target=manifests[stage]["success_target"],
        )
    finally:
        recorder.close()
        runner.close()

    after = publisher_count()
    report = build_report(config, stage, session["session_time_s"])
    report["publisher_count_before"] = before
    report["publisher_count_after"] = after
    report["stale_publisher"] = after != 0
    _write_json(output_dir / "collection_report.json", report)

    if stage == "pilot":
        quality = validate_pilot(config)
        quality["publisher_count_after"] = after
        if after != 0:
            quality["quality_passed"] = False
            quality["reasons"].append("a velocity publisher remained after collection")
        _write_json(output_dir / "quality.json", quality)
        report["quality_passed"] = quality["quality_passed"]

    return report


def _attempt(scenario, runner, recorder):
    started = time.monotonic()
    result = None
    error = ""
    status = "ERROR"

    try:
        try:
            runner.prepare(
                start=tuple(scenario["start"]),
                goal=tuple(scenario["goal"]),
            )
        except Exception as exc:
            status = _status_from_exception(exc)
            error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                result = runner.run(duration=float(scenario["duration_s"]))
                status = result.status
                if status == "GOAL_REACHED" and int(result.rejected_commands) != 0:
                    status = "REJECTED_COMMANDS"
            except Exception as exc:
                status = _status_from_exception(exc)
                error = f"{type(exc).__name__}: {exc}"
                result = None
    except Exception as exc:
        status = "ERROR"
        error = f"{type(exc).__name__}: {exc}"
        result = None

    if result is None:
        result = _empty_result(status)

    episode = episode_from_expert_result(
        result,
        episode_id=int(scenario["episode_id"]),
        scenario_id=scenario["scenario_id"],
        goal=tuple(scenario["goal"]),
    )
    if episode.termination_reason != status and status != "GOAL_REACHED":
        episode.termination_reason = status

    try:
        recorder.record(episode)
    except Exception as exc:
        # A rejected record is not a stored episode and is not a success.
        # The scenario stays out of the manifest, so a later run can retry it.
        error = f"{type(exc).__name__}: {exc}"
        status = "RECORD_REJECTED"
        episode.termination_reason = status

    navigation_time = time.monotonic() - started

    return {
        "scenario_id": scenario["scenario_id"],
        "episode_id": int(scenario["episode_id"]),
        "status": episode.termination_reason,
        "success": episode.termination_reason == "GOAL_REACHED",
        "navigation_time_s": round(navigation_time, 4),
        "accepted_commands": int(result.accepted_commands),
        "rejected_commands": int(result.rejected_commands),
        "error": error[:200],
    }


def _empty_result(status):
    from types import SimpleNamespace

    return SimpleNamespace(
        status=status,
        transitions=[],
        accepted_commands=0,
        rejected_commands=0,
    )


def _status_from_exception(exc):
    message = str(exc)
    if "PLANNING_FAILED" in message:
        return "PLANNING_FAILED"
    if "INVALID_PATH" in message:
        return "INVALID_PATH"
    if "timed out" in message.lower() or "timeout" in message.lower():
        return "TIMEOUT"
    return "ERROR"


def _recorded_scenario_ids(output_dir):
    path = Path(output_dir) / "manifest.json"
    if not path.exists():
        return set()
    return {episode["scenario_id"] for episode in load_dataset(output_dir)}


def _success_count(output_dir):
    path = Path(output_dir) / "manifest.json"
    if not path.exists():
        return 0
    return sum(1 for episode in load_dataset(output_dir) if episode["success"])


def _any_recorded_episodes(output_root):
    for split in SPLITS:
        manifest = output_root / split / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("episodes"):
            return True
    return False


def _refuse_foreign_version(version_path):
    if not version_path.exists():
        return
    payload = json.loads(version_path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "Refusing to write into a dataset from schema "
            f"{payload.get('schema_version')}"
        )


def _write_manifests(output_root, manifests, config):
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "dataset_version.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": int(config["seed"]),
            "world": str(config["world"]),
            "inflation_radius": float(config["inflation_radius"]),
        },
    )
    for split, manifest in manifests.items():
        _write_json(manifest_dir / f"{split}.json", manifest)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate disjoint Jackal expert scenarios and collect "
            "demonstrations. preview and preflight do not move the robot."
        )
    )
    parser.add_argument(
        "command",
        choices=("preview", "preflight", "pilot", "collect"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("train", "validation", "test"),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "preview":
        manifests = preview(config)
        print(json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "output_root": str(config["output_root"]),
                "scenarios": {
                    split: len(manifest["scenarios"])
                    for split, manifest in manifests.items()
                },
                "movement_enabled": False,
            },
            indent=2,
        ))
        print(NO_MOTION_MESSAGE)
        return 0

    if args.command == "preflight":
        from navigation.collect_expert_demonstrations import (
            run_collection_preflight,
        )
        return 0 if run_collection_preflight() else 1

    stage = "pilot" if args.command == "pilot" else args.stage
    if stage is None:
        parser.error("collect requires --stage train, validation, or test")

    try:
        report = run_stage(config, stage, allow_execution=True)
    except Exception as exc:
        print(f"COLLECT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(NO_MOTION_MESSAGE)
        return 1

    print(json.dumps(report, indent=2))
    if stage == "pilot":
        print("PILOT QUALITY:", "PASS" if report.get("quality_passed") else "FAIL")
        return 0 if report.get("quality_passed") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
