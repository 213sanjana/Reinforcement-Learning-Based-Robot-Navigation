"""Build an offline navigation report from recorded trial JSON.

This module reads files that already exist. It does not launch Gazebo, load a
policy, publish a command, or write into a historical experiment directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation.warehouse_metrics import aggregate_episodes


class NavigationReportError(ValueError):
    """A recorded trial directory cannot produce a consistent report."""


REPORT_VERSION = 1
RECORDED = "RECORDED"
NOT_RECORDED = "NOT_RECORDED"

TASK_FAMILIES = (
    "industrial_route_14",
    "industrial_short_routes",
    "small_warehouse_roam",
    "cafe_roam",
    "cafe_nav2_goal",
    "gridworld_manifest",
)

_FAMILY_REASONS = {
    "industrial_short_routes": (
        "This report only reads the 14-stop spawn-route trials. "
        "Short-route results are a separate task and are not copied here."
    ),
    "small_warehouse_roam": (
        "No small-warehouse roam record was an input. "
        "No small-warehouse performance figure is reported."
    ),
    "cafe_roam": (
        "No cafe PPO roam record was an input. "
        "No cafe roam performance figure is reported."
    ),
    "cafe_nav2_goal": (
        "Nav2 goal sending does not write a metric file in this repository. "
        "No Nav2 performance figure is reported."
    ),
    "gridworld_manifest": (
        "GridWorld manifest results use a different protocol and were not inputs. "
        "No grid score is copied into this report."
    ),
}

_AGGREGATE_KEYS = (
    "episodes",
    "successes",
    "collisions",
    "timeouts",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_path_length_m",
    "mean_duration_s",
    "mean_time_to_goal_s",
    "mean_linear_speed_mps",
    "mean_angular_speed_radps",
    "mean_min_clearance_m",
    "mean_rotations",
    "mean_terminal_goal_distance_m",
    "mean_episode_length",
    "mean_return",
)

_SUMMARY_TRIAL_KEYS = (
    "trial",
    "success",
    "collision",
    "timeout",
    "goal_distance",
    "termination_reason",
    "checkpoints_reached",
    "pose_xy",
)

_UNAVAILABLE = {
    "mean_time_to_goal_s": (
        "Trial files store duration_s but not time_to_goal_s. "
        "The historical summary leaves this mean null, and duration_s is not substituted for it."
    ),
    "mean_linear_speed_mps": "Trial files do not store mean_linear_speed_mps.",
    "mean_angular_speed_radps": "Trial files do not store mean_angular_speed_radps.",
    "mean_rotations": "Trial files do not store rotations.",
    "mean_return": "Trial files do not store episode_return.",
    "safety_rejection_count": (
        "The route driver can recover from an in-loop safety rejection, but it does not "
        "write that recovery count. Only a terminal reason of safety_rejected is stored."
    ),
    "collision_event_count": "collision is an episode flag, not a count of contact events.",
    "path_efficiency": "Straight-line distance is not stored, so path efficiency is not computed.",
    "goal_tolerance_m": (
        "Goal tolerance is not stored in the trial files, summary.json, or checkpoint metadata.json."
    ),
    "configuration": (
        "No config path is stored in the trial files, summary.json, or checkpoint metadata.json."
    ),
    "execution_mode": (
        "The trial files and summary.json do not store the evaluation execution mode. "
        "Checkpoint metadata execution describes the training run and is not copied as the evaluation mode."
    ),
}


def build_industrial_route_report(trial_dir: Path, repo_root: Path | None = None) -> dict:
    """Validate spawn-route trials against summary.json and return a report."""

    trial_dir = Path(trial_dir)
    root = Path.cwd() if repo_root is None else Path(repo_root)
    summary, records = _load_records(trial_dir)
    _validate(summary, records)
    metadata_path, metadata = _checkpoint_metadata(summary.get("checkpoint"), root)
    families = {
        "industrial_route_14": _recorded_family(
            trial_dir, summary, records, metadata, metadata_path, root
        )
    }
    for name in TASK_FAMILIES:
        if name != "industrial_route_14":
            families[name] = _unrecorded_family(name)
    return {
        "report_version": REPORT_VERSION,
        "live_execution": False,
        "active_family": "industrial_route_14",
        "source_directory": _display(trial_dir, root),
        "comparison": _comparison(),
        "task_families": families,
    }


def write_report(report: dict, output_dir: Path, *, source_dir: Path, repo_root: Path) -> Path:
    """Write report.json into a new directory outside the historical records."""

    output_dir = Path(output_dir)
    _assert_output_allowed(output_dir, Path(source_dir), Path(repo_root))
    output_dir.mkdir(parents=True)
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _load_records(trial_dir: Path) -> tuple[dict, list[dict]]:
    summary_path = trial_dir / "summary.json"
    if not summary_path.is_file():
        raise NavigationReportError(f"Missing summary.json in {trial_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise NavigationReportError("summary.json must be an object")
    missing = [key for key in _AGGREGATE_KEYS if key not in summary]
    if missing:
        raise NavigationReportError("summary.json is missing " + ", ".join(missing))
    paths = sorted(trial_dir.glob("trial_*.json"))
    if not paths:
        raise NavigationReportError(f"No trial_*.json files in {trial_dir}")
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise NavigationReportError(f"{path.name} must be an object")
        payload["source_file"] = path.name
        records.append(payload)
    records.sort(key=lambda record: int(record["trial"]))
    return summary, records


def _validate(summary: dict, records: list[dict]) -> None:
    episodes = summary["episodes"]
    if episodes != len(records):
        raise NavigationReportError(
            f"summary episodes {episodes} does not match {len(records)} trial files"
        )
    if summary.get("completed_trials") != len(records):
        raise NavigationReportError("completed_trials does not match the trial files")
    requested = summary.get("requested_trials")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < len(records):
        raise NavigationReportError("requested_trials is smaller than the number of trial files")
    expected_ids = list(summary.get("checkpoints") or [])
    if not expected_ids:
        raise NavigationReportError("summary.json has no checkpoint list")
    numbers = [record.get("trial") for record in records]
    if numbers != list(range(1, len(records) + 1)):
        raise NavigationReportError("trial numbers must be 1..N in filename order")
    for record in records:
        filename = record["source_file"]
        if filename != f"trial_{int(record['trial']):02d}.json":
            raise NavigationReportError(f"{filename} does not match trial number {record['trial']}")
        if list(record.get("checkpoints_expected") or []) != expected_ids:
            raise NavigationReportError(f"{filename} checkpoints_expected does not match the summary")
        reached = list(record.get("checkpoints_reached") or [])
        if reached != expected_ids[: len(reached)]:
            raise NavigationReportError(
                f"{filename} checkpoints_reached is not an ordered prefix of the route"
            )
    recomputed = aggregate_episodes(records)
    for key in _AGGREGATE_KEYS:
        if not _equal(summary[key], recomputed[key]):
            raise NavigationReportError(
                f"summary.json {key} does not match the trial files: "
                f"summary={summary[key]!r} recomputed={recomputed[key]!r}"
            )
    rows = summary.get("trials")
    if not isinstance(rows, list) or len(rows) != len(records):
        raise NavigationReportError("summary trials does not match the trial files")
    for row, record in zip(rows, records):
        for key in _SUMMARY_TRIAL_KEYS:
            if key not in row or not _equal(row[key], record.get(key)):
                raise NavigationReportError(
                    f"summary trial {record.get('trial')} {key} does not match {record['source_file']}"
                )


def _recorded_family(trial_dir, summary, records, metadata, metadata_path, root: Path) -> dict:
    expected = list(summary["checkpoints"])
    reached_counts = [len(record.get("checkpoints_reached") or []) for record in records]
    complete_names_failed = [
        record
        for record in records
        if list(record.get("checkpoints_reached") or []) == expected and not record.get("success")
    ]
    safety_rejected = [
        record["trial"]
        for record in records
        if record.get("termination_reason") == "safety_rejected"
    ]
    metrics = {key: summary[key] for key in _AGGREGATE_KEYS}
    metrics.update(
        {
            "checkpoints_expected": len(expected),
            "checkpoints_reached_per_trial": reached_counts,
            "trials_with_all_checkpoint_names": sum(count == len(expected) for count in reached_counts),
            "safety_rejected_episodes": len(safety_rejected),
            "safety_rejection_count": None,
            "collision_event_count": None,
            "path_efficiency": None,
        }
    )
    if complete_names_failed:
        described = ", ".join(
            f"{record['trial']} ({record.get('termination_reason')})" for record in complete_names_failed
        )
        name_note = (
            f"Trials {described} list every checkpoint name, but success is false. "
            "The report does not recompute success."
        )
    else:
        name_note = "No trial combines a complete checkpoint-name list with success false."
    if safety_rejected:
        safety_note = (
            "Terminal safety_rejected episodes: "
            + ", ".join(str(trial) for trial in safety_rejected)
            + ". This is not a count of rejected commands."
        )
    else:
        safety_note = "No trial ended with termination_reason safety_rejected."
    return {
        "status": RECORDED,
        "identity": _identity(trial_dir, summary, records, metadata, metadata_path, root),
        "validation": {
            "passed": True,
            "trial_files": len(records),
            "aggregate_matches_summary": True,
            "summary_rows_match_trial_files": True,
            "checkpoint_lists_match": True,
            "reached_checkpoints_are_ordered_prefixes": True,
        },
        "metrics": metrics,
        "unavailable": dict(_UNAVAILABLE),
        "notes": [
            "Success is the stored success flag. A complete checkpoint-name list is not success by itself.",
            name_note,
            safety_note,
        ],
        "trials": [_trial_view(record, len(expected)) for record in records],
    }


def _trial_view(record: dict, expected_count: int) -> dict:
    reached = list(record.get("checkpoints_reached") or [])
    return {
        "trial": record.get("trial"),
        "seed": record.get("seed"),
        "spawn": record.get("spawn"),
        "success": record.get("success"),
        "collision": record.get("collision"),
        "timeout": record.get("timeout"),
        "goal_distance": record.get("goal_distance"),
        "path_length_m": record.get("path_length_m"),
        "duration_s": record.get("duration_s"),
        "min_clearance_m": record.get("min_clearance_m"),
        "episode_length": record.get("episode_length"),
        "termination_reason": record.get("termination_reason"),
        "checkpoints_reached_count": len(reached),
        "checkpoints_expected_count": expected_count,
        "checkpoints_reached": reached,
        "pose_xy": record.get("pose_xy"),
        "route_start": record.get("route_start"),
        "destination": record.get("destination"),
        "source_file": record.get("source_file"),
    }


def _identity(trial_dir, summary, records, metadata, metadata_path, root: Path) -> dict:
    metadata = metadata or {}
    ppo = metadata.get("ppo") if isinstance(metadata.get("ppo"), dict) else {}
    policy = ppo.get("policy")
    metadata_found = bool(metadata)
    world_reason = None
    controller_reason = None
    if not metadata_found:
        world_reason = "Checkpoint metadata.json was not found, and the trial files do not store the world."
        controller_reason = "Checkpoint metadata.json was not found, and the trial files do not name a controller."
    elif not policy:
        controller_reason = "Checkpoint metadata.json has no ppo.policy, and the trial files do not name a controller."
    return {
        "world": _field(
            metadata.get("world_name") if metadata_found else None,
            "checkpoint metadata.json world_name" if metadata_found else None,
            world_reason,
            "The trial files do not store the world. This value is the checkpoint training record named by summary.json."
            if metadata_found
            else None,
        ),
        "controller": _field(
            "ppo" if policy else None,
            "checkpoint metadata.json ppo.policy" if policy else None,
            controller_reason,
            f"Checkpoint policy class is {policy}." if policy else None,
        ),
        "checkpoint": _field(summary.get("checkpoint"), "summary.json checkpoint", None, None),
        "configuration": _field(None, None, _UNAVAILABLE["configuration"], None),
        "seed": {
            "value": [record.get("seed") for record in records],
            "source": "trial JSON seed fields",
            "checkpoint_training_seed": _field(
                metadata.get("seed") if metadata_found else None,
                "checkpoint metadata.json seed" if metadata_found else None,
                None if metadata_found else "Checkpoint metadata.json was not found.",
                "Training seed from the checkpoint record, not the per-trial evaluation seeds."
                if metadata_found
                else None,
            ),
        },
        "route": {
            "id": trial_dir.name,
            "id_source": "trial directory name",
            "spawn": summary.get("spawn"),
            "spawn_source": "summary.json spawn",
            "checkpoints": list(summary.get("checkpoints") or []),
            "checkpoints_source": "summary.json checkpoints",
        },
        "goal_tolerance_m": _field(None, None, _UNAVAILABLE["goal_tolerance_m"], None),
        "execution_mode": _field(
            None,
            None,
            _UNAVAILABLE["execution_mode"],
            "checkpoint metadata.json execution is stored only as checkpoint_record.training_execution.",
        ),
        "checkpoint_record": _checkpoint_record(summary, metadata, metadata_path, root),
    }


def _checkpoint_record(summary, metadata, metadata_path, root: Path) -> dict:
    if not metadata:
        return {
            "path": summary.get("checkpoint"),
            "metadata_present": False,
            "reason": "Checkpoint metadata.json was not found. The policy zip was not opened.",
        }
    versions = metadata.get("versions") if isinstance(metadata.get("versions"), dict) else {}
    ppo = metadata.get("ppo") if isinstance(metadata.get("ppo"), dict) else {}
    return {
        "path": summary.get("checkpoint"),
        "metadata_path": _display(metadata_path, root) if metadata_path is not None else None,
        "metadata_present": True,
        "weights_loaded": False,
        "world_name": metadata.get("world_name"),
        "policy": ppo.get("policy"),
        "training_seed": metadata.get("seed"),
        "training_execution": metadata.get("execution"),
        "observation_interface": metadata.get("observation_interface"),
        "action_interface": metadata.get("action_interface"),
        "stage_id": metadata.get("stage_id"),
        "git_head": versions.get("git_head"),
        "note": (
            "These fields describe the saved checkpoint named by summary.json. "
            "training_execution is not the evaluation execution mode. The policy zip was not opened."
        ),
    }


def _checkpoint_metadata(checkpoint, root: Path) -> tuple[Path | None, dict | None]:
    if not isinstance(checkpoint, str) or not checkpoint:
        return None, None
    directory = Path(checkpoint)
    if not directory.is_absolute():
        directory = root / directory
    path = directory / "metadata.json"
    if not path.is_file():
        return path, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NavigationReportError("checkpoint metadata.json must be an object")
    return path, payload


def _unrecorded_family(name: str) -> dict:
    return {
        "status": NOT_RECORDED,
        "reason": _FAMILY_REASONS[name],
        "metrics": None,
    }


def _comparison() -> dict:
    return {
        "task_family": "industrial_route_14",
        "methods": {
            "ppo": RECORDED,
            "nav2": NOT_RECORDED,
            "astar": NOT_RECORDED,
        },
        "ranking": None,
        "reason": (
            "Only PPO has a recorded result for industrial_route_14. "
            "Nav2 and A* are NOT_RECORDED and are not ranked. "
            "Other task families are not compared with this route."
        ),
    }


def _field(value, source, reason, note):
    payload = {"value": value}
    if source:
        payload["source"] = source
    if value is None and reason:
        payload["reason"] = reason
    if note:
        payload["note"] = note
    return payload


def _assert_output_allowed(output_dir: Path, source_dir: Path, repo_root: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output = output_dir.resolve()
    source = source_dir.resolve()
    historical = (repo_root / "artifacts" / "warehouse_ppo").resolve()
    if output == source or _is_inside(output, source) or _is_inside(source, output):
        raise NavigationReportError(f"Refusing to write into the recorded trial directory {source_dir}")
    if output == historical or _is_inside(output, historical):
        raise NavigationReportError(f"Refusing to write under {historical}")


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _equal(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-9
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline report from recorded warehouse trial JSON. "
            "Does not launch Gazebo, load a policy, or overwrite historical records."
        ),
    )
    parser.add_argument("--trials", required=True, help="Directory containing summary.json and trial_*.json")
    parser.add_argument("--output", required=True, help="New directory for report.json")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.repo_root)
    source = Path(args.trials)
    report = build_industrial_route_report(source, repo_root=root)
    path = write_report(report, Path(args.output), source_dir=source, repo_root=root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
