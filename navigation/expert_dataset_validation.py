"""Validate collected Jackal expert demonstrations.

The validator loads every recorded split, checks the 40D observation and
2D action contract, and writes a machine-readable report. A failed
required check rejects the dataset. Preview manifests with no episodes
are included in the split-leakage check and are not counted as successes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from navigation.expert_demonstration_recorder import load_dataset
from navigation.expert_evaluation_plan import OBSTACLE_BOX


REPORT_VERSION = "6e.3.0"
GOAL_FEATURE_INDEX = 36
LIDAR_BINS = 36
GOAL_REACHED_LIMIT_M = 0.50
SPLITS = ("pilot", "train", "validation", "test")


def validate_collection(dataset_root, output_dir):
    """Load the collection, write plots and a report, and return the report."""

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = _load_manifests(dataset_root)
    recorded, load_errors = _load_recorded(dataset_root, manifests)
    schema = _load_schema(dataset_root, recorded)
    goal_scale = float(
        schema.get("observation_normalization", {}).get("goal_max_distance_m", 20.0)
    )
    action_low, action_high = _action_bounds(schema)

    checks = []
    findings = {
        "unusually_long": [],
        "incomplete": [],
        "duplicates": [],
    }

    if load_errors:
        checks.append(_check("dataset_load", False, load_errors))
    else:
        checks.append(_check("dataset_load", True, ["Recorded datasets loaded"]))

    episodes = [
        (split, episode)
        for split, rows in recorded.items()
        for episode in rows
    ]
    checks.extend(
        _episode_checks(episodes, manifests, action_low, action_high, goal_scale, findings)
    )
    checks.append(_split_check(manifests, recorded))
    checks.append(_duplicate_check(episodes, findings))

    plots = _write_plots(
        output_dir,
        episodes,
        manifests,
        goal_scale,
    )
    summary = _summary(episodes)
    passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": REPORT_VERSION,
        "dataset_root": str(dataset_root),
        "validation_passed": passed,
        "datasets_loaded": [
            split for split, rows in recorded.items() if rows
        ],
        "datasets_without_episodes": [
            split for split in SPLITS
            if split in manifests and not recorded.get(split)
        ],
        "observation_dimension": 40,
        "action_dimension": 2,
        "action_bounds": [action_low, action_high],
        "episodes": summary,
        "checks": checks,
        "findings": findings,
        "plots": [str(path) for path in plots],
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _episode_checks(episodes, manifests, action_low, action_high, goal_scale, findings):
    dimension_errors = []
    finite_errors = []
    bound_errors = []
    boundary_errors = []
    timestamp_errors = []
    sync_errors = []
    seen_ids = {}

    for split, episode in episodes:
        label = f"{split}/{episode['scenario_id']}"
        observations = np.asarray(episode["observations"])
        actions = np.asarray(episode["actions"])
        requested = np.asarray(episode["requested_actions"])
        velocities = np.asarray(episode["applied_velocities"])
        poses = np.asarray(episode["poses"])
        timestamps = np.asarray(episode["simulation_timestamps_ns"])
        count = int(observations.shape[0]) if observations.ndim == 2 else -1

        if observations.shape != (count, 40) or count < 0:
            dimension_errors.append(f"{label} observations are not [T, 40]")
        if actions.ndim != 2 or actions.shape[1] != 2:
            dimension_errors.append(f"{label} actions are not [T, 2]")
        if requested.ndim != 2 or requested.shape[1] != 2:
            dimension_errors.append(f"{label} requested actions are not [T, 2]")

        arrays = {
            "observations": observations,
            "actions": actions,
            "requested_actions": requested,
            "applied_velocities": velocities,
            "poses": poses,
        }
        for name, value in arrays.items():
            if value.size and not np.all(np.isfinite(value)):
                finite_errors.append(f"{label} {name} contains nonfinite values")

        if actions.size and (
            np.any(actions < action_low) or np.any(actions > action_high)
        ):
            bound_errors.append(f"{label} actions are outside [{action_low}, {action_high}]")
        if requested.size and (
            np.any(requested < action_low) or np.any(requested > action_high)
        ):
            bound_errors.append(
                f"{label} requested actions are outside [{action_low}, {action_high}]"
            )

        lengths = [
            int(observations.shape[0]) if observations.ndim == 2 else -1,
            int(actions.shape[0]) if actions.ndim == 2 else -1,
            int(requested.shape[0]) if requested.ndim == 2 else -1,
            int(velocities.shape[0]) if velocities.ndim == 2 else -1,
            int(poses.shape[0]) if poses.ndim == 2 else -1,
            int(timestamps.shape[0]) if timestamps.ndim == 1 else -1,
        ]
        if len(set(lengths)) != 1:
            sync_errors.append(f"{label} observation-action rows are not aligned: {lengths}")

        if timestamps.ndim != 1 or np.any(timestamps < 0):
            timestamp_errors.append(f"{label} has invalid simulation timestamps")
        elif timestamps.size > 1 and np.any(np.diff(timestamps.astype(np.int64)) < 0):
            timestamp_errors.append(f"{label} simulation timestamps are not monotonic")

        identity = (split, int(episode["episode_id"]))
        if identity in seen_ids:
            boundary_errors.append(f"{label} repeats episode id {identity[1]}")
        seen_ids[identity] = label

        scenario = _scenario(manifests, split, episode["scenario_id"])
        if scenario is None:
            boundary_errors.append(f"{label} is not in the {split} manifest")
        elif episode["scenario_id"] in _foreign_ids(manifests, split):
            boundary_errors.append(f"{label} also appears in another split")
        else:
            _length_and_completeness(
                label,
                episode,
                scenario,
                goal_scale,
                findings,
            )

    action_errors = [
        message for message in dimension_errors if "action" in message
    ] + bound_errors
    observation_errors = [
        message for message in dimension_errors if "action" not in message
    ]
    return [
        _check("observation_dimension", not observation_errors, observation_errors or ["40"]),
        _check(
            "action_dimension_and_bounds",
            not action_errors,
            action_errors or ["actions are 2D and inside the Gymnasium bounds"],
        ),
        _check("finite_values", not finite_errors, finite_errors or ["all numeric arrays are finite"]),
        _check("episode_boundaries", not boundary_errors, boundary_errors or ["episode ids and scenario ids stay inside their split"]),
        _check("monotonic_timestamps", not timestamp_errors, timestamp_errors or ["timestamps are monotonic within each episode"]),
        _check("observation_action_synchronization", not sync_errors, sync_errors or ["every row has one observation, action, velocity, pose, and timestamp"]),
    ]


def _length_and_completeness(label, episode, scenario, goal_scale, findings):
    attempts = episode.get("navigation_time_s")
    duration = float(scenario["duration_s"])
    if attempts is not None and float(attempts) > duration + 1.0:
        findings["unusually_long"].append(
            f"{label} navigated for {float(attempts):.1f}s with a {duration:.0f}s budget"
        )

    if not episode["success"]:
        return

    observations = np.asarray(episode["observations"])
    if observations.shape[0] < 1:
        findings["incomplete"].append(f"{label} is marked successful with no transitions")
        return

    final_distance = float(observations[-1, GOAL_FEATURE_INDEX]) * goal_scale
    if final_distance > GOAL_REACHED_LIMIT_M:
        findings["incomplete"].append(
            f"{label} is marked successful {final_distance:.2f} m from the goal"
        )


def _split_check(manifests, recorded):
    errors = []
    seen = {}
    for split, manifest in manifests.items():
        for scenario in manifest["scenarios"]:
            fingerprint = scenario["fingerprint"]
            previous = seen.get(fingerprint)
            if previous is not None:
                errors.append(
                    f"fingerprint {fingerprint} is in {previous} and {split}"
                )
            seen[fingerprint] = split

    for split, episodes in recorded.items():
        own = {item["scenario_id"] for item in manifests.get(split, {}).get("scenarios", [])}
        for episode in episodes:
            if episode["scenario_id"] not in own:
                errors.append(
                    f"{split}/{episode['scenario_id']} is outside its split manifest"
                )

    return _check(
        "scenario_split_separation",
        not errors,
        errors or ["scenario fingerprints are disjoint across splits"],
    )


def _duplicate_check(episodes, findings):
    seen = {}
    for split, episode in episodes:
        if int(np.asarray(episode["observations"]).shape[0]) == 0:
            continue
        digest = _trajectory_digest(episode)
        label = f"{split}/{episode['scenario_id']}"
        previous = seen.get(digest)
        if previous is not None:
            message = f"{label} repeats the trajectory in {previous}"
            findings["duplicates"].append(message)
        else:
            seen[digest] = label

    errors = list(findings["duplicates"])
    errors.extend(findings["incomplete"])
    errors.extend(findings["unusually_long"])
    return _check(
        "duplicate_and_episode_quality",
        not errors,
        errors or ["no duplicate, incomplete, or over-budget trajectories"],
    )


def _write_plots(output_dir, episodes, manifests, goal_scale):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    rows = [episode for _split, episode in episodes if np.asarray(episode["observations"]).size]
    plots = []
    if not rows:
        return plots

    actions = np.concatenate([np.asarray(episode["actions"]) for episode in rows], axis=0)
    observations = np.concatenate(
        [np.asarray(episode["observations"]) for episode in rows],
        axis=0,
    )
    lidar = observations[:, :LIDAR_BINS]
    distances = observations[:, GOAL_FEATURE_INDEX] * goal_scale
    finals = [
        float(np.asarray(episode["observations"])[-1, GOAL_FEATURE_INDEX]) * goal_scale
        for episode in rows
        if np.asarray(episode["observations"]).shape[0]
    ]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(actions[:, 0], bins=20, color="#2463a8")
    axes[0].set_title("Linear action")
    axes[0].set_xlabel("action")
    axes[1].hist(actions[:, 1], bins=20, color="#21a366")
    axes[1].set_title("Angular action")
    axes[1].set_xlabel("action")
    for axis in axes:
        axis.set_ylabel("transitions")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    plots.append(_save(figure, output_dir / "action_distribution.png"))

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(lidar.reshape(-1), bins=20, color="#2463a8")
    axes[0].set_title("LiDAR features")
    axes[0].set_xlabel("normalized range")
    axes[0].set_ylabel("bins")
    axes[1].plot(np.mean(lidar, axis=0), color="#21a366")
    axes[1].set_title("Mean LiDAR bin")
    axes[1].set_xlabel("bin")
    axes[1].set_ylabel("normalized range")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    plots.append(_save(figure, output_dir / "lidar_distribution.png"))

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(distances, bins=20, color="#2463a8")
    axes[0].set_title("Goal distance during episodes")
    axes[0].set_xlabel("meters")
    axes[1].hist(finals, bins=10, color="#e49322")
    axes[1].set_title("Final goal distance")
    axes[1].set_xlabel("meters")
    for axis in axes:
        axis.set_ylabel("count")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    plots.append(_save(figure, output_dir / "goal_distance_distribution.png"))

    samples = _sample_episodes(episodes, manifests)
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    center_x, center_y, size_x, size_y = OBSTACLE_BOX
    for axis, (split, episode) in zip(axes.reshape(-1), samples):
        scenario = _scenario(manifests, split, episode["scenario_id"]) or {}
        path = scenario.get("planned_path") or []
        axis.add_patch(Rectangle(
            (center_x - size_x / 2, center_y - size_y / 2),
            size_x,
            size_y,
            facecolor="#b33",
            alpha=0.35,
        ))
        if len(path) >= 2:
            axis.plot(
                [point[0] for point in path],
                [point[1] for point in path],
                color="#2463a8",
                linewidth=2,
            )
        start = scenario.get("start", [None, None])
        goal = episode["goal"]
        if start[0] is not None:
            axis.scatter([start[0]], [start[1]], c="#e49322", marker="s", s=40)
        axis.scatter([goal[0]], [goal[1]], c="#111111", marker="*", s=80)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.3)
        axis.set_title(
            f"{episode['scenario_id']}: {episode['termination_reason']}",
            fontsize=9,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.tight_layout()
    plots.append(_save(figure, output_dir / "sample_trajectories.png"))
    plt.close("all")
    return plots


def _sample_episodes(episodes, manifests):
    chosen = []
    wanted = (
        ("open", "short"),
        ("open", "medium"),
        ("obstacle_rich", "medium"),
        ("obstacle_rich", "long"),
    )
    used = set()
    for area, length in wanted:
        for split, episode in episodes:
            scenario = _scenario(manifests, split, episode["scenario_id"]) or {}
            if scenario.get("area_class") == area and scenario.get("length_class") == length:
                chosen.append((split, episode))
                used.add(episode["scenario_id"])
                break
    for split, episode in episodes:
        if not episode["success"] and episode["scenario_id"] not in used:
            if len(chosen) == 4:
                chosen[-1] = (split, episode)
            else:
                chosen.append((split, episode))
            break
    while len(chosen) < 4 and episodes:
        chosen.append(episodes[len(chosen) % len(episodes)])
    return chosen[:4]


def _summary(episodes):
    successes = 0
    failures = 0
    by_status = {}
    for _split, episode in episodes:
        status = episode["termination_reason"]
        by_status[status] = by_status.get(status, 0) + 1
        if episode["success"]:
            successes += 1
        else:
            failures += 1
    return {
        "recorded": len(episodes),
        "successes": successes,
        "failures": failures,
        "by_status": by_status,
    }


def _load_manifests(dataset_root):
    manifest_dir = dataset_root / "manifests"
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Missing manifests directory: {manifest_dir}")
    manifests = {}
    for name in SPLITS:
        path = manifest_dir / f"{name}.json"
        if path.is_file():
            manifests[name] = json.loads(path.read_text(encoding="utf-8"))
    if not manifests:
        raise FileNotFoundError("No split manifests found")
    return manifests


def _load_recorded(dataset_root, manifests):
    recorded = {}
    errors = []
    for split in manifests:
        episode_dir = dataset_root / split
        manifest_path = episode_dir / "manifest.json"
        recorded[split] = []
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not payload.get("episodes"):
            continue
        try:
            episodes = load_dataset(episode_dir)
        except Exception as exc:
            errors.append(f"{split}: {type(exc).__name__}: {exc}")
            continue
        attempts = {
            row["scenario_id"]: row
            for row in _read_jsonl(episode_dir / "attempts.jsonl")
        }
        for episode in episodes:
            logged = attempts.get(episode["scenario_id"], {})
            episode["navigation_time_s"] = logged.get("navigation_time_s")
        recorded[split] = episodes
    return recorded, errors


def _load_schema(dataset_root, recorded):
    for split, episodes in recorded.items():
        if not episodes:
            continue
        path = dataset_root / split / "schema.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _action_bounds(schema):
    space = schema.get("action_space", {})
    low = space.get("low", [-1.0, -1.0])
    high = space.get("high", [1.0, 1.0])
    return float(low[0]), float(high[0])


def _scenario(manifests, split, scenario_id):
    for scenario in manifests.get(split, {}).get("scenarios", []):
        if scenario["scenario_id"] == scenario_id:
            return scenario
    return None


def _foreign_ids(manifests, split):
    foreign = set()
    for name, manifest in manifests.items():
        if name == split:
            continue
        foreign.update(item["scenario_id"] for item in manifest["scenarios"])
    return foreign


def _trajectory_digest(episode):
    hasher = hashlib.sha256()
    for key in ("observations", "actions"):
        array = np.ascontiguousarray(episode[key])
        hasher.update(key.encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _check(name, passed, messages):
    return {
        "name": name,
        "passed": bool(passed),
        "messages": list(messages),
    }


def _save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    figure.clf()
    return path


def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate collected Jackal expert demonstration datasets."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = validate_collection(args.dataset, args.output)
    except Exception as exc:
        print(f"VALIDATION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "validation_passed": report["validation_passed"],
            "recorded": report["episodes"]["recorded"],
            "successes": report["episodes"]["successes"],
            "failures": report["episodes"]["failures"],
            "datasets_loaded": report["datasets_loaded"],
            "datasets_without_episodes": report["datasets_without_episodes"],
            "report": report["report_path"],
        },
        indent=2,
    ))
    print("VALIDATION:", "PASS" if report["validation_passed"] else "FAIL")
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
