"""Validated, non-overwriting export of expert navigation trajectories."""

import json
from pathlib import Path

import numpy as np


def export_expert_episode(result, output_dir, episode_id):
    """Save one completed expert episode as NPZ plus JSON metadata."""

    episode_id = int(episode_id)

    if episode_id < 0:
        raise ValueError("episode_id must be nonnegative")

    output_dir = Path(output_dir)
    transitions = list(result.transitions)

    if result.status != "GOAL_REACHED":
        raise ValueError(
            "Only GOAL_REACHED episodes can be exported as demonstrations"
        )

    if not transitions:
        raise ValueError("Cannot export an episode without transitions")

    if result.rejected_commands != 0:
        raise ValueError("Cannot export rejected-command episodes")

    if result.accepted_commands != len(transitions):
        raise ValueError("Accepted-command count does not match records")

    observations = np.stack(
        [np.asarray(t.observation, dtype=np.float32) for t in transitions]
    )

    requested_actions = np.stack(
        [np.asarray(t.action, dtype=np.float32) for t in transitions]
    )

    applied_actions = np.stack(
        [np.asarray(t.applied_action, dtype=np.float32) for t in transitions]
    )

    poses = np.asarray(
        [t.pose for t in transitions],
        dtype=np.float64,
    )

    velocities = np.asarray(
        [
            (t.published_linear, t.published_angular)
            for t in transitions
        ],
        dtype=np.float32,
    )

    timestamps = np.asarray(
        [t.simulation_time_ns for t in transitions],
        dtype=np.int64,
    )

    if observations.shape != (len(transitions), 40):
        raise ValueError("Invalid observation shape")

    if requested_actions.shape != (len(transitions), 2):
        raise ValueError("Invalid requested action shape")

    if applied_actions.shape != (len(transitions), 2):
        raise ValueError("Invalid applied action shape")

    if poses.shape != (len(transitions), 3):
        raise ValueError("Invalid pose shape")

    if velocities.shape != (len(transitions), 2):
        raise ValueError("Invalid velocity shape")

    for array in (
        observations,
        requested_actions,
        applied_actions,
        poses,
        velocities,
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError("Dataset contains nonfinite values")

    if np.any(np.abs(requested_actions) > 1.0):
        raise ValueError("Requested actions outside normalized range")

    if np.any(np.abs(applied_actions) > 1.0):
        raise ValueError("Applied actions outside normalized range")

    if np.any(timestamps < 0):
        raise ValueError("Invalid simulation timestamp")

    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Simulation timestamps are not monotonic")

    prefix = f"expert_episode_{episode_id:04d}"
    npz_path = output_dir / f"{prefix}.npz"
    json_path = output_dir / f"{prefix}.json"

    if npz_path.exists() or json_path.exists():
        raise FileExistsError(
            "Episode output already exists; refusing to overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "episode_id": episode_id,
        "status": result.status,
        "num_transitions": len(transitions),
        "accepted_commands": result.accepted_commands,
        "rejected_commands": result.rejected_commands,
        "observation_dim": 40,
        "action_dim": 2,
        "training_action_key": "applied_actions",
        "planned_path": [
            list(map(float, point))
            for point in result.planned_path
        ],
        "executed_poses": [
            list(map(float, pose))
            for pose in result.executed_poses
        ],
    }

    # Create both outputs exclusively and track ownership for cleanup.
    created_npz = False
    created_json = False

    try:
        with npz_path.open("xb") as handle:
            created_npz = True
            np.savez_compressed(
                handle,
                observations=observations,
                requested_actions=requested_actions,
                applied_actions=applied_actions,
                poses=poses,
                published_velocities=velocities,
                simulation_timestamps_ns=timestamps,
            )

        with json_path.open("x", encoding="utf-8") as handle:
            created_json = True
            json.dump(metadata, handle, indent=2)

    except BaseException:
        # Remove only files created by this export attempt.
        if created_json:
            json_path.unlink(missing_ok=True)

        if created_npz:
            npz_path.unlink(missing_ok=True)

        raise

    return npz_path, json_path
