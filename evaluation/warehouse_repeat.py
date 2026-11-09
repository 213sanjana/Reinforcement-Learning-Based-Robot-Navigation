"""Repeat the full warehouse checkpoint route from the original spawn.

This records the saved policy. It does not train, and it does not change the
reward. Each trial teleports the Jackal to the warehouse spawn, then follows
every checkpoint in order. A robot already near the dustbin is not treated
as a completed route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.warehouse_checkpoints import WAREHOUSE_ROUTE
from evaluation.warehouse_metrics import aggregate_episodes
from evaluation.warehouse_navigate import navigate_route
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_estop import live_authorized

SPAWN = (0.0, 0.0, 0.0)
MEASURED_KEYS = (
    "success",
    "collision",
    "timeout",
    "goal_distance",
    "termination_reason",
    "checkpoints_reached",
    "path_length_m",
    "duration_s",
    "min_clearance_m",
    "episode_length",
    "pose_xy",
    "route_start",
    "destination",
)


def route_goals():
    return [((point["x"], point["y"]), point["id"]) for point in WAREHOUSE_ROUTE]


def trial_record(result: dict, *, trial: int, seed: int) -> dict:
    """Keep the fields needed to compare repeated routes."""

    record = {"trial": trial, "seed": seed, "spawn": [SPAWN[0], SPAWN[1], SPAWN[2]]}
    for key in MEASURED_KEYS:
        record[key] = result.get(key)
    record["success"] = bool(record["success"])
    record["collision"] = bool(record["collision"])
    record["timeout"] = bool(record["timeout"]) or record["termination_reason"] == "time_limit"
    reached = record["checkpoints_reached"] or []
    record["checkpoints_reached"] = list(reached)
    record["checkpoints_expected"] = [point["id"] for point in WAREHOUSE_ROUTE]
    return record


def summarize_trials(records: list[dict]) -> dict:
    """Aggregate completed trials. An empty list is an error."""

    summary = aggregate_episodes(records)
    summary["spawn"] = [SPAWN[0], SPAWN[1], SPAWN[2]]
    summary["checkpoints"] = [point["id"] for point in WAREHOUSE_ROUTE]
    summary["trials"] = [
        {
            "trial": record["trial"],
            "success": record["success"],
            "collision": record["collision"],
            "timeout": record["timeout"],
            "goal_distance": record["goal_distance"],
            "termination_reason": record["termination_reason"],
            "checkpoints_reached": record["checkpoints_reached"],
            "pose_xy": record["pose_xy"],
        }
        for record in records
    ]
    return summary


def run_spawn_trials(
    config: dict,
    checkpoint: Path,
    output_dir: Path,
    *,
    trials: int,
    seed: int = 1,
    authorize_live: bool = False,
    navigate=navigate_route,
) -> dict:
    """Drive the full route from spawn and write one JSON file per trial."""

    if trials < 1 or trials > 10:
        raise ValueError("Repeat the route from 1 to 10 times")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    records = []
    failure = None
    for index in range(trials):
        trial_seed = seed + index
        try:
            result = navigate(
                config,
                route_goals(),
                checkpoint=Path(checkpoint),
                start=SPAWN,
                authorize_live=authorize_live,
                seed=trial_seed,
                resume=False,
                world_route=True,
            )
        except Exception as exc:
            result = {
                "success": False,
                "collision": False,
                "timeout": False,
                "termination_reason": type(exc).__name__,
                "goal_distance": None,
                "checkpoints_reached": [],
                "failure_message": str(exc),
            }
            failure = exc
        record = trial_record(result, trial=index + 1, seed=trial_seed)
        if isinstance(result, dict) and result.get("failure_message"):
            record["failure_message"] = result["failure_message"]
        _write_json(output_dir / f"trial_{index + 1:02d}.json", record)
        records.append(record)
        if failure is not None:
            break
    summary = summarize_trials(records)
    summary["checkpoint"] = str(checkpoint)
    summary["completed_trials"] = len(records)
    summary["requested_trials"] = trials
    _write_json(output_dir / "summary.json", summary)
    if failure is not None:
        raise failure
    return summary


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the full warehouse checkpoint route from the spawn pose. "
            "Records success, collisions, timeouts, and final goal distance. "
            "Does not train."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    summary = run_spawn_trials(
        load_warehouse_config(args.config),
        Path(args.checkpoint),
        Path(args.output),
        trials=args.trials,
        seed=args.seed,
        authorize_live=live_authorized(args.authorize_live),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["successes"] == summary["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
