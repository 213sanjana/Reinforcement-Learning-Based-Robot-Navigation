"""Run the saved PPO policy on several warehouse routes.

The routes use the existing industrial-warehouse obstacles. They differ in
start, destination, and which aisle they follow. This module does not train
and does not move the warehouse meshes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.warehouse_checkpoints import clearance_m
from evaluation.warehouse_metrics import aggregate_episodes
from evaluation.warehouse_navigate import navigate_route
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_estop import live_authorized

# Each goal is (x, y, id). Starts are (x, y, yaw) in the warehouse frame.
WAREHOUSE_ROUTES = (
    {
        "id": "familiar_east",
        "aisle": "open floor west of the east shelves",
        "obstacles": "east shelf faces",
        "start": (0.0, 0.0, 0.0),
        "goals": ((1.5, 0.0, "east_goal"),),
    },
    {
        "id": "south_shelf_aisle",
        "aisle": "open floor along the south shelf faces",
        "obstacles": "south shelf row",
        "start": (0.0, -3.0, 0.0),
        "goals": (
            (1.5, -1.2, "shelf_row"),
            (1.5, -4.0, "south_gap"),
            (1.5, -4.8, "south_shelf"),
        ),
    },
    {
        "id": "narrow_shelf_box_bay",
        "aisle": "1.90 m bay between the east shelf and the box row",
        "obstacles": "shelf face and facing box row",
        "start": (1.5, 1.97, 0.0),
        "goals": (
            (3.3, 1.97, "bay_west"),
            (4.5, 1.97, "bay_middle"),
            (5.7, 1.97, "bay_east"),
        ),
    },
    {
        "id": "north_box_lane",
        "aisle": "open lane east of the north box stacks",
        "obstacles": "north box rows and green dustbin",
        "start": (1.2, 3.2, 0.0),
        "goals": (
            (1.2, 4.6, "toward_dustbin"),
            (1.2, 5.6, "dustbin_approach"),
            (1.2, 6.6, "green_dustbin"),
        ),
    },
)


def route_points(route: dict):
    start = route["start"]
    points = [(float(start[0]), float(start[1]))]
    points.extend((float(goal[0]), float(goal[1])) for goal in route["goals"])
    return points


def assert_routes_clear(minimum_m: float = 0.90) -> None:
    """Reject a route point that sits inside the recorded obstacle clearance."""

    starts = []
    destinations = []
    for route in WAREHOUSE_ROUTES:
        for x, y in route_points(route):
            clearance = clearance_m(x, y)
            if clearance < minimum_m:
                raise ValueError(f"{route['id']} point ({x}, {y}) clearance is {clearance:.2f} m")
        starts.append(route["start"][:2])
        last = route["goals"][-1]
        destinations.append((last[0], last[1]))
    if len(set(starts)) != len(WAREHOUSE_ROUTES):
        raise ValueError("Each route needs its own start")
    if len(set(destinations)) != len(WAREHOUSE_ROUTES):
        raise ValueError("Each route needs its own destination")


def _goals(route: dict):
    return [((float(goal[0]), float(goal[1])), str(goal[2])) for goal in route["goals"]]


def route_record(route: dict, result: dict, *, seed: int) -> dict:
    reached = list(result.get("checkpoints_reached") or [])
    return {
        "route_id": route["id"],
        "aisle": route["aisle"],
        "obstacles": route["obstacles"],
        "seed": seed,
        "start": [float(value) for value in route["start"]],
        "destination": [float(route["goals"][-1][0]), float(route["goals"][-1][1])],
        "checkpoints_expected": [str(goal[2]) for goal in route["goals"]],
        "checkpoints_reached": reached,
        "success": bool(result.get("success")),
        "collision": bool(result.get("collision")),
        "timeout": bool(result.get("timeout")) or result.get("termination_reason") == "time_limit",
        "goal_distance": result.get("goal_distance"),
        "termination_reason": result.get("termination_reason"),
        "path_length_m": result.get("path_length_m"),
        "duration_s": result.get("duration_s"),
        "min_clearance_m": result.get("min_clearance_m"),
        "episode_length": result.get("episode_length"),
        "pose_xy": result.get("pose_xy"),
    }


def run_routes(
    config: dict,
    checkpoint: Path,
    output_dir: Path,
    *,
    route_id: str | None = None,
    seed: int = 1,
    authorize_live: bool = False,
    navigate=navigate_route,
) -> dict:
    """Drive each selected route once and write one JSON file per route."""

    assert_routes_clear()
    selected = [route for route in WAREHOUSE_ROUTES if route_id is None or route["id"] == route_id]
    if not selected:
        known = ", ".join(route["id"] for route in WAREHOUSE_ROUTES)
        raise ValueError(f"Unknown route {route_id}. Known routes: {known}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    records = []
    failure = None
    for index, route in enumerate(selected):
        try:
            result = navigate(
                config,
                _goals(route),
                checkpoint=Path(checkpoint),
                start=tuple(route["start"]),
                authorize_live=authorize_live,
                seed=seed + index,
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
        record = route_record(route, result, seed=seed + index)
        if isinstance(result, dict) and result.get("failure_message"):
            record["failure_message"] = result["failure_message"]
        _write_json(output_dir / f"{route['id']}.json", record)
        records.append(record)
        if failure is not None:
            break
    summary = aggregate_episodes(records)
    summary["routes"] = [
        {
            "route_id": record["route_id"],
            "aisle": record["aisle"],
            "obstacles": record["obstacles"],
            "start": record["start"],
            "destination": record["destination"],
            "success": record["success"],
            "collision": record["collision"],
            "timeout": record["timeout"],
            "goal_distance": record["goal_distance"],
            "termination_reason": record["termination_reason"],
            "checkpoints_reached": record["checkpoints_reached"],
        }
        for record in records
    ]
    summary["checkpoint"] = str(checkpoint)
    _write_json(output_dir / "summary.json", summary)
    if failure is not None:
        raise failure
    return summary


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the saved PPO policy on warehouse routes with different "
            "starts, destinations, and aisles. Does not train."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--route", choices=[route["id"] for route in WAREHOUSE_ROUTES])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    summary = run_routes(
        load_warehouse_config(args.config),
        Path(args.checkpoint),
        Path(args.output),
        route_id=args.route,
        seed=args.seed,
        authorize_live=live_authorized(args.authorize_live),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["successes"] == summary["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
