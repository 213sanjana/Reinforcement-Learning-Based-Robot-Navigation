"""Safe CLI for expert demonstration collection.

Preview mode is read-only and does not initialize ROS or Gazebo.
Execute mode validates a side-effect-free plan, runs the existing
preflight checks, and only then authorizes the orchestrator.
A failed plan or preflight issues no robot commands and no resets.
"""

import sys

import argparse
import json
import math
from pathlib import Path


DEFAULT_WORLD = (
    Path.home()
    / "rl_robot_navigation"
    / "jackal_rl_ws"
    / "worlds"
    / "jackal_obstacle_test_collision.sdf"
)

DEFAULT_OUTPUT = Path("artifacts/imitation/expert_demonstrations")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("Value must be finite")
    return result


def positive_float(value):
    result = finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Jackal expert demonstration collector"
    )

    parser.add_argument(
        "--mode",
        choices=("preview", "preflight", "execute"),
        default="preview",
    )

    parser.add_argument(
        "--world",
        type=Path,
        default=DEFAULT_WORLD,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--start",
        nargs=3,
        type=finite_float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "YAW"),
    )

    parser.add_argument(
        "--goal",
        nargs=2,
        type=finite_float,
        default=(3.0, 0.0),
        metavar=("X", "Y"),
    )

    parser.add_argument(
        "--duration",
        type=positive_float,
        default=45.0,
    )

    parser.add_argument(
        "--inflation-radius",
        type=finite_float,
        default=0.95,
    )

    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    if args.inflation_radius < 0:
        parser.error("--inflation-radius cannot be negative")

    if not args.world.is_file():
        parser.error(f"World file not found: {args.world}")

    return args


def existing_episode_ids(output_dir):
    """Find IDs used by either dataset file extension."""

    output_dir = Path(output_dir)
    ids = set()

    for extension in ("npz", "json"):
        for path in output_dir.glob(
            f"expert_episode_*.{extension}"
        ):
            suffix = path.stem.removeprefix("expert_episode_")

            if suffix.isdigit():
                ids.add(int(suffix))

    return ids


def allocate_episode_ids(output_dir, count):
    """Choose unused episode IDs without modifying files."""

    used = existing_episode_ids(output_dir)
    selected = []
    candidate = 0

    while len(selected) < count:
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)

        candidate += 1

    return selected


def preview(args):
    from navigation.expert_collection_plan import build_collection_plan

    episode_ids = allocate_episode_ids(
        args.output_dir, args.episodes
    )

    plan = build_collection_plan(args, episode_ids)

    collision = plan.collision_config

    config = {
        "mode": args.mode,
        "world": str(args.world),
        "output_dir": str(args.output_dir),
        "episodes": args.episodes,
        "episode_ids": [
            episode.episode_id for episode in plan.episodes
        ],
        "start": list(args.start),
        "goal": list(args.goal),
        "duration": args.duration,
        "inflation_radius": args.inflation_radius,
        "collision_monitoring_enabled": collision.enabled,
        "collision_topic": collision.topic,
        "robot_name": collision.robot_name,
        "movement_enabled": plan.movement_enabled,
        "simulation_reset_enabled": plan.simulation_reset_enabled,
    }

    print(json.dumps(config, indent=2))


NO_MOTION_MESSAGE = (
    "No robot commands or simulation resets were issued."
)


def run_collection_preflight():
    """Run graph then runtime checks. Return True only if both pass."""

    from navigation.expert_collection_preflight import run_preflight
    from navigation.expert_runtime_preflight import run_runtime_preflight

    try:
        graph_result = run_preflight()
    except Exception as exc:
        print(
            f"ROS GRAPH PREFLIGHT ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("PREFLIGHT: FAIL")
        return False

    for name, passed in graph_result.checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(
        "ROS GRAPH PREFLIGHT:",
        "PASS" if graph_result.passed else "FAIL",
    )

    if not graph_result.passed:
        print("RUNTIME PREFLIGHT: SKIPPED")
        print("PREFLIGHT: FAIL")
        return False

    try:
        runtime_result = run_runtime_preflight(timeout=5.0)
    except Exception as exc:
        print(
            f"RUNTIME PREFLIGHT ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("PREFLIGHT: FAIL")
        return False

    for name, passed in runtime_result.checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(
        "RUNTIME PREFLIGHT:",
        "PASS" if runtime_result.passed else "FAIL",
    )

    passed = graph_result.passed and runtime_result.passed

    print("PREFLIGHT:", "PASS" if passed else "FAIL")

    return passed


def _block_execution(message):
    print(f"EXECUTE ERROR: {message}", file=sys.stderr)
    print("EXECUTE: FAIL")
    print(NO_MOTION_MESSAGE)
    return 1


def execute(args):
    """Collect demonstrations only after plan validation and preflight."""

    from navigation.expert_collection_plan import build_collection_plan

    try:
        episode_ids = allocate_episode_ids(
            args.output_dir, args.episodes
        )
        plan = build_collection_plan(args, episode_ids)
    except Exception as exc:
        return _block_execution(f"{type(exc).__name__}: {exc}")

    if not run_collection_preflight():
        print("EXECUTE: BLOCKED")
        print(NO_MOTION_MESSAGE)
        return 1

    from navigation.expert_collection_orchestrator import collect_plan
    from navigation.expert_dataset_export import export_expert_episode
    from navigation.expert_live_runner_factory import create_live_runner

    def runner_factory(episode):
        return create_live_runner(
            episode,
            world_path=plan.world,
            inflation_radius=plan.inflation_radius,
            allow_live=True,
        )

    try:
        outcomes = collect_plan(
            plan,
            runner_factory,
            export_expert_episode,
            allow_execution=True,
        )
    except Exception as exc:
        print(
            f"EXECUTE ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("EXECUTE: FAIL")
        return 1

    summary = {
        "mode": "execute",
        "execution_authorized": True,
        "episodes_requested": len(plan.episodes),
        "episodes_completed": len(outcomes),
        "outcomes": [
            {
                "episode_id": outcome.episode_id,
                "status": outcome.status,
                "exported": outcome.exported,
                "output_files": [
                    str(path) for path in outcome.output_files
                ],
                "error": outcome.error,
            }
            for outcome in outcomes
        ],
    }

    print(json.dumps(summary, indent=2))

    success = (
        len(outcomes) == len(plan.episodes)
        and all(
            outcome.exported and outcome.status == "GOAL_REACHED"
            for outcome in outcomes
        )
    )

    print("EXECUTE:", "PASS" if success else "FAIL")

    return 0 if success else 1


def main(argv=None):
    args = parse_args(argv)

    if args.mode == "execute":
        return execute(args)

    if args.mode == "preflight":
        return 0 if run_collection_preflight() else 1

    preview(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
