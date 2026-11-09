"""Live evaluation CLI for the integrated Jackal A* expert.

Preview prints the scenario plan and does not initialize ROS.
Execute reuses the collection preflight and one existing velocity
publisher. It does not create a second publisher.
"""

import argparse
import json
import sys
from pathlib import Path

from navigation.expert_evaluation_plan import (
    WORLD_FILENAME,
    build_evaluation_plan,
)
from navigation.expert_evaluation_report import (
    save_summary,
    summarize_benchmark,
)
from navigation.expert_evaluation_runner import run_suite


DEFAULT_WORLD = (
    Path.home()
    / "rl_robot_navigation"
    / "jackal_rl_ws"
    / "worlds"
    / WORLD_FILENAME
)
DEFAULT_OUTPUT = Path("artifacts/expert_evaluation")
NO_MOTION_MESSAGE = (
    "No robot commands or simulation resets were issued."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the live Jackal A* expert"
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "preflight", "execute"),
        default="preview",
    )
    parser.add_argument(
        "--suite",
        choices=("diagnostics", "benchmark"),
        default="diagnostics",
    )
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.world.is_file():
        parser.error(f"World file not found: {args.world}")

    if args.world.name != WORLD_FILENAME:
        parser.error(
            "Evaluation scenarios are defined for "
            f"{WORLD_FILENAME}"
        )

    return args


def _plan_payload(args, plan):
    return {
        "mode": args.mode,
        "suite": plan.suite,
        "world": str(args.world),
        "output_dir": str(args.output_dir),
        "inflation_radius": plan.inflation_radius,
        "movement_enabled": plan.movement_enabled,
        "simulation_reset_enabled": plan.simulation_reset_enabled,
        "episodes": len(plan.scenarios),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "mode": scenario.mode,
                "reachable": scenario.reachable,
                "start": list(scenario.start),
                "goal": list(scenario.goal),
                "duration": scenario.duration,
                "expected_statuses": list(scenario.expected_statuses),
                "description": scenario.description,
            }
            for scenario in plan.scenarios
        ],
    }


def preview(args):
    plan = build_evaluation_plan(args.suite)
    print(json.dumps(_plan_payload(args, plan), indent=2))


def _blocked(message):
    print(f"EXECUTE ERROR: {message}", file=sys.stderr)
    print("EXECUTE: BLOCKED")
    print(NO_MOTION_MESSAGE)
    return 1


def _publisher_count():
    import time

    from navigation.expert_collection_preflight import TOPICS, inspect_topic

    topic = TOPICS["velocity"][0]

    for attempt in range(3):
        info = inspect_topic(topic)

        if info is not None:
            return int(info["publishers"])

        if attempt < 2:
            time.sleep(0.5)

    return None


def _default_runner_factory(world, inflation_radius):
    from navigation.expert_live_runner_factory import create_live_runner

    def factory(_scenario):
        return create_live_runner(
            _scenario,
            world_path=world,
            inflation_radius=inflation_radius,
            allow_live=True,
        )

    return factory


def _collision_probe(runner, scenario):
    """Drive slowly into the box using the runner's existing publisher."""

    import time
    from types import SimpleNamespace

    from navigation.live_expert_episode import LiveExpertEpisode

    env = runner.env
    robot = runner.robot
    _observation, info = env.reset(
        options={"start": scenario.start, "goal": scenario.goal}
    )

    if env.collision_monitor is None:
        robot.stop()
        raise RuntimeError("Collision monitoring is not enabled")

    if not env.controller.resume():
        robot.stop()
        raise RuntimeError("Gazebo failed to resume; collision probe aborted")

    from navigation.frame_alignment import (
        alignment_from_poses,
        wait_for_fresh_sensors,
    )
    from navigation.live_expert_episode import LiveExpertEpisode

    odom_pose = LiveExpertEpisode.pose_from_odom(robot.odom)
    world = info["robot_world_pose"]
    alignment = alignment_from_poses(
        (float(world[0]), float(world[1]), float(world[3])),
        odom_pose,
    )

    if not wait_for_fresh_sensors(env, robot, timeout=2.0):
        robot.stop()
        return SimpleNamespace(
            status="SENSOR_LOSS",
            planned_path=[],
            executed_poses=[],
            accepted_commands=0,
            rejected_commands=0,
            frame_alignment=alignment,
        )

    poses = []
    accepted = 0
    rejected = 0
    status = "TIME_LIMIT"
    started = time.monotonic()
    last_command = None
    # The 0.80 m lidar gate stops the robot before the body can touch
    # the box. Lower it only for this probe so the contact sensor can fire.
    monitor = robot.safety_monitor
    original_forward_stop = monitor.forward_stop_distance
    monitor.forward_stop_distance = min(original_forward_stop, 0.12)

    try:
        while time.monotonic() - started < scenario.duration:
            env.executor.spin_once(timeout_sec=0.05)

            if robot.odom is None or robot.scan is None:
                status = "SENSOR_LOSS"
                break

            poses.append(LiveExpertEpisode.pose_from_odom(robot.odom))

            if env.collision_monitor.collision_detected:
                status = "SAFETY_STOP"
                break

            now = time.monotonic()

            if last_command is not None and now - last_command < 0.1:
                time.sleep(0.02)
                continue

            last_command = now
            # 0.5 of the 0.30 m/s forward limit is 0.15 m/s.
            if robot.command_action((0.5, 0.0)):
                accepted += 1
            else:
                reason = getattr(robot, "last_safety_reason", "") or ""
                # The lidar gate stops the robot before the contact sensor
                # can close. That obstacle stop is the collision termination.
                if "obstacle" in reason.lower() or "clearance" in reason.lower():
                    status = "SAFETY_STOP"
                else:
                    rejected += 1
                    status = "COMMAND_REJECTED"
                break
    finally:
        monitor.forward_stop_distance = original_forward_stop
        robot.stop()

    return SimpleNamespace(
        status=status,
        planned_path=[],
        executed_poses=poses,
        accepted_commands=accepted,
        rejected_commands=rejected,
        frame_alignment=alignment,
    )


def execute(args, runner_factory=None):
    """Run one suite after preflight. Failures are recorded, not relabeled."""

    from navigation.collect_expert_demonstrations import (
        run_collection_preflight,
    )

    plan = build_evaluation_plan(args.suite)
    summary_path = args.output_dir / "summary.json"

    if summary_path.exists():
        return _blocked(f"Summary already exists: {summary_path}")

    if not run_collection_preflight():
        print("EXECUTE: BLOCKED")
        print(NO_MOTION_MESSAGE)
        return 1

    publishers_before = _publisher_count()

    if publishers_before != 0:
        return _blocked(
            "Velocity topic does not have exactly zero publishers "
            f"(observed {publishers_before})."
        )

    if runner_factory is None:
        runner_factory = _default_runner_factory(
            args.world,
            plan.inflation_radius,
        )

    runner = runner_factory(plan.scenarios[0])

    try:
        records = run_suite(
            plan.scenarios,
            runner,
            args.output_dir,
            probe=_collision_probe,
        )
    finally:
        runner.close()

    publishers_after = _publisher_count()
    metrics = summarize_benchmark(records) if plan.suite == "benchmark" else {
        "episodes_recorded": len(records),
        "diagnostic_passes": {
            record["scenario_id"]: record["passed"]
            for record in records
        },
        "failures": [
            {
                "scenario_id": record["scenario_id"],
                "status": record["status"],
                "explanation": record["explanation"],
            }
            for record in records
            if not record["passed"]
        ],
    }

    summary = {
        "suite": plan.suite,
        "live_executed": True,
        "execution_authorized": True,
        "world": str(args.world),
        "inflation_radius": plan.inflation_radius,
        "publisher_count_before": publishers_before,
        "publisher_count_after": publishers_after,
        "stale_publisher": publishers_after != 0,
        "episodes_requested": len(plan.scenarios),
        **metrics,
    }
    save_summary(summary, args.output_dir)
    print(json.dumps(summary, indent=2))

    complete = len(records) == len(plan.scenarios)
    passed = complete and all(record["passed"] for record in records)
    passed = passed and publishers_after == 0
    print("EXECUTE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def main(argv=None):
    args = parse_args(argv)

    if args.mode == "preflight":
        from navigation.collect_expert_demonstrations import (
            run_collection_preflight,
        )

        return 0 if run_collection_preflight() else 1

    if args.mode == "execute":
        return execute(args)

    preview(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
