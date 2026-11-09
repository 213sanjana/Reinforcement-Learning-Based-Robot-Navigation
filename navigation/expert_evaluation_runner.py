"""Run an already-authorized expert runner across evaluation scenarios.

This module does not construct a ROS node or a velocity publisher.
The caller supplies one existing runner and a stop check that uses
that runner's publisher.
"""

import time

from navigation.expert_evaluation_report import (
    assess_episode,
    episode_paths,
    goal_distance,
    max_cross_track_error,
    path_length,
    plot_planned_versus_executed,
    save_episode,
)
from navigation.expert_evaluation_plan import OBSTACLE_BOX
from navigation.frame_alignment import points_to_world


def verify_stop(robot):
    """Publish zero velocity through the robot's current callback chain."""

    captured = {}
    original = getattr(robot, "expert_publish_callback", None)

    def capture(linear, angular, stamp):
        captured["linear"] = float(linear)
        captured["angular"] = float(angular)
        captured["stamp"] = int(stamp)

        if original is not None:
            original(linear, angular, stamp)

    robot.expert_publish_callback = capture

    try:
        robot.stop()
    finally:
        robot.expert_publish_callback = original

    odom_linear = None
    odom_angular = None
    odom = getattr(robot, "odom", None)

    if odom is not None:
        twist = odom.twist.twist
        odom_linear = float(twist.linear.x)
        odom_angular = float(twist.angular.z)

    return {
        "stop_verified": (
            captured.get("linear") == 0.0
            and captured.get("angular") == 0.0
        ),
        "stop_command_linear": captured.get("linear"),
        "stop_command_angular": captured.get("angular"),
        "odom_linear_after_stop": odom_linear,
        "odom_angular_after_stop": odom_angular,
    }


def _status_from_exception(exc):
    message = str(exc)

    if "PLANNING_FAILED" in message:
        return "PLANNING_FAILED"

    if "INVALID_PATH" in message:
        return "INVALID_PATH"

    return "ERROR"


def _record(scenario, status, error, result, stop_info, navigation_time_s):
    planned = []
    executed = []
    accepted = 0
    rejected = 0

    alignment = getattr(result, "frame_alignment", None) if result is not None else None
    coordinate_frame = "odom"

    if result is not None:
        planned = [list(map(float, point[:2])) for point in result.planned_path]
        executed = [list(map(float, pose[:3])) for pose in result.executed_poses]
        accepted = int(result.accepted_commands)
        rejected = int(result.rejected_commands)

        if alignment is not None:
            planned = points_to_world(planned, alignment)
            executed = points_to_world(executed, alignment)
            coordinate_frame = "world"

        if status == "GOAL_REACHED" and rejected:
            status = "REJECTED_COMMANDS"

    measured = {
        "status": status,
        "goal_distance_m": goal_distance(executed, scenario.goal),
        "max_cross_track_error_m": max_cross_track_error(planned, executed),
        "stop_verified": bool(stop_info.get("stop_verified")),
    }
    passed, explanation = assess_episode(scenario, measured)

    return {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "mode": scenario.mode,
        "reachable": scenario.reachable,
        "description": scenario.description,
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "duration_s": scenario.duration,
        "expected_statuses": list(scenario.expected_statuses),
        "status": status,
        "passed": passed,
        "explanation": explanation,
        "error": error,
        "navigation_time_s": navigation_time_s,
        "path_length_m": path_length(executed),
        "planned_length_m": path_length(planned),
        "goal_distance_m": measured["goal_distance_m"],
        "max_cross_track_error_m": measured["max_cross_track_error_m"],
        "accepted_commands": accepted,
        "rejected_commands": rejected,
        "coordinate_frame": coordinate_frame,
        "planned_path": planned,
        "executed_poses": executed,
        **stop_info,
    }


def run_suite(
    scenarios,
    runner,
    output_dir,
    *,
    probe,
    stop_check=verify_stop,
    obstacle_box=OBSTACLE_BOX,
):
    """Evaluate scenarios on one runner and persist every outcome."""

    records = []

    for scenario in scenarios:
        json_path, plot_path = episode_paths(output_dir, scenario.scenario_id)

        if json_path.exists() or plot_path.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing evaluation result"
            )

        started = time.monotonic()
        result = None
        error = ""

        try:
            if scenario.mode == "collision_probe":
                result = probe(runner, scenario)
                status = result.status
            else:
                try:
                    runner.prepare(
                        start=scenario.start,
                        goal=scenario.goal,
                    )
                except Exception as exc:
                    status = _status_from_exception(exc)
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    result = runner.run(duration=scenario.duration)
                    status = result.status
        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            result = None

        navigation_time_s = time.monotonic() - started

        try:
            stop_info = stop_check(runner.robot)
        except Exception as exc:
            stop_info = {
                "stop_verified": False,
                "stop_command_linear": None,
                "stop_command_angular": None,
                "odom_linear_after_stop": None,
                "odom_angular_after_stop": None,
            }
            status = "CLEANUP_FAILED"
            cleanup = f"Cleanup failure: {type(exc).__name__}: {exc}"
            error = f"{error}; {cleanup}" if error else cleanup

        record = _record(
            scenario,
            status,
            error,
            result,
            stop_info,
            navigation_time_s,
        )

        try:
            plot_planned_versus_executed(record, plot_path, obstacle_box)
            record["plot"] = str(plot_path)
            record["plot_error"] = ""
        except Exception as exc:
            record["plot"] = ""
            record["plot_error"] = f"{type(exc).__name__}: {exc}"

        save_episode(record, output_dir)
        records.append(record)

        if status == "CLEANUP_FAILED":
            break

    return tuple(records)
