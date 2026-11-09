"""Score, explain, save, and plot expert evaluation episodes.

These helpers do not create ROS nodes or publish velocity commands.
Missing measurements stay null. They are not replaced with zeros.
"""

import json
import math
from pathlib import Path


GOAL_DISTANCE_LIMIT = 0.50
CROSS_TRACK_LIMIT = 0.75

STATUS_EXPLANATIONS = {
    "PLANNING_FAILED": (
        "A* did not return a path. The start or goal is outside the "
        "map or blocked by the inflated obstacle."
    ),
    "INVALID_PATH": "A* returned a path that failed occupancy validation.",
    "TIME_LIMIT": "The episode wall-clock limit expired before the goal.",
    "TIMEOUT": "The navigation guard exceeded its time limit.",
    "SAFETY_STOP": "Collision monitoring stopped the robot before another command.",
    "COLLISION": "The episode ended on a collision.",
    "SENSOR_LOSS": "LiDAR or odometry stopped updating.",
    "STALLED": "The robot stopped making progress along the planned route.",
    "COMMAND_REJECTED": "The existing action interface rejected a velocity command.",
    "REJECTED_COMMANDS": "At least one command was rejected, so the episode is not a demonstration.",
    "ERROR": "The episode failed before navigation reached a terminal status.",
    "CLEANUP_FAILED": "The robot or environment could not be closed cleanly.",
    "SIMULATION_STEP_FAILED": "Gazebo did not advance.",
    "DRY_RUN": "Motion was not enabled, so this result is not a live navigation.",
}


def _xy(point):
    return float(point[0]), float(point[1])


def path_length(points):
    """Length of an xy trajectory. None when no points were recorded."""

    if points is None:
        return None

    points = list(points)

    if not points:
        return None

    if len(points) == 1:
        return 0.0

    total = 0.0

    for start, end in zip(points, points[1:]):
        x0, y0 = _xy(start)
        x1, y1 = _xy(end)
        total += math.hypot(x1 - x0, y1 - y0)

    return float(total)


def goal_distance(poses, goal):
    if not poses:
        return None

    x, y = _xy(poses[-1])
    gx, gy = _xy(goal)
    return float(math.hypot(x - gx, y - gy))


def _segment_distance(point, start, end):
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy

    if length_sq == 0.0:
        return math.hypot(px - x0, py - y0)

    scale = ((px - x0) * dx + (py - y0) * dy) / length_sq
    scale = max(0.0, min(1.0, scale))
    return math.hypot(px - (x0 + scale * dx), py - (y0 + scale * dy))


def max_cross_track_error(planned_path, executed_poses):
    """Largest distance from an executed pose to the planned polyline."""

    if not planned_path or not executed_poses:
        return None

    segments = list(zip(planned_path, planned_path[1:]))
    errors = []

    for pose in executed_poses:
        point = _xy(pose)

        if not segments:
            errors.append(math.hypot(point[0] - planned_path[0][0], point[1] - planned_path[0][1]))
            continue

        errors.append(min(
            _segment_distance(point, _xy(start), _xy(end))
            for start, end in segments
        ))

    return float(max(errors))


def assess_episode(scenario, measured):
    """Decide whether the measured outcome matches the scenario contract."""

    status = measured["status"]
    expected = status in scenario.expected_statuses
    passed = expected
    explanation = ""

    if not measured.get("stop_verified", False):
        passed = False
        explanation = (
            "Termination did not publish a zero linear and angular command "
            "on the existing velocity publisher."
        )
    elif not expected:
        explanation = STATUS_EXPLANATIONS.get(
            status,
            f"Status {status} was not one of the expected outcomes.",
        )
    elif "GOAL_REACHED" in scenario.expected_statuses and status == "GOAL_REACHED":
        distance = measured.get("goal_distance_m")
        deviation = measured.get("max_cross_track_error_m")

        if distance is None or distance > GOAL_DISTANCE_LIMIT:
            passed = False
            explanation = (
                "The controller reported the goal, but the final pose is "
                f"outside {GOAL_DISTANCE_LIMIT:.2f} m."
            )
        elif deviation is None or deviation > CROSS_TRACK_LIMIT:
            passed = False
            explanation = (
                "The executed trajectory left the planned corridor "
                f"(limit {CROSS_TRACK_LIMIT:.2f} m)."
            )

    return passed, explanation


def summarize_benchmark(records):
    """Aggregate a repeatable reachable-route evaluation.

    Rates are null until at least one episode has actually been recorded.
    """

    recorded = list(records)

    if not recorded:
        return {
            "episodes_recorded": 0,
            "goal_reached_count": None,
            "success_count": None,
            "success_rate": None,
            "collision_count": None,
            "timeout_count": None,
            "timeout_rate": None,
            "mean_path_length_m": None,
            "mean_navigation_time_s": None,
            "failures": [],
        }

    goal_reached = sum(record["status"] == "GOAL_REACHED" for record in recorded)
    successes = sum(bool(record["passed"]) for record in recorded)
    collisions = sum(
        record["status"] in ("SAFETY_STOP", "COLLISION")
        for record in recorded
    )
    timeouts = sum(
        record["status"] in ("TIME_LIMIT", "TIMEOUT")
        for record in recorded
    )
    lengths = [
        record["path_length_m"]
        for record in recorded
        if record["path_length_m"] is not None
    ]
    times = [
        record["navigation_time_s"]
        for record in recorded
        if record["navigation_time_s"] is not None
    ]

    count = len(recorded)

    return {
        "episodes_recorded": count,
        "goal_reached_count": goal_reached,
        "success_count": successes,
        "success_rate": successes / count,
        "collision_count": collisions,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / count,
        "mean_path_length_m": (
            sum(lengths) / len(lengths) if lengths else None
        ),
        "mean_navigation_time_s": (
            sum(times) / len(times) if times else None
        ),
        "failures": [
            {
                "scenario_id": record["scenario_id"],
                "status": record["status"],
                "explanation": record["explanation"],
            }
            for record in recorded
            if not record["passed"]
        ],
    }


def episode_paths(output_dir, scenario_id):
    output_dir = Path(output_dir)
    stem = f"episode_{scenario_id}"
    return (
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.png",
    )


def save_episode(record, output_dir):
    """Write one episode record without overwriting an existing result."""

    json_path, _ = episode_paths(output_dir, record["scenario_id"])

    if json_path.exists():
        raise FileExistsError(
            f"Episode result already exists: {json_path}"
        )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2))
    return json_path


def save_summary(summary, output_dir):
    path = Path(output_dir) / "summary.json"

    if path.exists():
        raise FileExistsError(f"Summary already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return path


def plot_planned_versus_executed(record, output_path, obstacle_box):
    """Draw the planned route and the poses that were actually recorded."""

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(f"Plot already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    center_x, center_y, size_x, size_y = obstacle_box
    axis.add_patch(Rectangle(
        (center_x - size_x / 2, center_y - size_y / 2),
        size_x,
        size_y,
        facecolor="#b33",
        alpha=0.35,
        label="obstacle",
    ))

    planned = record.get("planned_path") or []
    executed = record.get("executed_poses") or []

    if len(planned) >= 2:
        axis.plot(
            [point[0] for point in planned],
            [point[1] for point in planned],
            "--",
            color="#21a366",
            linewidth=2,
            label="planned",
        )

    if executed:
        axis.plot(
            [point[0] for point in executed],
            [point[1] for point in executed],
            "-",
            color="#2463a8",
            linewidth=2,
            label="executed",
        )

    start = record["start"]
    goal = record["goal"]
    axis.scatter([start[0]], [start[1]], c="#e49322", marker="s", s=60, label="start", zorder=4)
    axis.scatter([goal[0]], [goal[1]], c="#111111", marker="*", s=120, label="goal", zorder=5)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(f"{record['scenario_id']}: {record['status']}")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path
