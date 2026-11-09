"""Additional safety checks for ROS 2 A* navigation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationSafetyResult:
    safe: bool
    reason: str
    path_distance: float = 0.0


def point_segment_distance(px, py, ax, ay, bx, by):
    """Shortest distance from a point to a line segment."""

    dx = bx - ax
    dy = by - ay

    length_squared = dx * dx + dy * dy

    if length_squared <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = max(
        0.0,
        min(
            1.0,
            ((px - ax) * dx + (py - ay) * dy)
            / length_squared,
        ),
    )

    closest_x = ax + t * dx
    closest_y = ay + t * dy

    return math.hypot(
        px - closest_x,
        py - closest_y,
    )


def distance_to_path(x, y, path):
    """Compute distance to the planned waypoint polyline."""

    if not path:
        return float("inf")

    if len(path) == 1:
        return math.hypot(
            x - path[0][0],
            y - path[0][1],
        )

    return min(
        point_segment_distance(
            x,
            y,
            a[0],
            a[1],
            b[0],
            b[1],
        )
        for a, b in zip(path[:-1], path[1:])
    )


def evaluate_navigation_safety(
    pose,
    path,
    sensor_ages,
    collision_detected,
    max_sensor_age=0.5,
    max_path_deviation=0.35,
):
    """Evaluate preconditions before issuing navigation commands."""

    if pose is None:
        return NavigationSafetyResult(
            False,
            "Robot pose unavailable",
        )

    if not all(math.isfinite(v) for v in pose):
        return NavigationSafetyResult(
            False,
            "Robot pose contains nonfinite values",
        )

    if sensor_ages is None or len(sensor_ages) != 2:
        return NavigationSafetyResult(
            False,
            "Sensor timestamps unavailable",
        )

    if any(
        not math.isfinite(age)
        or age < 0
        or age > max_sensor_age
        for age in sensor_ages
    ):
        return NavigationSafetyResult(
            False,
            "Sensor data stale or invalid",
        )

    if collision_detected:
        return NavigationSafetyResult(
            False,
            "Gazebo collision detected",
        )

    if not path:
        return NavigationSafetyResult(
            False,
            "Navigation path unavailable",
        )

    deviation = distance_to_path(
        pose[0],
        pose[1],
        path,
    )

    if deviation > max_path_deviation:
        return NavigationSafetyResult(
            False,
            "Robot deviated from A* path",
            deviation,
        )

    return NavigationSafetyResult(
        True,
        "Navigation preconditions satisfied",
        deviation,
    )
