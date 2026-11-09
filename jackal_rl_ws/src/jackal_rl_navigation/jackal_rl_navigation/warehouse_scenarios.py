"""Seeded start and goal selection for warehouse episodes.

A pair is accepted only when it satisfies the configured geometric
constraints. That check is not a live navigation result.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def _point(values, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must have length {length}")
    point = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{label} must be finite")
    return point


def _as_point(start) -> tuple[float, float, float]:
    if len(start) == 2:
        return (float(start[0]), float(start[1]), 0.0)
    return _point(start, 3, "start")


def pair_key(start, goal) -> tuple:
    start = _as_point(start)
    goal = _point(goal, 2, "goal")
    return tuple(round(value, 6) for value in (*start[:2], *goal))


def obstacle_contains(x: float, y: float, obstacles: Sequence[Mapping[str, object]], margin: float) -> bool:
    for obstacle in obstacles:
        center = obstacle.get("center")
        radius = obstacle.get("radius")
        if center is None or radius is None:
            rectangle = obstacle.get("rectangle")
            if rectangle is None:
                raise ValueError("Obstacle needs a circle or a rectangle")
            min_x, min_y, max_x, max_y = (float(value) for value in rectangle)
            if min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin:
                return True
            continue
        cx, cy = (float(value) for value in center)
        if math.hypot(x - cx, y - cy) <= float(radius) + margin:
            return True
    return False


def segment_blocked(start, goal, obstacles: Sequence[Mapping[str, object]], margin: float) -> bool:
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    for obstacle in obstacles:
        center = obstacle.get("center")
        radius = obstacle.get("radius")
        if center is None or radius is None:
            continue
        cx, cy = float(center[0]), float(center[1])
        limit = float(radius) + margin
        vx, vy = gx - sx, gy - sy
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-12:
            if math.hypot(sx - cx, sy - cy) <= limit:
                return True
            continue
        t = ((cx - sx) * vx + (cy - sy) * vy) / length_sq
        t = min(1.0, max(0.0, t))
        px, py = sx + t * vx, sy + t * vy
        if math.hypot(px - cx, py - cy) <= limit:
            return True
    return False


def rejection_reason(scenario: Mapping[str, object], constraints: Mapping[str, object]) -> str | None:
    """Return a reason to reject a pair, or None when it satisfies constraints."""

    try:
        start = _as_point(scenario["start"])
        goal = _point(scenario["goal"], 2, "goal")
    except (KeyError, TypeError, ValueError) as exc:
        return str(exc)
    minimum = float(constraints.get("min_separation_m", 0.5))
    margin = float(constraints.get("robot_radius_m", 0.25))
    tolerance = float(constraints.get("goal_tolerance_m", 0.30))
    distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    if distance <= tolerance:
        return "start is already inside the goal tolerance"
    if distance < minimum:
        return "start and goal are closer than min_separation_m"
    bounds = constraints.get("bounds_xy")
    if bounds is not None:
        min_x, min_y, max_x, max_y = (float(value) for value in bounds)
        for x, y in ((start[0], start[1]), goal):
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                return "endpoint is outside configured bounds"
    obstacles = list(constraints.get("obstacles") or [])
    if obstacle_contains(start[0], start[1], obstacles, margin):
        return "start is inside an obstacle"
    if obstacle_contains(goal[0], goal[1], obstacles, margin):
        return "goal is inside an obstacle"
    if segment_blocked(start, goal, obstacles, margin):
        return "straight path violates the configured obstacle clearance"
    return None


def assert_disjoint_splits(scenarios: Sequence[Mapping[str, object]]) -> None:
    seen: dict[tuple, str] = {}
    for scenario in scenarios:
        split = str(scenario.get("split", "train"))
        key = pair_key(scenario["start"], scenario["goal"])
        previous = seen.get(key)
        if previous is not None and previous != split:
            raise ValueError(
                f"Start-goal pair {key} is in both {previous} and {split}"
            )
        seen.setdefault(key, split)


def select_scenario(
    scenarios: Sequence[Mapping[str, object]],
    rng: np.random.Generator,
    split: str,
    constraints: Mapping[str, object],
) -> dict:
    """Sample one valid scenario for ``split`` using ``rng`` order."""

    pool = [dict(item) for item in scenarios if str(item.get("split", "train")) == split]
    if not pool:
        raise RuntimeError(f"No scenarios are configured for split {split!r}")
    order = rng.permutation(len(pool))
    rejected = []
    for index in order:
        scenario = pool[int(index)]
        reason = rejection_reason(scenario, constraints)
        if reason is None:
            scenario["start"] = _as_point(scenario["start"])
            scenario["goal"] = _point(scenario["goal"], 2, "goal")
            return scenario
        rejected.append(f"{scenario.get('id', index)}: {reason}")
    raise RuntimeError("No reachable start-goal pair: " + "; ".join(rejected))
