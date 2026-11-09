"""Choose the next open-space destination from the current LiDAR scan.

The saved policy still receives a destination in its 40-D observation. These
goals come from the scan, so the robot keeps driving through whichever world
is running instead of following a stored checkpoint list.
"""

from __future__ import annotations

import math

import numpy as np


def sector_bearing(sector: int, sectors: int = 36) -> float:
    """Center angle of one LiDAR sector. Sector 0 starts at -pi."""

    return -math.pi + (int(sector) + 0.5) * (2.0 * math.pi / sectors)


def exploration_goal(
    observation,
    pose,
    *,
    travel_m: float = 3.0,
    min_range_m: float = 1.5,
    previous_sector: int | None = None,
    preferred_sector: int | None = None,
) -> dict:
    """Place a goal along the clearest LiDAR direction from ``pose``."""

    vector = np.asarray(observation, dtype=np.float64)
    if vector.shape != (40,) or not np.all(np.isfinite(vector[:36])):
        raise ValueError("LiDAR observation must contain 36 finite ranges")
    if len(pose) < 3 or not np.all(np.isfinite(np.asarray(pose[:3], dtype=np.float64))):
        raise ValueError("Pose must be a finite x, y, yaw")
    ranges = vector[:36] * 12.0
    visible = [
        sector for sector in range(36)
        if abs(sector_bearing(sector)) <= math.pi / 2
    ]
    order = [int(index) for index in np.argsort(ranges)[::-1] if int(index) in visible]
    chosen = None
    if (
        preferred_sector is not None
        and int(preferred_sector) in visible
        and float(ranges[int(preferred_sector)]) >= min_range_m
    ):
        chosen = int(preferred_sector)
    if chosen is None:
        for sector in order:
            if previous_sector is not None and sector == int(previous_sector):
                continue
            if float(ranges[sector]) < min_range_m:
                continue
            chosen = sector
            break
    if chosen is None:
        chosen = max(visible, key=lambda sector: float(ranges[sector]))
    travel = min(float(travel_m), max(1.0, float(ranges[chosen]) * 0.5))
    heading = float(pose[2]) + sector_bearing(chosen)
    return {
        "goal": (
            float(pose[0]) + travel * math.cos(heading),
            float(pose[1]) + travel * math.sin(heading),
        ),
        "sector": chosen,
        "travel_m": travel,
    }


def roaming_goal(
    observation,
    pose,
    recent=(),
    *,
    travel_m: float = 4.5,
    min_range_m: float = 1.6,
):
    """Pick a clear direction that leaves the area the robot just drove through."""

    vector = np.asarray(observation, dtype=np.float64)
    if vector.shape != (40,) or not np.all(np.isfinite(vector[:36])):
        raise ValueError("LiDAR observation must contain 36 finite ranges")
    if len(pose) < 3 or not np.all(np.isfinite(np.asarray(pose[:3], dtype=np.float64))):
        raise ValueError("Pose must be a finite x, y, yaw")
    ranges = vector[:36] * 12.0
    x, y, yaw = (float(pose[0]), float(pose[1]), float(pose[2]))
    best = None
    best_score = -1.0e9
    for sector in range(36):
        bearing = sector_bearing(sector)
        # The laser covers the forward half. Rear bins are unmeasured.
        if abs(bearing) > math.pi / 2:
            continue
        distance = float(ranges[sector])
        if distance < min_range_m:
            continue
        travel = min(float(travel_m), max(2.5, distance * 0.65))
        heading = yaw + bearing
        goal = (x + travel * math.cos(heading), y + travel * math.sin(heading))
        spread = 0.0
        if recent:
            spread = min(
                math.hypot(goal[0] - float(point[0]), goal[1] - float(point[1]))
                for point in recent
            )
        score = spread + 0.2 * distance
        if abs(bearing) > math.radians(150):
            score -= 4.0
        if score > best_score:
            best_score = score
            best = {"goal": goal, "sector": sector, "travel_m": travel}
    if best is None:
        return exploration_goal(observation, pose, travel_m=travel_m, min_range_m=1.0)
    return best


def forward_sector(observation) -> int | None:
    """Longest sector inside the forward safety cone, when that cone is open."""

    vector = np.asarray(observation, dtype=np.float64)
    if vector.shape != (40,) or not np.all(np.isfinite(vector[15:21])):
        return None
    window = vector[15:21] * 12.0
    if float(np.max(window)) < 0.90:
        return None
    return 15 + int(np.argmax(window))


def avoidance_action(observation, stop_distance_m: float = 0.90):
    """Steer around a nearby obstacle without stopping first.

    A table beside the front wheel is outside the narrow safety cone. While
    the straight-ahead bins are still open, keep rolling and turn. Spin in
    place only when the path directly ahead is inside the safety stop.
    """

    vector = np.asarray(observation, dtype=np.float64)
    if vector.shape != (40,) or not np.all(np.isfinite(vector[:36])):
        return None
    ranges = vector[:36] * 12.0
    # Wider than the safety cone: a table beside the front wheel is outside
    # ±25 degrees, and the wheel still hits it.
    forward = float(np.min(ranges[12:24]))
    if forward >= float(stop_distance_m):
        return None
    left_front = float(np.min(ranges[18:24]))
    right_front = float(np.min(ranges[12:18]))
    if abs(left_front - right_front) >= 0.15:
        direction = -1.0 if left_front < right_front else 1.0
    else:
        right = float(np.min(ranges[6:12]))
        left = float(np.min(ranges[24:30]))
        direction = 1.0 if left >= right else -1.0
    if direction > 0.0:
        side_clear = float(np.min(ranges[22:30]))
    else:
        side_clear = float(np.min(ranges[6:14]))
    center = float(np.min(ranges[15:21]))
    if side_clear < 0.70:
        # The safety monitor would cancel this turn and the robot would sit.
        if center < 0.85:
            return np.array([-0.55, 0.0], dtype=np.float32)
        return None
    if center >= 0.85:
        return np.array([0.70, 0.45 * direction], dtype=np.float32)
    return np.array([0.0, 0.65 * direction], dtype=np.float32)
