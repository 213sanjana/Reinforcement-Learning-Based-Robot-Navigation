"""Deterministic start-goal scenarios for the validated Jackal world.

Scenarios are planned with the existing A* occupancy grid. Generating a
manifest does not start ROS, Gazebo, or a velocity publisher.
"""

from dataclasses import dataclass
import math

import numpy as np

from planners.gazebo_astar import GazeboAStarPlanner


SCHEMA_VERSION = "6e.2.0"

LENGTH_RANGES = {
    "short": (1.2, 2.8),
    "medium": (2.8, 5.0),
    "long": (5.0, 12.0),
}
TURN_RANGES = {
    "straight": (0.0, 0.45),
    "turning": (0.9, 20.0),
}
OPEN_CLEARANCE_M = 1.0
OBSTACLE_CLEARANCE_M = 0.75
BORDER_MARGIN_M = 0.35


@dataclass(frozen=True)
class _Region:
    name: str
    cells: tuple


def generate_manifests(config, planner=None):
    """Build disjoint pilot, train, validation, and test manifests."""

    planner = planner or planner_from_config(config)
    if int(planner.obstacles.__len__()) < 1:
        raise ValueError("World has no static box obstacle")

    regions = _regions(planner)
    used = set()
    manifests = {}
    master_seed = int(config["seed"])

    for index, split in enumerate(("pilot", "train", "validation", "test")):
        stage = config["stages"][split]
        quotas = config["quotas"][split]
        split_seed = master_seed + index * 10007
        scenarios = _fill_quotas(
            planner,
            regions,
            quotas,
            used,
            np.random.Generator(np.random.PCG64(split_seed)),
            split=split,
            split_seed=split_seed,
            max_draws=int(config.get("max_draws_per_scenario", 400)),
        )
        expected = int(stage["attempts"])
        if len(scenarios) != expected:
            raise RuntimeError(
                f"{split} generated {len(scenarios)} scenarios, expected {expected}"
            )
        manifests[split] = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "seed": master_seed,
            "split_seed": split_seed,
            "attempts": expected,
            "success_target": stage["success_target"],
            "world": str(config["world"]),
            "inflation_radius": float(config["inflation_radius"]),
            "scenarios": scenarios,
        }

    assert_disjoint(manifests)
    return manifests


def planner_from_config(config):
    return GazeboAStarPlanner(
        config["world"],
        bounds=tuple(config["bounds"]),
        resolution=float(config["resolution"]),
        inflation_radius=float(config["inflation_radius"]),
    )


def assert_disjoint(manifests):
    """Reject a fingerprint that appears in more than one scenario."""

    seen = {}
    identities = set()
    for split, manifest in manifests.items():
        for scenario in manifest["scenarios"]:
            fingerprint = scenario["fingerprint"]
            previous = seen.get(fingerprint)
            if previous is not None:
                raise ValueError(
                    f"Scenario leakage: {fingerprint} is in "
                    f"{previous} and {split}"
                )
            seen[fingerprint] = split

            identity = (split, scenario["scenario_id"])
            if identity in identities or scenario["scenario_id"] in {
                item[1] for item in identities
            }:
                raise ValueError(f"Duplicate scenario id {scenario['scenario_id']}")
            identities.add(identity)


def assert_scenario_feasible(planner, scenario):
    """Re-plan a stored scenario and require a clear, reachable path."""

    start = tuple(scenario["start"][:2])
    goal = tuple(scenario["goal"])
    start_cell = planner.world_to_grid(*start)
    goal_cell = planner.world_to_grid(*goal)

    if start_cell is None or goal_cell is None:
        raise ValueError("Scenario endpoint is outside the map")

    if planner.grid[start_cell] or planner.grid[goal_cell]:
        raise ValueError("Scenario endpoint is inside inflated obstacle")

    path = planner.plan(start, goal)
    if not path or not planner.validate_path(path):
        raise ValueError("Scenario is not reachable in free space")

    clearance = _clearance(path, planner.obstacles)
    if scenario["area_class"] == "open" and clearance < OPEN_CLEARANCE_M:
        raise ValueError("Open scenario lacks obstacle clearance")

    if (
        scenario["area_class"] == "obstacle_rich"
        and clearance > OBSTACLE_CLEARANCE_M
    ):
        raise ValueError("Obstacle-rich scenario does not approach the obstacle")

    return path


def _fill_quotas(
    planner,
    regions,
    quotas,
    used,
    rng,
    *,
    split,
    split_seed,
    max_draws,
):
    scenarios = []
    sample_index = 0

    for kind, count in quotas.items():
        area, length, turn = kind.split("_")
        if area == "obstacle":
            area = "obstacle_rich"

        for _ in range(int(count)):
            scenario = None
            for _draw in range(max_draws):
                sample_index += 1
                candidate = _draw_candidate(rng, regions, area, length)
                if candidate is None:
                    continue
                scenario = _accept(
                    planner,
                    candidate,
                    area,
                    length,
                    turn,
                    used,
                    split=split,
                    episode_id=len(scenarios),
                    split_seed=split_seed,
                    sample_index=sample_index,
                )
                if scenario is not None:
                    break

            if scenario is None:
                raise RuntimeError(
                    f"Unable to sample {split} {kind} after {max_draws} draws"
                )

            used.add(scenario["fingerprint"])
            scenarios.append(scenario)

    return scenarios


def _accept(
    planner,
    candidate,
    area,
    length_name,
    turn_name,
    used,
    *,
    split,
    episode_id,
    split_seed,
    sample_index,
):
    start_xy, goal_xy = candidate
    start_cell = planner.world_to_grid(*start_xy)
    goal_cell = planner.world_to_grid(*goal_xy)

    if (
        start_cell is None
        or goal_cell is None
        or start_cell == goal_cell
        or planner.grid[start_cell]
        or planner.grid[goal_cell]
    ):
        return None

    fingerprint = (
        f"{start_cell[0]}:{start_cell[1]}:{goal_cell[0]}:{goal_cell[1]}"
    )
    if fingerprint in used:
        return None

    path = planner.plan(start_xy, goal_xy)
    if not path or not planner.validate_path(path):
        return None

    length = _path_length(path)
    turn = _net_heading_change(path)
    clearance = _clearance(path, planner.obstacles)
    direct = planner.validate_path([start_xy, goal_xy])

    if not _inside(length, LENGTH_RANGES[length_name]):
        return None
    if not _inside(turn, TURN_RANGES[turn_name]):
        return None
    if turn_name == "straight" and not direct:
        return None
    if turn_name == "turning" and direct:
        return None
    if area == "open" and clearance < OPEN_CLEARANCE_M:
        return None
    if area == "obstacle_rich" and clearance > OBSTACLE_CLEARANCE_M:
        return None

    yaw = 0.0
    if len(path) > 1:
        yaw = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])

    duration = {"short": 70.0, "medium": 140.0, "long": 260.0}[length_name]
    scenario_id = f"{split}_{episode_id:04d}"

    return {
        "scenario_id": scenario_id,
        "episode_id": int(episode_id),
        "split": split,
        "seed": int(split_seed),
        "sample_index": int(sample_index),
        "fingerprint": fingerprint,
        "start": [round(start_xy[0], 4), round(start_xy[1], 4), round(yaw, 6)],
        "goal": [round(goal_xy[0], 4), round(goal_xy[1], 4)],
        "duration_s": duration,
        "planned_length_m": round(length, 4),
        "heading_change_rad": round(turn, 4),
        "min_clearance_m": round(clearance, 4),
        "length_class": length_name,
        "turn_class": turn_name,
        "area_class": area,
        "planned_path": [
            [round(point[0], 4), round(point[1], 4)] for point in path
        ],
    }


def _draw_candidate(rng, regions, area, length_name):
    low, high = LENGTH_RANGES[length_name]

    if area == "open":
        band = regions["north_open"] if rng.random() < 0.5 else regions["south_open"]
        if len(band) < 2:
            return None
        first = band[int(rng.integers(len(band)))]
        second = band[int(rng.integers(len(band)))]
        if not (low <= abs(first[0] - second[0]) < high):
            return None
        if abs(first[1] - second[1]) > 0.35:
            return None
        return (first, second)

    if length_name == "short":
        ring = regions["obstacle_ring"]
        if len(ring) < 2:
            return None
        first = ring[int(rng.integers(len(ring)))]
        second = ring[int(rng.integers(len(ring)))]
        separation = math.hypot(first[0] - second[0], first[1] - second[1])
        if not (low <= separation < high):
            return None
        if (first[0] - 1.5) * (second[0] - 1.5) > 0 and (
            first[1] * second[1] > 0
        ):
            return None
        return (first, second)

    pairs = (
        (regions["west_south"], regions["east_north"]),
        (regions["west_north"], regions["east_south"]),
        (regions["east_north"], regions["west_south"]),
        (regions["east_south"], regions["west_north"]),
    )
    left, right = pairs[int(rng.integers(len(pairs)))]
    if not left or not right:
        return None
    return (
        left[int(rng.integers(len(left)))],
        right[int(rng.integers(len(right)))],
    )


def _regions(planner):
    def collect(predicate):
        cells = []
        for row in range(planner.height):
            for col in range(planner.width):
                if planner.grid[row, col] != 0:
                    continue
                x, y = planner.grid_to_world(row, col)
                if not _inset(planner, x, y):
                    continue
                if predicate(x, y):
                    cells.append((x, y))
        return tuple(cells)

    return {
        "north_open": collect(lambda x, y: 1.30 <= y <= 2.55),
        "south_open": collect(lambda x, y: -2.55 <= y <= -1.30),
        "west_south": collect(lambda x, y: -1.55 <= x <= 0.55 and -2.2 <= y <= -0.7),
        "west_north": collect(lambda x, y: -1.55 <= x <= 0.55 and 0.7 <= y <= 2.2),
        "east_north": collect(lambda x, y: 2.45 <= x <= 4.55 and 0.7 <= y <= 2.2),
        "east_south": collect(lambda x, y: 2.45 <= x <= 4.55 and -2.2 <= y <= -0.7),
        "obstacle_ring": collect(
            lambda x, y: _clearance([(x, y)], planner.obstacles) <= OBSTACLE_CLEARANCE_M
        ),
    }


def _inset(planner, x, y):
    return (
        planner.xmin + BORDER_MARGIN_M <= x <= planner.xmax - BORDER_MARGIN_M
        and planner.ymin + BORDER_MARGIN_M <= y <= planner.ymax - BORDER_MARGIN_M
    )


def _path_length(path):
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
    )


def _travel_heading(origin, target):
    return math.atan2(target[1] - origin[1], target[0] - origin[0])


def _heading_over(origin, points):
    for point in points:
        if math.hypot(point[0] - origin[0], point[1] - origin[1]) >= 0.4:
            return _travel_heading(origin, point)
    return _travel_heading(origin, points[-1])


def _net_heading_change(path):
    """Change in travel direction, ignoring one-cell grid stair steps."""

    if len(path) < 2:
        return 0.0

    initial = _heading_over(path[0], path[1:])
    final = _heading_over(path[-1], list(reversed(path[:-1])))
    final = math.atan2(math.sin(final + math.pi), math.cos(final + math.pi))
    return abs(math.atan2(math.sin(final - initial), math.cos(final - initial)))


def _clearance(path, obstacles):
    distances = []
    for x, y in path:
        for cx, cy, sx, sy, _name in obstacles:
            dx = max(abs(x - cx) - sx / 2.0, 0.0)
            dy = max(abs(y - cy) - sy / 2.0, 0.0)
            distances.append(math.hypot(dx, dy))
    return min(distances) if distances else math.inf


def _inside(value, bounds):
    low, high = bounds
    return low <= value < high
