import math

from planners.gazebo_astar import GazeboAStarPlanner


def obstacle_clearance(point, obstacles):
    """Minimum distance from a point to any static box footprint."""
    x, y = point
    clearance = float("inf")

    for ox, oy, sx, sy, _ in obstacles:
        dx = max(abs(x - ox) - sx / 2.0, 0.0)
        dy = max(abs(y - oy) - sy / 2.0, 0.0)
        clearance = min(clearance, math.hypot(dx, dy))

    return clearance


def departure_is_clear(start, target, obstacles, min_clearance=0.30):
    """Check a straight departure using conservative spatial sampling."""
    distance = math.dist(start, target)
    steps = max(1, math.ceil(distance / 0.025))

    for i in range(steps + 1):
        t = i / steps
        point = (
            start[0] + t * (target[0] - start[0]),
            start[1] + t * (target[1] - start[1]),
        )

        if obstacle_clearance(point, obstacles) < min_clearance:
            return False

    return True


def plan_with_safe_start(
    world_path,
    start,
    goal,
    inflation_radius=0.95,
    min_clearance=0.30,
    search_radius=0.80,
):
    """Find a validated departure to a normally planned A* route."""

    planner = GazeboAStarPlanner(
        world_path,
        inflation_radius=inflation_radius,
    )

    # Prefer ordinary planning whenever possible.
    direct = planner.plan(start, goal)

    if direct and planner.validate_path(direct):
        return direct

    candidates = []

    for row in range(planner.height):
        for col in range(planner.width):
            if planner.grid[row, col] != 0:
                continue

            point = planner.grid_to_world(row, col)
            distance = math.dist(start, point)

            if distance > search_radius:
                continue

            if not departure_is_clear(
                start,
                point,
                planner.obstacles,
                min_clearance=min_clearance,
            ):
                continue

            candidates.append((distance, point))

    candidates.sort(key=lambda item: item[0])

    for _, candidate in candidates:
        path = planner.plan(candidate, goal)

        if not path or not planner.validate_path(path):
            continue

        # The first segment is an explicitly checked departure.
        # The remaining path uses the normal inflated occupancy grid.
        return [tuple(start)] + path

    return None
