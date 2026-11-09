"""Deterministic optimal four-connected A*; coordinates are (row, column)."""
import heapq
import itertools
import numpy as np

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def validate_grid(obstacles):
    grid = np.asarray(obstacles)
    if grid.ndim != 2 or not np.isin(grid, (0, 1)).all():
        raise ValueError('obstacles must be a binary 2D array')
    return grid


def position(value):
    a = np.asarray(value)
    if a.shape != (2,) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('position must contain two integers')
    return tuple(map(int, a))


def astar(obstacles, start, goal):
    """Return inclusive shortest path, or None for blocked/out-of-bounds/unreachable endpoints."""
    grid = validate_grid(obstacles)
    start, goal = position(start), position(goal)
    h, w = grid.shape
    def free(p):
        return 0 <= p[0] < h and 0 <= p[1] < w and grid[p] == 0
    if not free(start) or not free(goal):
        return None
    def heuristic(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    counter = itertools.count()
    queue = [(heuristic(start), next(counter), 0, start)]
    costs, parent = {start: 0}, {start: None}
    while queue:
        _, _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for dr, dc in ACTIONS:
            neighbor = current[0] + dr, current[1] + dc
            if free(neighbor) and cost + 1 < costs.get(neighbor, float('inf')):
                costs[neighbor], parent[neighbor] = cost + 1, current
                heapq.heappush(queue, (cost + 1 + heuristic(neighbor), next(counter), cost + 1, neighbor))
    return None


def path_to_actions(path):
    if not path:
        raise ValueError('path cannot be empty')
    points = [position(p) for p in path]
    actions = []
    for a, b in zip(points, points[1:]):
        delta = b[0] - a[0], b[1] - a[1]
        if delta not in ACTIONS:
            raise ValueError('path contains a non-adjacent move')
        actions.append(ACTIONS.index(delta))
    return actions
