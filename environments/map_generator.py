"""Solvable random grid maps using local seeded RNG and BFS."""
from collections import deque
from dataclasses import dataclass
import numpy as np

Position = tuple[int, int]

@dataclass(frozen=True)
class GridMap:
    obstacles: np.ndarray
    start: Position
    goal: Position


def shortest_path(obstacles: np.ndarray, start: Position, goal: Position) -> list[Position] | None:
    """Return a four-connected shortest path, or None if unreachable."""
    h, w = obstacles.shape
    if any(not (0 <= r < h and 0 <= c < w) or obstacles[r, c] for r, c in (start, goal)):
        return None
    queue = deque([start])
    parent: dict[Position, Position | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        r, c = current
        for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            nr, nc = neighbor
            if 0 <= nr < h and 0 <= nc < w and not obstacles[nr,nc] and neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    return None


def generate_map(height: int, width: int, obstacle_density: float,
                 rng: np.random.Generator, max_attempts: int = 1000) -> GridMap:
    """Sample obstacles and distinct reachable start/goal; fail explicitly if impossible."""
    if height < 2 or width < 2:
        raise ValueError('Grid height and width must both be at least 2')
    if not 0 <= obstacle_density < 1:
        raise ValueError('obstacle_density must be in [0, 1)')
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    count = int(round(height * width * obstacle_density))
    if height * width - count < 2:
        raise ValueError('Obstacle density leaves fewer than two free cells')
    for _ in range(max_attempts):
        occupied = np.zeros(height * width, dtype=np.int8)
        if count:
            occupied[rng.choice(height * width, size=count, replace=False)] = 1
        obstacles = occupied.reshape(height, width)
        free = np.argwhere(obstacles == 0)
        chosen = rng.choice(len(free), size=2, replace=False)
        start, goal = (tuple(int(v) for v in free[i]) for i in chosen)
        if shortest_path(obstacles, start, goal) is not None:
            return GridMap(obstacles=obstacles, start=start, goal=goal)
    raise RuntimeError(f'Could not generate a solvable map in {max_attempts} attempts; reduce obstacle_density')
