"""Additive Phase 2 adapter; Phase 1 RobotNavigationEnv remains unchanged."""
from __future__ import annotations
from collections import deque
import numpy as np
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path


def reachable_cells(obstacles: np.ndarray, goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = obstacles.shape
    if not (0 <= goal[0] < h and 0 <= goal[1] < w) or obstacles[goal]:
        raise ValueError('Goal must be a free cell')
    queue = deque([goal])
    seen = {goal}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            p = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and not obstacles[p] and p not in seen:
                seen.add(p)
                queue.append(p)
    return sorted(seen - {goal})


class FixedMapNavigationEnv(RobotNavigationEnv):
    def __init__(self, obstacles: np.ndarray, goal: tuple[int, int], **kwargs) -> None:
        grid = np.asarray(obstacles, dtype=np.int8)
        if grid.ndim != 2 or not np.isin(grid, [0, 1]).all():
            raise ValueError('obstacles must be a binary 2D array')
        self.fixed_obstacles = grid.copy()
        self.fixed_goal = tuple(map(int, goal))
        self.starts = reachable_cells(self.fixed_obstacles, self.fixed_goal)
        if not self.starts:
            raise ValueError('Map requires at least one reachable non-goal start')
        if 'height' in kwargs or 'width' in kwargs or 'obstacle_density' in kwargs:
            raise ValueError('Dimensions and obstacles are derived from the fixed map')
        super().__init__(height=grid.shape[0], width=grid.shape[1], obstacle_density=0, **kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Reuse Gymnasium seeding without regenerating a map.
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        if options is not None and set(options) - {'start'}:
            raise ValueError('Only reset option "start" is supported')
        if options and 'start' in options:
            start = tuple(map(int, options['start']))
            if start not in self.starts:
                raise ValueError('Start must be a reachable non-goal free cell')
        else:
            start = self.starts[int(self.np_random.integers(len(self.starts)))]
        self.obstacles = self.fixed_obstacles.copy()
        self.goal = self.fixed_goal
        self.agent = start
        self.steps = 0
        self._finished = False
        self._initialized = True
        if self.render_mode == 'human':
            self.render()
        return self._observation(), self._info()
