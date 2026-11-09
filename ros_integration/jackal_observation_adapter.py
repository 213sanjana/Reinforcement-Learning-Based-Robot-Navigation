"""Convert Jackal positions and known maps into GridWorld observations."""

import numpy as np

from ros_integration.jackal_grid_adapter import JackalGridAdapter


class JackalObservationAdapter:
    """Build GridWorld observations from a known occupancy grid."""

    def __init__(self, grid=None):
        self.grid = grid if grid is not None else JackalGridAdapter()

    def build(self, x, y, goal_row, goal_col, occupancy):
        """Build an observation, or raise on invalid map/position."""

        agent_cell = self.grid.world_to_grid(x, y)

        if agent_cell is None:
            raise ValueError("Robot position is outside the grid")

        occupancy = np.asarray(occupancy)

        expected_shape = (self.grid.height, self.grid.width)

        if occupancy.shape != expected_shape:
            raise ValueError(
                f"Expected occupancy shape {expected_shape}, "
                f"received {occupancy.shape}"
            )

        if not np.all((occupancy == 0) | (occupancy == 1)):
            raise ValueError("Occupancy values must be binary")

        if (
            not isinstance(goal_row, (int, np.integer))
            or not isinstance(goal_col, (int, np.integer))
            or isinstance(goal_row, (bool, np.bool_))
            or isinstance(goal_col, (bool, np.bool_))
            or not (0 <= goal_row < self.grid.height)
            or not (0 <= goal_col < self.grid.width)
        ):
            raise ValueError("Goal must be a valid grid cell")

        if occupancy[agent_cell] != 0:
            raise ValueError("Robot cell is occupied")

        if occupancy[goal_row, goal_col] != 0:
            raise ValueError("Goal cell is occupied")

        return {
            "agent": np.asarray(agent_cell, dtype=np.int32),
            "goal": np.asarray(
                [goal_row, goal_col], dtype=np.int32
            ),
            "obstacles": occupancy.astype(np.int8, copy=True),
        }
