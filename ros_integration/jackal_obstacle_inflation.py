
import numpy as np


class JackalObstacleInflation:
    """Inflate occupied cells using Euclidean grid distance."""

    def __init__(self, radius_cells=1):
        if not isinstance(radius_cells, int) or radius_cells < 0:
            raise ValueError("radius_cells must be a nonnegative integer")
        self.radius_cells = radius_cells

    def inflate(self, occupancy):
        source = np.asarray(occupancy)

        if source.ndim != 2:
            raise ValueError("Occupancy grid must be 2D")

        inflated = (source != 0).astype(np.int8)
        occupied = np.argwhere(source != 0)

        height, width = source.shape
        radius = self.radius_cells

        for row, col in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc > radius * radius:
                        continue

                    r = row + dr
                    c = col + dc

                    if 0 <= r < height and 0 <= c < width:
                        inflated[r, c] = 1

        return inflated
