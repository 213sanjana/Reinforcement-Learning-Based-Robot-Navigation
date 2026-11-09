
import math
import numpy as np


class JackalLidarGridMapper:
    """Project finite LiDAR returns into a world-aligned grid."""

    def __init__(self, grid):
        self.grid = grid

    def build(self, observation):
        occupancy = np.zeros(
            (self.grid.height, self.grid.width),
            dtype=np.int8,
        )

        ranges = np.asarray(observation.lidar_ranges, dtype=float)
        angles = np.asarray(observation.lidar_angles, dtype=float)

        if ranges.shape != angles.shape:
            raise ValueError("LiDAR ranges and angles must match")

        if ranges.ndim != 1:
            raise ValueError("LiDAR data must be one-dimensional")

        valid = np.isfinite(ranges) & (ranges > 0)

        for distance, angle in zip(ranges[valid], angles[valid]):
            world_angle = observation.heading + angle

            x = observation.position_x + distance * math.cos(world_angle)
            y = observation.position_y + distance * math.sin(world_angle)

            cell = self.grid.world_to_grid(x, y)

            if cell is not None:
                occupancy[cell] = 1

        return occupancy
