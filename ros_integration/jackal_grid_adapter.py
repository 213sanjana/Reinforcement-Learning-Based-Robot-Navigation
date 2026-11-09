"""Convert between Jackal world coordinates and GridWorld cells."""

import math


class JackalGridAdapter:
    """Map world XY coordinates to row/column grid coordinates."""

    def __init__(
        self,
        height=10,
        width=10,
        cell_size=0.5,
        origin_x=-2.5,
        origin_y=2.5,
    ):
        if (
            not isinstance(height, int)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or isinstance(width, bool)
            or height <= 0
            or width <= 0
        ):
            raise ValueError("Grid dimensions must be positive integers")

        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("Cell size must be positive and finite")

        if not math.isfinite(origin_x) or not math.isfinite(origin_y):
            raise ValueError("Grid origin must be finite")

        self.height = height
        self.width = width
        self.cell_size = float(cell_size)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

    def world_to_grid(self, x, y):
        """Convert world position to a grid cell or return None."""

        if not math.isfinite(x) or not math.isfinite(y):
            return None

        col = math.floor((x - self.origin_x) / self.cell_size)
        row = math.floor((self.origin_y - y) / self.cell_size)

        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col

        return None

    def grid_to_world(self, row, col):
        """Return the world coordinates of a cell's center."""

        if (
            not isinstance(row, int)
            or not isinstance(col, int)
            or isinstance(row, bool)
            or isinstance(col, bool)
            or not (0 <= row < self.height)
            or not (0 <= col < self.width)
        ):
            raise ValueError("Grid cell is outside map boundaries")

        x = self.origin_x + (col + 0.5) * self.cell_size
        y = self.origin_y - (row + 0.5) * self.cell_size

        return x, y
