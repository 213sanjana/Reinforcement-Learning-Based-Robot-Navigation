"""Static Gazebo SDF occupancy mapping and four-connected A* planning."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from planners.astar import astar


class GazeboAStarPlanner:
    """Plan world-coordinate paths around static SDF box obstacles."""

    def __init__(
        self,
        world_path,
        bounds=(-2.0, 5.0, -3.0, 3.0),
        resolution=0.1,
        inflation_radius=0.45,
    ):
        self.world_path = Path(world_path)
        self.bounds = tuple(float(v) for v in bounds)
        self.resolution = float(resolution)
        self.inflation_radius = float(inflation_radius)

        if len(self.bounds) != 4:
            raise ValueError("bounds must contain xmin,xmax,ymin,ymax")

        self.xmin, self.xmax, self.ymin, self.ymax = self.bounds

        if not all(math.isfinite(v) for v in self.bounds):
            raise ValueError("Map bounds must be finite")

        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Invalid map bounds")

        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("Resolution must be positive and finite")

        if not math.isfinite(self.inflation_radius) or self.inflation_radius < 0:
            raise ValueError("Inflation radius must be nonnegative and finite")

        self.width = math.ceil(
            (self.xmax - self.xmin) / self.resolution
        )
        self.height = math.ceil(
            (self.ymax - self.ymin) / self.resolution
        )

        self.obstacles = self._read_obstacles()

        self.grid = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        self._build_grid()

    def _read_obstacles(self):
        """Read supported, axis-aligned static SDF box collisions."""

        tree = ET.parse(self.world_path)
        root = tree.getroot()

        obstacles = []

        worlds = root.findall("world")

        if len(worlds) != 1:
            raise ValueError("Expected exactly one SDF world")

        world = worlds[0]

        for model in world.findall("model"):
            name = model.get("name", "")

            if model.findtext("static", "false").strip().lower() != "true":
                continue

            model_pose = self._parse_pose(model.findtext("pose"))

            if not self._yaw_is_zero(model_pose[5]):
                raise ValueError(
                    f"Rotated model unsupported: {name}"
                )

            for link in model.findall("link"):
                link_pose = self._parse_pose(link.findtext("pose"))

                if not self._yaw_is_zero(link_pose[5]):
                    raise ValueError(
                        f"Rotated link unsupported: {name}"
                    )

                for collision in link.findall("collision"):
                    geometry = collision.find("geometry")

                    if geometry is None:
                        continue

                    # Ground plane is traversable, not an obstacle.
                    if geometry.find("plane") is not None:
                        continue

                    box = geometry.find("box")

                    if box is None:
                        raise ValueError(
                            f"Unsupported static collision geometry in {name}"
                        )

                    size_text = box.findtext("size")

                    if size_text is None:
                        raise ValueError("Box collision missing size")

                    sx, sy, sz = map(float, size_text.split())

                    if not all(
                        math.isfinite(v) and v > 0
                        for v in (sx, sy, sz)
                    ):
                        raise ValueError("Invalid box dimensions")

                    collision_pose = self._parse_pose(
                        collision.findtext("pose")
                    )

                    if not self._yaw_is_zero(collision_pose[5]):
                        raise ValueError(
                            f"Rotated collision unsupported: {name}"
                        )

                    if any(
                        abs(p[i]) > 1e-8
                        for p in (
                            model_pose,
                            link_pose,
                            collision_pose,
                        )
                        for i in (3, 4)
                    ):
                        raise ValueError(
                            f"Roll/pitch unsupported in {name}"
                        )

                    x = (
                        model_pose[0]
                        + link_pose[0]
                        + collision_pose[0]
                    )
                    y = (
                        model_pose[1]
                        + link_pose[1]
                        + collision_pose[1]
                    )

                    obstacles.append(
                        (x, y, sx, sy, name)
                    )

        return obstacles

    @staticmethod
    def _parse_pose(text):
        if not text:
            return (0.0,) * 6

        values = tuple(map(float, text.split()))

        if len(values) != 6 or not all(
            math.isfinite(v) for v in values
        ):
            raise ValueError("Invalid SDF pose")

        return values

    @staticmethod
    def _yaw_is_zero(yaw):
        return abs(math.remainder(yaw, 2 * math.pi)) < 1e-8

    def _build_grid(self):
        """Mark cells whose centers lie within inflated boxes."""

        xs = self.xmin + (
            np.arange(self.width) + 0.5
        ) * self.resolution

        ys = self.ymin + (
            np.arange(self.height) + 0.5
        ) * self.resolution

        xx, yy = np.meshgrid(xs, ys)

        # Additional half-cell margin makes cell occupancy
        # conservative with respect to cell area.
        margin = (
            self.inflation_radius
            + self.resolution * math.sqrt(2) / 2
        )

        for x, y, sx, sy, _ in self.obstacles:
            dx = np.maximum(
                np.abs(xx - x) - sx / 2,
                0,
            )
            dy = np.maximum(
                np.abs(yy - y) - sy / 2,
                0,
            )

            self.grid[
                np.hypot(dx, dy) <= margin
            ] = 1

    def world_to_grid(self, x, y):
        """Convert world x,y into grid row,column."""

        x, y = float(x), float(y)

        if not math.isfinite(x) or not math.isfinite(y):
            return None

        if not (
            self.xmin <= x < self.xmax
            and self.ymin <= y < self.ymax
        ):
            return None

        col = int((x - self.xmin) / self.resolution)
        row = int((y - self.ymin) / self.resolution)

        if not (
            0 <= row < self.height
            and 0 <= col < self.width
        ):
            return None

        return row, col

    def grid_to_world(self, row, col):
        """Return world coordinates of a cell center."""

        if not (
            0 <= row < self.height
            and 0 <= col < self.width
        ):
            raise ValueError("Grid index out of bounds")

        x = self.xmin + (col + 0.5) * self.resolution
        y = self.ymin + (row + 0.5) * self.resolution

        return float(x), float(y)

    def plan(self, start, goal):
        """Return world-coordinate waypoints, or None."""

        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)

        if start_cell is None or goal_cell is None:
            return None

        cells = astar(
            self.grid,
            start_cell,
            goal_cell,
        )

        if cells is None:
            return None

        waypoints = [
            self.grid_to_world(row, col)
            for row, col in cells
        ]

        # Retain exact requested endpoints when their
        # corresponding cells are valid and unoccupied.
        if len(waypoints) == 1:
            if start == goal:
                return [tuple(map(float, start))]
            return waypoints

        waypoints[0] = tuple(map(float, start))
        waypoints[-1] = tuple(map(float, goal))

        return waypoints

    def validate_path(self, waypoints):
        """Validate occupancy of waypoints and axis-aligned segments."""

        if not waypoints:
            return False

        for x, y in waypoints:
            cell = self.world_to_grid(x, y)

            if cell is None or self.grid[cell] != 0:
                return False

        for a, b in zip(waypoints, waypoints[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]

            distance = math.hypot(dx, dy)

            steps = max(
                1,
                math.ceil(distance / (self.resolution / 4)),
            )

            for i in range(steps + 1):
                t = i / steps

                x = a[0] + t * dx
                y = a[1] + t * dy

                cell = self.world_to_grid(x, y)

                if cell is None or self.grid[cell] != 0:
                    return False

        return True
