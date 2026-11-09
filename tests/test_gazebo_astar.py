"""Tests for Gazebo SDF occupancy mapping and A*."""

from pathlib import Path

import numpy as np
import pytest

from planners.gazebo_astar import GazeboAStarPlanner


WORLD = (
    Path(__file__).resolve().parents[1]
    / "jackal_rl_ws"
    / "worlds"
    / "jackal_obstacle_test_collision.sdf"
)


@pytest.fixture
def planner():
    return GazeboAStarPlanner(WORLD)


def test_world_obstacle_extraction(planner):
    assert len(planner.obstacles) == 1

    x, y, sx, sy, name = planner.obstacles[0]

    assert name == "safety_test_box"
    assert x == pytest.approx(1.5)
    assert y == pytest.approx(0.0)
    assert sx == pytest.approx(0.5)
    assert sy == pytest.approx(0.5)


def test_grid_is_binary(planner):
    assert planner.grid.ndim == 2
    assert set(np.unique(planner.grid)).issubset({0, 1})


def test_obstacle_center_is_blocked(planner):
    cell = planner.world_to_grid(1.5, 0.0)

    assert cell is not None
    assert planner.grid[cell] == 1


def test_coordinate_conversion(planner):
    for point in [(0.0, 0.0), (2.0, 1.0), (-1.0, -1.0)]:
        cell = planner.world_to_grid(*point)

        assert cell is not None

        x, y = planner.grid_to_world(*cell)

        assert abs(x - point[0]) <= planner.resolution
        assert abs(y - point[1]) <= planner.resolution


def test_path_around_obstacle(planner):
    start = (0.0, 0.0)
    goal = (3.0, 0.0)

    path = planner.plan(start, goal)

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert planner.validate_path(path)

    # The direct route is blocked, so the path must detour.
    assert any(abs(y) > 0.7 for _, y in path)


def test_blocked_goal(planner):
    assert planner.plan((0.0, 0.0), (1.5, 0.0)) is None


def test_out_of_bounds(planner):
    assert planner.plan((-100.0, 0.0), (3.0, 0.0)) is None


def test_deterministic_path(planner):
    first = planner.plan((0.0, 0.0), (3.0, 0.0))
    second = planner.plan((0.0, 0.0), (3.0, 0.0))

    assert first == second


def test_clearance_region(planner):
    assert planner.grid[
        planner.world_to_grid(1.5, 0.5)
    ] == 1


def test_same_start_and_goal(planner):
    point = (0.0, 0.0)

    assert planner.plan(point, point) == [point]
