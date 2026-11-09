import numpy as np
import pytest
from environments.map_generator import generate_map, shortest_path
from planners.astar import astar, path_to_actions


def test_action_indices():
    assert path_to_actions([(1,1), (0,1), (1,1), (1,0), (1,1)]) == [0,1,2,3]


def test_detour_and_unreachable():
    grid = np.zeros((5,5), dtype=np.int8)
    grid[:4,2] = 1
    path = astar(grid, (1,1), (1,3))
    assert len(path) - 1 == 8
    assert all(grid[p] == 0 for p in path)
    assert path == astar(grid, (1,1), (1,3))
    grid[4,2] = 1
    assert astar(grid, (1,1), (1,3)) is None


@pytest.mark.parametrize('density', [0, .1, .35, .5])
def test_matches_bfs_on_100_maps(density):
    for seed in range(100):
        layout = generate_map(10, 10, density, np.random.default_rng(seed))
        actual = astar(layout.obstacles, layout.start, layout.goal)
        expected = shortest_path(layout.obstacles, layout.start, layout.goal)
        assert len(actual) == len(expected)
        assert actual[0] == layout.start and actual[-1] == layout.goal
        assert all(0 <= r < 10 and 0 <= c < 10 and not layout.obstacles[r,c] for r,c in actual)
        assert len(path_to_actions(actual)) == len(actual)-1


def test_endpoints_and_invalid_inputs():
    grid = np.zeros((3,3), dtype=np.int8)
    assert astar(grid, (1,1), (1,1)) == [(1,1)]
    assert astar(grid, (-1,0), (2,2)) is None
    grid[0,0] = 1
    assert astar(grid, (0,0), (2,2)) is None
    with pytest.raises(ValueError):
        astar(np.full((3,3), .5), (0,0), (2,2))
    with pytest.raises(ValueError):
        path_to_actions([(0,0), (2,0)])
    with pytest.raises(ValueError):
        astar(grid, (.5,1), (2,2))
