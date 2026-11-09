import pytest
from environments.gridworld_env import RobotNavigationEnv


def detour_env(enabled):
    env = RobotNavigationEnv(
        height=5,
        width=5,
        obstacle_density=0,
        bfs_shaping=enabled,
        shaping_gamma=0.99,
    )
    env.reset(seed=1)
    env.obstacles[:] = 0
    env.obstacles[0:4, 2] = 1
    env.agent = (1, 1)
    env.goal = (1, 3)
    return env


def test_necessary_detour_gets_progress_reward():
    # DOWN increases Manhattan distance but reduces BFS distance from 8 to 7.
    env = detour_env(True)
    try:
        _, reward, term, trunc, info = env.step(1)
        assert reward == pytest.approx(-1 + 2 * (8 - 0.99 * 7))
        assert reward > 0
        assert not info["collision"]
        assert not term and not trunc
    finally:
        env.close()


def test_original_reward_is_preserved():
    env = detour_env(False)
    try:
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(-3)
    finally:
        env.close()


def test_collision_uses_same_state_potential():
    env = detour_env(True)
    try:
        _, reward, _, _, info = env.step(3)
        assert info["collision"]
        assert reward == pytest.approx(-11 + 2 * (8 - 0.99 * 8))
        assert reward < 0
    finally:
        env.close()


def test_goal_has_zero_potential():
    env = detour_env(True)
    try:
        env.agent, env.goal = (4, 3), (4, 4)
        _, reward, term, trunc, _ = env.step(3)
        assert term and not trunc
        assert reward == pytest.approx(101)
    finally:
        env.close()
