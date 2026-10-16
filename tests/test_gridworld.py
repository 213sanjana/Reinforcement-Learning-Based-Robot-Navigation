import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path


def test_gymnasium_checker():
    env = RobotNavigationEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()

@pytest.mark.parametrize('seed', range(25))
def test_valid_solvable_maps(seed):
    env = RobotNavigationEnv(obstacle_density=0.35)
    obs, _ = env.reset(seed=seed)
    start, goal = tuple(obs['agent']), tuple(obs['goal'])
    assert start != goal
    assert obs['obstacles'][start] == obs['obstacles'][goal] == 0
    assert shortest_path(obs['obstacles'], start, goal) is not None
    assert env.observation_space.contains(obs)
    env.close()

def test_seeded_reproducibility():
    a, b = RobotNavigationEnv(), RobotNavigationEnv()
    oa, _ = a.reset(seed=2026)
    ob, _ = b.reset(seed=2026)
    for key in oa:
        np.testing.assert_array_equal(oa[key], ob[key])
    for action in [0, 3, 1, 2]:
        ta, tb = a.step(action), b.step(action)
        for key in ta[0]:
            np.testing.assert_array_equal(ta[0][key], tb[0][key])
        assert ta[1:] == tb[1:]
    a.close(); b.close()

def test_boundary_collision():
    env = RobotNavigationEnv(obstacle_density=0)
    env.reset(seed=2)
    env.agent = (0, 0)
    obs, reward, term, trunc, info = env.step(0)
    assert tuple(obs['agent']) == (0, 0)
    assert info['collision'] and reward < 0
    assert not term and not trunc
    env.close()

def test_obstacle_collision():
    env = RobotNavigationEnv(obstacle_density=0)
    env.reset(seed=2)
    env.obstacles[:] = 0
    env.agent, env.goal = (1, 1), (9, 9)
    env.obstacles[1, 2] = 1
    obs, reward, term, trunc, info = env.step(3)
    assert tuple(obs['agent']) == (1, 1)
    assert info['collision'] and reward < 0
    assert not term and not trunc
    env.close()

def test_goal_termination():
    env = RobotNavigationEnv(obstacle_density=0)
    env.reset(seed=3)
    env.agent, env.goal = (0, 0), (0, 1)
    obs, reward, term, trunc, _ = env.step(3)
    assert term and not trunc and reward > 0
    assert tuple(obs['agent']) == (0, 1)
    with pytest.raises(RuntimeError):
        env.step(3)
    env.close()

def test_time_limit_truncation():
    env = RobotNavigationEnv(obstacle_density=0, max_steps=1)
    env.reset(seed=3)
    env.agent, env.goal = (0, 0), (9, 9)
    _, _, term, trunc, _ = env.step(0)
    assert trunc and not term
    env.close()

def test_observation_space_after_steps():
    env = RobotNavigationEnv(max_steps=20)
    obs, _ = env.reset(seed=12)
    assert env.observation_space.contains(obs)
    for action in [0,1,2,3,0,1]:
        obs, _, term, trunc, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    env.close()

def test_rgb_dimensions():
    env = RobotNavigationEnv(height=7, width=11, cell_size=24, render_mode='rgb_array')
    env.reset(seed=8)
    frame = env.render()
    assert frame.shape == (7*24, 11*24, 3)
    assert frame.dtype == np.uint8
    env.close()

def test_invalid_settings():
    with pytest.raises(ValueError):
        RobotNavigationEnv(obstacle_density=1)
    with pytest.raises(ValueError):
        RobotNavigationEnv(max_steps=0)
    with pytest.raises(ValueError):
        RobotNavigationEnv(render_mode='bad')
