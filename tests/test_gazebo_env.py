"""Foundation tests for the Jackal Gymnasium environment."""

import ast
from pathlib import Path

import gymnasium as gym
import numpy as np

from jackal_rl_navigation.gazebo_env import JackalGazeboEnv


def test_gymnasium_inheritance():
    assert issubclass(JackalGazeboEnv, gym.Env)


def test_observation_space_definition():
    source = Path(
        "jackal_rl_ws/src/jackal_rl_navigation/"
        "jackal_rl_navigation/gazebo_env.py"
    ).read_text()

    tree = ast.parse(source)
    assert tree is not None
    assert "shape=(40,)" in source


def test_action_space_definition():
    source = Path(
        "jackal_rl_ws/src/jackal_rl_navigation/"
        "jackal_rl_navigation/gazebo_env.py"
    ).read_text()

    assert "shape=(2,)" in source


def test_gymnasium_space_compatibility():
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(40,),
        dtype=np.float32,
    )

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
    )

    assert observation_space.contains(
        np.zeros(40, dtype=np.float32)
    )

    assert action_space.contains(
        np.array([0.5, -0.5], dtype=np.float32)
    )

    assert not action_space.contains(
        np.array([2.0, 0.0], dtype=np.float32)
    )
