"""Load warehouse PPO, reward, environment, and curriculum YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from jackal_rl_navigation.warehouse_episode import EpisodeLimits
from jackal_rl_navigation.warehouse_reward import RewardConfig
from jackal_rl_navigation.warehouse_scenarios import assert_disjoint_splits


OBSERVATION_INTERFACE = "jackal_40d_v1"
ACTION_INTERFACE = "normalized_twist_v1"

PPO_DEFAULTS = {
    "learning_rate": 0.0003,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
    "normalize_observations": False,
    "device": "auto",
    "seed": 42,
    "total_timesteps": 1024,
    "checkpoint_interval": 1024,
    "execution": "offline",
}

OFFLINE_SCENARIOS = (
    {"id": "offline_train", "split": "train", "start": [0.0, 0.0, 0.0], "goal": [1.0, 0.0]},
    {"id": "offline_val", "split": "validation", "start": [0.0, 0.2, 0.0], "goal": [1.0, 0.4]},
    {"id": "offline_test", "split": "test", "start": [-0.2, 0.0, 0.0], "goal": [1.2, -0.2]},
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def load_warehouse_config(path: str | Path | None = None, *, root: Path | None = None) -> dict:
    """Merge the PPO file with its reward, environment, and curriculum files."""

    root = repo_root() if root is None else Path(root)
    if path is None:
        config = {
            "execution": "offline",
            "split": "train",
            "stage_id": "offline_kinematic",
            "scenarios": [dict(item) for item in OFFLINE_SCENARIOS],
            "obstacles": [],
            "require_listed_pair": False,
            "world_file": None,
            "world_name": "offline_kinematic",
            "reward": {},
            "ppo": dict(PPO_DEFAULTS),
        }
    else:
        file_path = _resolve(path, root)
        config = _load_yaml(file_path)
        base = file_path.parent
        for key, target in (
            ("reward_config", "reward"),
            ("environment_config", "environment"),
            ("curriculum_config", "curriculum"),
        ):
            if key in config:
                included_path = Path(config[key])
                if not included_path.is_absolute():
                    from_root = root / included_path
                    from_file = base / included_path
                    included_path = from_root if from_root.is_file() else from_file
                config[target] = _load_yaml(included_path)
        environment = dict(config.get("environment") or {})
        stage_id = config.get("stage_id") or environment.get("active_stage")
        stages = environment.get("stages") or {}
        stage = dict(stages.get(stage_id) or {})
        scenarios = stage.get("scenarios", environment.get("scenarios", []))
        config["stage_id"] = stage_id
        config["scenarios"] = scenarios
        config["obstacles"] = stage.get("obstacles", environment.get("obstacles", []))
        config["world_file"] = stage.get("world_file", environment.get("world_file"))
        config["world_name"] = stage.get("world_name", environment.get("world_name"))
        partition = stage.get("gazebo_partition", environment.get("gazebo_partition"))
        if partition is not None:
            config["gazebo_partition"] = partition
        config["bounds_xy"] = stage.get("bounds_xy", environment.get("bounds_xy"))
        route = stage.get("route", environment.get("route"))
        if route is not None:
            config["route"] = route
        config["require_listed_pair"] = stage.get(
            "require_listed_pair",
            environment.get("require_listed_pair", False),
        )
        for field in (
            "namespace",
            "lidar_topic",
            "odom_topic",
            "contact_topic",
            "odom_frame",
            "goal_tolerance_m",
            "max_episode_steps",
            "sensor_timeout_s",
            "control_frequency_hz",
            "min_separation_m",
            "robot_radius_m",
            "stationary_linear_limit_mps",
            "stationary_angular_limit_radps",
        ):
            if field in environment and field not in config:
                config[field] = environment[field]
        if "reward" not in config:
            config["reward"] = {}
        if "split" not in config:
            config["split"] = "train"
    config["reward_config"] = RewardConfig.from_mapping(config.get("reward"))
    config["limits"] = EpisodeLimits.from_mapping(config)
    assert_disjoint_splits(config.get("scenarios") or [])
    config["observation_interface"] = OBSERVATION_INTERFACE
    config["action_interface"] = ACTION_INTERFACE
    ppo = dict(PPO_DEFAULTS)
    ppo.update(config.get("ppo") or {})
    config["ppo"] = ppo
    if config["ppo"]["normalize_observations"] is True and config["ppo"].get("normalize_actions", False):
        raise ValueError("Actions are already in [-1, 1] and must not be normalized again")
    return config
