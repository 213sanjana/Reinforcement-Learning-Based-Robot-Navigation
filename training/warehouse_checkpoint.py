"""Checkpoint paths and metadata for warehouse PPO.

Runs go to a new directory. Historical experiment trees are refused.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import gymnasium
import numpy as np
import torch
import yaml

from jackal_rl_navigation.warehouse_config import ACTION_INTERFACE, OBSERVATION_INTERFACE


PROTECTED_PREFIXES = (
    "artifacts/ppo",
    "artifacts/behavior_cloning",
    "artifacts/experiments",
    "artifacts/project_audit",
    "artifacts/benchmarks",
    "artifacts/imitation",
    "artifacts/curriculum",
    "artifacts/qlearning",
    "artifacts/dqn",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def assert_output_allowed(path: Path, root: Path | None = None) -> Path:
    root = repo_root() if root is None else Path(root)
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite {resolved}")
    relative = _relative(resolved, root)
    if relative is not None:
        for prefix in PROTECTED_PREFIXES:
            if relative == prefix or relative.startswith(prefix + "/"):
                raise RuntimeError(f"Refusing to write inside protected evidence tree {prefix}")
        allowed = "artifacts/warehouse_ppo/runs"
        if not (relative == allowed or relative.startswith(allowed + "/")):
            raise RuntimeError(
                "In-repository outputs must be a new directory under artifacts/warehouse_ppo/runs"
            )
    return resolved


def assert_load_allowed(path: Path, root: Path | None = None) -> Path:
    root = repo_root() if root is None else Path(root)
    resolved = Path(path).expanduser().resolve()
    relative = _relative(resolved, root)
    if relative is not None:
        for prefix in PROTECTED_PREFIXES:
            if relative == prefix or relative.startswith(prefix + "/"):
                raise RuntimeError(
                    f"Refusing to load {resolved}. Historical checkpoints stay untouched."
                )
    return resolved


def new_run_directory(root: Path | None = None, *, stamp: str | None = None) -> Path:
    root = repo_root() if root is None else Path(root)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = os.urandom(4).hex()
    path = root / "artifacts" / "warehouse_ppo" / "runs" / f"run_{stamp}_{token}"
    return assert_output_allowed(path, root)


def software_versions() -> dict:
    import stable_baselines3

    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "git_head": commit,
    }


def select_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError("device must be auto, cpu, or cuda")


def write_metadata(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def write_config_copy(directory: Path, config: dict) -> None:
    serializable = {
        key: _plain(value)
        for key, value in config.items()
        if key not in {"reward_config", "limits"} and not str(key).startswith("_")
    }
    serializable["reward"] = {
        field: getattr(config["reward_config"], field)
        for field in config["reward_config"].__dataclass_fields__
    }
    limits = config["limits"]
    serializable["limits"] = {
        "goal_tolerance_m": limits.goal_tolerance_m,
        "max_episode_steps": limits.max_episode_steps,
        "sensor_timeout_s": limits.sensor_timeout_s,
        "control_frequency_hz": limits.control_frequency_hz,
    }
    path = directory / "training_config.yaml"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")


def metadata_payload(config: dict, device: str, seed: int) -> dict:
    return {
        "observation_interface": OBSERVATION_INTERFACE,
        "action_interface": ACTION_INTERFACE,
        "observation_shape": [40],
        "action_shape": [2],
        "normalize_observations": bool(config["ppo"]["normalize_observations"]),
        "normalize_actions": False,
        "device": device,
        "seed": int(seed),
        "stage_id": config.get("stage_id"),
        "world_name": config.get("world_name"),
        "execution": config.get("execution"),
        "split": config.get("split"),
        "versions": software_versions(),
        "ppo": dict(config["ppo"]),
    }


def assert_compatible(metadata: dict) -> None:
    if metadata.get("observation_interface") != OBSERVATION_INTERFACE:
        raise RuntimeError("Checkpoint observation interface does not match jackal_40d_v1")
    if metadata.get("action_interface") != ACTION_INTERFACE:
        raise RuntimeError("Checkpoint action interface does not match normalized_twist_v1")
    if list(metadata.get("observation_shape", [])) != [40] or list(metadata.get("action_shape", [])) != [2]:
        raise RuntimeError("Checkpoint tensor shapes do not match the Jackal warehouse interface")
    if metadata.get("normalize_actions"):
        raise RuntimeError("Checkpoint requests action normalization, which this interface rejects")
