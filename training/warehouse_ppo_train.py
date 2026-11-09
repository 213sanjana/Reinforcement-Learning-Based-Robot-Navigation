"""Train warehouse PPO on the offline kinematic env or a live Gazebo env.

The Gazebo path is refused unless both ``--authorize-live`` and
``WAREHOUSE_RL_LIVE_AUTHORIZATION=I_AUTHORIZE_JACKAL_MOTION`` are present.
This process never loads checkpoints under the historical artifact trees.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_env import JackalWarehouseRLEnv
from jackal_rl_navigation.warehouse_estop import AUTHORIZATION_ENV, AUTHORIZATION_VALUE, live_authorized
from training.warehouse_callbacks import WarehouseTrainingCallback
from training.warehouse_checkpoint import (
    assert_compatible,
    assert_load_allowed,
    assert_output_allowed,
    metadata_payload,
    new_run_directory,
    select_device,
    write_config_copy,
    write_metadata,
)


def make_env(config: dict, *, authorize_live: bool):
    def _factory():
        return JackalWarehouseRLEnv(config, authorize_live=authorize_live)

    return _factory


def build_vec_env(config: dict, *, authorize_live: bool, training: bool):
    if int(config["ppo"].get("n_envs", 1)) != 1:
        raise RuntimeError("The Jackal warehouse has one robot, so n_envs must be 1")
    vector = DummyVecEnv([make_env(config, authorize_live=authorize_live)])
    if config["ppo"]["normalize_observations"]:
        vector = VecNormalize(
            vector,
            norm_obs=True,
            norm_reward=False,
            training=training,
            gamma=float(config["ppo"]["gamma"]),
        )
    return vector


def build_model(config: dict, vector_env, device: str, seed: int) -> PPO:
    ppo = config["ppo"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensorboard_log = None
    try:
        import tensorboard  # noqa: F401
        tensorboard_log = str(config["_run_directory"] / "tensorboard")
    except ImportError:
        tensorboard_log = None
    return PPO(
        policy=ppo["policy"],
        env=vector_env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        seed=seed,
        device=device,
        verbose=0,
        tensorboard_log=tensorboard_log,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )


def save_run(model: PPO, vector_env, directory: Path) -> None:
    model.save(str(directory / "ppo_model"))
    if isinstance(vector_env, VecNormalize):
        vector_env.save(str(directory / "vecnormalize.pkl"))


def load_run(directory: Path, vector_env, device: str, *, training: bool):
    directory = assert_load_allowed(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert_compatible(metadata)
    model_path = directory / "ppo_model.zip"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if metadata.get("normalize_observations"):
        stats = directory / "vecnormalize.pkl"
        if not stats.is_file():
            raise FileNotFoundError("Normalization was enabled but vecnormalize.pkl is missing")
        base = vector_env.venv if isinstance(vector_env, VecNormalize) else vector_env
        vector_env = VecNormalize.load(str(stats), base)
        vector_env.training = bool(training)
        vector_env.norm_reward = False
    elif (directory / "vecnormalize.pkl").is_file():
        raise RuntimeError("Unexpected normalization file for an unnormalized run")
    model = PPO.load(str(model_path), env=vector_env, device=device)
    return model, vector_env


def train(config_path: str | Path | None = None, *, config: dict | None = None, output_dir: Path | None = None, authorize_live: bool = False, resume_from: Path | None = None) -> dict:
    config = load_warehouse_config(config_path) if config is None else config
    execution = config.get("execution", "offline")
    if execution == "stationary_observation":
        raise SystemExit(
            "Refusing training on the stationary observation path. "
            "step() stays blocked and no checkpoint is loaded."
        )
    if execution == "gazebo" and not authorize_live:
        raise SystemExit(
            "Refusing Gazebo PPO training. It would create the robot command "
            "interface. Re-run only with a separate motion authorization."
        )
    if execution != "gazebo" and authorize_live:
        raise RuntimeError("authorize_live is only valid when execution is gazebo")
    if output_dir is None:
        output_dir = new_run_directory()
    else:
        output_dir = assert_output_allowed(output_dir)
    output_dir.mkdir(parents=True)
    config = dict(config)
    config["_run_directory"] = output_dir
    seed = int(config["ppo"]["seed"])
    device = select_device(config["ppo"]["device"])
    write_config_copy(output_dir, config)
    write_metadata(output_dir, metadata_payload(config, device, seed))
    vector_env = build_vec_env(config, authorize_live=authorize_live, training=True)
    if resume_from is None:
        model = build_model(config, vector_env, device, seed)
        already = 0
    else:
        model, vector_env = load_run(Path(resume_from), vector_env, device, training=True)
        already = int(model.num_timesteps)
    remaining = int(config["ppo"]["total_timesteps"]) - already
    callback = WarehouseTrainingCallback(
        output_dir,
        checkpoint_interval=int(config["ppo"]["checkpoint_interval"]),
        min_episodes_for_best=int(config["ppo"].get("min_episodes_for_best", 5)),
    )
    result = {
        "output_dir": str(output_dir),
        "device": device,
        "execution": execution,
        "resumed_timesteps": already,
        "trained": False,
    }
    if remaining <= 0:
        save_run(model, vector_env, output_dir / "latest")
        result["stop_reason"] = "timestep budget already consumed"
        return result
    try:
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=resume_from is None)
        result["trained"] = True
    except RuntimeError as exc:
        result["stop_reason"] = str(exc)
        callback.stop_reason = str(exc)
    finally:
        save_run(model, vector_env, output_dir)
        if callback.stop_reason and not (output_dir / "stop_reason.txt").exists():
            (output_dir / "stop_reason.txt").write_text(callback.stop_reason + "\n", encoding="utf-8")
        vector_env.close()
    result["timesteps"] = int(model.num_timesteps)
    result["stop_reason"] = callback.stop_reason
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train warehouse PPO. Offline by config; Gazebo only when authorized.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    authorized = live_authorized(args.authorize_live)
    if args.authorize_live and not authorized:
        print(
            f"--authorize-live also requires {AUTHORIZATION_ENV}={AUTHORIZATION_VALUE}.",
        )
        return 2
    if args.resume_from and os.environ.get("WAREHOUSE_RL_ALLOW_RESUME") != "1":
        print("Resume is disabled unless WAREHOUSE_RL_ALLOW_RESUME=1. No checkpoint was loaded.")
        return 2
    result = train(
        args.config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        authorize_live=authorized,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )
    print(json.dumps({key: result[key] for key in result if key != "model"}, sort_keys=True))
    return 0 if result.get("trained") or result.get("stop_reason") else 1


if __name__ == "__main__":
    raise SystemExit(main())
