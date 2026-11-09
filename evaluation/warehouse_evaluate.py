"""Evaluate a warehouse checkpoint without updating normalization statistics.

Gazebo evaluation is refused unless live authorization is present. This module
does not report success for a method that was not rolled out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.warehouse_metrics import aggregate_episodes
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_env import JackalWarehouseRLEnv
from jackal_rl_navigation.warehouse_estop import live_authorized
from training.warehouse_checkpoint import assert_load_allowed, select_device
from training.warehouse_ppo_train import build_vec_env, load_run


def rollout_policy(env: JackalWarehouseRLEnv, policy, *, episodes: int, seed: int) -> list[dict]:
    records = []
    for index in range(int(episodes)):
        observation, info = env.reset(seed=seed + index)
        done = False
        last = dict(info)
        while not done:
            action = policy(observation, last)
            observation, _reward, terminated, truncated, last = env.step(action)
            done = bool(terminated or truncated)
        last = dict(last)
        last["episode_index"] = index
        records.append(last)
        if last.get("fatal"):
            break
    return records


def evaluate(config: dict, checkpoint: Path | None, *, episodes: int, seed: int, authorize_live: bool, policy=None) -> dict:
    execution = config.get("execution", "offline")
    if execution == "stationary_observation":
        raise SystemExit(
            "Refusing inference on the stationary observation path. "
            "step() and checkpoint loading stay blocked."
        )
    if execution == "gazebo" and not authorize_live:
        raise SystemExit("Refusing Gazebo evaluation because it would command the robot.")
    if policy is None:
        if checkpoint is None:
            raise ValueError("A checkpoint or an explicit policy is required")
        checkpoint = assert_load_allowed(checkpoint)
        device = select_device(config["ppo"]["device"])
        vector_env = build_vec_env(config, authorize_live=authorize_live, training=False)
        model, vector_env = load_run(checkpoint, vector_env, device, training=False)
        if hasattr(vector_env, "training"):
            vector_env.training = False
            vector_env.norm_reward = False

        normalizer = vector_env if hasattr(vector_env, "normalize_obs") else None
        inner = vector_env.venv.envs[0] if hasattr(vector_env, "venv") else vector_env.envs[0]

        def policy(observation, _info):
            model_observation = observation
            if normalizer is not None:
                batched = np.asarray(observation, dtype=np.float32)[None, :]
                model_observation = np.asarray(normalizer.normalize_obs(batched)[0], dtype=np.float32)
            action, _state = model.predict(model_observation, deterministic=True)
            return np.asarray(action, dtype=np.float32)

        try:
            records = rollout_policy(inner, policy, episodes=episodes, seed=seed)
        finally:
            vector_env.close()
    else:
        env = JackalWarehouseRLEnv(config, authorize_live=authorize_live)
        try:
            records = rollout_policy(env, policy, episodes=episodes, seed=seed)
        finally:
            env.close()
    summary = aggregate_episodes(records)
    summary["execution"] = execution
    summary["checkpoint"] = None if checkpoint is None else str(checkpoint)
    summary["claimed_success"] = False
    summary["note"] = (
        "Offline kinematic rollouts are not Gazebo results. "
        "A success rate here does not mean the Jackal reached a warehouse goal."
    )
    return {"summary": summary, "episodes": records}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a warehouse PPO checkpoint. Reads a run directory.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    config = load_warehouse_config(args.config)
    config["split"] = "test"
    result = evaluate(
        config,
        Path(args.checkpoint),
        episodes=args.episodes,
        seed=args.seed,
        authorize_live=live_authorized(args.authorize_live),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".episodes.json").write_text(
        json.dumps(result["episodes"], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
