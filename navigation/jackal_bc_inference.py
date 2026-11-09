"""Example forward pass for the Phase 6F.1 Jackal policy.

The command fits train-only observation statistics, saves them, writes an
initialized checkpoint, and scores one observation. It does not train and
it does not open the test split.

    python -m navigation.jackal_bc_inference --config configs/jackal_behavior_cloning_6f1.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from navigation.jackal_bc_data import make_behavior_cloning_loaders
from navigation.jackal_bc_policy import (
    ARCHITECTURE,
    build_policy,
    load_checkpoint,
    predict_numpy,
    resolve_device,
    save_checkpoint,
)
from navigation.jackal_bc_schema import PHASE, create_new_file


def load_config(path):
    """Load and check a Phase 6F.1 YAML configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def validate_config(config):
    """Raise unless this mapping is a Phase 6F.1 architecture config."""

    if not isinstance(config, dict) or config.get("phase") != PHASE:
        raise ValueError("Configuration phase must be 6f.1")

    dataset = _mapping(config, "dataset")
    model = _mapping(config, "model")
    normalization = _mapping(config, "normalization")
    data = _mapping(config, "data")
    runtime = _mapping(config, "runtime")
    checkpoint = _mapping(config, "checkpoint")
    inference = _mapping(config, "inference")

    if dataset.get("read_test") is not False:
        raise ValueError("Phase 6F.1 cannot read the test split")
    if dataset.get("train_split") != "train":
        raise ValueError("Train split must be named train")
    if dataset.get("validation_split") != "validation":
        raise ValueError("Validation split must be named validation")
    if dataset.get("test_split") != "test":
        raise ValueError("Test split must be named test")
    if dataset.get("action_key") != "actions":
        raise ValueError("Training action key must be actions")
    if not isinstance(dataset.get("success_only"), bool):
        raise ValueError("success_only must be a boolean")
    if not dataset.get("root"):
        raise ValueError("Dataset root is required")

    if model.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Architecture must be {ARCHITECTURE}")
    for name in (
        "lidar_bins",
        "state_dim",
        "action_dim",
        "lidar_kernel_size",
        "lidar_embedding",
        "state_hidden",
        "state_embedding",
    ):
        _positive_int(model.get(name), name)
    _positive_ints(model.get("lidar_channels"), "lidar_channels")
    _positive_ints(model.get("fusion_hidden"), "fusion_hidden")

    epsilon = normalization.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("Normalization epsilon must be numeric")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1e-2:
        raise ValueError("Normalization epsilon must be in (0, 1e-2]")
    if not normalization.get("output"):
        raise ValueError("Normalization output path is required")

    _positive_int(data.get("batch_size"), "batch_size")
    if data.get("context_length") != 1:
        raise ValueError("Phase 6F.1 context length is 1")
    workers = data.get("num_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")

    if runtime.get("deterministic") is not True:
        raise ValueError("Phase 6F.1 initialization is deterministic")
    _nonnegative_int(runtime.get("seed"), "seed")
    if runtime.get("device") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")

    if not checkpoint.get("output") or not inference.get("output"):
        raise ValueError("Checkpoint and inference output paths are required")
    return config


def run_initialized_inference(config, observation_path=None):
    """Fit train statistics, save an initialized policy, and score one observation."""

    config = validate_config(config)
    device = resolve_device(config["runtime"]["device"])
    stats_path = Path(config["normalization"]["output"])
    checkpoint_path = Path(config["checkpoint"]["output"])
    report_path = Path(config["inference"]["output"])
    _require_absent(stats_path, stats_path.with_suffix(".npz"), checkpoint_path, report_path)

    loaders = make_behavior_cloning_loaders(
        config["dataset"]["root"],
        batch_size=int(config["data"]["batch_size"]),
        seed=int(config["runtime"]["seed"]),
        success_only=config["dataset"]["success_only"],
        num_workers=int(config["data"]["num_workers"]),
        epsilon=float(config["normalization"]["epsilon"]),
    )
    policy = build_policy(config["model"], seed=int(config["runtime"]["seed"]))
    policy.install_normalizer(loaders.normalizer)

    stats_written = False
    try:
        loaders.normalizer.save(stats_path)
        stats_written = True
        save_checkpoint(checkpoint_path, policy, loaders.normalizer)
    except Exception:
        if stats_written and not checkpoint_path.exists():
            stats_path.unlink(missing_ok=True)
            stats_path.with_suffix(".npz").unlink(missing_ok=True)
        raise

    policy.to(device)
    policy.eval()
    if observation_path is None:
        observation = loaders.train_corpus.observations[0][0].numpy()
        source = {
            "split": "train",
            "scenario_id": loaders.train_corpus.included_records[0]["scenario_id"],
            "timestep": 0,
        }
    else:
        observation = np.load(observation_path)
        source = {"file": str(observation_path)}

    action = predict_numpy(policy, observation)
    transition_batch = next(iter(loaders.train))
    episode_batch = next(iter(loaders.train_episodes))
    report = {
        "phase": PHASE,
        "architecture": ARCHITECTURE,
        "trained": False,
        "device": str(device),
        "parameter_count": int(sum(item.numel() for item in policy.parameters())),
        "train_transitions": len(loaders.train.dataset),
        "validation_transitions": len(loaders.validation.dataset),
        "train_episodes": len(loaders.train_episodes.dataset),
        "validation_episodes": len(loaders.validation_corpus.observations),
        "normalizer_count": int(loaders.normalizer.count),
        "normalizer_source_split": loaders.normalizer.source_split,
        "normalizer_sha256": loaders.normalizer.observation_sha256,
        "test_split_read": False,
        "transition_batch_shape": list(transition_batch["observation"].shape),
        "episode_batch_shape": list(episode_batch["observations"].shape),
        "episode_batch_valid_steps": int(episode_batch["mask"].sum().item()),
        "episode_batch_padded_steps": int((~episode_batch["mask"]).sum().item()),
        "observation_source": source,
        "action": [float(value) for value in np.reshape(action, (-1, 2))[0]],
        "action_in_gymnasium_space": True,
        "checkpoint": str(checkpoint_path),
        "statistics": str(stats_path),
    }

    def _write(path):
        _write_json(path, report)

    create_new_file(report_path, _write)
    loaded, _payload = load_checkpoint(checkpoint_path, device=str(device))
    replay = predict_numpy(loaded, observation)
    if not np.allclose(
        np.reshape(action, (-1, 2)),
        np.reshape(replay, (-1, 2)),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError("Reloaded checkpoint does not reproduce the example action")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 6F.1 Jackal behavior-cloning example inference",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--observation",
        help="Optional .npy array shaped (40,) or (N, 40)",
    )
    args = parser.parse_args(argv)
    report = run_initialized_inference(load_config(args.config), args.observation)
    print(json.dumps({
        "action": report["action"],
        "action_in_gymnasium_space": report["action_in_gymnasium_space"],
        "device": report["device"],
        "checkpoint": report["checkpoint"],
        "statistics": report["statistics"],
    }, indent=2))
    return 0


def _require_absent(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"Refusing to overwrite {path}")


def _mapping(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} configuration must be a mapping")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_ints(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    return [_positive_int(item, name) for item in values]


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
