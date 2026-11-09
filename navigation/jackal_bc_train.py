"""Supervised training for the Jackal behavior-cloning policy.

The trainer loads the frozen Phase 6F.1 observation statistics and the
initialized ``jackal_bc_v1`` policy. Expert targets are the recorded
``actions`` column, already in the Gymnasium interval [-1, 1], ordered
``[linear, angular]``. Episode mini-batches are padded inside the batch.
Padded timesteps are excluded from the masked mean squared error and from
the per-dimension errors.

Validation loss selects the best checkpoint. The test split is not opened.

Run from the repository root:

    python -m navigation.jackal_bc_train --config configs/jackal_behavior_cloning_6f2.yaml

Resume an interrupted run without replacing its output directory:

    python -m navigation.jackal_bc_train --config configs/jackal_behavior_cloning_6f2.yaml --resume
"""

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from navigation.jackal_bc_data import (
    JackalBCDataLoader,
    JackalDemonstrationCorpus,
    ObservationNormalizer,
    assert_no_split_leakage,
)
from navigation.jackal_bc_policy import (
    ARCHITECTURE,
    FORMAT_VERSION,
    JackalBCPolicy,
    load_checkpoint,
    resolve_device,
)
from navigation.jackal_bc_schema import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    ACTION_NAMES,
    OBSERVATION_DIM,
    PHASE as INFERENCE_PHASE,
    create_new_file,
    describe_contract,
    feature_names,
)


LOG = logging.getLogger(__name__)

TRAINING_PHASE = "6f.2"
OBJECTIVE = "masked_mse"
SELECTION_METRIC = "validation_masked_mse"
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "train_linear_mae",
    "train_angular_mae",
    "validation_linear_mae",
    "validation_angular_mae",
    "train_linear_mse",
    "train_angular_mse",
    "validation_linear_mse",
    "validation_angular_mse",
    "learning_rate",
    "epoch_seconds",
    "best_epoch",
    "best_validation_loss",
    "early_stop_stale",
    "train_valid_steps",
    "validation_valid_steps",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_COMMAND = (
    "python -m navigation.jackal_bc_train "
    "--config configs/jackal_behavior_cloning_6f2.yaml"
)
RESUME_COMMAND = TRAIN_COMMAND + " --resume"


class EarlyStopping:
    """Stop after ``patience`` epochs without a strict validation improvement."""

    def __init__(self, patience, min_delta):
        self.patience = _positive_int(patience, "patience")
        self.min_delta = _nonnegative_float(min_delta, "min_delta")
        self.best = float("inf")
        self.stale = 0

    def update(self, loss):
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError("Validation loss must be finite")
        if loss < self.best - self.min_delta:
            self.best = loss
            self.stale = 0
            return False
        self.stale += 1
        return self.stale >= self.patience

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "stale": self.stale,
        }

    @classmethod
    def from_state(cls, payload):
        stopper = cls(payload["patience"], payload["min_delta"])
        stopper.best = float(payload["best"])
        stopper.stale = int(payload["stale"])
        return stopper


def load_config(path):
    """Load and check a Phase 6F.2 training configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def validate_config(config):
    """Raise unless ``config`` can drive supervised Jackal behavior cloning."""

    if not isinstance(config, dict) or config.get("phase") != TRAINING_PHASE:
        raise ValueError("Configuration phase must be 6f.2")

    dataset = _mapping(config, "dataset")
    model = _mapping(config, "model")
    normalization = _mapping(config, "normalization")
    initialization = _mapping(config, "initialization")
    data = _mapping(config, "data")
    training = _mapping(config, "training")
    runtime = _mapping(config, "runtime")
    output = _mapping(config, "output")

    if dataset.get("read_test") is not False:
        raise ValueError("Phase 6F.2 cannot read the test split")
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
    _path_string(dataset.get("root"), "dataset root")

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
    if int(model["lidar_bins"]) != 36 or int(model["state_dim"]) != 4:
        raise ValueError("Policy observation layout must stay 36 LiDAR + 4 state")
    if int(model["action_dim"]) != ACTION_DIM:
        raise ValueError("Policy action must stay 2-dimensional")

    if normalization.get("fit_statistics") is not False:
        raise ValueError(
            "Training must reuse frozen train-split statistics and must not fit them"
        )
    epsilon = normalization.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("Normalization epsilon must be numeric")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1e-2:
        raise ValueError("Normalization epsilon must be in (0, 1e-2]")
    _path_string(normalization.get("statistics"), "normalization statistics")
    _path_string(initialization.get("checkpoint"), "initialization checkpoint")

    _positive_int(data.get("batch_size"), "batch_size")
    if data.get("context_length") != 1:
        raise ValueError("Phase 6F.2 context length is 1")
    if data.get("mode") != "episode":
        raise ValueError("Training mode must be episode so padding can be masked")
    workers = data.get("num_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")

    if training.get("optimizer") != "adam":
        raise ValueError("Optimizer must be adam")
    if training.get("objective") != OBJECTIVE:
        raise ValueError("Objective must be masked_mse")
    if training.get("selection_metric") != SELECTION_METRIC:
        raise ValueError("Model selection must use validation masked MSE")
    _positive_float(training.get("learning_rate"), "learning_rate")
    _nonnegative_float(training.get("weight_decay"), "weight_decay")
    _positive_int(training.get("epochs"), "epochs")
    _positive_int(training.get("patience"), "patience")
    _nonnegative_float(training.get("min_delta"), "min_delta")
    _positive_int(training.get("valid_step_chunk"), "valid_step_chunk")

    if runtime.get("deterministic") is not True:
        raise ValueError("Phase 6F.2 training is deterministic")
    _nonnegative_int(runtime.get("seed"), "seed")
    if runtime.get("device") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    _path_string(output.get("directory"), "output directory")
    return config


def masked_action_errors(predictions, targets, mask):
    """Sum squared and absolute errors on valid timesteps.

    Column 0 is linear and column 1 is angular. Padded timesteps are not
    part of either sum.
    """

    if predictions.shape != targets.shape or int(predictions.shape[-1]) != ACTION_DIM:
        raise ValueError("Predictions and targets must be shaped (N, T, 2)")
    if mask.shape != predictions.shape[:2]:
        raise ValueError("Mask must have shape (N, T)")
    if mask.dtype != torch.bool:
        raise TypeError("Mask must be boolean")
    valid_steps = int(mask.sum().item())
    if valid_steps == 0:
        raise ValueError("Batch has no valid timesteps")

    selected = (predictions - targets)[mask]
    if not torch.isfinite(selected).all():
        raise FloatingPointError("Non-finite residuals on valid timesteps")
    squared = selected.square()
    absolute = selected.abs()
    return {
        "valid_steps": valid_steps,
        "element_sse": float(squared.sum().item()),
        "linear_sse": float(squared[:, 0].sum().item()),
        "angular_sse": float(squared[:, 1].sum().item()),
        "linear_sae": float(absolute[:, 0].sum().item()),
        "angular_sae": float(absolute[:, 1].sum().item()),
    }


def reduce_split_metrics(totals):
    """Turn summed valid-step errors into means. Padding is already excluded."""

    steps = int(totals["valid_steps"])
    if steps < 1:
        raise ValueError("Split has no valid timesteps")
    elements = steps * ACTION_DIM
    return {
        "loss": totals["element_sse"] / elements,
        "linear_mae": totals["linear_sae"] / steps,
        "angular_mae": totals["angular_sae"] / steps,
        "linear_mse": totals["linear_sse"] / steps,
        "angular_mse": totals["angular_sse"] / steps,
        "valid_steps": steps,
    }


def masked_regression_step(
    policy,
    observations,
    actions,
    mask,
    *,
    chunk_size,
    training,
    context,
):
    """Compute masked MSE on one padded episode batch.

    Valid timesteps are scored in chunks so the optimizer step still matches
    the mean over the whole mini-batch. Padded observations never enter the
    network, the loss, or the error sums.
    """

    if observations.ndim != 3 or int(observations.shape[-1]) != OBSERVATION_DIM:
        raise ValueError(f"{context}: expected observations shaped (N, T, 40)")
    if actions.shape != (observations.shape[0], observations.shape[1], ACTION_DIM):
        raise ValueError(f"{context}: expected actions shaped (N, T, 2)")
    if observations.dtype != torch.float32 or actions.dtype != torch.float32:
        raise TypeError(f"{context}: observations and actions must be float32")
    if mask.shape != observations.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(f"{context}: mask must be boolean with shape (N, T)")
    chunk_size = _positive_int(chunk_size, "valid_step_chunk")

    flat_observations = observations.reshape(-1, OBSERVATION_DIM)
    flat_actions = actions.reshape(-1, ACTION_DIM)
    flat_mask = mask.reshape(-1)
    indices = flat_mask.nonzero(as_tuple=False).flatten()
    valid_steps = int(indices.numel())
    if valid_steps == 0:
        raise ValueError(f"{context}: batch has no valid timesteps")

    valid_observations = flat_observations.index_select(0, indices)
    valid_actions = flat_actions.index_select(0, indices)
    if not torch.isfinite(valid_observations).all():
        raise FloatingPointError(
            f"{context}: non-finite observations in a valid timestep"
        )
    if not torch.isfinite(valid_actions).all():
        raise FloatingPointError(
            f"{context}: non-finite expert actions in a valid timestep"
        )
    outside = (valid_actions < ACTION_LOW) | (valid_actions > ACTION_HIGH)
    if bool(torch.any(outside)):
        raise ValueError(
            f"{context}: expert actions leave the Gymnasium interval [-1, 1]"
        )

    total_elements = valid_steps * ACTION_DIM
    flat_predictions = torch.zeros(
        flat_observations.shape[0],
        ACTION_DIM,
        device=observations.device,
        dtype=observations.dtype,
    )
    if training:
        policy.zero_grad(set_to_none=True)

    for start in range(0, valid_steps, chunk_size):
        chosen = indices[start:start + chunk_size]
        predictions = policy(flat_observations.index_select(0, chosen))
        targets = flat_actions.index_select(0, chosen)
        if predictions.shape != targets.shape:
            raise RuntimeError(f"{context}: prediction and target shapes diverged")
        if not torch.isfinite(predictions).all():
            raise FloatingPointError(f"{context}: non-finite action predictions")
        flat_predictions.index_copy_(0, chosen, predictions.detach())
        if training:
            chunk_loss = (predictions - targets).square().sum() / total_elements
            if not torch.isfinite(chunk_loss):
                raise FloatingPointError(f"{context}: non-finite masked MSE")
            chunk_loss.backward()

    if training:
        require_finite_gradients(policy, context)

    predictions = flat_predictions.view(
        observations.shape[0],
        observations.shape[1],
        ACTION_DIM,
    )
    errors = masked_action_errors(predictions, actions, mask)
    if errors["valid_steps"] != valid_steps:
        raise RuntimeError(f"{context}: valid-step count changed during the forward")
    errors["loss"] = errors["element_sse"] / float(total_elements)
    if not math.isfinite(errors["loss"]):
        raise FloatingPointError(f"{context}: non-finite masked MSE")
    return errors


def require_finite_gradients(policy, context):
    """Raise when backward produced a missing or non-finite gradient."""

    found = False
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        found = True
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"{context}: non-finite gradient in parameter {name}"
            )
    if not found:
        raise FloatingPointError(f"{context}: backward produced no gradients")


def run_epoch(policy, loader, device, *, chunk_size, training, optimizer, context):
    """Run one pass over an episode loader and reduce metrics by valid steps."""

    policy.train(training)
    totals = {
        "element_sse": 0.0,
        "linear_sse": 0.0,
        "angular_sse": 0.0,
        "linear_sae": 0.0,
        "angular_sae": 0.0,
        "valid_steps": 0,
    }
    expected_steps = len(loader.dataset.corpus.transition_index)
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(loader, start=1):
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            if int(mask.sum().item()) != int(lengths.sum().item()):
                raise RuntimeError(
                    f"{context}: padding mask does not match episode lengths"
                )
            errors = masked_regression_step(
                policy,
                observations,
                actions,
                mask,
                chunk_size=chunk_size,
                training=training,
                context=f"{context} batch {batch_index}",
            )
            if training:
                optimizer.step()
            for key in totals:
                totals[key] += errors[key]
    if totals["valid_steps"] != expected_steps:
        raise RuntimeError(
            f"{context}: counted {totals['valid_steps']} valid steps, "
            f"expected {expected_steps}. Padded timesteps must not be counted."
        )
    return reduce_split_metrics(totals)


def train(config, *, resume=False, epoch_callback=None):
    """Train or resume a Phase 6F.2 run. Return the machine-readable summary."""

    config = validate_config(copy.deepcopy(config))
    device = resolve_device(config["runtime"]["device"])
    output = resolve_path(config["output"]["directory"])
    assert_output_allowed(output)
    stats_path = resolve_path(config["normalization"]["statistics"])
    init_path = resolve_path(config["initialization"]["checkpoint"])
    dataset_root = resolve_path(config["dataset"]["root"])
    _require_file(stats_path, "Frozen train normalization statistics")
    _require_file(stats_path.with_suffix(".npz"), "Frozen train normalization arrays")
    _require_file(init_path, "Phase 6F.1 initialization checkpoint")

    state = None
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"Cannot resume; {output} does not exist")
        if (output / "summary.json").is_file():
            raise FileExistsError(
                f"Refusing to overwrite completed run at {output}"
            )
        resume_path = output / "resume_state.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume; {resume_path} does not exist"
            )
        state = _load_resume_state(resume_path)
        if not (output / "config.yaml").is_file():
            raise FileNotFoundError(
                f"Cannot resume; {output / 'config.yaml'} does not exist"
            )
        if not configs_compatible(state.get("config"), config):
            raise ValueError(
                "Resume config does not match the config that started this run. "
                "Refusing to overwrite its checkpoints."
            )
    else:
        _refuse_nonempty_directory(output)

    protected = _protected_inputs(stats_path, init_path)
    protected_hashes = {str(path): file_sha256(path) for path in protected}
    seed_everything(int(config["runtime"]["seed"]), True)
    if state is not None:
        if protected_hashes != state["protected_hashes"]:
            raise RuntimeError(
                "Frozen Phase 6F.1 inputs changed since this run started; "
                "refusing to resume"
            )
        restore_rng(state["rng"], device)

    train_corpus, validation_corpus = load_train_and_validation(
        dataset_root,
        success_only=config["dataset"]["success_only"],
    )
    normalizer = load_frozen_train_normalizer(
        stats_path,
        train_corpus,
        validation_corpus,
        epsilon=float(config["normalization"]["epsilon"]),
        success_only=config["dataset"]["success_only"],
    )
    train_loader, validation_loader = make_episode_loaders(
        train_corpus,
        validation_corpus,
        batch_size=int(config["data"]["batch_size"]),
        seed=int(config["runtime"]["seed"]),
        num_workers=int(config["data"]["num_workers"]),
    )
    if state is not None:
        train_loader.generator.set_state(state["train_generator"].clone())

    settings = config["training"]
    if state is None:
        policy = initialize_policy(config, init_path, normalizer, device)
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        stopper = EarlyStopping(settings["patience"], settings["min_delta"])
        history = []
        best_loss = float("inf")
        best_epoch = 0
        start_epoch = 1
        elapsed = 0.0
        devices = [str(device)]
        _claim_fresh_directory(output)
        _write_new_json(output / "config.yaml", None, yaml_payload=config)
        _write_new_json(output / "architecture.json", architecture_metadata(policy))
        _write_new_json(
            output / "preprocessing.json",
            preprocessing_metadata(normalizer, stats_path),
        )
    else:
        policy = JackalBCPolicy.from_network_config(state["network_config"])
        policy.load_state_dict(state["model_state_dict"], strict=True)
        policy.to(device)
        _assert_normalizer_frozen(policy, normalizer)
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        optimizer_state = state["optimizer_state_dict"]
        for group in optimizer_state.get("state", {}).values():
            for key, value in list(group.items()):
                if torch.is_tensor(value):
                    group[key] = value.to(device)
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state(optimizer, device)
        stopper = EarlyStopping.from_state(state["early_stopping"])
        if stopper.patience != int(settings["patience"]):
            raise ValueError("Resume early-stopping patience does not match")
        history = list(state["history"])
        best_loss = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        elapsed = float(state["elapsed_seconds"])
        devices = list(state["devices"])
        if str(device) not in devices:
            devices.append(str(device))
        write_history(output, history)
        if state.get("stop_requested") or start_epoch > int(settings["epochs"]):
            return _finalize(
                output=output,
                config=config,
                policy=policy,
                normalizer=normalizer,
                history=history,
                best_loss=best_loss,
                best_epoch=best_epoch,
                elapsed=elapsed,
                devices=devices,
                protected_hashes=protected_hashes,
                train_corpus=train_corpus,
                validation_corpus=validation_corpus,
                stats_path=stats_path,
                init_path=init_path,
                stop_reason=state.get("stop_reason") or "max_epochs",
                optimizer=optimizer,
                stopper=stopper,
                train_loader=train_loader,
                device=device,
            )

    LOG.info(
        "BC phase=%s device=%s train_transitions=%d validation_transitions=%d "
        "parameters=%d resume=%s",
        TRAINING_PHASE,
        device,
        len(train_corpus.transition_index),
        len(validation_corpus.transition_index),
        sum(parameter.numel() for parameter in policy.parameters()),
        resume,
    )
    session_started = time.perf_counter()
    stop_reason = None
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            policy,
            train_loader,
            device,
            chunk_size=int(settings["valid_step_chunk"]),
            training=True,
            optimizer=optimizer,
            context=f"train epoch {epoch}",
        )
        _assert_normalizer_frozen(policy, normalizer)
        validation_metrics = run_epoch(
            policy,
            validation_loader,
            device,
            chunk_size=int(settings["valid_step_chunk"]),
            training=False,
            optimizer=None,
            context=f"validation epoch {epoch}",
        )
        _assert_normalizer_frozen(policy, normalizer)
        improved = validation_metrics["loss"] < best_loss - float(settings["min_delta"])
        if improved:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
        stop = stopper.update(validation_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_linear_mae": train_metrics["linear_mae"],
            "train_angular_mae": train_metrics["angular_mae"],
            "validation_linear_mae": validation_metrics["linear_mae"],
            "validation_angular_mae": validation_metrics["angular_mae"],
            "train_linear_mse": train_metrics["linear_mse"],
            "train_angular_mse": train_metrics["angular_mse"],
            "validation_linear_mse": validation_metrics["linear_mse"],
            "validation_angular_mse": validation_metrics["angular_mse"],
            "learning_rate": float(settings["learning_rate"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "early_stop_stale": stopper.stale,
            "train_valid_steps": train_metrics["valid_steps"],
            "validation_valid_steps": validation_metrics["valid_steps"],
        }
        _require_finite_history_row(row)
        history.append(row)
        stop_reason = "early_stopping" if stop else None
        if epoch == int(settings["epochs"]) and stop_reason is None:
            stop_reason = "max_epochs"
        _commit_epoch(
            output=output,
            policy=policy,
            normalizer=normalizer,
            optimizer=optimizer,
            stopper=stopper,
            train_loader=train_loader,
            device=device,
            config=config,
            history=history,
            epoch=epoch,
            best_loss=best_loss,
            best_epoch=best_epoch,
            improved=improved,
            elapsed=elapsed + (time.perf_counter() - session_started),
            devices=devices,
            protected_hashes=protected_hashes,
            stop_requested=stop,
            stop_reason=stop_reason,
            stats_path=stats_path,
            init_path=init_path,
        )
        LOG.info(
            "epoch=%d/%d train_loss=%.6f validation_loss=%.6f "
            "validation_linear_mae=%.6f validation_angular_mae=%.6f best_epoch=%d",
            epoch,
            int(settings["epochs"]),
            row["train_loss"],
            row["validation_loss"],
            row["validation_linear_mae"],
            row["validation_angular_mae"],
            best_epoch,
        )
        if epoch_callback is not None:
            epoch_callback(epoch, history)
        if stop:
            LOG.info(
                "Early stopping: %d epochs without a validation improvement",
                int(settings["patience"]),
            )
            break

    _assert_inputs_unchanged(protected, protected_hashes)
    return _finalize(
        output=output,
        config=config,
        policy=policy,
        normalizer=normalizer,
        history=history,
        best_loss=best_loss,
        best_epoch=best_epoch,
        elapsed=elapsed + (time.perf_counter() - session_started),
        devices=devices,
        protected_hashes=protected_hashes,
        train_corpus=train_corpus,
        validation_corpus=validation_corpus,
        stats_path=stats_path,
        init_path=init_path,
        stop_reason=stop_reason or "max_epochs",
        optimizer=optimizer,
        stopper=stopper,
        train_loader=train_loader,
        device=device,
    )


def load_train_and_validation(dataset_root, *, success_only):
    """Open the train and validation demonstrations. The test split stays closed."""

    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"
    validation_dir = dataset_root / "validation"
    for directory, split in ((train_dir, "train"), (validation_dir, "validation")):
        if directory.name != split or split == "test":
            raise ValueError("The test split is excluded from supervised training")
    train_corpus = JackalDemonstrationCorpus(train_dir, "train", success_only)
    validation_corpus = JackalDemonstrationCorpus(
        validation_dir,
        "validation",
        success_only,
    )
    assert_no_split_leakage(train_corpus, validation_corpus)
    return train_corpus, validation_corpus


def load_frozen_train_normalizer(
    stats_path,
    train_corpus,
    validation_corpus,
    *,
    epsilon,
    success_only,
):
    """Load Phase 6F.1 train statistics and require an exact train-split match."""

    normalizer = ObservationNormalizer.load(stats_path)
    if normalizer.source_split != "train":
        raise ValueError("Observation statistics must come from the train split")
    if normalizer.success_only != bool(success_only):
        raise ValueError("Frozen statistics success_only flag does not match the config")
    if not np.isclose(float(normalizer.epsilon), float(epsilon), rtol=0.0, atol=0.0):
        raise ValueError("Frozen statistics epsilon does not match the config")
    if normalizer.observation_sha256 != train_corpus.observation_sha256:
        raise ValueError(
            "Frozen normalization statistics do not match the train split observations"
        )
    if set(normalizer.scenario_ids) != set(train_corpus.included_scenario_ids()):
        raise ValueError("Frozen normalizer scenarios do not match the train split")
    if set(normalizer.scenario_ids) & set(validation_corpus.scenario_ids):
        raise ValueError("Train normalization scenarios overlap validation")
    if normalizer.excluded_empty != train_corpus.excluded_empty:
        raise ValueError("Frozen statistics excluded_empty count does not match train")
    if normalizer.excluded_failures != train_corpus.excluded_failures:
        raise ValueError(
            "Frozen statistics excluded_failures count does not match train"
        )
    if int(normalizer.count) != len(train_corpus.transition_index):
        raise ValueError("Frozen statistics count does not match the train split")
    return normalizer


def make_episode_loaders(train_corpus, validation_corpus, *, batch_size, seed, num_workers):
    """Build padded episode loaders. Validation order stays fixed."""

    train_loader = JackalBCDataLoader(
        train_corpus.episodes(),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        num_workers=num_workers,
        mode="episode",
    )
    validation_loader = JackalBCDataLoader(
        validation_corpus.episodes(),
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        num_workers=num_workers,
        mode="episode",
    )
    if train_loader.split != "train" or validation_loader.split != "validation":
        raise RuntimeError("Training loaders must be the train and validation splits")
    return train_loader, validation_loader


def initialize_policy(config, checkpoint_path, normalizer, device):
    """Load the untrained Phase 6F.1 policy and its frozen normalizer."""

    policy, payload = load_checkpoint(checkpoint_path, device="cpu")
    if payload.get("trained") is not False:
        raise ValueError("Initialization checkpoint must be the untrained Phase 6F.1 policy")
    if payload.get("architecture") != ARCHITECTURE or payload.get("phase") != INFERENCE_PHASE:
        raise ValueError("Initialization checkpoint is not the Phase 6F.1 Jackal policy")
    if int(payload["network_config"]["seed"]) != int(config["runtime"]["seed"]):
        raise ValueError("Initialization checkpoint seed does not match the training seed")
    _assert_same_architecture(config["model"], payload["network_config"])
    _assert_normalizer_frozen(policy, normalizer)
    recorded = payload.get("normalization") or {}
    if recorded.get("observation_sha256") != normalizer.observation_sha256:
        raise ValueError(
            "Initialization checkpoint normalization does not match the frozen statistics"
        )
    if list(payload.get("action_names") or []) != list(ACTION_NAMES):
        raise ValueError("Initialization checkpoint action order is not [linear, angular]")
    policy.to(device)
    return policy


def architecture_metadata(policy):
    """Describe the network that training is allowed to update."""

    return {
        "phase": TRAINING_PHASE,
        "inference_phase": INFERENCE_PHASE,
        "architecture": ARCHITECTURE,
        "format_version": FORMAT_VERSION,
        "network_config": policy.network_config(),
        "parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_names": list(ACTION_NAMES),
        "output_activation": "tanh",
        "output_low": [ACTION_LOW, ACTION_LOW],
        "output_high": [ACTION_HIGH, ACTION_HIGH],
    }


def preprocessing_metadata(normalizer, stats_path):
    """Record frozen observation statistics and Gymnasium action scaling."""

    contract = describe_contract()
    observation = normalizer.as_dict()
    observation["fit_during_training"] = False
    observation["statistics_path"] = str(stats_path)
    observation["consumed_by"] = TRAINING_PHASE
    return {
        "phase": TRAINING_PHASE,
        "observation_normalization": observation,
        "action_scaling": {
            "method": "identity",
            "space": "gymnasium_box",
            "dtype": "float32",
            "order": list(ACTION_NAMES),
            "low": [ACTION_LOW, ACTION_LOW],
            "high": [ACTION_HIGH, ACTION_HIGH],
            "target_key": "actions",
            "regression_target": "gymnasium_action",
            "environment_action_space": contract["action_space"],
            "velocity_limits": contract["velocity_limits"],
        },
    }


def build_inference_payload(
    policy,
    normalizer,
    *,
    epoch,
    best_epoch,
    best_validation_loss,
    seed,
):
    """Build a checkpoint that ``load_checkpoint`` can score."""

    if normalizer.source_split != "train":
        raise ValueError("Checkpoint statistics must come from the train split")
    _assert_normalizer_frozen(policy, normalizer)
    if not math.isfinite(float(best_validation_loss)):
        raise FloatingPointError("Best validation loss must be finite")
    expected_mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
    expected_std = torch.as_tensor(normalizer.std, dtype=torch.float32)
    return {
        "format_version": FORMAT_VERSION,
        "phase": INFERENCE_PHASE,
        "training_phase": TRAINING_PHASE,
        "architecture": ARCHITECTURE,
        "trained": True,
        "network_config": policy.network_config(),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in policy.state_dict().items()
        },
        "normalization": {
            "mean": expected_mean.clone(),
            "std": expected_std.clone(),
            "count": int(normalizer.count),
            "epsilon": float(normalizer.epsilon),
            "source_split": "train",
            "observation_sha256": str(normalizer.observation_sha256),
            "scenario_fingerprint": normalizer.scenario_fingerprint(),
            "scenario_count": len(normalizer.scenario_ids),
            "success_only": bool(normalizer.success_only),
            "feature_names": list(feature_names()),
        },
        "action_names": list(ACTION_NAMES),
        "action_low": [ACTION_LOW, ACTION_LOW],
        "action_high": [ACTION_HIGH, ACTION_HIGH],
        "action_normalization": "identity",
        "action_target_key": "actions",
        "objective": OBJECTIVE,
        "selection_metric": "minimum validation masked mse",
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "seed": int(seed),
    }


def plot_learning_curves(history, path):
    """Write training and validation loss curves and per-dimension errors."""

    if not history:
        raise ValueError("Cannot plot an empty training history")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Training")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation",
    )
    best_epoch = int(history[-1]["best_epoch"])
    axes[0].axvline(best_epoch, color="0.35", linewidth=1, label=f"Best epoch {best_epoch}")
    axes[0].set(xlabel="Epoch", ylabel="Masked MSE", title="Learning curve")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        epochs,
        [row["train_linear_mae"] for row in history],
        linestyle="--",
        label="Train linear",
    )
    axes[1].plot(
        epochs,
        [row["train_angular_mae"] for row in history],
        linestyle="--",
        label="Train angular",
    )
    axes[1].plot(
        epochs,
        [row["validation_linear_mae"] for row in history],
        label="Validation linear",
    )
    axes[1].plot(
        epochs,
        [row["validation_angular_mae"] for row in history],
        label="Validation angular",
    )
    axes[1].set(xlabel="Epoch", ylabel="Masked MAE", title="Action prediction error")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    def _write(temporary):
        figure.savefig(temporary, dpi=160, format="png")

    atomic_replace(path, _write)
    plt.close(figure)
    return path


def write_history(output, history):
    """Rewrite the CSV and JSON history from the committed rows."""

    output = Path(output)

    def _write_csv(temporary):
        with Path(temporary).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for row in history:
                writer.writerow({key: row[key] for key in HISTORY_FIELDS})

    atomic_replace(output / "metrics.csv", _write_csv)
    atomic_replace(
        output / "history.json",
        lambda temporary: _write_json(temporary, history),
    )


def seed_everything(seed, deterministic):
    """Seed Python, NumPy, and PyTorch."""

    seed = _nonnegative_int(seed, "seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def capture_rng(device):
    """Capture generator state needed to continue a later epoch."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    """Restore a captured generator state onto the active device."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def configs_compatible(saved, incoming):
    """Return whether two configs describe the same run.

    The resolved device may change between an interruption and a resume.
    Every other setting must match so a resume cannot retarget the run.
    """

    if not isinstance(saved, dict) or not isinstance(incoming, dict):
        return False
    left = copy.deepcopy(saved)
    right = copy.deepcopy(incoming)
    left_runtime = dict(left.get("runtime") or {})
    right_runtime = dict(right.get("runtime") or {})
    left_runtime.pop("device", None)
    right_runtime.pop("device", None)
    left["runtime"] = left_runtime
    right["runtime"] = right_runtime
    return left == right


def assert_output_allowed(path):
    """Refuse to place a run inside Phase 6F.1 artifacts or the expert dataset."""

    resolved = Path(path).resolve()
    protected = (
        REPO_ROOT / "artifacts" / "behavior_cloning" / "jackal_6f1",
        REPO_ROOT / "artifacts" / "expert_dataset",
        REPO_ROOT / "navigation",
        REPO_ROOT / "configs",
        REPO_ROOT / "tests",
    )
    for item in protected:
        item = item.resolve()
        if resolved == item or _is_relative_to(resolved, item):
            raise ValueError(f"Refusing to write training outputs into {item}")


def resolve_path(value):
    """Resolve a config path from the repository root."""

    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def file_sha256(path):
    """Hash a file without loading it all at once."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_replace(path, writer):
    """Write ``path`` by replacing it with a completed temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 6F.2 supervised Jackal behavior-cloning training",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run in its existing output directory",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_directory is not None:
        config["output"]["directory"] = args.output_directory
    config = validate_config(config)
    summary = train(config, resume=args.resume)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _commit_epoch(
    *,
    output,
    policy,
    normalizer,
    optimizer,
    stopper,
    train_loader,
    device,
    config,
    history,
    epoch,
    best_loss,
    best_epoch,
    improved,
    elapsed,
    devices,
    protected_hashes,
        stop_requested,
        stop_reason,
        stats_path,
        init_path,
):
    payload = build_inference_payload(
        policy,
        normalizer,
        epoch=epoch,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        seed=int(config["runtime"]["seed"]),
    )
    atomic_replace(output / "final.pt", lambda temporary: torch.save(payload, temporary))
    if improved:
        atomic_replace(
            output / "best.pt",
            lambda temporary: torch.save(payload, temporary),
        )
    state = _resume_payload(
        policy=policy,
        optimizer=optimizer,
        stopper=stopper,
        train_loader=train_loader,
        device=device,
        history=history,
        epoch=epoch,
        best_loss=best_loss,
        best_epoch=best_epoch,
        elapsed=elapsed,
        devices=devices,
        protected_hashes=protected_hashes,
        stop_requested=stop_requested,
        stop_reason=stop_reason,
        network_config=policy.network_config(),
        config=config,
    )
    atomic_replace(
        output / "resume_state.pt",
        lambda temporary: torch.save(state, temporary),
    )
    write_history(output, history)
    plot_learning_curves(history, output / "training_curves.png")
    _assert_inputs_unchanged(
        _protected_inputs(stats_path, init_path),
        protected_hashes,
    )


def _finalize(
    *,
    output,
    config,
    policy,
    normalizer,
    history,
    best_loss,
    best_epoch,
    elapsed,
    devices,
    protected_hashes,
    train_corpus,
    validation_corpus,
    stats_path,
    init_path,
    stop_reason,
    optimizer,
    stopper,
    train_loader,
    device,
):
    if not history:
        raise RuntimeError("Training finished without a completed epoch")
    _assert_inputs_unchanged(_protected_inputs(stats_path, init_path), protected_hashes)
    _assert_normalizer_frozen(policy, normalizer)
    write_history(output, history)
    plot_learning_curves(history, output / "training_curves.png")
    final_row = history[-1]
    best_row = history[int(best_epoch) - 1]
    if int(best_row["epoch"]) != int(best_epoch):
        raise RuntimeError("Best epoch is missing from the training history")
    if float(best_row["validation_loss"]) != float(best_loss):
        raise RuntimeError("Best validation loss does not match the history")
    summary = {
        "phase": TRAINING_PHASE,
        "architecture": ARCHITECTURE,
        "epochs_completed": len(history),
        "max_epochs": int(config["training"]["epochs"]),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "best_validation_linear_mae": best_row["validation_linear_mae"],
        "best_validation_angular_mae": best_row["validation_angular_mae"],
        "final_train_loss": final_row["train_loss"],
        "final_validation_loss": final_row["validation_loss"],
        "final_train_linear_mae": final_row["train_linear_mae"],
        "final_train_angular_mae": final_row["train_angular_mae"],
        "final_validation_linear_mae": final_row["validation_linear_mae"],
        "final_validation_angular_mae": final_row["validation_angular_mae"],
        "final_train_linear_mse": final_row["train_linear_mse"],
        "final_train_angular_mse": final_row["train_angular_mse"],
        "final_validation_linear_mse": final_row["validation_linear_mse"],
        "final_validation_angular_mse": final_row["validation_angular_mse"],
        "stopped_early": stop_reason == "early_stopping",
        "stop_reason": stop_reason,
        "device": devices[-1],
        "devices": list(devices),
        "seed": int(config["runtime"]["seed"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "batch_size": int(config["data"]["batch_size"]),
        "objective": OBJECTIVE,
        "selection_metric": "minimum validation masked mse",
        "action_target_key": "actions",
        "action_order": list(ACTION_NAMES),
        "action_normalization": "identity",
        "observation_normalization_source_split": "train",
        "observation_normalization_sha256": normalizer.observation_sha256,
        "normalizer_count": int(normalizer.count),
        "test_split_read": False,
        "train_episodes": len(train_corpus.observations),
        "train_transitions": len(train_corpus.transition_index),
        "validation_episodes": len(validation_corpus.observations),
        "validation_transitions": len(validation_corpus.transition_index),
        "parameter_count": int(
            sum(parameter.numel() for parameter in policy.parameters())
        ),
        "duration_seconds": float(elapsed),
        "output_directory": str(output),
        "best_checkpoint": str(output / "best.pt"),
        "final_checkpoint": str(output / "final.pt"),
        "history": str(output / "history.json"),
        "metrics": str(output / "metrics.csv"),
        "learning_curves": str(output / "training_curves.png"),
        "architecture_metadata": str(output / "architecture.json"),
        "preprocessing_metadata": str(output / "preprocessing.json"),
        "initialization_checkpoint": str(init_path),
        "statistics": str(stats_path),
        "protected_input_sha256": dict(protected_hashes),
        "train_command": TRAIN_COMMAND,
        "resume_command": RESUME_COMMAND,
        "software": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
    }
    summary_path = output / "summary.json"
    if not summary_path.exists():
        _write_new_json(summary_path, summary)
    state = _resume_payload(
        policy=policy,
        optimizer=optimizer,
        stopper=stopper,
        train_loader=train_loader,
        device=device,
        history=history,
        epoch=int(final_row["epoch"]),
        best_loss=best_loss,
        best_epoch=best_epoch,
        elapsed=elapsed,
        devices=devices,
        protected_hashes=protected_hashes,
        stop_requested=True,
        stop_reason=stop_reason,
        network_config=policy.network_config(),
        config=config,
    )
    state["finished"] = True
    atomic_replace(
        output / "resume_state.pt",
        lambda temporary: torch.save(state, temporary),
    )
    _assert_inputs_unchanged(_protected_inputs(stats_path, init_path), protected_hashes)
    return summary


def _resume_payload(
    *,
    policy,
    optimizer,
    stopper,
    train_loader,
    device,
    history,
    epoch,
    best_loss,
    best_epoch,
    elapsed,
    devices,
    protected_hashes,
    stop_requested,
    stop_reason,
    network_config,
    config,
):
    return {
        "phase": TRAINING_PHASE,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "history": list(history),
        "elapsed_seconds": float(elapsed),
        "devices": list(devices),
        "protected_hashes": dict(protected_hashes),
        "stop_requested": bool(stop_requested),
        "stop_reason": stop_reason,
        "finished": False,
        "network_config": network_config,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in policy.state_dict().items()
        },
        "optimizer_state_dict": _cpu_optimizer_state(optimizer),
        "early_stopping": stopper.state_dict(),
        "train_generator": train_loader.generator.get_state().clone(),
        "rng": capture_rng(device),
        "config": config,
    }


def _cpu_optimizer_state(optimizer):
    """Copy optimizer state onto CPU without moving the live tensors."""

    state = optimizer.state_dict()
    packed_state = {}
    for parameter_id, group in state.get("state", {}).items():
        packed_group = {}
        for key, value in group.items():
            if torch.is_tensor(value):
                packed_group[key] = value.detach().cpu().clone()
            else:
                packed_group[key] = value
        packed_state[parameter_id] = packed_group
    return {
        "state": packed_state,
        "param_groups": state["param_groups"],
    }


def _move_optimizer_state(optimizer, device):
    for group in optimizer.state.values():
        for key, value in list(group.items()):
            if torch.is_tensor(value):
                group[key] = value.to(device)


def _assert_normalizer_frozen(policy, normalizer):
    expected_mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
    expected_std = torch.as_tensor(normalizer.std, dtype=torch.float32)
    if not torch.equal(policy.obs_mean.detach().cpu(), expected_mean):
        raise RuntimeError("Observation mean changed or does not match the frozen statistics")
    if not torch.equal(policy.obs_std.detach().cpu(), expected_std):
        raise RuntimeError("Observation scale changed or does not match the frozen statistics")
    if int(policy.normalizer_count.item()) != int(normalizer.count):
        raise RuntimeError("Observation normalizer count changed during training")


def _assert_same_architecture(model_config, network_config):
    for name in (
        "lidar_bins",
        "state_dim",
        "action_dim",
        "lidar_kernel_size",
        "lidar_embedding",
        "state_hidden",
        "state_embedding",
    ):
        if int(model_config[name]) != int(network_config[name]):
            raise ValueError(
                f"Config {name} does not match the initialization checkpoint"
            )
    for name in ("lidar_channels", "fusion_hidden"):
        recorded = [int(value) for value in network_config[name]]
        requested = [int(value) for value in model_config[name]]
        if recorded != requested:
            raise ValueError(
                f"Config {name} does not match the initialization checkpoint"
            )


def _assert_inputs_unchanged(paths, hashes):
    current = {str(path): file_sha256(path) for path in paths}
    if current != hashes:
        raise RuntimeError(
            "Phase 6F.1 normalization or initialization files changed during training"
        )


def _protected_inputs(stats_path, init_path):
    return (Path(stats_path), Path(stats_path).with_suffix(".npz"), Path(init_path))


def _require_finite_history_row(row):
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"Non-finite training history value for {key}")


def _load_resume_state(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot read resume state at {path}: {exc}") from exc
    if not isinstance(payload, dict) or "epoch" not in payload or "history" not in payload:
        raise RuntimeError(f"Resume state at {path} is incomplete")
    if payload.get("phase") != TRAINING_PHASE:
        raise RuntimeError(f"Resume state at {path} is not a Phase 6F.2 run")
    return payload


def _refuse_nonempty_directory(path):
    path = Path(path)
    if not path.exists():
        return
    names = sorted(item.name for item in path.iterdir())
    if names:
        raise FileExistsError(
            f"Refusing to overwrite {path}; it already contains {', '.join(names)}"
        )


def _claim_fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    _refuse_nonempty_directory(path)


def _require_file(path, label):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _write_new_json(path, payload, yaml_payload=None):
    def _write(temporary):
        if yaml_payload is not None:
            text = yaml.safe_dump(yaml_payload, sort_keys=False)
            Path(temporary).write_text(text, encoding="utf-8")
            return
        _write_json(temporary, payload)

    return create_new_file(path, _write)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _mapping(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} configuration must be a mapping")
    return value


def _path_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} path is required")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _nonnegative_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return value


def _positive_ints(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    return [_positive_int(item, name) for item in values]


def _is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
