"""Phase 6F.2 supervised Jackal behavior-cloning training."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from navigation.expert_demonstration_recorder import ExpertDemonstrationRecorder
from navigation.expert_demonstration_schema import (
    DemonstrationEpisode,
    jackal_policy_spaces,
)
from navigation.jackal_bc_data import (
    ObservationNormalizer,
    load_dataset,
)
from navigation.jackal_bc_policy import (
    JackalBCPolicy,
    load_checkpoint,
    predict_numpy,
)
from navigation.jackal_bc_schema import require_env_actions
from navigation.jackal_bc_train import (
    TRAINING_PHASE,
    _cpu_optimizer_state,
    load_config,
    load_train_and_validation,
    main,
    make_episode_loaders,
    masked_action_errors,
    masked_regression_step,
    train,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "jackal_behavior_cloning_6f2.yaml"
PHASE_6F1 = ROOT / "artifacts" / "behavior_cloning" / "jackal_6f1"

TRAIN_ROWS = (
    (0, "train_0000", 4, 0.0, 0.2, True),
    (1, "train_0001", 2, 1.0, 0.4, True),
    (2, "train_0002", 0, 0.0, 0.0, False),
    (3, "train_0003", 3, 0.5, -0.1, False),
)
VALIDATION_ROWS = (
    (0, "validation_0000", 3, 10.0, 0.1, True),
    (1, "validation_0001", 5, -0.25, -0.3, True),
)


def make_episode(episode_id, scenario_id, steps, fill, action, success):
    goal = np.array([1.0, -1.0], dtype=np.float64)
    if steps == 0:
        zeros = {
            "observations": np.zeros((0, 40), dtype=np.float32),
            "actions": np.zeros((0, 2), dtype=np.float32),
            "requested_actions": np.zeros((0, 2), dtype=np.float32),
            "applied_velocities": np.zeros((0, 2), dtype=np.float32),
            "poses": np.zeros((0, 3), dtype=np.float64),
            "simulation_timestamps_ns": np.zeros((0,), dtype=np.int64),
        }
        return DemonstrationEpisode(
            episode_id=episode_id,
            scenario_id=scenario_id,
            termination_reason="PLANNING_FAILED",
            goal=goal,
            **zeros,
        )

    observations = np.full((steps, 40), 0.25, dtype=np.float32)
    observations[:, 0] = np.float32(fill)
    actions = np.zeros((steps, 2), dtype=np.float32)
    actions[:, 0] = np.float32(action)
    actions[:, 1] = np.float32(-0.2)
    requested = np.zeros((steps, 2), dtype=np.float32)
    requested[:, 0] = np.float32(-0.8)
    requested[:, 1] = np.float32(0.1)
    velocities = np.zeros((steps, 2), dtype=np.float32)
    velocities[:, 0] = np.float32(0.05)
    poses = np.zeros((steps, 3), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, 0.1, steps)
    timestamps = np.arange(steps, dtype=np.int64) * 100_000_000 + 1_000
    return DemonstrationEpisode(
        episode_id=episode_id,
        scenario_id=scenario_id,
        termination_reason="GOAL_REACHED" if success else "SAFETY_STOP",
        goal=goal,
        observations=observations,
        actions=actions,
        requested_actions=requested,
        applied_velocities=velocities,
        poses=poses,
        simulation_timestamps_ns=timestamps,
    )


def record_split(directory, rows):
    observation_space, action_space = jackal_policy_spaces()
    with ExpertDemonstrationRecorder(
        directory,
        observation_space,
        action_space,
    ) as recorder:
        for row in rows:
            recorder.record(make_episode(*row))


def file_sha256(path):
    hasher = hashlib.sha256()
    hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()


def phase_6f1_hashes():
    return {
        path.name: file_sha256(path)
        for path in sorted(PHASE_6F1.iterdir())
        if path.is_file()
    }


def make_run(tmp_path, **training_overrides):
    data_root = tmp_path / "data"
    record_split(data_root / "train", TRAIN_ROWS)
    record_split(data_root / "validation", VALIDATION_ROWS)
    train_corpus, _validation = load_train_and_validation(data_root, success_only=False)
    normalizer = ObservationNormalizer.from_observations(
        train_corpus.stacked_observations(),
        split="train",
        epsilon=1e-6,
        scenario_ids=train_corpus.included_scenario_ids(),
        success_only=False,
        excluded_empty=train_corpus.excluded_empty,
        excluded_failures=train_corpus.excluded_failures,
    )
    stats_path = tmp_path / "observation_stats.json"
    normalizer.save(stats_path)
    config = copy.deepcopy(load_config(CONFIG_PATH))
    policy = JackalBCPolicy(
        lidar_bins=config["model"]["lidar_bins"],
        state_dim=config["model"]["state_dim"],
        action_dim=config["model"]["action_dim"],
        lidar_channels=tuple(config["model"]["lidar_channels"]),
        lidar_kernel_size=config["model"]["lidar_kernel_size"],
        lidar_embedding=config["model"]["lidar_embedding"],
        state_hidden=config["model"]["state_hidden"],
        state_embedding=config["model"]["state_embedding"],
        fusion_hidden=tuple(config["model"]["fusion_hidden"]),
        seed=int(config["runtime"]["seed"]),
    )
    policy.install_normalizer(normalizer)
    from navigation.jackal_bc_policy import save_checkpoint

    init_path = tmp_path / "initialized_policy.pt"
    save_checkpoint(init_path, policy, normalizer)
    config["dataset"]["root"] = str(data_root)
    config["normalization"]["statistics"] = str(stats_path)
    config["initialization"]["checkpoint"] = str(init_path)
    config["output"]["directory"] = str(tmp_path / "run")
    config["runtime"]["device"] = "cpu"
    config["data"]["batch_size"] = 2
    config["training"]["epochs"] = 2
    config["training"]["valid_step_chunk"] = 3
    config["training"].update(training_overrides)
    return validate_config(config)


def ready_policy():
    values = np.random.default_rng(0).normal(size=(8, 40)).astype(np.float64)
    normalizer = ObservationNormalizer.from_observations(
        values,
        split="train",
        epsilon=1e-6,
        scenario_ids=["train_unit"],
        success_only=False,
        excluded_empty=0,
        excluded_failures=0,
    )
    policy = JackalBCPolicy(seed=5)
    policy.install_normalizer(normalizer)
    return policy


def padded_batch():
    observations = torch.zeros(2, 4, 40, dtype=torch.float32)
    observations[:, :, 0] = torch.tensor([0.2, 0.4], dtype=torch.float32)[:, None]
    actions = torch.zeros(2, 4, 2, dtype=torch.float32)
    actions[0, :, 0] = 0.5
    actions[0, :, 1] = -0.25
    actions[1, :, 0] = -0.5
    actions[1, :, 1] = 0.25
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :1] = True
    return observations, actions, mask


def test_default_configuration_matches_the_phase_contract():
    config = load_config(CONFIG_PATH)

    assert config["phase"] == "6f.2"
    assert config["runtime"]["seed"] == 642
    assert config["runtime"]["device"] == "auto"
    assert config["training"]["optimizer"] == "adam"
    assert config["training"]["learning_rate"] == pytest.approx(3e-4)
    assert config["training"]["epochs"] == 100
    assert config["training"]["patience"] == 10
    assert config["training"]["objective"] == "masked_mse"
    assert config["training"]["selection_metric"] == "validation_masked_mse"
    assert config["data"]["batch_size"] == 256
    assert config["data"]["mode"] == "episode"
    assert config["data"]["context_length"] == 1
    assert config["dataset"]["action_key"] == "actions"
    assert config["dataset"]["read_test"] is False
    assert config["normalization"]["fit_statistics"] is False
    assert config["model"]["architecture"] == "jackal_bc_v1"
    assert config["model"]["action_dim"] == 2


def test_configuration_rejects_test_access_and_refitting(tmp_path):
    config = make_run(tmp_path)
    read_test = copy.deepcopy(config)
    read_test["dataset"]["read_test"] = True
    with pytest.raises(ValueError, match="cannot read the test"):
        validate_config(read_test)

    requested = copy.deepcopy(config)
    requested["dataset"]["action_key"] = "requested_actions"
    with pytest.raises(ValueError, match="must be actions"):
        validate_config(requested)

    refit = copy.deepcopy(config)
    refit["normalization"]["fit_statistics"] = True
    with pytest.raises(ValueError, match="must not fit"):
        validate_config(refit)

    transition = copy.deepcopy(config)
    transition["data"]["mode"] = "transition"
    with pytest.raises(ValueError, match="episode"):
        validate_config(transition)


def test_masked_errors_ignore_padding_and_keep_action_order():
    predictions = torch.zeros(1, 2, 2)
    targets = torch.zeros(1, 2, 2)
    targets[0, 0, 0] = 0.5
    targets[0, 0, 1] = -0.25
    targets[0, 1] = 9.0
    mask = torch.tensor([[True, False]])

    errors = masked_action_errors(predictions, targets, mask)
    poisoned = targets.clone()
    poisoned[0, 1] = 100.0
    poisoned_errors = masked_action_errors(predictions, poisoned, mask)

    assert errors == poisoned_errors
    assert errors["valid_steps"] == 1
    assert errors["linear_sae"] == pytest.approx(0.5)
    assert errors["angular_sae"] == pytest.approx(0.25)
    assert errors["linear_sse"] == pytest.approx(0.25)
    assert errors["angular_sse"] == pytest.approx(0.0625)
    assert errors["element_sse"] / 2 == pytest.approx(0.15625)
    with pytest.raises(ValueError, match="no valid timesteps"):
        masked_action_errors(predictions, targets, torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(FloatingPointError, match="Non-finite residuals"):
        finite_targets = targets.clone()
        finite_targets[0, 0, 0] = float("nan")
        masked_action_errors(predictions, finite_targets, mask)


def test_regression_step_ignores_padded_nans_and_checks_finite_data():
    policy = ready_policy()
    observations, actions, mask = padded_batch()
    clean = masked_regression_step(
        policy,
        observations,
        actions,
        mask,
        chunk_size=2,
        training=False,
        context="unit",
    )
    observations = observations.clone()
    actions = actions.clone()
    observations[~mask] = float("nan")
    actions[~mask] = float("inf")
    poisoned = masked_regression_step(
        policy,
        observations,
        actions,
        mask,
        chunk_size=1,
        training=False,
        context="unit",
    )

    assert clean["loss"] == pytest.approx(poisoned["loss"])
    assert clean["valid_steps"] == 4
    assert clean["linear_sse"] == pytest.approx(poisoned["linear_sse"])
    assert clean["angular_sse"] == pytest.approx(poisoned["angular_sse"])

    broken_observations = observations.clone()
    broken_observations[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite observations"):
        masked_regression_step(
            policy,
            broken_observations,
            actions,
            mask,
            chunk_size=4,
            training=False,
            context="unit",
        )
    broken_actions = torch.zeros_like(actions)
    broken_actions[0, 0, 1] = 1.5
    with pytest.raises(ValueError, match="Gymnasium"):
        masked_regression_step(
            policy,
            torch.nan_to_num(observations),
            broken_actions,
            mask,
            chunk_size=4,
            training=False,
            context="unit",
        )


def test_chunked_gradients_match_a_single_reduction():
    observations, actions, mask = padded_batch()
    observations[~mask] = 50.0
    left = ready_policy()
    right = JackalBCPolicy(seed=9)
    right.load_state_dict(left.state_dict())

    left_errors = masked_regression_step(
        left,
        observations,
        actions,
        mask,
        chunk_size=1,
        training=True,
        context="chunk",
    )
    right_errors = masked_regression_step(
        right,
        observations,
        actions,
        mask,
        chunk_size=100,
        training=True,
        context="full",
    )

    assert left_errors["loss"] == pytest.approx(right_errors["loss"])
    assert left_errors["linear_sae"] == pytest.approx(right_errors["linear_sae"])
    assert left_errors["angular_sae"] == pytest.approx(right_errors["angular_sae"])
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(left_parameter.grad, right_parameter.grad)


def test_saved_optimizer_state_does_not_alias_live_tensors():
    policy = ready_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    observations, actions, mask = padded_batch()
    masked_regression_step(
        policy,
        observations,
        actions,
        mask,
        chunk_size=4,
        training=True,
        context="optimizer",
    )
    optimizer.step()
    live = next(iter(optimizer.state.values()))["exp_avg"]
    saved = _cpu_optimizer_state(optimizer)
    saved_tensor = next(iter(saved["state"].values()))["exp_avg"]

    assert saved_tensor.data_ptr() != live.data_ptr()
    assert saved_tensor.device.type == "cpu"
    live_before = live.detach().clone()
    with torch.no_grad():
        live.add_(1)
    assert not torch.equal(saved_tensor, live)
    torch.testing.assert_close(live - 1, live_before)


def test_nonfinite_gradients_are_reported_before_the_optimizer_step():
    policy = ready_policy()
    observations, actions, mask = padded_batch()
    parameter = next(policy.parameters())
    before = parameter.detach().clone()

    def poison(gradient):
        gradient = gradient.clone()
        gradient.view(-1)[0] = float("nan")
        return gradient

    parameter.register_hook(poison)
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        masked_regression_step(
            policy,
            observations,
            actions,
            mask,
            chunk_size=4,
            training=True,
            context="grad",
        )
    torch.testing.assert_close(parameter, before)

    broken = ready_policy()
    with torch.no_grad():
        broken.fusion[-1].bias[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite action predictions"):
        masked_regression_step(
            broken,
            observations,
            actions,
            mask,
            chunk_size=4,
            training=True,
            context="pred",
        )


def test_cuda_request_fails_clearly_when_cuda_is_unavailable(tmp_path, monkeypatch):
    config = make_run(tmp_path)
    config["runtime"]["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        train(config)
    assert not Path(config["output"]["directory"]).exists()


def test_training_cannot_write_into_phase_6f1_artifacts(tmp_path):
    before = phase_6f1_hashes()
    config = make_run(tmp_path)
    config["output"]["directory"] = str(PHASE_6F1)

    with pytest.raises(ValueError, match="Refusing to write"):
        train(config)
    assert phase_6f1_hashes() == before


def test_episode_targets_are_actions_not_requested_actions(tmp_path):
    config = make_run(tmp_path)
    train_corpus, validation_corpus = load_train_and_validation(
        config["dataset"]["root"],
        success_only=False,
    )
    train_loader, validation_loader = make_episode_loaders(
        train_corpus,
        validation_corpus,
        batch_size=2,
        seed=642,
        num_workers=0,
    )

    seen = []
    for batch in train_loader:
        for index, length in enumerate(batch["lengths"].tolist()):
            seen.append(length)
            linear = batch["actions"][index, :length, 0]
            angular = batch["actions"][index, :length, 1]
            assert torch.all(angular == -0.2)
            assert torch.all(linear != -0.8)
            assert length in (4, 2, 3)
    assert sorted(seen) == [2, 3, 4]
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)
    assert isinstance(validation_loader.sampler, torch.utils.data.SequentialSampler)


def test_training_smoke(tmp_path, monkeypatch):
    config = make_run(tmp_path)
    opened = []
    original = load_dataset

    def wrapped(path):
        opened.append(Path(path).name)
        if Path(path).name == "test":
            raise AssertionError("test split was opened")
        return original(path)

    def fail_fit(cls, *args, **kwargs):
        raise AssertionError("training must not fit normalization statistics")

    monkeypatch.setattr("navigation.jackal_bc_data.load_dataset", wrapped)
    monkeypatch.setattr(
        ObservationNormalizer,
        "from_observations",
        classmethod(fail_fit),
    )
    before = phase_6f1_hashes()
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0

    output = Path(config["output"]["directory"])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((output / "preprocessing.json").read_text(encoding="utf-8"))
    architecture = json.loads((output / "architecture.json").read_text(encoding="utf-8"))
    curves = (output / "training_curves.png").read_bytes()

    assert opened == ["train", "validation"]
    assert summary["phase"] == TRAINING_PHASE
    assert summary["epochs_completed"] == 2
    selected = min(history, key=lambda row: (row["validation_loss"], row["epoch"]))
    assert summary["best_epoch"] == selected["epoch"]
    assert summary["best_validation_loss"] == pytest.approx(selected["validation_loss"])
    assert summary["test_split_read"] is False
    assert summary["action_target_key"] == "actions"
    assert summary["action_order"] == ["linear", "angular"]
    assert summary["action_normalization"] == "identity"
    assert summary["observation_normalization_source_split"] == "train"
    assert summary["device"] == "cpu"
    assert summary["train_transitions"] == 9
    assert summary["validation_transitions"] == 8
    assert summary["parameter_count"] == 98498
    assert history[0]["train_valid_steps"] == 9
    assert history[0]["validation_valid_steps"] == 8
    assert all(math_finite(row["train_loss"]) for row in history)
    assert all(math_finite(row["validation_loss"]) for row in history)
    assert preprocessing["action_scaling"]["target_key"] == "actions"
    assert preprocessing["action_scaling"]["order"] == ["linear", "angular"]
    assert preprocessing["observation_normalization"]["fit_during_training"] is False
    assert preprocessing["observation_normalization"]["source_split"] == "train"
    assert architecture["architecture"] == "jackal_bc_v1"
    assert curves.startswith(b"\x89PNG")
    assert phase_6f1_hashes() == before

    _observation_space, action_space = jackal_policy_spaces()
    for name in ("best.pt", "final.pt"):
        policy, payload = load_checkpoint(output / name, device="cpu")
        assert payload["trained"] is True
        assert payload["training_phase"] == "6f.2"
        assert payload["phase"] == "6f.1"
        assert payload["action_names"] == ["linear", "angular"]
        assert payload["action_target_key"] == "actions"
        assert payload["normalization"]["source_split"] == "train"
        action = predict_numpy(policy, np.zeros(40, dtype=np.float32))
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert action_space.contains(action)
        require_env_actions(action)
        assert float(action[0]) != float("nan")

    with pytest.raises(FileExistsError, match="overwrite"):
        train(config)


def test_best_checkpoint_is_kept_when_validation_does_not_improve(tmp_path):
    config = make_run(tmp_path, epochs=2, min_delta=1.0e9)
    hashes = {}

    def record(epoch, _history):
        if epoch == 1:
            hashes["best"] = file_sha256(Path(config["output"]["directory"]) / "best.pt")

    summary = train(config, epoch_callback=record)
    best, payload = load_checkpoint(
        Path(config["output"]["directory"]) / "best.pt",
        device="cpu",
    )
    final, final_payload = load_checkpoint(
        Path(config["output"]["directory"]) / "final.pt",
        device="cpu",
    )

    assert hashes["best"] == file_sha256(Path(config["output"]["directory"]) / "best.pt")
    assert payload["epoch"] == 1
    assert payload["best_epoch"] == 1
    assert final_payload["epoch"] == 2
    assert summary["best_epoch"] == 1
    assert summary["epochs_completed"] == 2
    del best, final


def test_early_stopping_uses_validation_loss(tmp_path):
    config = make_run(tmp_path, epochs=5, patience=1, min_delta=1.0e9)
    summary = train(config)

    assert summary["epochs_completed"] == 2
    assert summary["stopped_early"] is True
    assert summary["stop_reason"] == "early_stopping"
    assert summary["best_epoch"] == 1
    with pytest.raises(FileExistsError, match="completed run"):
        train(config, resume=True)


def test_resume_continues_without_resetting_history_or_the_best_checkpoint(tmp_path):
    uninterrupted = make_run(tmp_path / "full", epochs=2, patience=10)
    reference = train(uninterrupted)
    interrupted = make_run(tmp_path / "partial", epochs=2, patience=10)

    def halt(epoch, _history):
        if epoch >= 1:
            raise RuntimeError("simulated interrupt")

    with pytest.raises(RuntimeError, match="simulated interrupt"):
        train(interrupted, epoch_callback=halt)
    output = Path(interrupted["output"]["directory"])
    assert not (output / "summary.json").exists()
    assert (output / "best.pt").is_file()
    best_hash = file_sha256(output / "best.pt")
    partial_history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError, match="overwrite"):
        train(interrupted, resume=False)

    resumed = train(interrupted, resume=True)
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))

    assert [row["epoch"] for row in history] == [1, 2]
    assert history[0]["train_loss"] == partial_history[0]["train_loss"]
    assert history[0]["validation_loss"] == partial_history[0]["validation_loss"]
    assert history[1]["train_loss"] == reference["final_train_loss"]
    assert history[1]["validation_loss"] == reference["final_validation_loss"]
    assert resumed["epochs_completed"] == 2
    assert resumed["best_epoch"] == reference["best_epoch"]
    if reference["best_epoch"] == 1:
        assert file_sha256(output / "best.pt") == best_hash


def test_resume_preserves_early_stopping_patience(tmp_path):
    config = make_run(tmp_path, epochs=10, patience=3, min_delta=1.0e9)

    def halt(epoch, _history):
        if epoch >= 2:
            raise RuntimeError("simulated interrupt")

    with pytest.raises(RuntimeError, match="simulated interrupt"):
        train(config, epoch_callback=halt)
    summary = train(config, resume=True)

    assert summary["epochs_completed"] == 4
    assert summary["stop_reason"] == "early_stopping"
    assert summary["best_epoch"] == 1


def test_frozen_statistics_must_match_the_train_split(tmp_path):
    config = make_run(tmp_path)
    stats_path = Path(config["normalization"]["statistics"])
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["observation_sha256"] = "0" * 64
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the train split"):
        train(config)


def math_finite(value):
    return value == value and value not in (float("inf"), float("-inf"))
