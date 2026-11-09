"""Phase 6G.1 behavior-cloning actor transfer into PPO."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs
from torch import nn

from agents.jackal_pretrained_ppo_policy import (
    SUPPORTED_SB3_VERSION,
    JackalPolicyContractEnv,
    JackalPretrainedPPOPolicy,
    check_sb3_version,
)
from navigation.jackal_bc_data import ObservationNormalizer
from navigation.jackal_bc_policy import JackalBCPolicy, load_checkpoint, save_checkpoint
from navigation.jackal_bc_ppo_transfer import (
    file_sha256,
    initialize_jackal_ppo,
    load_config,
    load_jackal_bc_for_transfer,
    transfer_actor,
    validate_equivalence,
)
from navigation.jackal_bc_schema import require_env_actions
from navigation.expert_demonstration_schema import jackal_policy_spaces


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "jackal_ppo_transfer_6g1.yaml"
BEST_CHECKPOINT = ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "best.pt"
PROTECTED = (
    BEST_CHECKPOINT,
    ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "final.pt",
    ROOT / "artifacts" / "behavior_cloning" / "jackal_6f2" / "preprocessing.json",
    ROOT / "artifacts" / "behavior_cloning" / "jackal_6f1" / "observation_stats.json",
    ROOT / "artifacts" / "behavior_cloning" / "jackal_6f1" / "observation_stats.npz",
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_version_spaces_and_environment_contract():
    assert check_sb3_version() == SUPPORTED_SB3_VERSION
    observation_space, action_space = jackal_policy_spaces()
    assert observation_space.shape == (40,)
    assert action_space.shape == (2,)
    assert not is_image_space(observation_space)
    batch = torch.arange(80, dtype=torch.float32).reshape(2, 40)
    processed = preprocess_obs(batch, observation_space, normalize_images=False)
    assert torch.equal(processed, batch)
    env = JackalPolicyContractEnv()
    observation, _info = env.reset(seed=0)
    assert observation.shape == (40,) and env.resets == 1 and env.steps == 0
    with pytest.raises(RuntimeError, match="does not step"):
        env.step(np.zeros(2, dtype=np.float32))
    assert env.steps == 1
    for bad_observation in (
        spaces.Box(0, 1, shape=(40,), dtype=np.float32),
        spaces.Box(-np.inf, np.inf, shape=(39,), dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            JackalPretrainedPPOPolicy(
                bad_observation,
                action_space,
                lambda _progress: 0.0,
                network_config=_network_config(),
            )


def test_synthetic_transfer_matches_bc_and_leaves_the_critic_independent(tmp_path):
    checkpoint = _save_synthetic_checkpoint(tmp_path / "bc.pt")
    source_hash = file_sha256(checkpoint)
    protected = {path: file_sha256(path) for path in PROTECTED}
    bc_policy, _payload = load_jackal_bc_for_transfer(checkpoint, device="cpu")
    model = _ppo(bc_policy)
    try:
        critic_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.policy.vf_features_extractor.named_parameters()
        }
        log_std_before = model.policy.log_std.detach().cpu().clone()
        report = transfer_actor(bc_policy, model.policy)
        assert report["transferred_parameter_elements"] == sum(
            parameter.numel() for parameter in bc_policy.parameters()
        )
        assert report["transferred_parameter_tensors"] > 0
        assert report["transferred_buffer_tensors"] == 3
        assert report["critic_parameters_unchanged_by_transfer"]
        assert report["critic_parameters_equal_to_actor_trunk"] == []
        assert torch.equal(model.policy.log_std.detach().cpu(), log_std_before)
        assert torch.equal(
            model.policy.log_std.detach().cpu(),
            torch.zeros_like(log_std_before),
        )
        for name, parameter in model.policy.vf_features_extractor.named_parameters():
            assert torch.equal(parameter.detach().cpu(), critic_before[name])
            actor = dict(model.policy.pi_features_extractor.named_parameters())[name]
            assert not torch.equal(parameter.detach(), actor.detach())
        observations = np.random.default_rng(642).normal(size=(8, 40)).astype(np.float32)
        verification = validate_equivalence(bc_policy, model, observations, 0.0, 0.0)
        assert verification["deterministic_mean_exact"]
        assert verification["deterministic_mean_max_absolute_error"] == 0.0
        assert verification["second_normalization"] is False
        assert verification["squash_output"] is False
        assert verification["unscale_action_applied"] is False
        deterministic = model.predict(observations, deterministic=True)[0]
        with torch.no_grad():
            model.policy.log_std.add_(1.5)
            model.policy.value_net.weight.add_(1.0)
            model.policy.vf_features_extractor.trunk.state_encoder.network[0].weight.add_(1.0)
        unchanged = model.predict(observations, deterministic=True)[0]
        np.testing.assert_array_equal(deterministic, unchanged)
        _assert_predict_clips_gaussian_samples(model, observations[0])
        assert file_sha256(checkpoint) == source_hash
        assert {path: file_sha256(path) for path in PROTECTED} == protected
        assert model.num_timesteps == 0 and model.policy.optimizer.state == {}
        assert model.get_env().envs[0].unwrapped.steps == 0
    finally:
        model.get_env().close()


def test_incompatible_checkpoints_and_invalid_observations_fail(tmp_path):
    checkpoint = _save_synthetic_checkpoint(tmp_path / "bc.pt")
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cases = []
    missing = copy.deepcopy(original)
    del missing["model_state_dict"]["fusion.4.bias"]
    cases.append(missing)
    extra = copy.deepcopy(original)
    extra["model_state_dict"]["fusion.6.weight"] = torch.zeros(2, 2)
    cases.append(extra)
    shape = copy.deepcopy(original)
    shape["model_state_dict"]["fusion.4.weight"] = torch.zeros(3, 64)
    cases.append(shape)
    nonfinite = copy.deepcopy(original)
    nonfinite["model_state_dict"]["fusion.4.bias"] = torch.tensor([0.0, float("nan")])
    cases.append(nonfinite)
    swapped = copy.deepcopy(original)
    swapped["action_names"] = ["angular", "linear"]
    cases.append(swapped)
    scaled = copy.deepcopy(original)
    scaled["action_normalization"] = "velocity"
    cases.append(scaled)
    for index, payload in enumerate(cases):
        path = tmp_path / f"bad_{index}.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError):
            load_jackal_bc_for_transfer(path)

    bc_policy, _payload = load_jackal_bc_for_transfer(checkpoint)
    model = _ppo(bc_policy)
    try:
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.policy.pi_features_extractor.named_parameters()
        }
        model.policy.action_net = nn.Linear(64, 3)
        with pytest.raises(ValueError, match="Incompatible tensor"):
            transfer_actor(bc_policy, model.policy)
        after = dict(model.policy.pi_features_extractor.named_parameters())
        for name, value in before.items():
            assert torch.equal(value, after[name].detach().cpu())
    finally:
        model.get_env().close()

    incompatible = load_checkpoint(checkpoint, device="cpu")[0]
    incompatible.fusion = nn.Sequential(nn.Linear(96, 2), nn.Tanh())
    rejected = _ppo(bc_policy)
    try:
        with pytest.raises(ValueError, match="tanh"):
            transfer_actor(incompatible, rejected.policy)
    finally:
        rejected.get_env().close()

    accepted = _ppo(bc_policy)
    try:
        transfer_actor(bc_policy, accepted.policy)
        tensor = torch.zeros(2, 40)
        tensor[0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            accepted.policy.raw_and_mean(tensor)
        with pytest.raises(ValueError, match="40"):
            accepted.policy.raw_and_mean(torch.zeros(2, 39))
        with pytest.raises(ValueError):
            accepted.predict(np.zeros(39, dtype=np.float32), deterministic=True)
        with pytest.raises(ValueError, match="finite"):
            accepted.predict(np.full(40, np.nan, dtype=np.float32), deterministic=True)
    finally:
        accepted.get_env().close()


def test_failed_validation_publishes_nothing_and_learn_is_blocked(tmp_path, monkeypatch):
    checkpoint = _save_synthetic_checkpoint(tmp_path / "bc.pt")
    protected = {path: file_sha256(path) for path in PROTECTED}
    output = tmp_path / "run"

    def fail(*_args, **_kwargs):
        raise ValueError("intentional equivalence failure")

    monkeypatch.setattr(
        "navigation.jackal_bc_ppo_transfer.validate_equivalence",
        fail,
    )
    with pytest.raises(ValueError, match="equivalence failure"):
        initialize_jackal_ppo(_config(checkpoint, tmp_path), output)
    assert not output.exists()
    monkeypatch.undo()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("PPO.learn must not run during initialization")

    monkeypatch.setattr(PPO, "learn", forbidden)
    summary = initialize_jackal_ppo(_config(checkpoint, tmp_path), output)
    assert summary["status"] == "passed"
    assert summary["training_started"] is False
    assert summary["ppo_num_timesteps"] == 0
    assert summary["deterministic_mean_exact"] is True
    assert (output / "initialized_ppo.zip").is_file()
    assert (output / "transfer_mapping.json").is_file()
    assert (output / "verification.json").is_file()
    reloaded = PPO.load(output / "initialized_ppo.zip", device="cpu")
    bc_policy, _payload = load_checkpoint(checkpoint, device="cpu")
    observations = np.load(output / "equivalence_observations.npz")["observations"]
    try:
        verification = validate_equivalence(bc_policy, reloaded, observations, 0.0, 0.0)
    finally:
        if reloaded.get_env() is not None:
            reloaded.get_env().close()
    assert verification["deterministic_mean_exact"]
    assert {path: file_sha256(path) for path in PROTECTED} == protected
    with pytest.raises(FileExistsError):
        initialize_jackal_ppo(_config(checkpoint, tmp_path), output)


def test_phase_6f2_checkpoint_transfer_and_fresh_process(tmp_path):
    protected = {path: file_sha256(path) for path in PROTECTED}
    config = load_config(CONFIG_PATH)
    output = tmp_path / "run_6f2"
    summary = initialize_jackal_ppo(config, output)
    assert summary["status"] == "passed"
    assert summary["source_checkpoint_sha256"] == protected[BEST_CHECKPOINT]
    assert summary["deterministic_mean_exact"] is True
    assert summary["transferred_parameter_elements"] == 98498
    assert summary["environment_steps"] == 0
    references = json.loads((output / "source_references.json").read_text())
    assert references["evaluation_context"]["used_for_weights"] is False
    assert references["evaluation_context"]["checkpoint_sha256_matches_source"] is True
    assert references["preprocessing"]["action_scaling"] == "identity"
    mapping = json.loads((output / "transfer_mapping.json").read_text())
    assert mapping["newly_initialized_groups"] == [
        "gaussian_log_std",
        "critic_trunk",
        "value_head",
    ]
    assert all(row["exact_match"] for row in mapping["transferred"])
    assert {path: file_sha256(path) for path in PROTECTED} == protected
    fresh = json.loads((output / "verification.json").read_text())["fresh_process"]
    assert fresh["loaded"] is True
    assert fresh["deterministic_action_exact"] is True


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "navigation.jackal_bc_ppo_transfer", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--output-directory" in result.stdout


def _assert_predict_clips_gaussian_samples(model, observation):
    def outside_box(self):
        del self
        return torch.tensor([[5.0, -4.0]], dtype=torch.float32)

    original = DiagGaussianDistribution.sample
    DiagGaussianDistribution.sample = outside_box
    try:
        deterministic = model.predict(observation, deterministic=True)[0]
        clipped, _state = model.predict(observation, deterministic=False)
    finally:
        DiagGaussianDistribution.sample = original
    np.testing.assert_array_equal(
        clipped,
        np.array([1.0, -1.0], dtype=np.float32),
    )
    require_env_actions(clipped)
    assert deterministic.shape == (2,)
    assert np.max(np.abs(deterministic)) <= 1.0


def _save_synthetic_checkpoint(path):
    observations = np.random.default_rng(0).normal(size=(8, 40))
    normalizer = ObservationNormalizer.from_observations(
        observations,
        split="train",
        epsilon=1e-6,
        scenario_ids=("train_0000", "train_0001"),
        success_only=False,
        excluded_empty=0,
        excluded_failures=0,
    )
    policy = JackalBCPolicy(seed=3)
    policy.install_normalizer(normalizer)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.25)
    save_checkpoint(path, policy, normalizer)
    return path


def _ppo(bc_policy):
    return PPO(
        JackalPretrainedPPOPolicy,
        JackalPolicyContractEnv(),
        policy_kwargs={
            "network_config": bc_policy.network_config(),
            "log_std_init": 0.0,
        },
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=3e-5,
        seed=642,
        device="cpu",
        verbose=0,
    )


def _config(checkpoint, tmp_path):
    config = load_config(CONFIG_PATH)
    config["source"] = {
        "checkpoint": str(checkpoint),
        "require_trained": False,
        "architecture": None,
        "preprocessing": None,
        "evaluation_context": None,
    }
    config["protected"] = [str(path) for path in PROTECTED]
    config["output"] = {"directory": str(tmp_path)}
    config["validation"] = {
        "observations": 8,
        "seed": 642,
        "atol": 0.0,
        "rtol": 0.0,
    }
    config["ppo"]["n_steps"] = 8
    config["ppo"]["batch_size"] = 4
    config["ppo"]["n_epochs"] = 1
    return config


def _network_config():
    return JackalBCPolicy(seed=0).network_config()
