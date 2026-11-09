"""Focused BC-anchor tests.

These tests build a fresh Jackal PPO policy. They do not load the Phase 6G.1
checkpoint, call learn(), or start ROS 2 or Gazebo.
"""

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.jackal_bc_anchored_ppo import (
    JackalBCAnchoredPPO,
    _assert_same_device,
    _require_supported_device,
    adopt_initialized_model,
    actor_mean_anchor,
    anchored_objective,
    assert_reference_identity,
    reference_fingerprint,
)
from agents.jackal_pretrained_ppo_policy import JackalPolicyContractEnv, JackalPretrainedPPOPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = {
    "source_checkpoint": "artifacts/ppo/jackal_6g1/run_20261022_230534/initialized_ppo.zip",
    "source_checkpoint_sha256": "dff19be924f920490915d8f625d44f32da79452caaf3f7698160d2b8fc287750",
    "sb3_version": "2.9.0",
}


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _network_config():
    payload = json.loads((ROOT / "artifacts/behavior_cloning/jackal_6f2/architecture.json").read_text())
    return payload["network_config"]


def _install_identity_normalizer(policy):
    for extractor in (policy.pi_features_extractor, policy.vf_features_extractor):
        trunk = extractor.trunk
        with torch.no_grad():
            trunk.obs_mean.zero_()
            trunk.obs_std.fill_(1.0)
            trunk.normalizer_count.fill_(1)


def _build_ppo(device="cpu", learning_rate=3.0e-5):
    model = PPO(
        JackalPretrainedPPOPolicy,
        JackalPolicyContractEnv(),
        policy_kwargs={"network_config": _network_config(), "log_std_init": 0.0},
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=learning_rate,
        seed=642,
        device=device,
        verbose=0,
    )
    _install_identity_normalizer(model.policy)
    return model


def _observations():
    generator = torch.Generator().manual_seed(642)
    return torch.randn(4, 40, generator=generator, dtype=torch.float32)


def _critic_names(policy):
    return [
        name for name, _parameter in policy.named_parameters()
        if name.startswith("vf_features_extractor") or name.startswith("value_net")
    ]


def test_anchor_math_freezes_reference_and_trains_only_the_actor():
    model = adopt_initialized_model(_build_ppo(), coefficient=0.1, provenance=PROVENANCE)
    reference = model.reference_policy
    model.policy.set_training_mode(True)
    assert reference.training is False
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert_reference_identity(model)
    assert {id(parameter) for parameter in reference.parameters()}.isdisjoint(
        {id(parameter) for parameter in model.policy.parameters()}
    )
    observations = _observations()
    identical, _current, _reference = actor_mean_anchor(model.policy, reference, observations)
    identical_value = float(identical.detach())
    assert identical_value == pytest.approx(0.0)
    del identical
    with torch.no_grad():
        model.policy.action_net.bias.add_(0.25)
    diverged, current_mean, reference_mean = actor_mean_anchor(model.policy, reference, observations)
    assert diverged.item() > identical_value
    assert reference_mean.requires_grad is False
    assert current_mean.requires_grad is True
    model.policy.optimizer.zero_grad()
    diverged.backward()
    assert model.policy.action_net.bias.grad is not None
    assert float(model.policy.action_net.bias.grad.abs().sum()) > 0
    assert model.policy.log_std.grad is None
    for name in _critic_names(model.policy):
        parameter = dict(model.policy.named_parameters())[name]
        assert parameter.grad is None
    for parameter in reference.parameters():
        assert parameter.grad is None
    before_reference = {key: value.clone() for key, value in reference.state_dict().items()}
    before_critic = {
        name: parameter.detach().clone()
        for name, parameter in model.policy.named_parameters()
        if name in _critic_names(model.policy)
    }
    before_bias = model.policy.action_net.bias.detach().clone()
    model.policy.optimizer.step()
    assert torch.equal(model.policy.action_net.bias.detach(), before_bias) is False
    for key, value in reference.state_dict().items():
        assert torch.equal(value, before_reference[key])
    for name, parameter in model.policy.named_parameters():
        if name in before_critic:
            assert torch.equal(parameter.detach(), before_critic[name])
    assert reference_fingerprint(reference) == model.anchor_provenance["reference_sha256"]


def test_zero_coefficient_does_not_change_the_ppo_loss_object():
    ppo_loss = torch.tensor(1.25, requires_grad=True)
    anchor_loss = torch.tensor(0.5, requires_grad=True)
    assert anchored_objective(ppo_loss, anchor_loss, 0) is ppo_loss
    combined = anchored_objective(ppo_loss, anchor_loss, 0.1)
    assert torch.allclose(combined, ppo_loss + 0.1 * anchor_loss)
    source = inspect.getsource(JackalBCAnchoredPPO.train)
    anchor_source = inspect.getsource(JackalBCAnchoredPPO._include_anchor)
    assert "_include_anchor" in source
    assert "anchored_objective" in anchor_source
    assert "self.vf_coef * value_loss" in source
    assert "clip_grad_norm_" in source
    assert "learn(" not in source


def test_save_load_keeps_the_original_reference_after_actor_updates(tmp_path):
    model = adopt_initialized_model(_build_ppo(), coefficient=0.1, provenance=PROVENANCE)
    original = reference_fingerprint(model.reference_policy)
    first = tmp_path / "initial.zip"
    model.save(first)
    with torch.no_grad():
        model.policy.action_net.bias.add_(0.3)
    second = tmp_path / "updated.zip"
    model.save(second)
    restored = JackalBCAnchoredPPO.load(second, device="cpu")
    assert restored.anchor_provenance["reference_sha256"] == original
    assert reference_fingerprint(restored.reference_policy) == original
    assert torch.equal(
        restored.reference_policy.action_net.bias.detach().cpu(),
        restored.policy.action_net.bias.detach().cpu(),
    ) is False
    with pytest.raises(RuntimeError, match="already anchored"):
        adopt_initialized_model(restored, coefficient=0.1, provenance=PROVENANCE)
    with torch.no_grad():
        restored.reference_policy.action_net.bias.add_(1.0)
    with pytest.raises(RuntimeError, match="fingerprint"):
        assert_reference_identity(restored)


def test_plain_ppo_zip_cannot_supply_the_reference(tmp_path):
    model = _build_ppo()
    path = tmp_path / "plain.zip"
    model.save(path)
    with pytest.raises(ValueError, match="saved BC reference"):
        JackalBCAnchoredPPO.load(path, device="cpu")


def test_anchor_metrics_log_means_without_a_rollout(caplog):
    model = adopt_initialized_model(_build_ppo(), coefficient=0.1, provenance=PROVENANCE)
    model.set_logger(Logger(folder=None, output_formats=[]))
    caplog.set_level("INFO")
    model._publish_anchor_metrics([
        {"loss": 0.25, "current": [0.2, -0.4], "reference": [0.1, -0.2]},
    ])
    assert model.anchor_metrics["anchor_loss"] == pytest.approx(0.25)
    assert model.anchor_metrics["anchor_coefficient"] == pytest.approx(0.1)
    assert model.anchor_metrics["current_actor_mean"] == pytest.approx([0.2, -0.4])
    assert model.anchor_metrics["reference_actor_mean"] == pytest.approx([0.1, -0.2])
    assert model.anchor_metrics["actor_mean_difference"] == pytest.approx([0.1, -0.2])
    assert "BC anchor loss=" in caplog.text
    assert "current_mean=" in caplog.text
    assert "reference_mean=" in caplog.text
    assert "difference=" in caplog.text


def _assert_device_contract(model, device_type):
    assert_reference_identity(model)
    seen = set()
    for module in (model.policy, model.reference_policy):
        for tensor in (*module.parameters(), *module.buffers()):
            seen.add(tensor.device.type)
            assert tensor.device.type == device_type
    assert seen == {device_type}
    reference_ids = {id(parameter) for parameter in model.reference_policy.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert reference_ids.isdisjoint(optimizer_ids)
    assert all(not parameter.requires_grad for parameter in model.reference_policy.parameters())
    assert model.reference_policy.training is False


def _fill_rollout(model):
    buffer = model.rollout_buffer
    buffer.reset()
    device = next(model.policy.parameters()).device
    n_envs = model.n_envs
    for step in range(model.n_steps):
        obs = np.zeros((n_envs, 40), dtype=np.float32)
        obs[:, 0] = np.float32(0.05 * (step + 1))
        action = np.zeros((n_envs, 2), dtype=np.float32)
        reward = np.ones(n_envs, dtype=np.float32)
        episode_start = np.zeros(n_envs, dtype=np.float32)
        if step == 0:
            episode_start[0] = np.float32(1.0)
        obs_tensor = torch.as_tensor(obs, device=device)
        action_tensor = torch.as_tensor(action, device=device)
        with torch.no_grad():
            values = model.policy.predict_values(obs_tensor)
            log_prob = model.policy.get_distribution(obs_tensor).log_prob(action_tensor)
        buffer.add(obs, action, reward, episode_start, values, log_prob)
    with torch.no_grad():
        last_values = model.policy.predict_values(
            torch.zeros((n_envs, 40), dtype=torch.float32, device=device)
        )
    buffer.compute_returns_and_advantage(last_values, np.zeros(n_envs, dtype=bool))


def _optimizer_signature(optimizer):
    rows = []
    for index, parameter in enumerate(optimizer.param_groups[0]["params"]):
        state = optimizer.state.get(parameter)
        if not state:
            rows.append((index, None))
            continue
        step = state["step"]
        step_value = int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step)
        rows.append((
            index,
            state["exp_avg"].detach().cpu().clone(),
            state["exp_avg_sq"].detach().cpu().clone(),
            step_value,
        ))
    return rows


def _signatures_equal(left, right):
    assert len(left) == len(right)
    assert any(row[1] is not None for row in left)
    for before, after in zip(left, right):
        assert before[0] == after[0]
        if before[1] is None:
            assert after[1] is None
            continue
        assert torch.equal(before[1], after[1])
        assert torch.equal(before[2], after[2])
        assert before[3] == after[3]


def _run_synthetic_update(model, directory):
    model.set_logger(Logger(folder=None, output_formats=[]))
    device_type = next(model.policy.parameters()).device.type
    _assert_device_contract(model, device_type)
    fingerprint = reference_fingerprint(model.reference_policy)
    with torch.no_grad():
        model.policy.action_net.bias.add_(0.2)
    before_bias = model.policy.action_net.bias.detach().clone()
    updates = int(model._n_updates)
    _fill_rollout(model)
    model.train()
    assert int(model._n_updates) == updates + int(model.n_epochs)
    assert model.anchor_metrics is not None
    assert reference_fingerprint(model.reference_policy) == fingerprint
    assert torch.equal(model.policy.action_net.bias.detach(), before_bias) is False
    _assert_device_contract(model, device_type)
    signature = _optimizer_signature(model.policy.optimizer)
    path = Path(directory) / "synthetic_update.zip"
    model.save(path)
    restored = JackalBCAnchoredPPO.load(path, device=device_type)
    restored.set_logger(Logger(folder=None, output_formats=[]))
    assert reference_fingerprint(restored.reference_policy) == fingerprint
    assert restored.anchor_provenance["reference_sha256"] == fingerprint
    _signatures_equal(signature, _optimizer_signature(restored.policy.optimizer))
    _assert_device_contract(restored, device_type)
    _fill_rollout(restored)
    restored.train()
    assert int(restored._n_updates) == int(model._n_updates) + int(restored.n_epochs)
    assert reference_fingerprint(restored.reference_policy) == fingerprint
    _assert_device_contract(restored, device_type)
    return restored


def _snapshot(root):
    root = Path(root)
    records = {".": ("dir", root.stat().st_mtime_ns)}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        relative = str(path.relative_to(root))
        if path.is_symlink():
            records[relative] = ("symlink", str(path.readlink()), stat.st_mtime_ns)
        elif path.is_dir():
            records[relative] = ("dir", stat.st_mtime_ns)
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records[relative] = ("file", stat.st_size, stat.st_mtime_ns, digest)
    return records


def test_synthetic_train_save_load_keeps_reference_and_optimizer(tmp_path):
    model = adopt_initialized_model(
        _build_ppo(learning_rate=1.0e-2), coefficient=0.1, provenance=PROVENANCE,
    )
    _run_synthetic_update(model, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_synthetic_train_keeps_reference_on_the_actor_device(tmp_path):
    model = adopt_initialized_model(
        _build_ppo(device="cuda", learning_rate=1.0e-2),
        coefficient=0.1,
        provenance=PROVENANCE,
    )
    _run_synthetic_update(model, tmp_path)


def test_unsupported_or_mismatched_device_fails_clearly():
    with pytest.raises(RuntimeError, match="does not support device"):
        _require_supported_device("meta")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            _require_supported_device("cuda")
    policy = torch.nn.Linear(1, 1)
    reference = torch.nn.Linear(1, 1)
    with torch.no_grad():
        reference.weight = torch.nn.Parameter(torch.empty(1, 1, device="meta"), requires_grad=False)
    with pytest.raises(RuntimeError, match="not on the trainable actor device"):
        _assert_same_device(reference, policy)


def test_rejected_reference_resume_writes_nothing(monkeypatch):
    from curriculum.jackal_curriculum import JackalCurriculum
    from navigation import jackal_ppo_diagnostic as diagnostic
    from navigation.jackal_curriculum_train import ANCHORED_OUTPUT_NAMESPACE, load_config, run

    def boom(*_args, **_kwargs):
        raise AssertionError("rejected resume continued into execution")

    monkeypatch.setattr(diagnostic, "verify_source_hashes", boom)
    monkeypatch.setattr(diagnostic, "create_model", boom)
    monkeypatch.setattr(diagnostic, "preflight_simulator", boom)
    monkeypatch.setattr(PPO, "learn", boom)
    model = adopt_initialized_model(_build_ppo(), coefficient=0.1, provenance=PROVENANCE)
    namespace = ANCHORED_OUTPUT_NAMESPACE
    created_namespace = not namespace.exists()
    target = namespace / "audit_rejected_reference_resume"
    if target.exists():
        raise AssertionError(f"refusing to reuse unexpected directory {target}")
    namespace.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    try:
        checkpoint = target / "synthetic.zip"
        model.save(checkpoint)
        curriculum = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
        assert curriculum.steps == model.num_timesteps
        config = load_config(ROOT / "configs" / "jackal_curriculum_bc_anchored.yaml")
        (target / "marker.txt").write_text("audit-zero-write")
        (target / "summary.json").write_text('{"status": "previous"}\n')
        (target / "latest.json").write_text(json.dumps({
            "config_sha256": diagnostic.config_sha256(config),
            "checkpoint": str(checkpoint.resolve()),
            "sha256": diagnostic.file_sha256(checkpoint),
            "curriculum": curriculum.state_dict(),
            "anchor_reference_sha256": "ab" * 32,
        }))
        before = _snapshot(target)
        result = run(
            config, target, mode="pilot", resume=True,
            environment_factory=lambda: pytest.fail("env"),
        )
        assert result["status"] == "failed"
        assert "reference" in result["error"]["message"]
        assert _snapshot(target) == before
        assert not (target / "failures.jsonl").exists()
        assert not any(path.name.startswith("config_") for path in target.iterdir())
    finally:
        shutil.rmtree(target)
        if created_namespace:
            current = namespace
            while current != ROOT and current.exists() and not any(current.iterdir()):
                current.rmdir()
                current = current.parent
