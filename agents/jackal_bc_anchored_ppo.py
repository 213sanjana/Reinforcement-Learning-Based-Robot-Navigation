"""PPO whose actor mean stays near the verified BC initialization.

Stable-Baselines3 2.9.0 minimizes one scalar in ``PPO.train``:

    policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

``policy_loss`` is already the negated clipped surrogate. The anchor term is
added to that same minimized scalar and is not added to the value loss:

    total = ppo_loss + anchor_coefficient * mean((current_mean - reference_mean)^2)

``current_mean`` and ``reference_mean`` are the deterministic tanh means from
``JackalPretrainedPPOPolicy.raw_and_mean``. Both action components are averaged
together. The reference forward runs under ``torch.no_grad``, its parameters
have ``requires_grad`` false, and those parameters are not in the PPO optimizer.

The reference is captured once from the actor that was just loaded. Resume
restores that saved reference. It does not copy the resumed actor again.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from agents.jackal_pretrained_ppo_policy import (
    SUPPORTED_SB3_VERSION,
    JackalPretrainedPPOPolicy,
    check_sb3_version,
)


LOGGER = logging.getLogger(__name__)


def validate_anchor_coefficient(value):
    """Return a finite nonnegative coefficient. Zero disables the added term."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("anchor coefficient must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("anchor coefficient must be finite and nonnegative")
    return value


def anchored_objective(ppo_loss, anchor_loss, coefficient):
    """Add the anchor penalty to the minimized PPO scalar.

    A zero coefficient returns ``ppo_loss`` unchanged so the autograd graph
    stays the audited PPO graph.
    """

    coefficient = validate_anchor_coefficient(coefficient)
    if coefficient == 0.0:
        return ppo_loss
    return ppo_loss + coefficient * anchor_loss


def actor_mean_anchor(current_policy, reference_policy, observations):
    """Mean squared difference of deterministic tanh action means.

    Observations are the raw rollout batch. ``raw_and_mean`` applies the
    frozen normalizer inside each trunk and does not add a second one.
    The reference must already be on the trainable actor's device.
    """

    device = _assert_same_device(reference_policy, current_policy)
    if observations.device != device or observations.dtype != th.float32:
        try:
            observations = observations.to(device=device, dtype=th.float32)
        except Exception as exc:
            raise RuntimeError(
                f"BC anchor observations cannot be placed on {device} as float32"
            ) from exc
    reference_policy.eval()
    _current_raw, current_mean = current_policy.raw_and_mean(observations)
    with th.no_grad():
        _reference_raw, reference_mean = reference_policy.raw_and_mean(observations)
    if reference_mean.requires_grad:
        raise RuntimeError("Gradients reached the frozen BC reference actor")
    reference_mean = reference_mean.detach()
    if current_mean.shape != reference_mean.shape or current_mean.shape[-1] != 2:
        raise RuntimeError("Actor means must be batches of [linear, angular]")
    anchor_loss = th.mean((current_mean - reference_mean) ** 2)
    return anchor_loss, current_mean, reference_mean


def reference_fingerprint(policy):
    """SHA-256 of the reference parameters and normalizer buffers."""

    digest = hashlib.sha256()
    state = policy.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _require_supported_device(device):
    """Accept cpu, or cuda when it is available. Fail before a later mismatch."""

    device = th.device(device)
    if device.type == "cpu":
        return device
    if device.type == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError(
                f"BC reference actor cannot use {device}: CUDA is not available"
            )
        return device
    raise RuntimeError(
        f"BC reference actor does not support device {device}. "
        "Use cpu or an available CUDA device."
    )


def _parameter_device(module):
    try:
        parameter = next(module.parameters())
    except StopIteration as exc:
        raise RuntimeError("BC actor has no parameters; device cannot be determined") from exc
    return _require_supported_device(parameter.device)


def _assert_same_device(reference, policy):
    expected = _parameter_device(policy)
    mismatches = []
    for kind, items in (
        ("parameter", reference.named_parameters()),
        ("buffer", reference.named_buffers()),
    ):
        for name, tensor in items:
            if tensor.device != expected:
                mismatches.append(f"{kind} {name} is on {tensor.device}")
    if mismatches:
        raise RuntimeError(
            "BC reference actor is not on the trainable actor device "
            f"{expected}: " + "; ".join(mismatches)
        )
    return expected


def _place_reference(reference, policy):
    device = _parameter_device(policy)
    try:
        reference.to(device)
    except Exception as exc:
        raise RuntimeError(f"BC reference actor could not be placed on {device}") from exc
    _assert_same_device(reference, policy)
    return _freeze_reference(reference)


def _freeze_reference(policy):
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    def _stay_eval(mode=True):
        return th.nn.Module.train(policy, False)

    policy.train = _stay_eval
    optimizer = getattr(policy, "optimizer", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            group["params"] = []
    return policy


def _assert_reference_isolated(model):
    reference = getattr(model, "reference_policy", None)
    if reference is None:
        raise RuntimeError("BC reference actor is missing")
    if reference.training:
        raise RuntimeError("BC reference actor is not in evaluation mode")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise RuntimeError("BC reference actor has a trainable parameter")
    reference_ids = {id(parameter) for parameter in reference.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    if reference_ids & optimizer_ids:
        raise RuntimeError("BC reference actor is in the PPO optimizer")


def assert_reference_identity(model):
    """Raise when the frozen reference no longer matches its saved fingerprint."""

    _assert_reference_isolated(model)
    _assert_same_device(model.reference_policy, model.policy)
    provenance = getattr(model, "anchor_provenance", None)
    expected = None if not isinstance(provenance, dict) else provenance.get("reference_sha256")
    actual = reference_fingerprint(model.reference_policy)
    if not isinstance(expected, str) or expected != actual:
        raise RuntimeError(
            "BC reference actor does not match the fingerprint captured at initialization"
        )
    return actual


def _make_reference_shell(policy):
    if not isinstance(policy, JackalPretrainedPPOPolicy):
        raise TypeError("BC anchoring requires JackalPretrainedPPOPolicy")
    shell = type(policy)(**policy._get_constructor_parameters())
    device = _parameter_device(policy)
    try:
        shell.to(device)
    except Exception as exc:
        raise RuntimeError(f"BC reference actor could not be placed on {device}") from exc
    shell.load_state_dict(policy.state_dict())
    return _place_reference(shell, policy)


def _mean_action(mean):
    reduced = mean.detach().mean(dim=0).cpu()
    return [float(reduced[0]), float(reduced[1])]


class JackalBCAnchoredPPO(PPO):
    """PPO subclass that adds the BC action-mean anchor inside ``train``."""

    def __init__(self, policy, env=None, anchor_coefficient=0.0, **kwargs):
        self.anchor_coefficient = validate_anchor_coefficient(anchor_coefficient)
        self.anchor_provenance = None
        self.reference_policy = None
        self.anchor_metrics = None
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.reference_policy is None:
            self.reference_policy = _make_reference_shell(self.policy)
        else:
            _place_reference(self.reference_policy, self.policy)
        _freeze_reference(self.reference_policy)

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return [*state_dicts, "reference_policy"], variables

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "reference_policy"]

    def _include_anchor(self, loss, observations):
        coefficient = float(self.anchor_coefficient)
        if coefficient == 0.0:
            with th.no_grad():
                anchor_loss, current_mean, reference_mean = actor_mean_anchor(
                    self.policy, self.reference_policy, observations
                )
            total = loss
        else:
            anchor_loss, current_mean, reference_mean = actor_mean_anchor(
                self.policy, self.reference_policy, observations
            )
            total = anchored_objective(loss, anchor_loss, coefficient)
        stats = {
            "loss": float(anchor_loss.detach().cpu()),
            "current": _mean_action(current_mean),
            "reference": _mean_action(reference_mean),
        }
        return total, stats

    def _publish_anchor_metrics(self, rows):
        if not rows:
            self.anchor_metrics = None
            return
        current = [
            float(np.mean([row["current"][index] for row in rows]))
            for index in (0, 1)
        ]
        reference = [
            float(np.mean([row["reference"][index] for row in rows]))
            for index in (0, 1)
        ]
        difference = [current[0] - reference[0], current[1] - reference[1]]
        self.anchor_metrics = {
            "anchor_loss": float(np.mean([row["loss"] for row in rows])),
            "anchor_coefficient": float(self.anchor_coefficient),
            "current_actor_mean": current,
            "reference_actor_mean": reference,
            "actor_mean_difference": difference,
        }
        LOGGER.info(
            "BC anchor loss=%.6g coefficient=%.6g current_mean=%s reference_mean=%s difference=%s",
            self.anchor_metrics["anchor_loss"],
            self.anchor_metrics["anchor_coefficient"],
            current,
            reference,
            difference,
        )
        self.logger.record("train/anchor_loss", self.anchor_metrics["anchor_loss"])
        self.logger.record("train/anchor_coefficient", self.anchor_metrics["anchor_coefficient"])
        self.logger.record("train/current_actor_mean_linear", current[0])
        self.logger.record("train/current_actor_mean_angular", current[1])
        self.logger.record("train/reference_actor_mean_linear", reference[0])
        self.logger.record("train/reference_actor_mean_angular", reference[1])
        self.logger.record("train/actor_mean_difference_linear", difference[0])
        self.logger.record("train/actor_mean_difference_angular", difference[1])

    def train(self) -> None:
        """Audited SB3 2.9.0 PPO update plus the action-mean anchor.

        The clipping rule, value loss, entropy term, advantage normalization,
        KL early stop, gradient clipping, and ``_n_updates`` accounting match
        that release. The anchor is added to the minimized total before
        ``backward`` and gradient clipping.
        """

        check_sb3_version()
        assert_reference_identity(self)
        _freeze_reference(self.reference_policy)
        self.policy.set_training_mode(True)
        _freeze_reference(self.reference_policy)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        anchor_rows = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss, anchor_stats = self._include_anchor(loss, rollout_data.observations)
                anchor_rows.append(anchor_stats)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self._publish_anchor_metrics(anchor_rows)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    @classmethod
    def load(cls, path, *args, **kwargs):
        try:
            model = super().load(path, *args, **kwargs)
        except ValueError as exc:
            if "reference_policy" in str(exc):
                raise ValueError(
                    "Anchored resume requires the saved BC reference. "
                    "The updated actor was not copied into the reference."
                ) from exc
            raise
        model.anchor_coefficient = validate_anchor_coefficient(model.anchor_coefficient)
        _place_reference(model.reference_policy, model.policy)
        assert_reference_identity(model)
        return model


def adopt_initialized_model(model, *, coefficient, provenance):
    """Freeze a reference from ``model``'s current actor and switch its class.

    Calling this again is refused. Resume must load the saved reference
    instead of photographing the updated actor.
    """

    check_sb3_version()
    if isinstance(model, JackalBCAnchoredPPO):
        raise RuntimeError(
            "Refusing to recapture the BC reference from an already anchored actor"
        )
    if not isinstance(model, PPO):
        raise TypeError("BC anchoring requires a Stable-Baselines3 PPO model")
    if not isinstance(provenance, dict):
        raise TypeError("Anchor provenance must be a dictionary")
    checkpoint = provenance.get("source_checkpoint")
    digest = provenance.get("source_checkpoint_sha256")
    sb3_version = provenance.get("sb3_version")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Anchor provenance requires the source checkpoint path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Anchor provenance requires the source checkpoint SHA-256")
    if sb3_version != SUPPORTED_SB3_VERSION:
        raise RuntimeError(
            "Jackal BC anchoring was audited for stable-baselines3 "
            f"{SUPPORTED_SB3_VERSION}; provenance says {sb3_version}"
        )
    coefficient = validate_anchor_coefficient(coefficient)
    reference = _make_reference_shell(model.policy)
    model.__class__ = JackalBCAnchoredPPO
    model.anchor_coefficient = coefficient
    model.reference_policy = reference
    model.anchor_metrics = None
    model.anchor_provenance = {
        "source_checkpoint": checkpoint,
        "source_checkpoint_sha256": digest,
        "sb3_version": SUPPORTED_SB3_VERSION,
        "coefficient": coefficient,
        "reference_sha256": reference_fingerprint(reference),
    }
    assert_reference_identity(model)
    LOGGER.info(
        "BC reference frozen coefficient=%.6g fingerprint=%s checkpoint=%s",
        coefficient,
        model.anchor_provenance["reference_sha256"],
        checkpoint,
    )
    return model
