"""Deterministic continuous policy for Jackal behavior cloning.

LiDAR sectors and the four robot-state features pass through separate
encoders. Their embeddings are fused into a 2D action, then squashed with
tanh so every output lies in the Gymnasium interval (-1, 1).

Observation z-score buffers are fitted elsewhere on the train split and
then frozen. The forward pass applies them and does not update them.
"""

import math

import numpy as np
import torch
from torch import nn

from navigation.jackal_bc_schema import (
    ACTION_DIM,
    ACTION_NAMES,
    LIDAR_BINS,
    OBSERVATION_DIM,
    PHASE,
    STATE_DIM,
    create_new_file,
    feature_names,
    require_env_actions,
    split_observation,
)


ARCHITECTURE = "jackal_bc_v1"
FORMAT_VERSION = 1


class CircularConv1d(nn.Module):
    """Same-length convolution that wraps the LiDAR ring."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        kernel_size = _positive_int(kernel_size, "lidar_kernel_size")
        if kernel_size % 2 == 0:
            raise ValueError("LiDAR kernel size must be odd")
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, values):
        if self.padding:
            values = torch.cat(
                (
                    values[..., -self.padding:],
                    values,
                    values[..., :self.padding],
                ),
                dim=-1,
            )
        return self.conv(values)


class LidarEncoder(nn.Module):
    """Encode 36 ordered LiDAR sectors with circular convolution."""

    def __init__(self, bins, channels, kernel_size, embedding_dim):
        super().__init__()
        self.bins = _positive_int(bins, "lidar_bins")
        channels = tuple(_positive_int(item, "lidar_channels") for item in channels)
        if not channels:
            raise ValueError("LiDAR encoder requires at least one channel width")
        self.channels = channels
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.append(CircularConv1d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Linear(in_channels * self.bins, embedding_dim)
        self.embedding_dim = _positive_int(embedding_dim, "lidar_embedding")

    def forward(self, lidar):
        if lidar.ndim != 2 or int(lidar.shape[1]) != self.bins:
            raise ValueError(f"Expected LiDAR shaped (N, {self.bins})")
        hidden = self.convolution(lidar[:, None, :])
        return torch.relu(self.projection(hidden.flatten(1)))


class RobotStateEncoder(nn.Module):
    """Encode goal distance, goal heading, and body velocity."""

    def __init__(self, state_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.state_dim = _positive_int(state_dim, "state_dim")
        hidden_dim = _positive_int(hidden_dim, "state_hidden")
        self.embedding_dim = _positive_int(embedding_dim, "state_embedding")
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        if state.ndim != 2 or int(state.shape[1]) != self.state_dim:
            raise ValueError(f"Expected robot state shaped (N, {self.state_dim})")
        return self.network(state)


class JackalBCPolicy(nn.Module):
    """Map one 40D Jackal observation to a 2D navigation action."""

    def __init__(
        self,
        lidar_bins=LIDAR_BINS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lidar_channels=(16, 32),
        lidar_kernel_size=5,
        lidar_embedding=64,
        state_hidden=32,
        state_embedding=32,
        fusion_hidden=(128, 64),
        seed=0,
    ):
        super().__init__()
        if int(lidar_bins) != LIDAR_BINS or int(state_dim) != STATE_DIM:
            raise ValueError("Policy observation layout must stay 36 LiDAR + 4 state")
        if int(action_dim) != ACTION_DIM:
            raise ValueError("Policy action must stay 2-dimensional")

        self.lidar_bins = LIDAR_BINS
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.lidar_channels = tuple(int(item) for item in lidar_channels)
        self.lidar_kernel_size = int(lidar_kernel_size)
        self.lidar_embedding = int(lidar_embedding)
        self.state_hidden = int(state_hidden)
        self.state_embedding = int(state_embedding)
        self.fusion_hidden = tuple(int(item) for item in fusion_hidden)
        self.init_seed = _nonnegative_int(seed, "seed")
        self.architecture = ARCHITECTURE

        self.lidar_encoder = LidarEncoder(
            self.lidar_bins,
            self.lidar_channels,
            self.lidar_kernel_size,
            self.lidar_embedding,
        )
        self.state_encoder = RobotStateEncoder(
            self.state_dim,
            self.state_hidden,
            self.state_embedding,
        )
        fused = self.lidar_embedding + self.state_embedding
        self.fusion = _fusion(fused, self.fusion_hidden, self.action_dim)
        self.register_buffer("obs_mean", torch.full((OBSERVATION_DIM,), float("nan")))
        self.register_buffer("obs_std", torch.full((OBSERVATION_DIM,), float("nan")))
        self.register_buffer("normalizer_count", torch.zeros((), dtype=torch.long))
        initialize_deterministically(self, self.init_seed)

    def install_normalizer(self, normalizer):
        """Copy frozen train-split statistics into the forward pass."""

        if normalizer.source_split != "train":
            raise ValueError("Observation statistics must come from the train split")
        if tuple(normalizer.feature_names) != feature_names():
            raise ValueError("Normalizer features do not match the 40D layout")
        if int(normalizer.count) < 2:
            raise ValueError("Normalizer does not contain training observations")

        mean = torch.as_tensor(normalizer.mean, dtype=self.obs_mean.dtype)
        std = torch.as_tensor(normalizer.std, dtype=self.obs_std.dtype)
        if mean.shape != (OBSERVATION_DIM,) or std.shape != (OBSERVATION_DIM,):
            raise ValueError("Normalizer statistics must have 40 features")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Normalizer statistics must be finite")
        if not torch.all(std > 0):
            raise ValueError("Normalizer standard deviations must be positive")

        with torch.no_grad():
            self.obs_mean.copy_(mean.to(device=self.obs_mean.device))
            self.obs_std.copy_(std.to(device=self.obs_std.device))
            self.normalizer_count.fill_(int(normalizer.count))

    @property
    def normalizer_is_ready(self):
        return (
            int(self.normalizer_count.item()) > 0
            and bool(torch.isfinite(self.obs_mean).all())
            and bool(torch.isfinite(self.obs_std).all())
            and bool(torch.all(self.obs_std > 0))
        )

    def forward(self, observations):
        """Return tanh actions for observations shaped (N, 40) or (N, T, 40)."""

        if observations.ndim == 3:
            batch, time, width = observations.shape
            flat = observations.reshape(batch * time, width)
            actions = self._forward_batch(flat)
            return actions.reshape(batch, time, self.action_dim)
        if observations.ndim == 2:
            return self._forward_batch(observations)
        raise ValueError("Expected observations shaped (N, 40) or (N, T, 40)")

    def forward_valid(self, observations, mask):
        """Score a padded episode batch without using padded steps."""

        if observations.ndim != 3 or int(observations.shape[-1]) != OBSERVATION_DIM:
            raise ValueError("Expected padded observations shaped (N, T, 40)")
        if mask.shape != observations.shape[:2]:
            raise ValueError("Mask must have shape (N, T)")
        if mask.dtype != torch.bool:
            raise TypeError("Mask must be boolean")

        batch, time, _ = observations.shape
        flat_observations = observations.reshape(batch * time, OBSERVATION_DIM)
        flat_mask = mask.reshape(batch * time)
        actions = observations.new_zeros(batch * time, self.action_dim)
        if bool(flat_mask.any()):
            selected = self._forward_batch(flat_observations[flat_mask])
            actions = actions.masked_scatter(
                flat_mask.unsqueeze(-1).expand_as(actions),
                selected.reshape(-1),
            )
        return actions.reshape(batch, time, self.action_dim)

    def network_config(self):
        """Return the architecture fields required to rebuild this module."""

        return {
            "lidar_bins": self.lidar_bins,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lidar_channels": list(self.lidar_channels),
            "lidar_kernel_size": self.lidar_kernel_size,
            "lidar_embedding": self.lidar_embedding,
            "state_hidden": self.state_hidden,
            "state_embedding": self.state_embedding,
            "fusion_hidden": list(self.fusion_hidden),
            "seed": self.init_seed,
        }

    @classmethod
    def from_network_config(cls, config):
        return cls(
            lidar_bins=config["lidar_bins"],
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            lidar_channels=tuple(config["lidar_channels"]),
            lidar_kernel_size=config["lidar_kernel_size"],
            lidar_embedding=config["lidar_embedding"],
            state_hidden=config["state_hidden"],
            state_embedding=config["state_embedding"],
            fusion_hidden=tuple(config["fusion_hidden"]),
            seed=config["seed"],
        )

    def _forward_batch(self, observations):
        if observations.ndim != 2 or int(observations.shape[1]) != OBSERVATION_DIM:
            raise ValueError("Expected observations shaped (N, 40)")
        if observations.dtype != torch.float32:
            raise TypeError("Observations must be float32")
        if not self.normalizer_is_ready:
            raise RuntimeError("Train-split observation normalizer is not installed")
        if not torch.isfinite(observations).all():
            raise ValueError("Observations must be finite")

        normalized = (observations - self.obs_mean) / self.obs_std
        lidar, state = split_observation(normalized)
        features = torch.cat(
            (self.lidar_encoder(lidar), self.state_encoder(state)),
            dim=-1,
        )
        return torch.tanh(self.fusion(features))


def initialize_deterministically(module, seed):
    """Replace parameter values from a CPU generator seeded with ``seed``."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_nonnegative_int(seed, "seed"))
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
            if parameter.ndim < 2:
                values.zero_()
            else:
                fan_in = int(math.prod(parameter.shape[1:]))
                bound = fan_in ** -0.5
                values.uniform_(-bound, bound, generator=generator)
            parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))


def build_policy(model_config, seed):
    """Build the Phase 6F.1 architecture from a configuration mapping."""

    if model_config.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Architecture must be {ARCHITECTURE}")
    return JackalBCPolicy(
        lidar_bins=model_config["lidar_bins"],
        state_dim=model_config["state_dim"],
        action_dim=model_config["action_dim"],
        lidar_channels=tuple(model_config["lidar_channels"]),
        lidar_kernel_size=model_config["lidar_kernel_size"],
        lidar_embedding=model_config["lidar_embedding"],
        state_hidden=model_config["state_hidden"],
        state_embedding=model_config["state_embedding"],
        fusion_hidden=tuple(model_config["fusion_hidden"]),
        seed=seed,
    )


def resolve_device(request="auto"):
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a torch device."""

    if request not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if request == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        chosen = request
    if chosen == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(chosen)


def predict_numpy(policy, observations):
    """Run one forward pass and return a float32 Gymnasium action."""

    array = np.asarray(observations, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, OBSERVATION_DIM)
    if array.ndim != 2 or int(array.shape[1]) != OBSERVATION_DIM:
        raise ValueError("Expected a float observation shaped (40,) or (N, 40)")
    if not np.all(np.isfinite(array)):
        raise ValueError("Observations must be finite")

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    tensor = tensor.to(device=policy.obs_mean.device)
    with torch.no_grad():
        actions = policy(tensor)
    output = np.ascontiguousarray(
        actions.detach().cpu().numpy(),
        dtype=np.float32,
    )
    require_env_actions(output)
    if single:
        return output[0]
    return output


def save_checkpoint(path, policy, normalizer):
    """Save an initialized or future trained policy without replacing files."""

    if normalizer.source_split != "train":
        raise ValueError("Checkpoint statistics must come from the train split")
    if int(policy.normalizer_count.item()) != int(normalizer.count):
        raise ValueError("Policy normalizer does not match the statistics object")

    expected_mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
    expected_std = torch.as_tensor(normalizer.std, dtype=torch.float32)
    if not torch.equal(policy.obs_mean.detach().cpu(), expected_mean):
        raise ValueError("Policy mean does not match the statistics object")
    if not torch.equal(policy.obs_std.detach().cpu(), expected_std):
        raise ValueError("Policy scale does not match the statistics object")

    payload = {
        "format_version": FORMAT_VERSION,
        "phase": PHASE,
        "architecture": ARCHITECTURE,
        "trained": False,
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
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
    }

    def _write(temporary):
        torch.save(payload, temporary)

    return create_new_file(path, _write)


def load_checkpoint(path, device="cpu"):
    """Load a Jackal BC checkpoint onto ``device``."""

    selected = resolve_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_payload(payload)
    policy = JackalBCPolicy.from_network_config(payload["network_config"])
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    if not policy.normalizer_is_ready:
        raise ValueError("Checkpoint is missing train-split normalization")
    policy.to(selected)
    policy.eval()
    return policy, payload


def _fusion(in_dim, hidden, out_dim):
    widths = tuple(_positive_int(item, "fusion_hidden") for item in hidden)
    if not widths:
        raise ValueError("Fusion MLP requires at least one hidden width")
    layers = []
    previous = in_dim
    for width in widths:
        layers.append(nn.Linear(previous, width))
        layers.append(nn.ReLU())
        previous = width
    layers.append(nn.Linear(previous, out_dim))
    return nn.Sequential(*layers)


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Unsupported Jackal BC checkpoint format")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported Jackal BC checkpoint format")
    if payload.get("architecture") != ARCHITECTURE or payload.get("phase") != PHASE:
        raise ValueError("Unsupported Jackal BC checkpoint format")
    normalization = payload.get("normalization") or {}
    if normalization.get("source_split") != "train":
        raise ValueError("Checkpoint statistics must come from the train split")
    if list(normalization.get("feature_names") or []) != list(feature_names()):
        raise ValueError("Checkpoint feature names do not match the 40D layout")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)
