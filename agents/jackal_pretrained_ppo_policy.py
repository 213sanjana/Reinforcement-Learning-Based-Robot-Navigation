"""Stable-Baselines3 PPO actor whose mean matches the Jackal BC policy.

The behavior-cloning network emits ``tanh`` actions in ``(-1, 1)`` with
order ``[linear, angular]``. PPO represents a continuous policy as a
diagonal Gaussian. This class keeps that Gaussian, copies the BC trunk
through the final linear layer, and uses ``tanh`` of that linear output
as the Gaussian mean.

Three quantities stay distinct:

* actor raw output: the unbounded linear layer, before ``tanh``
* distribution action: ``Normal(tanh(raw), exp(log_std))``
* ``predict`` action: the distribution action after Stable-Baselines3
  clips it to the Gymnasium box ``[-1, 1]``

``squash_output`` stays false, so PPO does not run ``unscale_action``.
The frozen BC normalizer runs once inside the trunk. Image normalization
is disabled, and this observation is not an image, so Stable-Baselines3
preprocessing does not z-score it again.

The critic reuses the trunk architecture with its own initialized
weights. It has no behavior-cloning value head to copy.
"""

from importlib.metadata import version

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from navigation.expert_demonstration_schema import (
    jackal_policy_spaces,
    validate_jackal_policy_spaces,
)
from navigation.jackal_bc_policy import (
    ARCHITECTURE,
    CircularConv1d,
    JackalBCPolicy,
)
from navigation.jackal_bc_schema import (
    ACTION_DIM,
    OBSERVATION_DIM,
    split_observation,
)


SUPPORTED_SB3_VERSION = "2.9.0"
PHASE = "6g.1"


def check_sb3_version():
    """Raise unless the audited Stable-Baselines3 release is installed."""

    installed = version("stable-baselines3")
    if installed != SUPPORTED_SB3_VERSION:
        raise RuntimeError(
            "Jackal PPO transfer was audited for stable-baselines3 "
            f"{SUPPORTED_SB3_VERSION}; found {installed}"
        )
    return installed


class JackalPolicyContractEnv(gym.Env):
    """Space-only stand-in for ``JackalGazeboEnv``.

    PPO needs an environment object to construct observation and action
    spaces. This stand-in exposes those spaces and never starts ROS,
    Gazebo, or a velocity command.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space, self.action_space = jackal_policy_spaces()
        self.resets = 0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, {}

    def step(self, action):
        self.steps += 1
        raise RuntimeError(
            "Phase 6G.1 does not step the Jackal environment or publish commands"
        )


class JackalActorTrunk(nn.Module):
    """BC network through the last hidden ReLU, including its normalizer."""

    def __init__(self, network_config):
        super().__init__()
        template = JackalBCPolicy.from_network_config(dict(network_config))
        if template.architecture != ARCHITECTURE:
            raise ValueError(f"Architecture must be {ARCHITECTURE}")
        fusion_modules = list(template.fusion.children())
        if not fusion_modules or not isinstance(fusion_modules[-1], nn.Linear):
            raise ValueError("BC fusion must end in a linear action layer")
        final = fusion_modules[-1]
        if int(final.out_features) != ACTION_DIM or final.bias is None:
            raise ValueError("BC action layer must be a biased linear map to 2 actions")

        self.lidar_encoder = template.lidar_encoder
        self.state_encoder = template.state_encoder
        self.fusion_trunk = nn.Sequential(*fusion_modules[:-1])
        self.action_in_features = int(final.in_features)
        self.action_out_features = int(final.out_features)
        self.network_config = template.network_config()
        self.register_buffer("obs_mean", template.obs_mean.detach().clone())
        self.register_buffer("obs_std", template.obs_std.detach().clone())
        self.register_buffer(
            "normalizer_count",
            template.normalizer_count.detach().clone(),
        )

    @property
    def normalizer_is_ready(self):
        return (
            int(self.normalizer_count.item()) > 0
            and bool(torch.isfinite(self.obs_mean).all())
            and bool(torch.isfinite(self.obs_std).all())
            and bool(torch.all(self.obs_std > 0))
        )

    def forward(self, observations):
        """Return the 64-D fusion hidden state for a raw ``(N, 40)`` batch."""

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
        return self.fusion_trunk(features)


class JackalActorFeatures(BaseFeaturesExtractor):
    """Stable-Baselines3 feature extractor wrapping :class:`JackalActorTrunk`."""

    def __init__(self, observation_space, network_config):
        if not isinstance(network_config, dict):
            raise TypeError("network_config must be a dictionary")
        trunk = JackalActorTrunk(network_config)
        super().__init__(observation_space, features_dim=trunk.action_in_features)
        self.trunk = trunk

    def forward(self, observations):
        return self.trunk(observations)


class JackalPretrainedPPOPolicy(ActorCriticPolicy):
    """BC actor mean, independent critic, diagonal-Gaussian exploration."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        network_config,
        log_std_init=0.0,
        use_sde=False,
    ):
        check_sb3_version()
        validate_jackal_policy_spaces(observation_space, action_space)
        if use_sde:
            raise ValueError(
                "Jackal continuous transfer uses a diagonal Gaussian, not gSDE"
            )
        if isinstance(log_std_init, bool) or not isinstance(log_std_init, (int, float)):
            raise TypeError("log_std_init must be a finite number")
        if not np.isfinite(float(log_std_init)):
            raise ValueError("log_std_init must be finite")
        if not isinstance(network_config, dict):
            raise TypeError("network_config must be a dictionary")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch={"pi": [], "vf": []},
            activation_fn=nn.ReLU,
            ortho_init=False,
            features_extractor_class=JackalActorFeatures,
            features_extractor_kwargs={"network_config": dict(network_config)},
            share_features_extractor=False,
            normalize_images=False,
            use_sde=False,
            log_std_init=float(log_std_init),
            squash_output=False,
        )
        self.network_config = self.pi_features_extractor.trunk.network_config
        _audit_policy_layout(self)

    def _get_constructor_parameters(self):
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "lr_schedule": self._dummy_schedule,
            "network_config": self.network_config,
            "log_std_init": self.log_std_init,
            "use_sde": False,
        }

    def _get_action_dist_from_latent(self, latent_pi):
        """Map the BC linear output through ``tanh`` into the Gaussian mean."""

        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("Jackal PPO actor requires a diagonal Gaussian")
        raw = self.action_net(latent_pi)
        mean = torch.tanh(raw)
        return self.action_dist.proba_distribution(mean, self.log_std)

    def raw_and_mean(self, observations):
        """Return ``(pre_tanh, tanh_mean)`` for a float32 batch ``(N, 40)``.

        The observation is the raw 40-D environment vector. This method
        rejects a second normalization from Stable-Baselines3.
        """

        if observations.ndim != 2 or int(observations.shape[1]) != OBSERVATION_DIM:
            raise ValueError("Expected observations shaped (N, 40)")
        if not torch.isfinite(observations).all():
            raise ValueError("Observations must be finite")
        preprocessed = preprocess_obs(
            observations,
            self.observation_space,
            normalize_images=self.normalize_images,
        )
        if preprocessed.dtype != observations.dtype or not torch.equal(
            preprocessed, observations
        ):
            raise RuntimeError(
                "Stable-Baselines3 preprocessing changed the Jackal observation"
            )
        latent = self.mlp_extractor.forward_actor(
            self.pi_features_extractor(preprocessed)
        )
        if not torch.equal(latent, self.pi_features_extractor(preprocessed)):
            raise RuntimeError("Actor MLP changed the BC hidden features")
        raw = self.action_net(latent)
        mean = torch.tanh(raw)
        return raw, mean


def audit_bc_actor_layout(policy):
    """Raise when ``policy`` is not the Jackal BC actor this transfer knows."""

    if not isinstance(policy, JackalBCPolicy) or policy.architecture != ARCHITECTURE:
        raise ValueError(f"Source policy architecture must be {ARCHITECTURE}")
    if any(isinstance(module, nn.Tanh) for module in policy.modules()):
        raise ValueError(
            "BC tanh must stay a parameter-free output, outside the transferred modules"
        )
    if int(policy.action_dim) != ACTION_DIM or int(policy.lidar_bins) != 36:
        raise ValueError("Source policy does not implement the 40-D Jackal contract")

    convolution = list(policy.lidar_encoder.convolution.children())
    if len(convolution) != 2 * len(policy.lidar_channels):
        raise ValueError("LiDAR encoder depth does not match lidar_channels")
    for module in convolution[0::2]:
        if not isinstance(module, CircularConv1d):
            raise ValueError("LiDAR front-end is not circular convolution")
    for module in convolution[1::2]:
        if not isinstance(module, nn.ReLU):
            raise ValueError("LiDAR activation is not ReLU")
    if not isinstance(policy.lidar_encoder.projection, nn.Linear):
        raise ValueError("LiDAR projection is not a linear layer")

    state_layers = list(policy.state_encoder.network.children())
    if [type(module) for module in state_layers] != [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
    ]:
        raise ValueError("Robot-state encoder is not Linear-ReLU-Linear-ReLU")

    fusion = list(policy.fusion.children())
    if len(fusion) != 2 * len(policy.fusion_hidden) + 1:
        raise ValueError("Fusion depth does not match fusion_hidden")
    for module in fusion[:-1:2]:
        if not isinstance(module, nn.Linear):
            raise ValueError("Fusion hidden layer is not linear")
    for module in fusion[1::2]:
        if not isinstance(module, nn.ReLU):
            raise ValueError("Fusion activation is not ReLU")
    final = fusion[-1]
    if (
        not isinstance(final, nn.Linear)
        or int(final.out_features) != ACTION_DIM
        or final.bias is None
    ):
        raise ValueError("Fusion must end in a biased linear map to 2 actions")
    return final


def _audit_policy_layout(policy):
    if policy.share_features_extractor:
        raise ValueError("Actor and critic feature extractors must be separate")
    if policy.pi_features_extractor is policy.vf_features_extractor:
        raise ValueError("Actor and critic feature extractors must be separate")
    if policy.normalize_images or policy.ortho_init or policy.squash_output or policy.use_sde:
        raise ValueError("PPO preprocessing, initialization, or exploration contract changed")
    if not isinstance(policy.action_dist, DiagGaussianDistribution):
        raise ValueError("Jackal PPO actor requires a diagonal Gaussian")
    if list(policy.mlp_extractor.policy_net.children()) or list(
        policy.mlp_extractor.value_net.children()
    ):
        raise ValueError("Extra PPO MLP layers would change the BC actor")

    actor = policy.pi_features_extractor.trunk
    critic = policy.vf_features_extractor.trunk
    if actor.network_config != critic.network_config:
        raise ValueError("Actor and critic trunks were built from different configs")
    if int(policy.action_net.in_features) != int(actor.action_in_features):
        raise ValueError("action_net does not match the BC final linear input")
    if int(policy.action_net.out_features) != ACTION_DIM or policy.action_net.bias is None:
        raise ValueError("action_net must be a biased linear map to 2 actions")
    if tuple(policy.log_std.shape) != (ACTION_DIM,):
        raise ValueError("Gaussian log_std must have one value per action")
    if int(policy.value_net.in_features) != int(policy.features_dim):
        raise ValueError("Value head does not read the critic trunk")
    if int(policy.value_net.out_features) != 1 or policy.value_net.bias is None:
        raise ValueError("Value head must be a biased linear map to one value")
    if actor.action_out_features != ACTION_DIM:
        raise ValueError("BC action layer width is not 2")
