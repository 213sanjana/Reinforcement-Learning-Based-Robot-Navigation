"""SB3 2.9.0 actor matching BC exactly, with an independent fresh critic."""
from importlib.metadata import version
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.bc_network import BCNetwork
from imitation.dataset import encode_observation

SUPPORTED_SB3_VERSION = '2.9.0'
ACTION_DELTAS = ((-1,0),(1,0),(0,-1),(0,1))


def check_sb3_version():
    installed=version('stable-baselines3')
    if installed != SUPPORTED_SB3_VERSION:
        raise RuntimeError(f'Transfer audited for stable-baselines3 {SUPPORTED_SB3_VERSION}; found {installed}. Do not silently upgrade/downgrade; inspect compatibility first.')
    return installed


def check_encoded_space(observation_space,action_space=None):
    if not isinstance(observation_space,spaces.Box) or observation_space.dtype != np.dtype(np.float32):
        raise ValueError('Expected a float32 Box observation space')
    if len(observation_space.shape) != 3 or observation_space.shape[0] != 3 or min(observation_space.shape[1:]) < 3:
        raise ValueError('Expected encoded observations shaped (3,H,W), H/W >= 3')
    if not np.all(observation_space.low==0) or not np.all(observation_space.high==1):
        raise ValueError('Encoded observation bounds must be [0,1], not [0,255]')
    if action_space is not None:
        if not isinstance(action_space,spaces.Discrete) or action_space.n != 4 or action_space.start != 0:
            raise ValueError('Expected Discrete(4) with action indices 0..3')


class EncodedNavigationObservation(gym.ObservationWrapper):
    """Preserve dynamics/actions; convert raw Dict into BC's binary float32 CHW planes."""
    def __init__(self,env):
        super().__init__(env)
        if not isinstance(env.observation_space,spaces.Dict) or set(env.observation_space.spaces) != {'agent','goal','obstacles'}:
            raise ValueError('Expected the audited GridWorld Dict observation space')
        if tuple(getattr(env.unwrapped,'ACTIONS',())) != ACTION_DELTAS:
            raise ValueError('Environment action semantics differ from BC')
        h,w=env.observation_space['obstacles'].shape
        self.observation_space=spaces.Box(0.,1.,shape=(3,h,w),dtype=np.float32)
        check_encoded_space(self.observation_space,env.action_space)

    def observation(self,observation):
        encoded=encode_observation(observation)
        if not self.observation_space.contains(encoded):
            raise ValueError('Encoded observation violates declared space')
        return encoded


class BCFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,hidden_size=256):
        check_encoded_space(observation_space)
        if not isinstance(hidden_size,int) or hidden_size < 1:
            raise ValueError('hidden_size must be a positive integer')
        super().__init__(observation_space,features_dim=hidden_size)
        self.height,self.width=observation_space.shape[1:]
        template=BCNetwork(self.height,self.width,hidden_size)
        self.features=template.features
        self.shared=template.shared

    def forward(self,observations):
        if observations.ndim != 4 or tuple(observations.shape[1:]) != (3,self.height,self.width):
            raise ValueError('Invalid PPO observation shape')
        return self.shared(self.features(observations))


class PretrainedPPOPolicy(ActorCriticPolicy):
    """BC actor path, identity extra MLPs; separate randomly initialized value path."""
    def __init__(self,observation_space,action_space,lr_schedule,hidden_size=256,use_sde=False):
        check_sb3_version()
        check_encoded_space(observation_space,action_space)
        if use_sde:
            raise ValueError('Discrete navigation uses a categorical policy, not gSDE')
        super().__init__(observation_space,action_space,lr_schedule,
                         net_arch=dict(pi=[],vf=[]),activation_fn=nn.ReLU,ortho_init=False,
                         features_extractor_class=BCFeaturesExtractor,
                         features_extractor_kwargs={'hidden_size':hidden_size},
                         share_features_extractor=False,normalize_images=False,use_sde=False)
        self.hidden_size=hidden_size

    def _get_constructor_parameters(self):
        # Support direct policy save/load as well as PPO.save/load.
        return dict(observation_space=self.observation_space,action_space=self.action_space,
                    lr_schedule=self._dummy_schedule,hidden_size=self.hidden_size,use_sde=False)
