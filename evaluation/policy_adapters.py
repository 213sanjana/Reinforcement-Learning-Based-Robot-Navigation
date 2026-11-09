"""Inference-only adapters. All receive the same raw navigation observation."""
from typing import Protocol
import numpy as np
from imitation.dataset import encode_observation
from planners.astar import astar, path_to_actions
from evaluation.scenario_manifest import digest, file_hash


class EvaluationPolicy(Protocol):
    def reset(self, scenario): ...
    def act(self, observation) -> int: ...
    def fingerprint(self) -> str: ...


class AStarAdapter:
    def reset(self, scenario):
        # Planner uses exactly the map/endpoints also present in CNN channels.
        path = astar(np.asarray(scenario['obstacles']), tuple(scenario['start']), tuple(scenario['goal']))
        self.actions = iter(path_to_actions(path))

    def act(self, observation):
        return next(self.actions)

    def fingerprint(self):
        return 'astar_manhattan_unit_cost_v1'


def module_hash(module):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


class NeuralAdapter:
    def __init__(self, kind, checkpoint, environment, device='cpu'):
        import torch
        from imitation.bc_network import resolve_device
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.kind = kind; self.path = checkpoint; self.device = resolve_device(device)
        self.checkpoint_hash = file_hash(checkpoint)
        self.model = None
        h, w = environment['height'], environment['width']
        if kind == 'dqn':
            # Load only inference network; no replay buffer/optimizer/exploration.
            from agents.dqn_agent import encode_observation as dqn_encode
            from models.cnn_dqn_network import CNNDQNNetwork
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload['config']['architecture'] != 'cnn' or (payload['height'], payload['width']) != (h, w):
                raise ValueError('Only compatible CNN DQN checkpoints supported')
            self.module = CNNDQNNetwork(h, w, payload['config']['hidden_size']).to(self.device)
            self.module.load_state_dict(payload['online'], strict=True)
            self.alternate_encoder = dqn_encode
        elif kind == 'bc':
            from imitation.bc_network import load_bc_checkpoint
            self.module, payload = load_bc_checkpoint(checkpoint, str(self.device))
            if (self.module.height, self.module.width) != (h, w):
                raise ValueError('BC dimensions mismatch')
            if payload['channels'] != ['obstacles','agent','goal'] or payload['actions'] != [[-1,0],[1,0],[0,-1],[0,1]]:
                raise ValueError('BC action/channel mismatch')
        elif kind == 'ppo':
            from stable_baselines3 import PPO
            from agents.pretrained_ppo_policy import check_encoded_space, check_sb3_version
            check_sb3_version()
            self.model = PPO.load(checkpoint, device=str(self.device))
            check_encoded_space(self.model.observation_space, self.model.action_space)
            if self.model.observation_space.shape != (3,h,w) or self.model.policy.normalize_images:
                raise ValueError('PPO observation preprocessing mismatch')
            self.module = self.model.policy
            self.module.set_training_mode(False)
        else:
            raise ValueError('Unknown adapter kind: ' + kind)
        if any(not torch.isfinite(v).all() for v in self.module.state_dict().values()):
            raise ValueError('Non-finite policy state')
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.initial_state = self.fingerprint()

    def reset(self, scenario):
        pass  # Audited policies are feed-forward; no per-episode hidden state.

    def fingerprint(self):
        counters = None if self.model is None else [self.model.num_timesteps, self.model._n_updates]
        return digest(dict(weights=module_hash(self.module), counters=counters))

    def act(self, observation):
        import torch
        encoded = encode_observation(observation)
        if self.kind == 'dqn' and not np.array_equal(encoded, self.alternate_encoder(observation)):
            raise ValueError('DQN encoding differs from common observation')
        with torch.inference_mode():
            if self.model is not None:
                action, _ = self.model.predict(encoded, deterministic=True)
                return int(action)
            logits = self.module(torch.as_tensor(encoded[None], device=self.device))
            if tuple(logits.shape) != (1,4) or not torch.isfinite(logits).all():
                raise ValueError('Invalid policy outputs')
            return int(logits.argmax(1).item())

    def verify(self):
        if self.fingerprint() != self.initial_state or file_hash(self.path) != self.checkpoint_hash:
            raise RuntimeError('Evaluation modified policy/checkpoint state')


def load_policy(name, specification, environment, device):
    if specification['kind'] == 'astar':
        return AStarAdapter()
    return NeuralAdapter(specification['kind'], specification['checkpoint'], environment, device)
