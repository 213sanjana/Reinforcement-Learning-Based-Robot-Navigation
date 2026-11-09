"""Policies see observations and an independent RNG seed, never benchmark labels."""
import numpy as np
from evaluation.policy_adapters import NeuralAdapter
from evaluation.scenario_manifest import file_hash, digest
from planners.astar import astar, path_to_actions


class RandomBaseline:
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return int(self.rng.integers(4))

    def fingerprint(self):
        return 'uniform_discrete4_v1'  # RNG progression is intentional, not learning.


class PlannerBaseline:
    def reset(self, seed):
        pass

    def act(self, observation):
        # Replan from currently observed state; no saved optimal length/trajectory.
        route = astar(observation['obstacles'], tuple(observation['agent']), tuple(observation['goal']))
        if route is None or len(route) < 2:
            raise RuntimeError('No planner action for current state')
        return path_to_actions(route)[0]

    def fingerprint(self):
        return 'astar_replan_each_step_v1'


def metadata(kind, path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    common = dict(path=str(path), sha256=file_hash(path), kind=kind)
    if kind == 'dqn':
        common.update(training_steps=payload['steps'], updates=payload['updates'],
                      height=payload['height'], width=payload['width'], agent_config=payload['config'],
                      training_density='Not stored in checkpoint; uploaded experiment YAML says 0.05',
                      selection='Configured Phase 5A checkpoint; no selection or modification during this benchmark')
    else:
        # Preserve JSON-compatible training/provenance fields, never weight arrays.
        def safe(value):
            if value is None or isinstance(value, (bool,str,int,float)): return value
            if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
            if isinstance(value, (tuple,list)): return [safe(v) for v in value]
            return {'type':type(value).__name__}
        common['checkpoint_metadata'] = {k:safe(v) for k,v in payload.items()
            if k not in ('model_state_dict','optimizer_state_dict','optimizer')}
    return common


def load_baseline(name, manifest, device):
    if name == 'random':
        return RandomBaseline(), dict(kind='random',distribution='uniform over all four actions; no masking')
    if name == 'astar':
        return PlannerBaseline(), dict(kind='astar',planning='Replan per action from current observation')
    spec = manifest['config']['policies'][name]
    if spec['kind'] != name or name not in ('dqn','bc'):
        raise ValueError('Expected original DQN/BC policy specification')
    adapter = NeuralAdapter(name, spec['checkpoint'], manifest['config']['environment'], device)
    return adapter, metadata(name,spec['checkpoint'])


def action_observation(obs):
    # Do not expose manifest IDs, shortest lengths, seeds, rewards or outcomes.
    return {k:np.array(obs[k],copy=True) for k in ('obstacles','agent','goal')}
