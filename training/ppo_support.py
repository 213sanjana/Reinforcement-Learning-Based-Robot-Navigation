"""Shared manifest environments, provenance, evaluation and continuation state for Phase 4B."""
from collections import deque
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from agents.pretrained_ppo_policy import EncodedNavigationObservation,PretrainedPPOPolicy,check_sb3_version
from environments.gridworld_env import RobotNavigationEnv
from environments.fixed_map_env import FixedMapNavigationEnv
from environments.map_generator import shortest_path
from imitation.dataset import map_identity
from imitation.transfer_to_ppo import (file_sha256,load_strict_bc,check_transferred_parameters,
                                       critic_snapshot,tensor_sha256,validate_equivalence)
from evaluation.bc_benchmark import detect_loops,summarize


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def actor_hash(policy):
    digest=hashlib.sha256()
    for prefix,module in [('features',policy.pi_features_extractor.features),
                          ('shared',policy.pi_features_extractor.shared),('actor',policy.action_net)]:
        for key,value in module.state_dict().items():
            digest.update((prefix+'.'+key).encode()); digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def finite_policy(model):
    for name,value in model.policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite policy tensor: {name}')


def load_pool(path,split,environment):
    with np.load(path,allow_pickle=False) as source:
        metadata=json.loads(str(source['metadata']))
        if metadata.get('split') != split or metadata.get('channels') != ['obstacles','agent','goal']:
            raise ValueError(f'Expected original {split} map manifest')
        if metadata.get('actions') != [[-1,0],[1,0],[0,-1],[0,1]]:
            raise ValueError('Map manifest action semantics mismatch')
        if metadata['environment'] != environment:
            raise ValueError('Reward/environment config must match the verified initialization')
        arrays={key:source[key] for key in ('obstacles','starts','goals','map_ids','map_seeds','path_lengths')}
    count=len(arrays['map_ids'])
    if count < 1 or len(set(arrays['map_ids'].tolist())) != count:
        raise ValueError('Empty or duplicate layout pool')
    for i,grid in enumerate(arrays['obstacles']):
        if map_identity(grid) != arrays['map_ids'][i]:
            raise ValueError('Invalid layout identity')
        route=shortest_path(grid,tuple(arrays['starts'][i]),tuple(arrays['goals'][i]))
        if route is None or len(route)-1 != arrays['path_lengths'][i] or len(route)-1 > environment['max_steps']:
            raise ValueError('Invalid or unreachable manifest task')
    return dict(arrays=arrays,environment=environment,path=str(Path(path)),
                sha256=file_sha256(path),split=split)


def load_data(config,environment,transfer_report):
    # Paths are tied to the actual BC input fingerprints, not guessed seed ranges.
    bc,payload=load_strict_bc(transfer_report['source_checkpoint'],'cpu')
    train=load_pool(config['data']['train'],'train',environment)
    validation=load_pool(config['data']['validation'],'validation',environment)
    for name,pool in [('train',train),('validation',validation)]:
        if pool['sha256'] != payload['dataset_provenance'][name]['sha256']:
            raise ValueError(f'{name} dataset differs from BC checkpoint provenance')
    if set(train['arrays']['map_ids']).intersection(validation['arrays']['map_ids']):
        raise ValueError('Training/validation layout leakage')
    count=int(config['validation']['episodes'])
    if not 1 <= count <= len(validation['arrays']['map_ids']):
        raise ValueError('Invalid validation episode count')
    indices=np.random.default_rng(config['validation']['seed']).choice(len(validation['arrays']['map_ids']),count,replace=False)
    selected={key:value[indices] for key,value in validation['arrays'].items()}
    signature=digest_json(dict(environment=environment,indices=indices.tolist(),
                              map_ids=selected['map_ids'].tolist(),starts=selected['starts'].tolist(),
                              goals=selected['goals'].tolist(),map_seeds=selected['map_seeds'].tolist()))
    scenarios=dict(arrays=selected,environment=environment,indices=indices.tolist(),signature=signature,
                   source_path=validation['path'],source_sha256=validation['sha256'])
    return train,scenarios


class TrainingMapPoolEnv(RobotNavigationEnv):
    """Original GridWorld step/reward implementation; reset samples only verified train tasks."""
    def __init__(self,pool,pending_rng=None):
        self.pool=pool
        self.pending_rng=pending_rng
        super().__init__(**pool['environment'])
        self.resets=0

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        if options:
            raise ValueError('Training map pool has no reset overrides')
        if self.pending_rng is not None:
            self.np_random.bit_generator.state=copy.deepcopy(self.pending_rng)
            self.pending_rng=None
        a=self.pool['arrays']; self.task_index=int(self.np_random.integers(len(a['map_ids'])))
        self.obstacles=a['obstacles'][self.task_index].copy()
        self.agent=tuple(map(int,a['starts'][self.task_index])); self.goal=tuple(map(int,a['goals'][self.task_index]))
        self.steps,self._finished,self._initialized=0,False,True
        self.episode_reward=0.; self.episode_collisions=0; self.resets+=1
        return self._observation(),self._info()

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        self.episode_reward+=reward; self.episode_collisions+=int(info['collision'])
        a=self.pool['arrays']; i=self.task_index
        info.update(is_success=bool(terminated),episode_reward=float(self.episode_reward),
                    episode_collisions=int(self.episode_collisions),episode_length=int(self.steps),
                    timeout=bool(truncated),map_id=str(a['map_ids'][i]),map_seed=int(a['map_seeds'][i]),task_index=i)
        return obs,reward,terminated,truncated,info

    def stream_state(self):
        return copy.deepcopy(self.np_random.bit_generator.state)


@contextmanager
def preserve_rng_and_mode(model):
    python_state=random.getstate(); numpy_state=np.random.get_state(); torch_state=torch.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mode=model.policy.training
    try:
        yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.policy.set_training_mode(mode)


def evaluate_model(model,scenarios):
    """Greedy held-out evaluation, with no action correction or training RNG effects."""
    before=actor_hash(model.policy); value_before=critic_snapshot(model.policy)
    rows=[]; arrays=scenarios['arrays']; cfg=scenarios['environment']
    with preserve_rng_and_mode(model),torch.inference_mode():
        for i,grid in enumerate(arrays['obstacles']):
            env=EncodedNavigationObservation(FixedMapNavigationEnv(grid,tuple(arrays['goals'][i]),
                max_steps=cfg['max_steps'],bfs_shaping=cfg.get('bfs_shaping',False),shaping_gamma=cfg.get('shaping_gamma',.99)))
            try:
                obs,_=env.reset(options={'start':arrays['starts'][i]})
                route=[arrays['starts'][i].tolist()]; actions=[]; total=0.; collisions=0
                while True:
                    action,_=model.predict(obs,deterministic=True)
                    obs,reward,term,trunc,info=env.step(int(action))
                    actions.append(int(action)); route.append(env.unwrapped.agent)
                    total+=reward; collisions+=int(info['collision'])
                    if term or trunc:
                        break
                length=len(actions); optimal=int(arrays['path_lengths'][i])
                rows.append(dict(scenario_index=i,map_id=str(arrays['map_ids'][i]),map_seed=int(arrays['map_seeds'][i]),
                                 success=bool(term),timeout=bool(trunc),steps=length,collisions=collisions,reward=float(total),
                                 optimal_steps=optimal,spl=float(term)*optimal/max(length,optimal),
                                 successful_path_efficiency=optimal/length if term else None,
                                 route=[list(map(int,p)) for p in route],actions=actions,**detect_loops(route,actions)))
            finally:
                env.close()
    value_after=critic_snapshot(model.policy)
    if actor_hash(model.policy) != before or any(not torch.equal(value_before[k],value_after[k]) for k in value_before):
        raise RuntimeError('Evaluation modified policy weights')
    return summarize(rows),rows


def read_verified_initialization(path,device):
    check_sb3_version()
    path=Path(path); report=json.loads(path.with_suffix('.transfer.json').read_text())
    if report['status'] != 'passed' or report['training_started'] or report['ppo_num_timesteps'] != 0:
        raise ValueError('Phase 4A acceptance report is not a passed fresh initialization')
    if file_sha256(path) != report['initialization_sha256']:
        raise ValueError('Initialization ZIP does not match verified transfer report')
    if file_sha256(report['source_checkpoint']) != report['source_checkpoint_sha256']:
        raise ValueError('Original BC checkpoint changed')
    model=PPO.load(path,device=device)
    bc,_=load_strict_bc(report['source_checkpoint'],device)
    check_transferred_parameters(bc,model.policy)
    with np.load(path.with_suffix('.observations.npz'),allow_pickle=False) as source:
        observations=source['observations']
    validate_equivalence(bc,model,observations,report['validation']['atol'],report['validation']['rtol'])
    if model.num_timesteps != 0 or model._n_updates != 0 or model.policy.optimizer.state:
        raise ValueError('Initialization has already been trained')
    return model,report


def save_checkpoint(model,path,state,is_best=False):
    """Run-owned checkpoint update; sidecar is required for continuation."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.temporary.zip')
    model.save(temporary); temporary.replace(path)
    checksum=file_sha256(path)
    if is_best:
        state.update(best_checkpoint=str(path.resolve()),best_checkpoint_sha256=checksum,best_step=int(model.num_timesteps))
    snapshot=copy.deepcopy(state)
    snapshot.update(checkpoint_sha256=checksum,checkpoint_step=int(model.num_timesteps))
    sidecar=path.with_suffix('.run.json')
    temporary_json=sidecar.with_suffix('.json.tmp')
    temporary_json.write_text(json.dumps(snapshot,indent=2)+'\n'); temporary_json.replace(sidecar)


def load_continuation(path,config_fingerprint,method):
    path=Path(path); state=json.loads(path.with_suffix('.run.json').read_text())
    if state['checkpoint_sha256'] != file_sha256(path):
        raise ValueError('Continuation checkpoint fingerprint mismatch')
    if state['config_fingerprint'] != config_fingerprint or state['method'] != method:
        raise ValueError('Continuation must retain method, datasets, validation scenarios and PPO hyperparameters')
    best=Path(state['best_checkpoint'])
    if not best.is_file() or file_sha256(best) != state['best_checkpoint_sha256']:
        raise ValueError('Retain the parent experiment: its immutable best checkpoint is required')
    return state
