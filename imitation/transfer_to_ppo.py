"""Strict BC-to-SB3 PPO initialization; compares full actor outputs and never calls learn()."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import preprocess_obs
from agents.pretrained_ppo_policy import (ACTION_DELTAS,EncodedNavigationObservation,
                                         PretrainedPPOPolicy,check_sb3_version)
from environments.gridworld_env import RobotNavigationEnv
from imitation.bc_network import ARCHITECTURE,BCNetwork,resolve_device
from imitation.dataset import encode_observation
from imitation.train_bc import seed_everything


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def load_strict_bc(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('format_version') != 1 or data.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported BC format/architecture')
    if data.get('channels') != ['obstacles','agent','goal'] or data.get('actions') != [list(a) for a in ACTION_DELTAS]:
        raise ValueError('BC observation/action semantics mismatch')
    network=BCNetwork(**data['network_config'])
    actual=data['model_state_dict']; expected=network.state_dict()
    if set(actual) != set(expected):
        raise ValueError(f'BC parameter keys mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}')
    for key,reference in expected.items():
        value=actual[key]
        if not isinstance(value,torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f'BC parameter shape/dtype mismatch: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite BC parameter: {key}')
    network.load_state_dict(actual,strict=True)
    env=data['environment']
    if (env['height'],env['width']) != (network.height,network.width):
        raise ValueError('BC environment/network grid size mismatch')
    network.to(resolve_device(device)).eval()
    return network,data


def actor_modules(policy):
    if not isinstance(policy,PretrainedPPOPolicy):
        raise ValueError('Expected the compatible PretrainedPPOPolicy')
    if policy.normalize_images or policy.share_features_extractor or policy.ortho_init:
        raise ValueError('PPO policy preprocessing/sharing/initialization contract changed')
    if policy.features_extractor is not policy.pi_features_extractor or policy.pi_features_extractor is policy.vf_features_extractor:
        raise ValueError('Unexpected SB3 feature-extractor aliases')
    if len(policy.mlp_extractor.policy_net) or len(policy.mlp_extractor.value_net):
        raise ValueError('Unexpected extra policy/value MLP layers')
    return [('features','pi_features_extractor.features',policy.pi_features_extractor.features),
            ('shared','pi_features_extractor.shared',policy.pi_features_extractor.shared),
            ('actor','action_net',policy.action_net)]


def critic_snapshot(policy):
    return {**{'vf_features_extractor.'+k:v.detach().cpu().clone() for k,v in policy.vf_features_extractor.state_dict().items()},
            **{'value_net.'+k:v.detach().cpu().clone() for k,v in policy.value_net.state_dict().items()}}


def transfer_actor(bc,policy):
    groups=actor_modules(policy)
    value_before=critic_snapshot(policy)
    actor_ids={id(p) for _,_,module in groups for p in module.parameters()}
    critic_ids={id(p) for p in policy.vf_features_extractor.parameters()} | {id(p) for p in policy.value_net.parameters()}
    if actor_ids & critic_ids or actor_ids | critic_ids != {id(p) for p in policy.parameters()}:
        raise ValueError('Actor/critic parameter partition is incomplete or shared')
    consumed=set(); rows=[]
    for source_prefix,target_prefix,target in groups:
        source=getattr(bc,source_prefix)
        source_state=source.state_dict(); target_state=target.state_dict()
        if set(source_state) != set(target_state):
            raise ValueError(f'Parameter key mismatch in {source_prefix}')
        for key,value in source_state.items():
            destination=target_state[key]
            if value.shape != destination.shape or value.dtype != destination.dtype:
                raise ValueError(f'Transfer shape/dtype mismatch: {source_prefix}.{key}')
        target.load_state_dict(source_state,strict=True)
        for key,value in source_state.items():
            destination=target.state_dict()[key]
            if not torch.equal(value.detach().cpu(),destination.detach().cpu()):
                raise ValueError(f'Parameter transfer was not exact: {source_prefix}.{key}')
            full=f'{source_prefix}.{key}'; consumed.add(full)
            rows.append(dict(source=full,target=f'{target_prefix}.{key}',shape=list(value.shape),
                             elements=value.numel(),dtype=str(value.dtype),sha256=tensor_sha256(value),exact_match=True))
    if consumed != set(bc.state_dict()):
        raise ValueError('Not every BC state entry transferred')
    value_after=critic_snapshot(policy)
    if value_before.keys() != value_after.keys() or any(not torch.equal(value_before[k],value_after[k]) for k in value_before):
        raise ValueError('Critic weights changed during actor transfer')
    if policy.optimizer.state:
        raise ValueError('Expected a fresh PPO optimizer with no BC optimizer state')
    optimizer_ids={id(p) for group in policy.optimizer.param_groups for p in group['params']}
    if optimizer_ids != {id(p) for p in policy.parameters()}:
        raise ValueError('PPO optimizer does not reference every current policy parameter')
    return dict(all_bc_parameters_transferred=True,transferred_tensors=len(rows),
                transferred_elements=sum(r['elements'] for r in rows),parameters=rows,
                critic_independent=True,critic_unchanged_by_transfer=True,
                critic_tensors=len(value_before),critic_elements=sum(v.numel() for v in value_before.values()),
                critic_sha256={k:tensor_sha256(v) for k,v in value_before.items()},optimizer_fresh=True,optimizer_parameter_coverage_exact=True)


def check_transferred_parameters(bc,policy):
    total=0
    for source_prefix,_,target in actor_modules(policy):
        source=getattr(bc,source_prefix).state_dict(); destination=target.state_dict()
        if source.keys() != destination.keys():
            raise ValueError('Reloaded actor parameter keys differ')
        for key in source:
            if not torch.equal(source[key].detach().cpu(),destination[key].detach().cpu()):
                raise ValueError(f'Actor parameter mismatch: {source_prefix}.{key}')
            total+=1
    if total != len(bc.state_dict()):
        raise ValueError('Incomplete actor parameter verification')


def validate_equivalence(bc,model,observations,atol=1e-6,rtol=1e-5):
    if observations.dtype != np.float32 or observations.ndim != 4 or len(observations)<1:
        raise ValueError('Validation batch must be nonempty float32 NCHW')
    if not np.isfinite(observations).all() or not np.isin(observations,(0,1)).all():
        raise ValueError('Validation observations must be binary and finite')
    if not np.all(observations[:,1:].sum(axis=(2,3))==1) or np.any(observations[:,0:1]*observations[:,1:]):
        raise ValueError('Validation agent/goal planes must be one-hot on free cells')
    policy=model.policy; policy.set_training_mode(False); bc.eval()
    device=next(policy.parameters()).device
    if next(bc.parameters()).device != device:
        raise ValueError('Compare BC and PPO on the same device')
    x=torch.from_numpy(observations).to(device)
    comparisons={}
    with torch.inference_mode():
        preprocessed=preprocess_obs(x,policy.observation_space,normalize_images=policy.normalize_images)
        if not torch.equal(x,preprocessed):
            raise ValueError('SB3 preprocessing altered BC input planes')
        bc_convolution=bc.features(x)
        ppo_convolution=policy.pi_features_extractor.features(preprocessed)
        bc_features=bc.shared(bc_convolution)
        ppo_features,_=policy.extract_features(x)
        bc_logits=bc(x)
        ppo_logits=policy.action_net(policy.mlp_extractor.forward_actor(ppo_features))
        bc_probabilities=torch.softmax(bc_logits,dim=1)
        ppo_probabilities=policy.get_distribution(x).distribution.probs
        for name,left,right in [('cnn_features',bc_convolution,ppo_convolution),
                                ('hidden_features',bc_features,ppo_features),
                                ('raw_logits',bc_logits,ppo_logits),
                                ('action_probabilities',bc_probabilities,ppo_probabilities)]:
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f'Non-finite {name}')
            if not torch.allclose(left,right,atol=atol,rtol=rtol):
                raise ValueError(f'{name} mismatch; max absolute error={float((left-right).abs().max())}')
            comparisons[name]={'max_absolute_error':float((left-right).abs().max().item()),'match':True}
        values=policy.predict_values(x)
        if values.shape != (len(x),1) or not torch.isfinite(values).all():
            raise ValueError('Non-finite/invalid independent critic output')
        bc_actions=bc_logits.argmax(1).cpu().numpy()
    predicted,_=model.predict(observations,deterministic=True)
    if not np.array_equal(predicted,bc_actions):
        raise ValueError('PPO.predict deterministic actions differ from BC')
    # Also exercise the unbatched public prediction path.
    single,_=model.predict(observations[0],deterministic=True)
    with torch.inference_mode():
        bc_single=int(bc(x[:1]).argmax(1).item())
    if int(single) != bc_single:
        raise ValueError('Unbatched deterministic action mismatch')
    check_transferred_parameters(bc,policy)
    return dict(batch_size=len(observations),atol=atol,rtol=rtol,preprocessing_exact=True,
                comparisons=comparisons,deterministic_action_agreement=1.0,
                single_observation_action_agreement=True,critic_outputs_finite=True)


def fixed_observations(payload,settings):
    dataset=Path(settings['observation_dataset'])
    expected=payload['dataset_provenance']['train']['sha256']
    if file_sha256(dataset) != expected:
        raise ValueError('Observation dataset differs from BC training provenance')
    with np.load(dataset,allow_pickle=False) as source:
        metadata=json.loads(str(source['metadata']))
        if metadata.get('split') != 'train' or metadata.get('channels') != payload['channels'] or metadata.get('actions') != payload['actions']:
            raise ValueError('Validation inputs must use the original train split and encoding contract')
        observations=source['observations']
        if observations.dtype != np.float32:
            raise ValueError('Training observations must already be float32; no implicit dtype conversion')
        count=settings['dataset_samples']
        if not 1 <= count <= len(observations):
            raise ValueError('dataset_samples exceeds available training observations')
        rng=np.random.default_rng(settings['environment_seed'])
        indices=rng.choice(len(observations),size=count,replace=False)
        batch=[np.asarray(observations[indices],dtype=np.float32)]
    # Compare raw-environment encoding to the original DQN encoder as an independent reference.
    from agents.dqn_agent import encode_observation as dqn_encode
    env=RobotNavigationEnv(**payload['environment'])
    wrapped=EncodedNavigationObservation(env)
    extra=[]; action_rng=np.random.default_rng(settings['environment_seed']+1)
    try:
        obs,_=env.reset(seed=settings['environment_seed'])
        for i in range(settings['environment_samples']):
            encoded=wrapped.observation(obs)
            if not np.array_equal(encoded,dqn_encode(obs)):
                raise ValueError('Raw environment encoding differs from audited DQN/BC convention')
            extra.append(encoded)
            obs,_,term,trunc,_=env.step(int(action_rng.integers(4)))
            if term or trunc:
                obs,_=env.reset(seed=settings['environment_seed']+i+1)
    finally:
        wrapped.close()
    if extra:
        batch.append(np.stack(extra))
    fixed=np.concatenate(batch)
    return fixed,dict(dataset=str(dataset),dataset_sha256=expected,dataset_indices=indices.tolist(),
                      dataset_samples=count,environment_samples=len(extra),environment_seed=settings['environment_seed'],
                      environment_action_seed=settings['environment_seed']+1,
                      original_dqn_encoder_agreement=True)


def initialize_ppo(bc_checkpoint,output,config,device=None):
    sb3_version=check_sb3_version()
    output=Path(output)
    if output.suffix != '.zip':
        raise ValueError('PPO initialization output must end in .zip')
    report_path=output.with_suffix('.transfer.json')
    batch_path=output.with_suffix('.observations.npz')
    for path in (output,report_path,batch_path):
        if path.exists():
            raise FileExistsError(f'{path} exists; choose a new output filename')
    if output.resolve()==Path(bc_checkpoint).resolve():
        raise ValueError('PPO output must differ from BC source')
    settings=config['validation']
    if settings['environment_samples'] < 1 or settings['dataset_samples'] < 1 or settings['atol'] < 0 or settings['rtol'] < 0:
        raise ValueError('Invalid transfer validation settings')
    seed=int(config['runtime']['seed'])
    seed_everything(seed,deterministic=True)
    request=device or config['runtime']['device']
    resolved=resolve_device(request)
    source_hash=file_sha256(bc_checkpoint)
    bc,payload=load_strict_bc(bc_checkpoint,str(resolved))
    observations,batch_provenance=fixed_observations(payload,settings)
    env=EncodedNavigationObservation(RobotNavigationEnv(**payload['environment']))
    model=None
    try:
        model=PPO(PretrainedPPOPolicy,env,policy_kwargs={'hidden_size':bc.hidden_size},
                  device=str(resolved),seed=seed,verbose=0,**config['ppo'])
        transfer=transfer_actor(bc,model.policy)
        validation=validate_equivalence(bc,model,observations,settings['atol'],settings['rtol'])
        if model.num_timesteps != 0 or model._n_updates != 0 or model.rollout_buffer.pos != 0:
            raise ValueError('PPO learning unexpectedly started')
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.ppo_transfer_',dir=output.parent) as directory:
            staged=Path(directory)/'initialization.zip'
            # save() serializes the model; no training or rollout collection occurs.
            model.save(staged)
            reloaded=PPO.load(staged,device=str(resolved))
            reload_validation=validate_equivalence(bc,reloaded,observations,settings['atol'],settings['rtol'])
            fresh=critic_snapshot(reloaded.policy)
            if any(transfer['critic_sha256'][key] != tensor_sha256(value) for key,value in fresh.items()):
                raise ValueError('Critic changed after serialization/reload')
            if any(not torch.equal(value.detach().cpu(),reloaded.policy.state_dict()[key].detach().cpu())
                   for key,value in model.policy.state_dict().items()):
                raise ValueError('PPO state changed after serialization/reload')
            if reloaded.num_timesteps != 0 or reloaded._n_updates != 0 or reloaded.policy.optimizer.state:
                raise ValueError('Reloaded PPO is not a fresh initialization')
            if source_hash != file_sha256(bc_checkpoint):
                raise ValueError('Source BC checkpoint changed')
            report=dict(status='passed',training_started=False,ppo_num_timesteps=0,ppo_updates=0,
                        source_checkpoint=str(bc_checkpoint),source_checkpoint_sha256=source_hash,
                        source_checkpoint_preserved=True,bc_checkpoint_epoch=int(payload['epoch']),
                        bc_architecture=payload['architecture'],network_config=payload['network_config'],
                        source_environment=payload['environment'],stable_baselines3_version=sb3_version,
                        torch_version=str(torch.__version__),numpy_version=str(np.__version__),device=str(resolved),
                        actor_critic_shared=False,critic_initialization='Separate fresh CNN, hidden layer and value head',
                        raw_observation_space='Dict: agent int32[2], goal int32[2], obstacles int8[H,W]',
                        ppo_observation_space=dict(type='Box',shape=list(model.observation_space.shape),dtype='float32',low=0.,high=1.),
                        action_space=dict(type='Discrete',n=4,start=0,deltas=[list(a) for a in ACTION_DELTAS]),
                        normalize_images=False,additional_actor_layers=[],parameter_transfer=transfer,
                        validation=validation,reload_validation=reload_validation,
                        observation_batch_provenance=batch_provenance,config=config,
                        output=str(output),initialization_sha256=file_sha256(staged),
                        fixed_observation_batch=str(batch_path))
            # Publish only after both original and reloaded models pass all checks.
            with output.open('xb') as stream,staged.open('rb') as source:
                shutil.copyfileobj(source,stream)
            with batch_path.open('xb') as stream:
                np.savez_compressed(stream,observations=observations)
            with report_path.open('x',encoding='utf8') as stream:
                json.dump(report,stream,indent=2); stream.write('\n')
        return report
    finally:
        if model is not None:
            model.get_env().close()
        else:
            env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bc-checkpoint',default='artifacts/imitation/behavior_cloning/best.pt')
    parser.add_argument('--output',default='artifacts/ppo/pretrained_initialization.zip')
    parser.add_argument('--config',default='configs/pretrained_ppo.yaml')
    parser.add_argument('--device',choices=['auto','cpu','cuda'])
    args=parser.parse_args()
    with open(args.config,encoding='utf8') as stream:
        config=yaml.safe_load(stream)
    report=initialize_ppo(args.bc_checkpoint,args.output,config,args.device)
    print(json.dumps({key:report[key] for key in ('status','training_started','ppo_num_timesteps','bc_checkpoint_epoch',
                                                'stable_baselines3_version','device','source_checkpoint_preserved','validation',
                                                'reload_validation','output')},indent=2))
    print('Full tensor-by-tensor audit:',Path(args.output).with_suffix('.transfer.json'))


if __name__=='__main__':
    main()
