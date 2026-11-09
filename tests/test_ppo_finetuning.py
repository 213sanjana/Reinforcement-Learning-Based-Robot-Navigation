"""Fine-tuning, matching budgets, continuation, held-out isolation and safety gates."""
import copy
import csv
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from agents.pretrained_ppo_policy import EncodedNavigationObservation
from imitation.bc_network import ARCHITECTURE,BCNetwork
from imitation.dataset import load_dataset
from imitation.expert_generator import generate_dataset
from imitation.transfer_to_ppo import file_sha256,initialize_ppo
from training.ppo_support import (TrainingMapPoolEnv,actor_hash,critic_snapshot,evaluate_model,
                                 finite_policy,load_data,read_verified_initialization)
from training.train_pretrained_ppo import train
from evaluation.evaluate_pretrained_ppo import evaluate,load_scenarios


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    root=tmp_path_factory.mktemp('ppo_finetuning_assets')
    paths=[]; seen=set()
    for i,split in enumerate(('train','validation')):
        path=root/f'{split}.npz'
        generate_dataset(path,episodes=10,seed=1000000+i*2000000,split=split,excluded_ids=seen)
        a,_=load_dataset(path); seen.update(a['map_ids'].tolist()); paths.append(path)
    torch.manual_seed(11); network=BCNetwork(10,10,16)
    env=dict(height=10,width=10,obstacle_density=.1,max_steps=150,bfs_shaping=False)
    payload=dict(format_version=1,architecture=ARCHITECTURE,
                 network_config=dict(height=10,width=10,hidden_size=16,n_actions=4),
                 model_state_dict=network.state_dict(),environment=env,epoch=0,
                 channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]],
                 dataset_provenance={name:{'path':str(path),'sha256':file_sha256(path)} for name,path in zip(('train','validation'),paths)})
    source=root/'bc.pt'; torch.save(payload,source)
    p=dict(learning_rate=.0003,n_steps=16,batch_size=8,n_epochs=1,gamma=.99,gae_lambda=.95,
           clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,target_kl=.02)
    transfer=dict(runtime={'seed':2027,'device':'cpu'},ppo=p,
                  validation=dict(observation_dataset=str(paths[0]),dataset_samples=16,environment_samples=8,
                                  environment_seed=40000000,atol=1e-6,rtol=1e-5))
    initialization=root/'initialization.zip'; initialize_ppo(source,initialization,transfer,device='cpu')
    cfg=dict(initialization=str(initialization),data={'train':str(paths[0]),'validation':str(paths[1])},
             training=dict(total_timesteps=160,seed=42,device='cpu',output_root=str(root/'runs'),checkpoint_interval=32),
             ppo=p,validation=dict(episodes=3,seed=50211,interval=64,early_steps=[16,32],
                                  regression_drop=1.,regression_patience=2,regression_min_steps=32))
    return cfg


def fresh(assets):
    return copy.deepcopy(assets)


def test_manifest_training_reset_only_uses_train_layouts(assets):
    cfg=fresh(assets); model,report=read_verified_initialization(cfg['initialization'],'cpu')
    pool,validation=load_data(cfg,report['source_environment'],report)
    env=TrainingMapPoolEnv(pool)
    try:
        for seed in range(20):
            obs,_=env.reset(seed=seed)
            assert env.pool['arrays']['map_ids'][env.task_index] not in set(validation['arrays']['map_ids'])
            for action in (0,1,2,3):
                obs,reward,term,trunc,info=env.step(action)
                assert info['episode_reward']==env.episode_reward
                if term or trunc:
                    break
    finally:
        env.close()


def test_finetuning_updates_parameters_and_logs_without_touching_sources(assets,tmp_path):
    cfg=fresh(assets); source_hash=file_sha256(cfg['initialization'])
    run=tmp_path/'pretrained'
    result=train(cfg,steps=160,output_directory=run,device='cpu')
    assert result['status']=='completed' and result['actual_env_timesteps']==160
    assert source_hash==file_sha256(cfg['initialization']) and result['source_artifacts_preserved']
    for name in ('initial.zip','best.zip','final.zip','final.run.json','episodes.csv','updates.csv',
                 'validation.csv','validation_scenarios.npz','training.png','summary.json'):
        assert (run/name).stat().st_size>0
    initial=PPO.load(run/'initial.zip',device='cpu'); final=PPO.load(run/'final.zip',device='cpu')
    assert actor_hash(initial.policy)!=actor_hash(final.policy)
    assert final._n_updates>0 and final.policy.optimizer.state
    assert json.loads((run/'run_manifest.json').read_text())['starting_actor_sha256']==json.loads((run/'run_manifest.json').read_text())['verified_actor_sha256']
    updates=list(csv.DictReader((run/'updates.csv').open()))
    assert len(updates)==10 and all(row['loss'] for row in updates)
    episodes=list(csv.DictReader((run/'episodes.csv').open()))
    assert episodes and all(float(row['collisions'])>=0 for row in episodes)
    with pytest.raises(FileExistsError):
        train(cfg,steps=160,output_directory=run,device='cpu')


def test_rounding_scratch_pairing_and_rng_preservation(assets,tmp_path):
    cfg=fresh(assets)
    first=train(cfg,steps=31,output_directory=tmp_path/'pretrained',device='cpu')
    second=train(cfg,steps=31,mode='scratch',output_directory=tmp_path/'scratch',device='cpu')
    assert first['actual_env_timesteps']==second['actual_env_timesteps']==32
    a=json.loads((tmp_path/'pretrained/run_manifest.json').read_text())
    b=json.loads((tmp_path/'scratch/run_manifest.json').read_text())
    assert a['validation_signature']==b['validation_signature']
    assert b['starting_actor_sha256']!=b['verified_actor_sha256']
    p=PPO.load(tmp_path/'pretrained/initial.zip',device='cpu')
    s=PPO.load(tmp_path/'scratch/initial.zip',device='cpu')
    left,right=critic_snapshot(p.policy),critic_snapshot(s.policy)
    assert all(torch.equal(left[key],right[key]) for key in left)
    p.policy.set_training_mode(True)
    before=torch.get_rng_state().clone(); python_state=random.getstate(); numpy_state=np.random.get_state()
    scenarios=load_scenarios(tmp_path/'pretrained')
    evaluate_model(p,scenarios)
    assert torch.equal(before,torch.get_rng_state()) and random.getstate()==python_state and p.policy.training
    assert np.array_equal(numpy_state[1],np.random.get_state()[1])
    paired=evaluate(tmp_path/'pretrained',tmp_path/'evaluation',tmp_path/'scratch',device='cpu')
    assert paired['comparison']['budget_matched'] and not paired['test_dataset_used']
    assert set(paired['summary'])=={'bc_initialization','pretrained_ppo','scratch_ppo'}


def test_continuation_keeps_optimizer_and_global_budget(assets,tmp_path):
    cfg=fresh(assets)
    first=train(cfg,steps=32,output_directory=tmp_path/'first',device='cpu')
    source=tmp_path/'first/final.zip'; checksum=file_sha256(source)
    model=PPO.load(source,device='cpu'); previous_updates=model._n_updates
    continuation=train(cfg,steps=64,resume=source,output_directory=tmp_path/'continued',device='cpu')
    assert continuation['status']=='completed' and continuation['starting_step']==32
    assert continuation['actual_env_timesteps']==64 and continuation['segment_env_timesteps']==32
    assert continuation['cumulative_consumed_env_timesteps']==64
    model=PPO.load(tmp_path/'continued/final.zip',device='cpu')
    assert model._n_updates>previous_updates and model.policy.optimizer.state and file_sha256(source)==checksum
    changed=fresh(assets); changed['ppo']['learning_rate']=.0001
    with pytest.raises(ValueError,match='Continuation'):
        train(changed,steps=96,resume=source,output_directory=tmp_path/'bad',device='cpu')


def test_early_regression_stops_on_update_boundary_and_keeps_initial_best(assets,tmp_path,monkeypatch):
    import training.train_pretrained_ppo as module
    original=module.evaluate_model
    calls=[]
    def regression(model,scenarios):
        summary,rows=original(model,scenarios); calls.append(model.num_timesteps)
        summary['success_rate']=1. if len(calls)==1 else 0.
        return summary,rows
    monkeypatch.setattr(module,'evaluate_model',regression)
    cfg=fresh(assets); cfg['validation'].update(regression_drop=.1,regression_patience=1,regression_min_steps=16)
    result=train(cfg,steps=160,output_directory=tmp_path/'regression',device='cpu')
    assert result['status']=='stopped_regression' and result['actual_env_timesteps']==16
    assert result['best_step']==0 and (tmp_path/'regression/best.zip').exists()


def test_nonfinite_update_restores_last_finite_checkpoint(assets,tmp_path,monkeypatch):
    original=PPO.train
    def broken(self):
        original(self)
        with torch.no_grad():
            next(self.policy.parameters()).fill_(float('nan'))
    monkeypatch.setattr(PPO,'train',broken)
    result=train(fresh(assets),steps=16,output_directory=tmp_path/'failed',device='cpu')
    assert result['status']=='failed_numerical' and result['final_checkpoint_saved']
    assert result['final_checkpoint_step']==0
    finite_policy(PPO.load(tmp_path/'failed/final.zip',device='cpu'))
    assert (tmp_path/'failed/failure.json').exists()


def test_train_validation_leakage_is_rejected(assets,tmp_path):
    cfg=fresh(assets); cfg['data']['validation']=cfg['data']['train']
    with pytest.raises(ValueError,match='validation'):
        train(cfg,steps=32,output_directory=tmp_path/'bad',device='cpu')
    assert not (tmp_path/'bad').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in hosted CPU runtime')
def test_cuda_smoke_training_and_reload(assets,tmp_path):
    result=train(fresh(assets),steps=16,output_directory=tmp_path/'cuda',device='cuda')
    assert result['status']=='completed'
    finite_policy(PPO.load(tmp_path/'cuda/final.zip',device='cuda'))


def test_forked_earlier_checkpoint_cost_is_not_hidden(assets,tmp_path):
    cfg=fresh(assets)
    train(cfg,steps=32,output_directory=tmp_path/'parent',device='cpu')
    fork=train(cfg,steps=16,resume=tmp_path/'parent/initial.zip',output_directory=tmp_path/'fork',device='cpu')
    assert fork['actual_env_timesteps']==16 and fork['cumulative_consumed_env_timesteps']==48
    train(cfg,steps=16,mode='scratch',output_directory=tmp_path/'scratch',device='cpu')
    report=evaluate(tmp_path/'fork',tmp_path/'mismatched_comparison',tmp_path/'scratch',device='cpu')
    assert report['comparison']['budget_matched'] is False


def test_interruption_keeps_only_completed_finite_checkpoint(assets,tmp_path,monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt()
    monkeypatch.setattr(PPO,'train',interrupt)
    result=train(fresh(assets),steps=16,output_directory=tmp_path/'interrupted',device='cpu')
    assert result['status']=='interrupted' and result['actual_env_timesteps']==16
    assert result['final_checkpoint_step']==0 and result['cumulative_consumed_env_timesteps']==16
    finite_policy(PPO.load(tmp_path/'interrupted/final.zip',device='cpu'))


def test_extended_training_requires_completed_diagnostic(assets,tmp_path):
    with pytest.raises(ValueError,match='diagnostic'):
        train(fresh(assets),steps=300000,output_directory=tmp_path/'extended',device='cpu')
    assert not (tmp_path/'extended').exists()
