"""Exact actor transfer, preprocessing, action probabilities, save/reload and no-training checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import preprocess_obs
from agents.pretrained_ppo_policy import (EncodedNavigationObservation,PretrainedPPOPolicy,
                                         check_encoded_space,check_sb3_version)
from environments.gridworld_env import RobotNavigationEnv
from imitation.bc_network import ARCHITECTURE,BCNetwork
from imitation.dataset import load_dataset
from imitation.expert_generator import generate_dataset
from imitation.transfer_to_ppo import (critic_snapshot,file_sha256,initialize_ppo,load_strict_bc,
                                       transfer_actor,validate_equivalence)


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def assets(tmp_path):
    dataset=tmp_path/'train.npz'
    generate_dataset(dataset,episodes=10,seed=1000000,split='train')
    torch.manual_seed(11)
    bc=BCNetwork(10,10,16)
    with torch.no_grad():
        bc.actor.bias.copy_(torch.tensor([-12.,3.,-5.,7.]))
    environment=dict(height=10,width=10,obstacle_density=.1,max_steps=150,bfs_shaping=False)
    payload=dict(format_version=1,architecture=ARCHITECTURE,
                 network_config=dict(height=10,width=10,hidden_size=16,n_actions=4),
                 model_state_dict=bc.state_dict(),environment=environment,epoch=0,
                 channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]],
                 dataset_provenance={'train':{'path':str(dataset),'sha256':file_sha256(dataset)}})
    checkpoint=tmp_path/'bc.pt'; torch.save(payload,checkpoint)
    config={'runtime':{'seed':2027,'device':'cpu'},
            'ppo':dict(learning_rate=.00003,n_steps=16,batch_size=8,n_epochs=2,gamma=.99,
                       gae_lambda=.95,clip_range=.1,ent_coef=0.,vf_coef=.5,max_grad_norm=.5),
            'validation':dict(observation_dataset=str(dataset),dataset_samples=16,environment_samples=16,
                              environment_seed=40000000,atol=1e-6,rtol=1e-5)}
    return checkpoint,dataset,config


def test_version_and_incompatible_spaces():
    assert check_sb3_version()=='2.9.0'
    correct=spaces.Box(0,1,shape=(3,10,10),dtype=np.float32)
    check_encoded_space(correct,spaces.Discrete(4))
    for bad in (spaces.Box(0,255,shape=(3,10,10),dtype=np.uint8),
                spaces.Box(0,1,shape=(10,10,3),dtype=np.float32)):
        with pytest.raises(ValueError):
            check_encoded_space(bad,spaces.Discrete(4))
    for actions in (spaces.Discrete(5),spaces.Discrete(4,start=1),spaces.Box(-1,1,shape=(2,))):
        with pytest.raises(ValueError,match='Discrete'):
            check_encoded_space(correct,actions)


@pytest.mark.parametrize('density',[.1,.3,.5])
def test_wrapper_matches_original_encoding_and_no_rescaling(density):
    from agents.dqn_agent import encode_observation
    raw=RobotNavigationEnv(obstacle_density=density)
    wrapped=EncodedNavigationObservation(raw)
    try:
        encoded,_=wrapped.reset(seed=42)
        obs=raw._observation()
        np.testing.assert_array_equal(encoded,encode_observation(obs))
        assert wrapped.observation_space.contains(encoded) and encoded.dtype==np.float32
        x=torch.from_numpy(encoded).unsqueeze(0)
        torch.testing.assert_close(preprocess_obs(x,wrapped.observation_space,normalize_images=False),x,rtol=0,atol=0)
        next_obs,_,_,_,_=wrapped.step(0)
        np.testing.assert_array_equal(next_obs,encode_observation(raw._observation()))
    finally:
        wrapped.close()


def test_wrapper_rejects_action_semantics():
    env=RobotNavigationEnv(); env.ACTIONS=((1,0),(-1,0),(0,-1),(0,1))
    with pytest.raises(ValueError,match='semantics'):
        EncodedNavigationObservation(env)
    env.close()


def test_strict_checkpoint_loading_rejects_missing_extra_shape_dtype_nonfinite_and_semantics(assets,tmp_path):
    checkpoint,_,_=assets
    original=torch.load(checkpoint,weights_only=True)
    cases=[]
    bad=copy.deepcopy(original); del bad['model_state_dict']['actor.bias']; cases.append(bad)
    bad=copy.deepcopy(original); bad['model_state_dict']['unexpected']=torch.ones(1); cases.append(bad)
    bad=copy.deepcopy(original); bad['model_state_dict']['actor.bias']=torch.zeros(5); cases.append(bad)
    bad=copy.deepcopy(original); bad['model_state_dict']['actor.bias']=torch.zeros(4,dtype=torch.float64); cases.append(bad)
    bad=copy.deepcopy(original); bad['model_state_dict']['actor.bias'][0]=float('nan'); cases.append(bad)
    bad=copy.deepcopy(original); bad['actions'][0]=[1,0]; cases.append(bad)
    for i,data in enumerate(cases):
        path=tmp_path/f'bad_{i}.pt'; torch.save(data,path)
        with pytest.raises(ValueError):
            load_strict_bc(path)


def test_actor_exact_and_value_network_independent(assets):
    checkpoint,dataset,config=assets
    bc,payload=load_strict_bc(checkpoint)
    env=EncodedNavigationObservation(RobotNavigationEnv(**payload['environment']))
    model=PPO(PretrainedPPOPolicy,env,policy_kwargs={'hidden_size':16},device='cpu',**config['ppo'])
    try:
        before=critic_snapshot(model.policy)
        report=transfer_actor(bc,model.policy)
        assert report['all_bc_parameters_transferred'] and report['transferred_tensors']==10
        assert report['transferred_elements']==sum(p.numel() for p in bc.parameters())
        after=critic_snapshot(model.policy)
        assert all(torch.equal(before[k],after[k]) for k in before)
        assert model.policy.pi_features_extractor is not model.policy.vf_features_extractor
        arrays,_=load_dataset(dataset)
        result=validate_equivalence(bc,model,arrays['observations'][:16])
        assert result['deterministic_action_agreement']==1.
        assert result['comparisons']['raw_logits']['max_absolute_error']==0.
        # Extra parameter-free nonlinearities must not pass a parameter-only audit.
        original=model.policy.mlp_extractor.policy_net
        model.policy.mlp_extractor.policy_net=torch.nn.Sequential(torch.nn.Tanh())
        with pytest.raises(ValueError,match='extra'):
            transfer_actor(bc,model.policy)
        model.policy.mlp_extractor.policy_net=original
        # Tampering with one actor tensor must be detected, not tolerated as a partial load.
        with torch.no_grad():
            model.policy.action_net.bias[0]+=1
        with pytest.raises(ValueError,match='mismatch'):
            validate_equivalence(bc,model,arrays['observations'][:16])
    finally:
        model.get_env().close()


def test_initialization_zip_report_reload_and_no_training(assets,tmp_path,monkeypatch):
    checkpoint,dataset,config=assets
    def forbidden(*args,**kwargs):
        raise AssertionError('PPO.learn must never run during initialization')
    monkeypatch.setattr(PPO,'learn',forbidden)
    before=file_sha256(checkpoint)
    output=tmp_path/'ppo.zip'
    report=initialize_ppo(checkpoint,output,config)
    assert report['status']=='passed' and not report['training_started']
    assert report['ppo_num_timesteps']==0 and report['ppo_updates']==0
    assert report['source_checkpoint_preserved'] and before==file_sha256(checkpoint)
    for path in (output,output.with_suffix('.transfer.json'),output.with_suffix('.observations.npz')):
        assert path.stat().st_size>0
    assert json.loads(output.with_suffix('.transfer.json').read_text())['status']=='passed'
    model=PPO.load(output,device='cpu')
    bc,_=load_strict_bc(checkpoint)
    with np.load(output.with_suffix('.observations.npz')) as source:
        batch=source['observations']
    assert validate_equivalence(bc,model,batch)['deterministic_action_agreement']==1.
    assert model.num_timesteps==0 and not model.policy.optimizer.state
    # Public reload/inference also works in a clean interpreter.
    code='from stable_baselines3 import PPO; import numpy as np; m=PPO.load('+repr(str(output))+',device="cpu"); x=np.load('+repr(str(output.with_suffix('.observations.npz')))+')["observations"]; assert m.predict(x,deterministic=True)[0].shape==(len(x),); assert m.num_timesteps==0'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with pytest.raises(FileExistsError):
        initialize_ppo(checkpoint,output,config)
    assert before==file_sha256(checkpoint)


def test_validation_failure_publishes_no_initialization(assets,tmp_path,monkeypatch):
    import imitation.transfer_to_ppo as module
    checkpoint,_,config=assets
    def fail(*args,**kwargs):
        raise ValueError('intentional equivalence failure')
    monkeypatch.setattr(module,'validate_equivalence',fail)
    output=tmp_path/'failed.zip'
    with pytest.raises(ValueError,match='equivalence failure'):
        initialize_ppo(checkpoint,output,config)
    assert not output.exists() and not output.with_suffix('.transfer.json').exists()


def test_dataset_provenance_and_split_rejection(assets,tmp_path):
    checkpoint,_,config=assets
    wrong=tmp_path/'validation.npz'; generate_dataset(wrong,episodes=5,seed=3000000,split='validation')
    config['validation']['observation_dataset']=str(wrong)
    with pytest.raises(ValueError,match='provenance'):
        initialize_ppo(checkpoint,tmp_path/'failed.zip',config)


def test_cli_help():
    result=subprocess.run([sys.executable,'-m','imitation.transfer_to_ppo','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--bc-checkpoint' in result.stdout and '--device' in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in hosted runtime')
def test_cuda_transfer_probability_parity(assets,tmp_path):
    checkpoint,_,config=assets
    report=initialize_ppo(checkpoint,tmp_path/'gpu.zip',config,device='cuda')
    assert report['validation']['deterministic_action_agreement']==1.
    assert report['reload_validation']['deterministic_action_agreement']==1.
