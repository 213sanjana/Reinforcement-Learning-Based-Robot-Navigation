"""Phase 4C decisions, reset-only maps, provenance, recovery and paired evaluation."""
import copy
import json
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from curriculum.curriculum_manager import CurriculumManager,wilson_lower
from curriculum.map_pools import build_pools,load_pools,split_for,CurriculumMapEnv
from imitation.dataset import map_identity,load_dataset
from imitation.bc_network import ARCHITECTURE,BCNetwork
from imitation.expert_generator import generate_dataset
from imitation.transfer_to_ppo import file_sha256,initialize_ppo
from training.train_pretrained_ppo import train as train4b
from training.train_curriculum_ppo import train,read_checkpoint,validate_config
from training.ppo_support import actor_hash
from evaluation.evaluate_curriculum import evaluate


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def counts(p,n=200):
    successes=round(p*n)
    return dict(episodes=n,successes=successes,success_rate=successes/n)


def manager(**kwargs):
    return CurriculumManager([.1,.2,.3,.4,.5],64,min_level_steps=16,**kwargs)


def test_promotion_uses_lower_bound_repetition_unique_updates_and_large_samples():
    m=manager()
    assert wilson_lower(160,200)<.8
    for _ in range(3):
        m.charge(16); assert m.observe([counts(.8)]*5)=='none'
    assert m.streak==0
    m=manager(); m.charge(16)
    assert m.observe([counts(1,20)]*5)=='none' and m.streak==0
    for _ in range(3):
        m.charge(16); decision=m.observe([counts(.9)]*5)
        streak=m.streak; assert m.observe([counts(.9)]*5)=='none' and m.streak==streak
    assert decision=='promoted' and m.level==1 and m.level_steps==0


def test_forgetting_blocks_promotion_and_level_budget_terminates():
    m=manager(); m.previous_reference={'0':.98}; m.level=1
    for _ in range(4):
        m.charge(16); m.observe([counts(.5),counts(1),counts(1),counts(1),counts(1)])
    assert m.streak==0 and m.enforce_budget()=='level_budget_exhausted'
    with pytest.raises(ValueError): m.charge(16)


def test_manager_roundtrip_mixture_and_mastery():
    m=manager(required_passes=2); m.level=4; m.level_steps=16
    m.charge(16); m.observe([counts(1)]*5)
    recovered=CurriculumManager.from_state(json.loads(json.dumps(m.state_dict())))
    assert recovered.state_dict()==m.state_dict()
    assert np.allclose(m.weights(),[.05,.05,.05,.05,.8])
    recovered.charge(16); assert recovered.observe([counts(1)]*5)=='mastered'
    f=manager(mode='fixed'); assert np.array_equal(f.weights(),[0,0,0,0,1])
    f.charge(64); f.observe([counts(1)]*5); assert f.enforce_budget()=='running'


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    root=tmp_path_factory.mktemp('curriculum_assets'); paths=[]; seen=set()
    for i,split in enumerate(('train','validation','test')):
        path=root/f'{split}.npz'; generate_dataset(path,episodes=8,seed=1000000+i*2000000,split=split,excluded_ids=seen)
        a,_=load_dataset(path); seen.update(a['map_ids']); paths.append(path)
    torch.manual_seed(9); net=BCNetwork(10,10,16)
    env=dict(height=10,width=10,obstacle_density=.1,max_steps=150,bfs_shaping=False)
    payload=dict(format_version=1,architecture=ARCHITECTURE,network_config=dict(height=10,width=10,hidden_size=16,n_actions=4),
        model_state_dict=net.state_dict(),environment=env,epoch=0,channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]],
        dataset_provenance={name:{'path':str(path),'sha256':file_sha256(path)} for name,path in zip(('train','validation'),paths)})
    bc=root/'bc.pt'; torch.save(payload,bc)
    p=dict(learning_rate=.00003,n_steps=16,batch_size=8,n_epochs=1,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,target_kl=.02)
    transfer=dict(runtime={'seed':2027,'device':'cpu'},ppo=p,validation=dict(observation_dataset=str(paths[0]),dataset_samples=8,
        environment_samples=4,environment_seed=40000000,atol=1e-6,rtol=1e-5))
    init=root/'initialization.zip'; initialize_ppo(bc,init,transfer,device='cpu')
    fourb=dict(initialization=str(init),data=dict(train=str(paths[0]),validation=str(paths[1])),ppo=p,
        training=dict(total_timesteps=16,seed=42,device='cpu',output_root=str(root),checkpoint_interval=16),
        validation=dict(episodes=4,seed=50211,interval=16,early_steps=[16],regression_drop=1.,regression_patience=2,regression_min_steps=16))
    train4b(fourb,steps=16,output_directory=root/'phase4b',device='cpu')
    cfg=dict(starting_checkpoint=str(root/'phase4b/final.zip'),verified_initialization=str(init),
        maps=dict(directory=str(root/'pools'),seed=7302026,split_salt='curriculum_test',train_maps_per_level=8,
            validation_maps_per_level=4,min_path_length=6,max_candidates_per_pool=10000,exclude_manifests=list(map(str,paths))),
        training=dict(total_timesteps=128,seed=42,device='cpu',output_root=str(root/'runs'),checkpoint_interval=16),ppo=p,
        curriculum=dict(densities=[.1,.3],success_threshold=.8,required_passes=2,min_level_steps=16,level_budget=64,
            forgetting_drop=.15,easier_weight=.2,harder_weight=.1,fixed_level=1),
        validation=dict(episodes=4,min_episodes=4,interval=16))
    return cfg,env


def test_maps_deterministic_disjoint_exclude_old_layouts_and_reserve_test(assets,tmp_path):
    cfg,env=copy.deepcopy(assets); pools,manifest=build_pools(cfg,env)
    cfg['maps']['directory']=str(tmp_path/'same'); same,other=build_pools(cfg,env)
    assert manifest['signature']==other['signature']; ids=[]
    for level in range(2):
        for split in ('train','validation'):
            a=pools[level][split]['arrays']; b=same[level][split]['arrays']
            assert all(np.array_equal(a[k],b[k]) for k in a); ids.extend(a['map_ids'])
            for grid in a['obstacles']:
                identity=map_identity(grid)
                assert map_identity(np.fliplr(np.rot90(grid)))==identity
                assert split_for(identity,cfg['maps']['split_salt'])==split
    assert len(ids)==len(set(ids))
    for path in cfg['maps']['exclude_manifests']:
        with np.load(path) as data: assert not set(ids)&set(data['map_ids'])
    assert 'test' not in pools[0]


def test_sampling_changes_density_only_at_reset_and_is_reproducible(assets):
    cfg,environment=copy.deepcopy(assets); pools,_=build_pools(cfg,environment)
    m=CurriculumManager([.1,.3],64,min_level_steps=16,min_episodes=4,fixed_level=1)
    a=CurriculumMapEnv(pools,m,0,0); b=CurriculumMapEnv(pools,m,0,0)
    first,_=a.reset(seed=42); second,_=b.reset(seed=42)
    assert all(np.array_equal(first[k],second[k]) for k in first)
    grid=a.obstacles.copy(); m.level=1; a.step(0)
    assert np.array_equal(a.obstacles,grid) and a.obstacle_density==.1
    a.reset(); assert a.obstacle_density==.3 and a.obstacles.sum()==30
    m.level=0; selected=[]
    mixed=CurriculumMapEnv(pools,m,0,.5)
    for i in range(100): mixed.reset(seed=i); selected.append(mixed.sampled_level)
    assert set(selected)=={0,1}
    a.close(); b.close(); mixed.close()


def test_generation_attempts_bounded_and_tampering_rejected(assets,tmp_path):
    cfg,env=copy.deepcopy(assets); cfg['maps']['directory']=str(tmp_path/'bounded'); cfg['maps']['max_candidates_per_pool']=1
    with pytest.raises(RuntimeError,match='bounded attempts'): build_pools(cfg,env)
    assert (tmp_path/'bounded/INCOMPLETE.txt').exists()
    cfg,env=copy.deepcopy(assets); cfg['maps']['directory']=str(tmp_path/'corrupt'); build_pools(cfg,env)
    path=tmp_path/'corrupt/level_0_train.npz'; path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='checksum'): load_pools(path.parent)


def test_real_training_resume_optimizer_counters_source_protection_and_pairing(assets,tmp_path):
    cfg,_=copy.deepcopy(assets); before=file_sha256(cfg['starting_checkpoint'])
    first=train(cfg,steps=32,output_directory=tmp_path/'first',device='cpu')
    assert first['status']=='total_budget_exhausted' and first['completed_phase4c_steps']==32
    model=PPO.load(tmp_path/'first/final.zip'); initial=PPO.load(tmp_path/'first/initial.zip')
    assert model._n_updates>initial._n_updates and model.policy.optimizer.state
    assert actor_hash(model.policy)!=actor_hash(initial.policy)
    again=train(cfg,steps=48,output_directory=tmp_path/'resume',resume=tmp_path/'first/final.zip',device='cpu')
    assert again['completed_phase4c_steps']==48 and again['cumulative_consumed_phase4c_steps']==48
    assert PPO.load(tmp_path/'resume/final.zip')._n_updates>model._n_updates
    assert file_sha256(cfg['starting_checkpoint'])==before
    fixed=train(cfg,steps=48,mode='fixed',output_directory=tmp_path/'fixed',device='cpu')
    assert fixed['completed_phase4c_steps']==48
    result=evaluate(tmp_path/'resume',tmp_path/'eval',tmp_path/'fixed',device='cpu')
    assert result['comparison']['budget_matched'] and not result['test_performance_used'] and len(result['metrics'])==6
    assert (tmp_path/'eval/comparison.png').is_file()
    for name in ('initial.zip','best.zip','last_finite.zip','final.zip','updates.csv','episodes.csv','validation.csv','curriculum.png'):
        assert (tmp_path/'first'/name).stat().st_size>0
    with pytest.raises(FileExistsError): train(cfg,steps=32,output_directory=tmp_path/'first',device='cpu')
    changed=copy.deepcopy(cfg); changed['ppo']['learning_rate']=.0001
    with pytest.raises(ValueError,match='Resume must retain'):
        train(changed,steps=48,output_directory=tmp_path/'badresume',resume=tmp_path/'first/final.zip',device='cpu')


def test_budget_stop_and_no_false_promotion(assets,tmp_path):
    cfg,_=copy.deepcopy(assets); cfg['curriculum']['success_threshold']=1.
    result=train(cfg,steps=128,output_directory=tmp_path/'bounded',device='cpu')
    assert result['status']=='level_budget_exhausted' and result['completed_phase4c_steps']==64 and result['current_level']==0
    assert (tmp_path/'bounded/levels/level_0_budget_exhausted.zip').exists()
    with pytest.raises(ValueError,match='Cannot resume'):
        train(cfg,steps=256,output_directory=tmp_path/'illegal',resume=tmp_path/'bounded/final.zip',device='cpu')


def test_simulated_passes_promote_at_distinct_updates_and_save_levels(assets,tmp_path,monkeypatch):
    cfg,_=copy.deepcopy(assets); cfg['curriculum']['success_threshold']=.1
    import training.train_curriculum_ppo as module
    original=module.evaluate_model
    def success(model,scenarios):
        summary,rows=original(model,scenarios); summary.update(episodes=4,successes=4,success_rate=1.)
        return summary,rows
    monkeypatch.setattr(module,'evaluate_model',success)
    result=train(cfg,steps=128,output_directory=tmp_path/'promote',device='cpu')
    assert result['status']=='mastered' and result['completed_phase4c_steps']==64
    assert (tmp_path/'promote/levels/level_0_passed.zip').exists() and (tmp_path/'promote/levels/level_1_passed.zip').exists()
    assert read_checkpoint(tmp_path/'promote/final.zip')['manager']['status']=='mastered'


def test_nonfinite_failure_retains_completed_checkpoint(assets,tmp_path,monkeypatch):
    cfg,_=copy.deepcopy(assets); original=PPO.train
    def broken(self):
        original(self)
        with torch.no_grad(): next(self.policy.parameters()).fill_(float('nan'))
    monkeypatch.setattr(PPO,'train',broken)
    result=train(cfg,steps=32,output_directory=tmp_path/'nan',device='cpu')
    assert result['status']=='failed_numerical' and result['final_saved_phase_step']==0 and result['segment_consumed_steps']==16
    model=PPO.load(tmp_path/'nan/final.zip'); assert all(torch.isfinite(p).all() for p in model.policy.parameters())


def test_config_rejects_small_samples_bad_budgets_and_source_leakage(assets,tmp_path):
    cfg,_=copy.deepcopy(assets); cfg['validation']['min_episodes']=200
    with pytest.raises(ValueError,match='sample'): validate_config(cfg)
    cfg,_=copy.deepcopy(assets); cfg['curriculum']['level_budget']=63
    with pytest.raises(ValueError,match='align'): validate_config(cfg)
    cfg,_=copy.deepcopy(assets); cfg['maps']['exclude_manifests']=cfg['maps']['exclude_manifests'][1:]
    with pytest.raises(ValueError,match='inherited data'): train(cfg,steps=16,output_directory=tmp_path/'leaky',device='cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_smoke_reload(assets,tmp_path):
    cfg,_=copy.deepcopy(assets)
    result=train(cfg,steps=16,output_directory=tmp_path/'cuda',device='cuda')
    assert result['completed_phase4c_steps']==16
    model=PPO.load(tmp_path/'cuda/final.zip',device='cuda'); assert all(torch.isfinite(p).all() for p in model.policy.parameters())


def test_resume_preserves_promotion_streak_and_does_not_count_baseline(assets,tmp_path,monkeypatch):
    cfg,_=copy.deepcopy(assets); cfg['curriculum']['success_threshold']=.1
    import training.train_curriculum_ppo as module
    original=module.evaluate_model
    def success(model,scenarios):
        summary,rows=original(model,scenarios); summary.update(episodes=4,successes=4,success_rate=1.)
        return summary,rows
    monkeypatch.setattr(module,'evaluate_model',success)
    first=train(cfg,steps=16,output_directory=tmp_path/'first',device='cpu')
    assert first['manager']['streak']==1 and first['current_level']==0
    second=train(cfg,steps=32,output_directory=tmp_path/'second',resume=tmp_path/'first/final.zip',device='cpu')
    assert second['current_level']==1 and second['manager']['streak']==0 and second['manager']['level_steps']==0
    with pytest.raises(ValueError,match='Resume must retain'):
        train(cfg,steps=48,mode='fixed',output_directory=tmp_path/'wrongmode',resume=tmp_path/'second/final.zip',device='cpu')


def test_interruption_fallback_and_recovery_charge_consumed_work(assets,tmp_path,monkeypatch):
    cfg,_=copy.deepcopy(assets); original=PPO.train
    def interrupt(self): raise KeyboardInterrupt()
    monkeypatch.setattr(PPO,'train',interrupt)
    first=train(cfg,steps=32,output_directory=tmp_path/'interrupted',device='cpu')
    assert first['status']=='interrupted' and first['final_saved_phase_step']==0 and first['segment_consumed_steps']==16
    monkeypatch.setattr(PPO,'train',original)
    second=train(cfg,steps=32,output_directory=tmp_path/'recovered',resume=tmp_path/'interrupted/final.zip',device='cpu')
    assert second['completed_phase4c_steps']==16 and second['cumulative_consumed_phase4c_steps']==32


def test_fresh_training_is_reproducible_and_checkpoint_hash_is_enforced(assets,tmp_path):
    cfg,_=copy.deepcopy(assets)
    a=train(cfg,steps=16,output_directory=tmp_path/'a',device='cpu')
    b=train(cfg,steps=16,output_directory=tmp_path/'b',device='cpu')
    assert actor_hash(PPO.load(tmp_path/'a/final.zip').policy)==actor_hash(PPO.load(tmp_path/'b/final.zip').policy)
    assert a['manager']==b['manager']
    path=tmp_path/'a/final.zip'; path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash'): read_checkpoint(path)
