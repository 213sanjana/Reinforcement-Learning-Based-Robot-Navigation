"""PPO comparison against synthetic run artifacts; no historical values invented."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from agents.pretrained_ppo_policy import PretrainedPPOPolicy,EncodedNavigationObservation
from environments.gridworld_env import RobotNavigationEnv
from evaluation.scenario_manifest import generate,save_json,file_hash
from evaluation.evaluate_ppo_variants import evaluate,validate_saved,inspect_checkpoint
from evaluation.compare_ppo_variants import expert_cost,history_comparison,compare,VARIANTS


def write(path,obj):path.write_text(json.dumps(obj))

@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(1)
    data=tmp_path/'train.npz';np.savez(data,map_ids=['one','two'],episode_offsets=[0,3,7])
    bc=tmp_path/'bc.pt';torch.save(dict(dataset_provenance=dict(train=dict(path=str(data),sha256=file_hash(data)))),bc)
    init=tmp_path/'init.zip';init.write_bytes(b'fixture initialization identity')
    write(init.with_suffix('.transfer.json'),dict(source_checkpoint=str(bc),source_checkpoint_sha256=file_hash(bc),
        initialization_sha256=file_hash(init),status='passed',training_started=False))
    cfg=yaml.safe_load(Path('configs/benchmark_protocol.yaml').read_text())
    cfg.update(identity_sources=[],seed_sources=[],episodes_per_density=1)
    env=EncodedNavigationObservation(RobotNavigationEnv(**cfg['environment']))
    for name in VARIANTS:
        run=tmp_path/name;run.mkdir();curr=name=='ppo_curriculum'
        model=PPO(PretrainedPPOPolicy,env,policy_kwargs=dict(hidden_size=8),n_steps=8,batch_size=4,device='cpu')
        for choice,step in [('best',24 if curr else 8),('final',32 if curr else 32)]:
            model.num_timesteps=step;path=run/(choice+'.zip');model.save(path)
            side=dict(checkpoint_sha256=file_hash(path))
            if curr:side.update(model_step=step,phase_step=step-8,start_model_step=8,fingerprint='c')
            else:side.update(checkpoint_step=step,config_fingerprint='b')
            write(path.with_suffix('.curriculum.json' if curr else '.run.json'),side)
        m=dict(seed=42,initialization=str(init),initialization_sha256=file_hash(init),
            verified_initialization=str(init),environment=dict(cfg['environment'],obstacle_density=.1),
            train_dataset_sha256=file_hash(data),validation_signature='same',lineage_cost_before_segment=48,starting_step=16,
            method='scratch' if name=='ppo_scratch' else 'pretrained',config_fingerprint='b')
        if curr:m.update(mode='curriculum',fingerprint='c',scenario_signatures=['a','b','c'],
            source_model_step=8,source_checkpoint=str(tmp_path/'ppo_bc/best.zip'),source_sha256=file_hash(tmp_path/'ppo_bc/best.zip'))
        s=dict(status='completed',cumulative_consumed_env_timesteps=64,actual_env_timesteps=32,
            lineage_cost_before_segment=48,starting_step=16,total_inherited_plus_phase4c_steps=88,
            observed_model_step=32,source_phase4b_consumed_steps=64,cumulative_consumed_phase4c_steps=24)
        write(run/'run_manifest.json',m);write(run/'summary.json',s)
        (run/'config.yaml').write_text(yaml.safe_dump(dict(ppo=dict(n_steps=8,batch_size=4),training=dict(seed=42))))
        if curr:(run/'validation.csv').write_text('phase_step,model_step,evaluated_level,density,success_rate,mean_spl\n16,24,0,0.1,0.5,0.5\n24,32,1,0.3,0.4,0.4\n')
        else:(run/'validation.csv').write_text('global_step,success_rate,mean_spl,kind\n16,0.2,0.2,baseline\n32,0.9,0.9,periodic\n')
        cfg['policies'][name]['checkpoint']=str(run/'best.zip')
    env.close()
    manifest=generate(cfg);path=tmp_path/'scenarios.json';save_json(path,manifest)
    return path,manifest,tmp_path


def test_full_paired_best_final_evaluation(assets):
    path,m,root=assets;before=file_hash(path)
    result=evaluate(path,root/'out')
    assert result==dict(valid=True,variants=3,selections=['best','final'],scenarios_per_checkpoint=3,episodes=18)
    assert file_hash(path)==before
    assert validate_saved(root/'out',m)==result
    report=compare(root/'out')
    for name in VARIANTS:
        assert set(report['policies'][name])=={'best','final'}
        for choice in ('best','final'):
            entry=report['policies'][name][choice]
            assert entry['episodes']==3 and entry['deterministic']
            assert sum(x['episodes'] for x in entry['summary'].values())==3
    assert report['policies']['ppo_bc']['best']['checkpoint']['checkpoint_model_timesteps']==8
    assert report['policies']['ppo_bc']['best']['checkpoint']['completed_search_online_interactions']==64
    assert report['policies']['ppo_curriculum']['final']['checkpoint']['completed_search_online_interactions']==88
    with pytest.raises(FileExistsError):evaluate(path,root/'out')


def test_expert_counts_and_scratch_zero(assets):
    path,m,root=assets
    for name in VARIANTS:
        run=Path(m['config']['policies'][name]['checkpoint']).parent
        rm=json.loads((run/'run_manifest.json').read_text())
        counts=expert_cost(m,rm,name)
        assert counts['trajectories']==(0 if name=='ppo_scratch' else 2)
        assert counts['transitions']==(0 if name=='ppo_scratch' else 7)


def test_sidecar_hash_tamper(assets):
    path,m,root=assets;p=root/'ppo_bc/best.run.json'
    r=json.loads(p.read_text());r['checkpoint_sha256']='bad';write(p,r)
    with pytest.raises(ValueError,match='sidecar hash'):evaluate(path,root/'out')
    assert not (root/'out').exists()


def test_counter_tamper(assets):
    path,m,root=assets;p=root/'ppo_scratch/final.run.json'
    r=json.loads(p.read_text());r['checkpoint_step']=999;write(p,r)
    with pytest.raises(ValueError,match='timestep'):evaluate(path,root/'out')


def test_search_cost_ledger_tamper(assets):
    path,m,root=assets;p=root/'ppo_curriculum/summary.json'
    r=json.loads(p.read_text());r['total_inherited_plus_phase4c_steps']=1;write(p,r)
    with pytest.raises(ValueError,match='cost ledger'):evaluate(path,root/'out')


def test_missing_historical_log_fails(assets):
    path,m,root=assets;(root/'ppo_bc/validation.csv').unlink()
    with pytest.raises(FileNotFoundError):evaluate(path,root/'out')


def test_source_data_tamper(assets):
    path,m,root=assets
    with (root/'train.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='dataset hash'):evaluate(path,root/'out')


def test_csv_corruption_detected(assets):
    path,m,root=assets;evaluate(path,root/'out',selection='final')
    with (root/'out/ppo_bc/final/episodes.csv').open('a') as f:f.write('corrupt')
    with pytest.raises(ValueError,match='checksum'):validate_saved(root/'out',m)


def test_checkpoint_unchanged_and_later_tamper_detected(assets):
    path,m,root=assets;p=root/'ppo_curriculum/final.zip';before=file_hash(p)
    evaluate(path,root/'out',selection='final');assert before==file_hash(p)
    with p.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):validate_saved(root/'out',m)


def test_sample_efficiency_requires_matching_signature():
    def row(name,sig,steps,sr):return dict(policy=name,validation_signature=sig,density=.1,consumed_online_steps=steps,success_rate=sr)
    h=dict(ppo_scratch=[row('ppo_scratch','a',10,.1)],ppo_bc=[row('ppo_bc','a',10,.9)],ppo_curriculum=[row('ppo_curriculum','different',10,.5)])
    result=history_comparison(h)
    assert result['matched_scratch_bc_validation_points'][0]['difference']==.8
    assert list(result['curves']['ppo_bc'].values())[0]['first_logged_success_at_least_threshold']==10
    h['ppo_bc'][0]['validation_signature']='other'
    assert history_comparison(h)['matched_scratch_bc_validation_points']==[]


def test_architecture_mismatch(assets):
    path,m,root=assets
    env=EncodedNavigationObservation(RobotNavigationEnv(**m['config']['environment']))
    model=PPO(PretrainedPPOPolicy,env,policy_kwargs=dict(hidden_size=16),n_steps=8,batch_size=4,device='cpu');model.num_timesteps=24
    p=root/'ppo_curriculum/best.zip';model.save(p)
    side=p.with_suffix('.curriculum.json');s=json.loads(side.read_text());s['checkpoint_sha256']=file_hash(p);write(side,s);env.close()
    with pytest.raises(ValueError,match='architectures differ'):evaluate(path,root/'out')


def test_best_only_output(assets):
    path,m,root=assets
    r=evaluate(path,root/'out',selection='best')
    assert r['episodes']==9 and r['selections']==['best']
    assert not (root/'out/ppo_bc/final').exists()
