"""Paired evaluation protocol, truthful metrics, loops, leakage and weight immutability."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from environments.fixed_map_env import FixedMapNavigationEnv
from imitation.bc_network import ARCHITECTURE,BCNetwork
from imitation.dataset import load_dataset,save_dataset
from imitation.expert_generator import generate_dataset
from planners.astar import astar,path_to_actions
from evaluation.bc_benchmark import (benchmark,detect_loops,file_sha256,model_sha256,
                                     reserved_layout_ids,run_episode,summarize)


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def assets(tmp_path):
    paths=[]; seen=set()
    for i,split in enumerate(('train','validation','test')):
        path=tmp_path/f'{split}.npz'
        generate_dataset(path,episodes=5,seed=1000000+i*2000000,split=split,excluded_ids=seen)
        a,_=load_dataset(path); seen.update(a['map_ids'].tolist()); paths.append(path)
    torch.manual_seed(11)
    network=BCNetwork(10,10,16)
    environment=dict(height=10,width=10,obstacle_density=.1,max_steps=150,bfs_shaping=False)
    payload=dict(format_version=1,architecture=ARCHITECTURE,
                 network_config=dict(height=10,width=10,hidden_size=16,n_actions=4),
                 model_state_dict=network.state_dict(),environment=environment,epoch=1,
                 channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]],
                 dataset_provenance={name:{'sha256':file_sha256(path)} for name,path in zip(('train','validation'),paths)})
    checkpoint=tmp_path/'fixture.pt'; torch.save(payload,checkpoint)
    return checkpoint,paths


def invoke(assets,output,**kwargs):
    checkpoint,paths=assets
    return benchmark(checkpoint,*paths,episodes=5,device='cpu',output_directory=output,route_count=1,**kwargs)


def test_loops_do_not_flag_straight_movement():
    straight=[[0,i] for i in range(6)]
    assert not detect_loops(straight,[3]*5)['loop_detected']
    stationary=detect_loops([[0,0]]*5,[0]*4)
    assert stationary['loop_detected'] and stationary['loop_kind']=='stationary_action'
    cycle=detect_loops([[0,0],[0,1]]*3+[[0,0]],[3,2]*3)
    assert cycle['loop_detected'] and cycle['loop_period']==2


def test_loop_detection_does_not_stop_or_repair_rollout():
    env=FixedMapNavigationEnv(np.zeros((3,3),dtype=np.int8),(2,2),max_steps=10)
    try:
        row=run_episode(env,(0,0),lambda obs:0,4,0,1,'fixture','bc')
    finally:
        env.close()
    assert row['loop_detected'] and row['loop_first_step']==3
    assert row['steps']==10 and row['collisions']==10 and row['timeout']
    assert row['reward']==-110 and row['spl']==0 and not row['success']
    assert len(row['route'])==11 and len(row['actions'])==10


def test_astar_truthful_success_and_efficiency():
    env=FixedMapNavigationEnv(np.zeros((3,3),dtype=np.int8),(2,2),max_steps=10)
    planned=iter(path_to_actions(astar(env.fixed_obstacles,(0,0),(2,2))))
    try:
        row=run_episode(env,(0,0),lambda obs:next(planned),4,0,1,'fixture','astar')
    finally:
        env.close()
    assert row['success'] and not row['timeout'] and row['steps']==4
    assert row['collisions']==0 and row['spl']==row['successful_path_efficiency']==1
    assert row['reward']==104 and not row['loop_detected']
    stats=summarize([row])
    assert stats['success_rate']==1 and stats['timeout_rate']==0


@pytest.mark.parametrize('density',[.1,.3,.5])
def test_paired_density_benchmark_and_artifacts(assets,tmp_path,density):
    before=file_sha256(assets[0]); output=tmp_path/'benchmark'
    report=invoke(assets,output,density=density)
    assert report['weights_unchanged'] and file_sha256(assets[0])==before
    assert report['model_sha256_before']==report['model_sha256_after']
    assert report['reserved_layout_overlap']==0
    assert report['summary']['astar']['success_rate']==1
    assert report['summary']['astar']['mean_spl']==1
    assert report['distribution_shift']==(density != .1)
    for name in ('episodes.csv','summary.csv','results.json','scenarios.npz','comparison.png'):
        assert (output/name).stat().st_size > 0
    assert len(list((output/'routes').glob('*.png')))==1
    rows=json.loads((output/'results.json').read_text())['results']
    assert len(rows)==15
    for i in range(5):
        paired=[r for r in rows if r['scenario_index']==i]
        assert {r['policy'] for r in paired}=={'bc','astar','random'}
        assert len({(r['map_seed'],r['map_id'],tuple(r['start']),tuple(r['goal'])) for r in paired})==1
        for row in paired:
            assert row['success'] != row['timeout']
            assert len(row['actions'])==row['steps']
            assert row['route'][0]==row['start']
            assert not row['success'] or row['route'][-1]==row['goal']
    with (output/'episodes.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==15
    a,_=load_dataset(output/'scenarios.npz')
    assert all(int(grid.sum())==round(100*density) for grid in a['obstacles'])
    assert set(a['map_ids']).isdisjoint(reserved_layout_ids(assets[1]))


def test_frozen_scenarios_reproduce_all_rollouts_and_no_overwrites(assets,tmp_path):
    first=tmp_path/'first'; second=tmp_path/'second'
    invoke(assets,first)
    invoke(assets,second,scenario_file=first/'scenarios.npz')
    a=json.loads((first/'results.json').read_text()); b=json.loads((second/'results.json').read_text())
    assert a['results']==b['results'] and a['summary']==b['summary']
    with pytest.raises(FileExistsError):
        invoke(assets,first)


def test_rejects_train_maps_and_bad_seed_records(assets,tmp_path):
    checkpoint,paths=assets
    with pytest.raises(ValueError,match='benchmark scenarios'):
        invoke(assets,tmp_path/'bad',scenario_file=paths[0])
    invoke(assets,tmp_path/'good')
    arrays,meta=load_dataset(tmp_path/'good/scenarios.npz')
    arrays['map_seeds'][0]+=1
    changed=tmp_path/'bad_seed.npz'; save_dataset(changed,arrays,meta)
    with pytest.raises(ValueError,match='seed'):
        invoke(assets,tmp_path/'bad2',scenario_file=changed)


def test_rejects_checkpoint_dataset_mismatch(assets,tmp_path):
    checkpoint,paths=assets
    # Use a different valid train file with identical schema but different provenance.
    new=tmp_path/'different.npz'; generate_dataset(new,episodes=5,seed=6000000,split='train')
    with pytest.raises(ValueError,match='provenance'):
        benchmark(checkpoint,new,paths[1],paths[2],episodes=5,device='cpu',output_directory=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_cli_help_contains_density_and_scenario_reuse():
    result=subprocess.run([sys.executable,'-m','evaluation.bc_benchmark','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--density' in result.stdout and '--scenario-file' in result.stdout
