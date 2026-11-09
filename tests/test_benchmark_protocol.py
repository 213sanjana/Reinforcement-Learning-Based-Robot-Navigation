"""Infrastructure tests use tiny synthetic pools; no historical test performance."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from evaluation.scenario_manifest import generate, validate, save_json, map_identity, digest
from evaluation.benchmark_protocol import run_smoke, run_episode
from evaluation.policy_adapters import AStarAdapter
from evaluation.result_schema import wilson


@pytest.fixture
def cfg():
    c = yaml.safe_load(Path('configs/benchmark_protocol.yaml').read_text())
    c.update(episodes_per_density=4, identity_sources=[], seed_sources=[])
    return c


def test_repeated_generation(cfg, tmp_path):
    a = generate(cfg); b = generate(cfg)
    assert a == b
    save_json(tmp_path/'a.json',a); save_json(tmp_path/'b.json',b)
    assert (tmp_path/'a.json').read_bytes() == (tmp_path/'b.json').read_bytes()
    assert validate(a)['scenarios'] == 12
    assert len({r['map_id'] for r in a['scenarios']}) == 12
    assert {r['density'] for r in a['scenarios']} == {.1,.3,.5}


def test_ten_episode_smoke(cfg):
    report = run_smoke(generate(cfg),['astar'])
    assert len(report['results']) == 10
    assert all(r['success'] and r['collisions']==0 and r['spl']==1 for r in report['results'])
    assert [sum(r['density']==d for r in report['results']) for d in [.1,.3,.5]] == [4,3,3]


@pytest.mark.parametrize('field,value', [('map_seed',1),('start',[0,0]),('goal',[0,1]),
    ('obstacle_layout_hash','bad'),('horizon',151),('optimal_steps',999),('scenario_id','bad')])
def test_tampering(cfg,field,value):
    m = generate(cfg); m['scenarios'][0][field] = value
    m['manifest_id'] = digest({k:v for k,v in m.items() if k!='manifest_id'})
    with pytest.raises(ValueError):
        validate(m)


def test_no_overwrite(tmp_path):
    p=tmp_path/'x.json'; save_json(p,{'old':1})
    with pytest.raises(FileExistsError):
        save_json(p,{'new':1})
    assert json.loads(p.read_text()) == {'old':1}


def test_missing_required_source(cfg):
    cfg['identity_sources']=[dict(pattern='nonexistent_pool.npz',role='train')]
    with pytest.raises(FileNotFoundError):
        generate(cfg)


def test_existing_split_leakage(cfg,tmp_path):
    grid=np.zeros((10,10),dtype=np.int8); grid[1,2]=1
    for name,g in [('train',grid),('validation',np.rot90(grid))]:
        np.savez(tmp_path/f'{name}.npz',obstacles=[g],map_ids=[map_identity(g)])
    cfg['identity_sources']=[dict(pattern=str(tmp_path/f'{name}.npz'),role=name) for name in ['train','validation']]
    with pytest.raises(ValueError,match='cross-split'):
        generate(cfg)


def test_excluded_layout_not_reused(cfg,tmp_path):
    m=generate(cfg); r=m['scenarios'][0]
    p=tmp_path/'pool.npz'; np.savez(p,obstacles=[r['obstacles']],map_ids=[r['map_id']])
    cfg['identity_sources']=[dict(pattern=str(p),role='train')]
    assert r['map_id'] not in {x['map_id'] for x in generate(cfg)['scenarios']}


def test_source_changed(cfg,tmp_path):
    p=tmp_path/'pool.npz'; grid=np.zeros((10,10),dtype=np.int8)
    np.savez(p,obstacles=[grid],map_ids=[map_identity(grid)])
    cfg['identity_sources']=[dict(pattern=str(p),role='train')]; m=generate(cfg)
    grid[1,2]=1; np.savez(p,obstacles=[grid],map_ids=[map_identity(grid)])
    with pytest.raises(ValueError):
        validate(m)


def test_collision_timeout_uses_common_reward():
    class Up:
        def reset(self,s): pass
        def act(self,o): return 0
    row=dict(obstacles=np.zeros((10,10),dtype=int).tolist(),start=[0,0],goal=[0,1],
             optimal_steps=1,horizon=5,map_seed=1,map_id='x',scenario_id='s',density=0.)
    r=run_episode(Up(),'up',row)
    assert r['timeout'] and r['episode_length']==5 and r['collisions']==5
    assert r['reward']==-55 and r['spl']==0 and r['path_efficiency'] is None


def test_goal_at_horizon_is_success():
    row=dict(obstacles=np.zeros((10,10),dtype=int).tolist(),start=[0,0],goal=[0,1],
             optimal_steps=1,horizon=1,map_seed=1,map_id='x',scenario_id='s',density=0.)
    r=run_episode(AStarAdapter(),'astar',row)
    assert r['success'] and not r['timeout'] and r['reward']==101


def test_wilson():
    lo,hi=wilson(0,10); assert lo==0 and 0.27 < hi < .28
    lo,hi=wilson(10,10); assert .72 < lo < .73 and hi==pytest.approx(1)


def test_bad_contract(cfg):
    cfg['contract']['actions'].reverse()
    with pytest.raises(ValueError): generate(cfg)


def test_neural_adapters(tmp_path,cfg):
    torch=pytest.importorskip('torch')
    from agents.dqn_agent import DQNAgent,DQNConfig
    from imitation.bc_network import BCNetwork
    from evaluation.policy_adapters import NeuralAdapter
    torch.set_num_threads(1)
    obs=dict(obstacles=np.zeros((10,10),dtype=np.int8),agent=np.array([0,0]),goal=np.array([1,1]))
    agent=DQNAgent(10,10,DQNConfig(architecture='cnn'),device='cpu')
    path=tmp_path/'dqn.pt'; agent.save(path)
    adapter=NeuralAdapter('dqn',path,cfg['environment'])
    assert adapter.act(obs)==agent.act(obs,explore=False)
    assert len({adapter.act(obs) for _ in range(3)})==1; adapter.verify()
    model=BCNetwork()
    path=tmp_path/'bc.pt'
    torch.save(dict(format_version=1,architecture='cnn_bc_v1',network_config={},
        model_state_dict=model.state_dict(),channels=['obstacles','agent','goal'],
        actions=[[-1,0],[1,0],[0,-1],[0,1]]),path)
    adapter=NeuralAdapter('bc',path,cfg['environment'])
    before=adapter.fingerprint(); adapter.act(obs); adapter.verify()
    assert adapter.fingerprint()==before and not adapter.module.training


def test_ppo_adapter(tmp_path,cfg):
    torch=pytest.importorskip('torch')
    sb3=pytest.importorskip('stable_baselines3')
    from agents.pretrained_ppo_policy import PretrainedPPOPolicy, EncodedNavigationObservation
    from environments.gridworld_env import RobotNavigationEnv
    from evaluation.policy_adapters import NeuralAdapter
    torch.set_num_threads(1)
    env=EncodedNavigationObservation(RobotNavigationEnv(**cfg['environment']))
    model=sb3.PPO(PretrainedPPOPolicy,env,n_steps=8,batch_size=4,device='cpu')
    path=tmp_path/'ppo.zip'; model.save(path)
    adapter=NeuralAdapter('ppo',path,cfg['environment'])
    obs,_=env.unwrapped.reset(seed=2)
    from imitation.dataset import encode_observation
    assert adapter.act(obs)==int(model.predict(encode_observation(obs),deterministic=True)[0])
    adapter.verify(); env.close()


def test_invalid_action_rejected(cfg):
    class Invalid:
        def reset(self,s): pass
        def act(self,o): return 4
    with pytest.raises(ValueError,match='Invalid policy action'):
        run_episode(Invalid(),'invalid',generate(cfg)['scenarios'][0])


def test_policy_mutation_detected(cfg,monkeypatch):
    import evaluation.benchmark_protocol as module
    class Mutating(AStarAdapter):
        count=0
        def act(self,o):
            self.count+=1
            return super().act(o)
        def fingerprint(self): return str(self.count)
    monkeypatch.setattr(module,'load_policy',lambda *a:Mutating())
    with pytest.raises(RuntimeError,match='Policy changed'):
        module.run_smoke(generate(cfg),['astar'])


def test_same_scenarios_for_every_policy(cfg):
    cfg['policies']['astar_copy']={'kind':'astar'}
    report=run_smoke(generate(cfg),['astar','astar_copy'])
    assert [r['scenario_id'] for r in report['results'][:10]] == [r['scenario_id'] for r in report['results'][10:]]


def test_unsolvable_or_invalid_grid_rejected(cfg):
    m=generate(cfg)
    m['scenarios'][0]['obstacles']=[[1]*10 for _ in range(10)]
    m['manifest_id']=digest({k:v for k,v in m.items() if k!='manifest_id'})
    with pytest.raises(ValueError): validate(m)
