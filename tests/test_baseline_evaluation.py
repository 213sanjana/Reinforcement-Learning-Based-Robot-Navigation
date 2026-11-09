"""Synthetic tasks and temporary checkpoints; no trained results invented."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from evaluation.scenario_manifest import generate, file_hash
from evaluation.baseline_adapters import RandomBaseline, PlannerBaseline
from evaluation.evaluate_baselines import rollout, evaluate, validate_results

@pytest.fixture
def manifest():
    cfg=yaml.safe_load(Path('configs/benchmark_protocol.yaml').read_text())
    cfg.update(identity_sources=[],seed_sources=[],episodes_per_density=4)
    return generate(cfg)

def test_random_seed_reproducibility():
    a=RandomBaseline();b=RandomBaseline()
    a.reset(np.random.SeedSequence([1,2]));b.reset(np.random.SeedSequence([1,2]))
    assert [a.act({}) for _ in range(100)]==[b.act({}) for _ in range(100)]

def test_planner_only_uses_current_observation(manifest):
    for s in manifest['scenarios']:
        row=rollout(PlannerBaseline(),'astar',s,2)
        assert row['success'] and row['path_length']==s['optimal_steps'] and row['collisions']==0
        assert row['inference_seconds']>=0 and row['environment_step_seconds']>=0

def test_collision_repeats_and_reward():
    class Up:
        def reset(self,seed): assert isinstance(seed,np.random.SeedSequence)
        def act(self,obs):
            assert set(obs)=={'obstacles','agent','goal'}
            return 0
    s=dict(obstacles=np.zeros((10,10),dtype=int).tolist(),start=[0,0],goal=[0,1],horizon=5,map_seed=1,map_id='m',scenario_id='s',density=.1,optimal_steps=1)
    r=rollout(Up(),'up',s,1)
    assert r['collisions']==r['invalid_actions']==5
    assert r['repeated_invalid_actions']==4 and r['max_invalid_streak']==5
    assert r['path_length']==0 and r['episode_length']==5 and r['reward']==-55
    assert r['timeout'] and r['path_efficiency'] is None

def test_policy_cannot_mutate_environment(manifest):
    class Mutate(PlannerBaseline):
        def act(self,obs):
            action=super().act(obs);obs['obstacles'][:]=1;return action
    assert rollout(Mutate(),'astar',manifest['scenarios'][0],1)['success']

@pytest.fixture
def checkpoints(tmp_path,manifest):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from agents.dqn_agent import DQNAgent,DQNConfig
    from imitation.bc_network import BCNetwork
    dqn=tmp_path/'best_dqn.pt';bc=tmp_path/'best_bc.pt'
    DQNAgent(10,10,DQNConfig(architecture='cnn'),device='cpu').save(dqn)
    model=BCNetwork()
    torch.save(dict(format_version=1,architecture='cnn_bc_v1',network_config={},epoch=2,channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]],model_state_dict=model.state_dict()),bc)
    manifest['config']['policies']['dqn']['checkpoint']=str(dqn)
    manifest['config']['policies']['bc']['checkpoint']=str(bc)
    return generate(manifest['config'])

def test_complete_run_csv_summary_pairing_and_no_overwrite(checkpoints,tmp_path):
    paths=[checkpoints['config']['policies'][n]['checkpoint'] for n in ('dqn','bc')]
    before=[file_hash(p) for p in paths];out=tmp_path/'results'
    status=evaluate(checkpoints,out)
    assert status['valid'] and status['episodes']==48
    assert validate_results(checkpoints,out)==status
    report=json.loads((out/'summary.json').read_text())
    assert report['summary']['astar']['0.1']['success_rate']==1
    assert report['provenance']['bc']['checkpoint']['checkpoint_metadata']['epoch']==2
    with (out/'episodes.csv').open() as f: rows=list(csv.DictReader(f))
    groups={n:[r['scenario_id'] for r in rows if r['policy']==n] for n in ('random','astar','dqn','bc')}
    assert all(v==groups['random'] for v in groups.values())
    assert [file_hash(p) for p in paths]==before
    with pytest.raises(FileExistsError):evaluate(checkpoints,out)

def test_result_corruption_detected(checkpoints,tmp_path):
    out=tmp_path/'results';evaluate(checkpoints,out)
    with (out/'episodes.csv').open('a') as f:f.write('corrupt\n')
    with pytest.raises(ValueError,match='checksum'):validate_results(checkpoints,out)

def test_summary_corruption_detected(checkpoints,tmp_path):
    out=tmp_path/'results';evaluate(checkpoints,out)
    p=out/'summary.json';r=json.loads(p.read_text());r['summary']['astar']['0.1']['success_rate']=0;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Summary'):validate_results(checkpoints,out)

def test_missing_checkpoint_fails_without_success_report(manifest,tmp_path):
    manifest['config']['policies']['dqn']['checkpoint']=str(tmp_path/'absent.pt')
    m=generate(manifest['config']);out=tmp_path/'results'
    with pytest.raises(FileNotFoundError):evaluate(m,out)
    assert not out.exists()

def test_invalid_action_rejected(manifest):
    class Invalid:
        def reset(self,seed):pass
        def act(self,obs):return 4
    with pytest.raises(ValueError,match='out-of-range'):rollout(Invalid(),'bad',manifest['scenarios'][0],1)

def test_mutation_rejected(checkpoints,tmp_path,monkeypatch):
    import evaluation.evaluate_baselines as module
    class Mutate(PlannerBaseline):
        count=0
        def act(self,o):self.count+=1;return super().act(o)
        def fingerprint(self):return str(self.count)
    monkeypatch.setattr(module,'load_baseline',lambda *args:(Mutate(),{}))
    with pytest.raises(RuntimeError,match='state changed'):evaluate(checkpoints,tmp_path/'bad')

def test_pairing_corruption_detected_even_with_updated_checksum(checkpoints,tmp_path):
    out=tmp_path/'results';evaluate(checkpoints,out)
    path=out/'episodes.csv'
    with path.open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows[0]['scenario_id']='wrong'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    p=out/'summary.json';r=json.loads(p.read_text());r['episodes_csv_sha256']=file_hash(path);p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Unpaired'):validate_results(checkpoints,out)

def test_timing_and_astar_invalid_results_rejected(manifest):
    from evaluation.evaluate_baselines import check_row
    row=rollout(PlannerBaseline(),'astar',manifest['scenarios'][0],1)
    bad=dict(row,inference_seconds=-1.)
    with pytest.raises(ValueError,match='timing'):check_row(bad)
    bad=dict(row,mean_inference_ms=row['mean_inference_ms']+1.)
    with pytest.raises(ValueError,match='Timing'):check_row(bad)

def test_checkpoint_modified_after_evaluation_rejected(checkpoints,tmp_path):
    out=tmp_path/'results';evaluate(checkpoints,out)
    path=Path(checkpoints['config']['policies']['dqn']['checkpoint'])
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='Checkpoint provenance'):validate_results(checkpoints,out)
