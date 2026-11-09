"""Statistical invariants and integration with actual Phase 5B/C saved-result loaders."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from evaluation.scenario_manifest import generate,save_json
from evaluation.baseline_adapters import PlannerBaseline
from evaluation.evaluate_baselines import rollout
from evaluation.statistical_analysis import mean_interval,paired,holm,exact_mcnemar,describe,analyze,run,validate_analysis
from evaluation.reliability_analysis import scenario_features,strata

@pytest.fixture
def samples():
    cfg=yaml.safe_load(Path('configs/benchmark_protocol.yaml').read_text())
    cfg.update(identity_sources=[],seed_sources=[],episodes_per_density=2)
    manifest=generate(cfg)
    rows=[rollout(PlannerBaseline(),'astar',s,1) for s in manifest['scenarios']]
    return manifest,dict(astar=rows,copy=copy.deepcopy(rows))

def test_bootstrap_reproducible():
    a=mean_interval([1,2,3,4],1000,1,'key')
    assert a==mean_interval([1,2,3,4],1000,1,'key') and a['mean']==2.5
    assert a['bootstrap95'][0]<=2.5<=a['bootstrap95'][1]

def test_constant_and_empty():
    a=mean_interval([0]*10,100,1,'x');assert a['bootstrap95']==[0,0] and a['degenerate']
    a=mean_interval([],100,1,'x');assert a['n']==0 and a['mean'] is None and a['bootstrap95'] is None

def test_nonfinite_rejected():
    with pytest.raises(ValueError):mean_interval([1,float('nan')],100,1,'x')

def test_wilson_not_zero_width_at_all_success(samples):
    s=describe(samples[1]['astar'],100,1,'x')
    assert s['success_rate']==1 and s['success_wilson95'][0]<1
    assert s['success_wilson95'][1]==pytest.approx(1)

def test_failure_efficiency_structural_missing(samples):
    rows=copy.deepcopy(samples[1]['astar'])
    for r in rows:r.update(success=False,timeout=True,path_efficiency=None,spl=0)
    s=describe(rows,100,1,'x')
    assert s['n']==6 and s['successes']==0 and s['failures']==6
    assert s['successful_efficiency']['n']==0 and s['successful_efficiency']['mean'] is None
    assert s['missing_efficiency_due_to_failure']==6 and s['metrics']['spl']['mean']==0

def test_identical_pairs(samples):
    a=samples[1]['astar'];r=paired(a,a,100,1,'x')
    assert r['n_pairs']==6 and r['both_success']==6 and r['discordant_success_pairs']==0
    assert r['exact_mcnemar_p']==1 and all(m['bootstrap95']==[0,0] for m in r['metrics'].values())

def test_unaligned_pairs(samples):
    a=samples[1]['astar']
    with pytest.raises(ValueError):paired(a,list(reversed(a)),100,1,'x')

def test_exact_mcnemar():
    assert exact_mcnemar(0,0)==1
    assert exact_mcnemar(5,0)==pytest.approx(.0625)
    assert exact_mcnemar(10,0)==pytest.approx(.001953125)
    assert exact_mcnemar(1,1)==1

def test_holm():
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])

def test_joint_success_subset(samples):
    a=copy.deepcopy(samples[1]['astar']);b=copy.deepcopy(a)
    b[0].update(success=False,timeout=True,path_efficiency=None,spl=0)
    r=paired(a,b,100,1,'x')
    assert r['n_pairs']==6 and r['jointly_successful_efficiency_difference']['n']==5 and r['a_only_success']==1

def test_observation_only_features():
    s=dict(scenario_id='s',density=.1,obstacles=np.zeros((3,3),dtype=int).tolist(),start=[0,0],goal=[2,2],optimal_steps=4)
    f=scenario_features(s)
    assert f['manhattan_distance']==4 and f['detour_ratio']==1
    assert f['reachable_free_cells']==9 and f['reachable_dead_end_fraction']==0

def test_empty_strata_retained(samples):
    features,groups=strata(samples[0]['scenarios'])
    assert len(features)==6 and len(groups)==33
    assert sum(len(g['scenario_ids']) for g in groups if g['axis']=='density')==6
    assert any(not g['scenario_ids'] for g in groups)

def test_reproducible_analysis_and_multiplicity(samples):
    m,d=samples;a=analyze(m,d,{}, {},100,1)
    assert a==analyze(m,d,{}, {},100,1) and not a['reliability']['training_seed_variability_estimated']
    primary=[r for r in a['paired_comparisons'] if r['role']=='primary_density_success']
    assert len(primary)==3 and all(r['statistics']['multiplicity_family_size']==3 for r in primary)
    assert all(not r['statistics']['holm_reject_at_0_05'] for r in primary)

def test_missing_scenarios(samples):
    m,d=samples;d['copy'].pop()
    with pytest.raises(ValueError):analyze(m,d,{}, {},100,1)

def test_bad_settings(samples):
    m,d=samples
    with pytest.raises(ValueError):analyze(m,d,{}, {},99,1)

def test_saved_analysis_recomputation(samples,tmp_path,monkeypatch):
    import evaluation.statistical_analysis as mod
    m,d=samples;monkeypatch.setattr(mod,'load_sources',lambda _: (m,d,{},{}))
    out=tmp_path/'stats';run('fixture',out,100,1)
    assert validate_analysis('fixture',out)['reproduced']
    with pytest.raises(FileExistsError):run('fixture',out,100,1)
    with (out/'summary.csv').open('a') as f:f.write('changed')
    with pytest.raises(ValueError,match='checksum'):validate_analysis('fixture',out)

def test_real_saved_result_loaders(tmp_path):
    import importlib.util
    import torch
    from evaluation.evaluate_baselines import evaluate as eval_base
    from evaluation.evaluate_ppo_variants import evaluate as eval_ppo
    from evaluation.statistical_analysis import load_sources
    from agents.dqn_agent import DQNAgent,DQNConfig
    from imitation.bc_network import BCNetwork
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('ppo_fixtures','tests/test_ppo_comparison.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    path,m,root=fixture.assets.__wrapped__(tmp_path)
    dqn=root/'dqn.pt';bc=root/'baseline_bc.pt'
    DQNAgent(10,10,DQNConfig(architecture='cnn'),device='cpu').save(dqn)
    model=BCNetwork()
    torch.save(dict(format_version=1,architecture='cnn_bc_v1',network_config={},epoch=0,
        channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]],model_state_dict=model.state_dict()),bc)
    cfg=m['config'];cfg['policies']['dqn']['checkpoint']=str(dqn);cfg['policies']['bc']['checkpoint']=str(bc)
    m=generate(cfg);path.unlink();save_json(path,m)
    eval_base(m,root/'baselines');eval_ppo(path,root/'ppo_variants')
    manifest,data,provenance,sources=load_sources(root)
    assert len(data)==10 and all(len(rows)==3 for rows in data.values())
    out=root/'statistics';run(root,out,100,1)
    assert validate_analysis(root,out)['reproduced']
    report=json.loads((out/'analysis.json').read_text())
    assert sum(r['role']=='primary_density_success' for r in report['paired_comparisons'])==135
