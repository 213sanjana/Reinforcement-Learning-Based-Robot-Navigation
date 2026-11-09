"""Evidence integrity, plotted intervals, checked replay and report reproducibility."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from visualization.benchmark_plots import generate,validate_figures,interval,replay
from evaluation.generate_research_report import generate as report_generate
from evaluation.baseline_adapters import PlannerBaseline
from evaluation.evaluate_baselines import rollout
from evaluation.compare_ppo_variants import read_json

@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    root=tmp_path_factory.mktemp('report_evidence')
    spec=importlib.util.spec_from_file_location('statistics_fixtures','tests/test_statistics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.test_real_saved_result_loaders(root)
    generate(root,root/'figures')
    report_generate(root,root/'reports/gridworld_benchmark.md')
    return root


def test_actual_generated_outputs(experiment):
    root=experiment;index=validate_figures(root,root/'figures')
    assert index['route_replays_verified']>0
    for name in ('success_rates.png','collisions.png','path_efficiency_spl.png','successful_path_efficiency.png','historical_sample_efficiency.png'):
        assert (root/'figures'/name).stat().st_size>1000
    assert len(index['policy_series'])==10
    assert len(list((root/'figures').glob('*_routes.png')))==10


def test_trajectory_replays_match_saved(experiment):
    traces=read_json(experiment/'figures/trajectory_replays.json')
    assert all(r['replay_matches_saved'] for r in traces['records'])
    assert any(not r['available'] for r in traces['availability'])
    for r in traces['records']:
        assert len(r['route'])==len(r['actions'])+1
        assert sum(r['blocked'])==r['saved_outcome']['collisions']


def test_report_contains_real_counts_and_limits(experiment):
    text=(experiment/'reports/gridworld_benchmark.md').read_text()
    assert '30 saved episode rows' in text
    assert 'training-seed variability' in text and 'Hypotheses, not established mechanisms' in text
    assert 'Historical curves contain only logged validation observations' in text
    assert 'ppo_bc/best' in text and 'ppo_bc/final' in text
    assert 'N/A' in text and 'Not reached in available logs' in text


def test_report_reproduces(experiment):
    result=report_generate(experiment,experiment/'reports/gridworld_benchmark.md',validate_only=True)
    assert result['reproduced']


def test_no_overwrite(experiment):
    with pytest.raises(FileExistsError):generate(experiment,experiment/'figures')
    with pytest.raises(FileExistsError):report_generate(experiment,experiment/'reports/gridworld_benchmark.md')


def test_plot_numbers_are_from_statistics(experiment):
    plotted=read_json(experiment/'figures/plotted_statistics.json')
    stats=read_json(experiment/'statistics/analysis.json')
    assert plotted['rows']==[r for r in stats['descriptive'] if r['axis']=='density']
    for r in plotted['rows']:
        value,ci=interval(r,'success')
        assert value==r['statistics']['success_rate'] and ci==r['statistics']['success_wilson95']


def test_missing_figures_aborts(experiment,tmp_path):
    with pytest.raises(FileNotFoundError):report_generate(experiment,tmp_path/'report.md',figures=tmp_path/'missing')
    assert not (tmp_path/'report.md').exists()


def test_replay_mismatch_fails():
    s=dict(obstacles=np.zeros((3,3),dtype=int).tolist(),start=[0,0],goal=[0,1],horizon=5,map_seed=1,map_id='m',scenario_id='s',density=.1,optimal_steps=1)
    saved=rollout(PlannerBaseline(),'astar',s,1);saved['collisions']=1
    with pytest.raises(ValueError,match='Replay mismatch'):replay(PlannerBaseline(),'astar',saved,s,1)


def test_figure_tampering_detected(experiment,tmp_path):
    import shutil
    folder=tmp_path/'figures';shutil.copytree(experiment/'figures',folder)
    with (folder/'success_rates.png').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='output changed'):validate_figures(experiment,folder)


def test_report_tampering_detected(experiment,tmp_path):
    import shutil
    path=tmp_path/'report.md';shutil.copyfile(experiment/'reports/gridworld_benchmark.md',path)
    shutil.copyfile(experiment/'reports/gridworld_benchmark.md.provenance.json',path.with_suffix('.md.provenance.json'))
    with path.open('a') as f:f.write('Changed')
    with pytest.raises(ValueError):report_generate(experiment,path,validate_only=True)
