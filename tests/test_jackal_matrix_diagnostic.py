"""Scripted Phase 6H.2 extension tests; no live ROS or Gazebo processes."""
import json
import math
from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import PPO
from navigation import jackal_generalization_diagnostic as d
from navigation import jackal_generalization_report as report
from test_jackal_generalization_diagnostic import Env, models

class MatrixEnv(Env):
    def __init__(self,fail_at=None):
        super().__init__();self.resets=0;self.calls=0;self.fail_at=fail_at
    def reset(self,*,seed=None,options=None):
        self.resets+=1
        obs,info=super().reset(seed=seed,options=options)
        self.robot.distance=math.dist(options['start'][:2],options['goal'])
        info['robot_world_pose']=[*options['start'][:2],.3,options['start'][2]]
        if self.resets==self.fail_at:info['robot_world_pose'][3]+=.2
        return self.robot.get_observation(),info
    def step(self,action):
        self.calls+=1
        return super().step(action)


def run(tmp_path,monkeypatch,models,fail_at=None):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(PPO,'learn',lambda *a,**k:pytest.fail('Training forbidden'))
    env=MatrixEnv(fail_at)
    result=d.run_diagnostic(tmp_path/'matrix',environment_factory=lambda:env,models=models,readiness=lambda:None,mode='matrix')
    return result,env,tmp_path/'matrix'


def test_plan_matches_preparation():
    prepared=json.loads(Path('artifacts/ppo/jackal_6h/diagnostics/prep_20261026_172838UTC_6h2_matrix/preparation_manifest.json').read_text())
    expected=[{k:v for k,v in e.items() if k!='index'} for e in prepared['episode_order']]
    assert d.episode_plan('matrix')==expected
    assert len({e['episode_id'] for e in expected})==52
    assert len(d.episode_plan('pilot'))==4
    assert d.BUDGETS=={'pilot':600,'matrix':6000}
    assert d.OUTER_TIMEOUTS=={'pilot':660,'matrix':6060}


def test_all_52_evaluation_only_and_report(tmp_path,monkeypatch,models):
    result,env,p=run(tmp_path,monkeypatch,models)
    assert result['status']=='completed',result.get('failure')
    assert env.resets==52 and result['planned_episodes']==52
    assert [e['episode_id'] for e in result['episodes']]==[e['episode_id'] for e in d.episode_plan('matrix')]
    assert result['not_started_episode_ids']==[] and result['hooks_restored']
    assert env.closed and result['training_steps']==0 and result['model_state_unchanged']
    assert len((p/'episodes.jsonl').read_text().splitlines())==52
    # Scripted shutdown ledger, not a claim of live process cleanup.
    support=p.with_name(p.name+'_supervisor');support.mkdir()
    (support/'cleanup.json').write_text(json.dumps(dict(remaining_owned_pids=[],full_matrix_launched=True)))
    summary=json.loads((p/'summary.json').read_text());summary['command_topic_counts_after_close']=[0,1]
    (p/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(report,'plot_episode',lambda *args:None)
    assert report.report(p,tmp_path)
    analysis=json.loads((p/'analysis.json').read_text())
    assert len(analysis['episodes'])==52 and len(analysis['paired_comparisons'])==26
    assert all(e['valid'] for e in analysis['episodes'])
    assert len(analysis['aggregates'])==28
    assert 'first control step failed' not in (p/'report.md').read_text()
    with pytest.raises(FileExistsError):report.report(p,tmp_path)
    with pytest.raises(FileExistsError):d.run_diagnostic(p,mode='matrix')


def test_first_reset_failure_preserved_no_retry(tmp_path,monkeypatch,models):
    result,env,p=run(tmp_path,monkeypatch,models,fail_at=6)
    assert result['status']=='failed' and env.resets==6
    assert len(result['episodes'])==6 and len(result['not_started_episode_ids'])==46
    assert result['episodes'][-1]['evaluator']['completed'] is False
    assert result['episodes'][-1]['reset'] is None  # Never reuse previous successful reset.
    assert result['hooks_restored'] and result['stop_verified'] and env.closed
    assert (p/'failures.jsonl').exists()
    monkeypatch.setattr(report,'plot_episode',lambda *a:None)
    assert not report.report(p,tmp_path)
    audit=json.loads((p/'analysis.json').read_text())
    assert sum(e['status']=='not_started' for e in audit['episodes'])==46
    assert audit['episodes'][5]['status']=='invalid'


@pytest.mark.parametrize('failure',['missing_subscriber','stale','nonfinite'])
def test_gate_failure_zero_calls(tmp_path,monkeypatch,models,failure):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path);env=MatrixEnv()
    if failure=='missing_subscriber':env.robot.count_subscribers=lambda topic:0
    if failure=='stale':env.robot.stale=True
    if failure=='nonfinite':env.robot.get_observation=lambda:np.full(40,np.nan,dtype=np.float32)
    result=d.run_diagnostic(tmp_path/'failed',environment_factory=lambda:env,models=models,readiness=lambda:None,mode='matrix')
    assert result['status']=='failed' and env.resets==1 and env.calls==0
    assert result['hooks_restored'] and env.closed and result['stop_verified']


def test_scripted_deadline_aborts_and_cleans(tmp_path,monkeypatch,models):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path);env=MatrixEnv();clock=[100.]
    monkeypatch.setattr(d.time,'monotonic',lambda:clock[0])
    original=env.reset
    def delayed(**kw):
        result=original(**kw);clock[0]+=6001;return result
    env.reset=delayed
    result=d.run_diagnostic(tmp_path/'deadline',environment_factory=lambda:env,models=models,readiness=lambda:None,mode='matrix')
    assert result['status']=='failed' and env.resets==1 and env.calls==0
    assert 'budget' in result['episodes'][0]['evaluator']['error']
    assert result['hooks_restored'] and env.closed


def test_supervisor_matrix_mode_deadline_and_owned_cleanup(tmp_path,monkeypatch):
    from navigation import jackal_diagnostic_supervisor as s
    monkeypatch.setattr(s,'OUTPUT_ROOT',tmp_path);monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s.os,'environ',s.SETTINGS.copy())
    monkeypatch.setattr(s.sys,'argv',['supervisor','--mode','matrix','--output',str(tmp_path/'run')])
    states=iter([{}, {990001:(1,'ros2 launch'),990002:(990001,'bridge'),990003:(1,'unrelated')}])
    monkeypatch.setattr(s,'processes',lambda:next(states));monkeypatch.setattr(s.time,'sleep',lambda _:None)
    waits=[];launched=[];killed=[]
    class Process:
        def __init__(self,pid):self.pid=pid;self.done=False
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):
            waits.append(timeout)
            if self.pid==990004 and not self.done:
                self.done=True;raise s.subprocess.TimeoutExpired('scripted',timeout)
            self.done=True;return 0
    children=iter([Process(990001),Process(990004)])
    def popen(cmd,**kw):launched.append(cmd);return next(children)
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    monkeypatch.setattr(s.os,'kill',lambda pid,sig:killed.append(pid))
    monkeypatch.setattr(s.os,'killpg',lambda pid,sig:killed.append(pid))
    assert s.main()==1
    assert len(launched)==2 and 'matrix' in launched[1] and 6060 in waits
    assert 990003 not in killed and 990002 in killed
    cleanup=json.loads((tmp_path/'run_supervisor/cleanup.json').read_text())
    assert cleanup['full_matrix_launched'] and cleanup['worker_budget_s']==6000
    assert 'no restart' in cleanup['failure']


def test_live_path_alarm_configuration_without_live_calls(tmp_path,monkeypatch,models):
    """Exercise production alarm branch with every external boundary replaced."""
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(d.os,'environ',d.SETTINGS.copy())
    monkeypatch.setattr(d,'_make_jackal_env',lambda base:MatrixEnv(fail_at=1))
    monkeypatch.setattr(d.signal,'signal',lambda *args:'previous-handler')
    alarms=[]
    monkeypatch.setattr(d.signal,'setitimer',lambda clock,seconds:alarms.append(seconds))
    from navigation import jackal_ppo_train
    monkeypatch.setattr(jackal_ppo_train,'_topic_counts',lambda topic:(0,1))
    result=d.run_diagnostic(tmp_path/'alarm',models=models,readiness=lambda:None,mode='matrix')
    assert result['status']=='failed' and alarms==[6000,0]
    assert result['hooks_restored'] and result['shutdown']['closed']


def test_checkpoint_failure_before_any_environment(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    def bad():raise ValueError('Checkpoint identity mismatch')
    monkeypatch.setattr(d,'verify_checkpoints',bad)
    result=d.run_diagnostic(tmp_path/'bad_hash',environment_factory=lambda:pytest.fail('No environment'),mode='matrix')
    assert result['status']=='failed' and result['episodes']==[]
    assert len(result['not_started_episode_ids'])==52


def test_report_detects_missing_dispatch(tmp_path,monkeypatch,models):
    result,env,p=run(tmp_path,monkeypatch,models)
    f=p/'action_intents.jsonl'
    lines=f.read_text().splitlines();f.write_text('\n'.join(lines[:-1])+'\n')
    monkeypatch.setattr(report,'plot_episode',lambda *args:None)
    assert not report.report(p,tmp_path)
    analysis=json.loads((p/'analysis.json').read_text())
    assert analysis['episodes'][-1]['status']=='invalid'
    assert 'intent lifecycle incomplete' in analysis['episodes'][-1]['errors']


def test_matrix_preview_never_executes(monkeypatch,capsys):
    monkeypatch.setattr(d,'run_diagnostic',lambda *a,**k:pytest.fail('No execution'))
    assert d.main(['preview-matrix'])==0
    manifest=json.loads(capsys.readouterr().out)
    assert len(manifest['episode_order'])==52 and manifest['wall_time_cap_s']==6000
    assert manifest['outer_timeout_s']==6060 and manifest['max_steps']==15600


def test_collision_aborts_without_next_reset(tmp_path,monkeypatch,models):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path);env=MatrixEnv()
    original=env.step
    def collided(action):
        obs,reward,terminated,truncated,info=original(action)
        info.update(collision_detected=True,goal_reached=False,termination_reason='collision')
        return obs,reward,True,False,info
    env.step=collided
    result=d.run_diagnostic(tmp_path/'collision',environment_factory=lambda:env,models=models,readiness=lambda:None,mode='matrix')
    assert result['status']=='failed' and env.resets==1
    assert result['episodes'][0]['evaluator']['collision']
    assert result['stop_verified'] and result['hooks_restored'] and env.closed
