"""Deterministic dispatch-deadline tests. All control/publication is scripted."""
import copy
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest
from stable_baselines3 import PPO
from navigation import jackal_generalization_diagnostic as d
from test_jackal_generalization_diagnostic import Env, wrapper, models


class CountingEnv(Env):
    def __init__(self):
        super().__init__();self.control_calls=0;self.dispatched_actions=[]
    def step(self,action):
        self.control_calls+=1;self.dispatched_actions.append(np.array(action).copy())
        return super().step(action)


@pytest.fixture
def rig(tmp_path,monkeypatch,models,request):
    clock=[time.monotonic_ns()]
    monkeypatch.setattr(d.time,'monotonic_ns',lambda:clock[0])
    env=CountingEnv();wrapped=wrapper(tmp_path,env)
    observation,_=wrapped.reset(options=dict(start=(0,-1.8,0),goal=(.75,-1.8)))
    def advance(seconds):clock[0]+=round(seconds*1e9)
    value=SimpleNamespace(env=env,w=wrapped,observation=observation,advance=advance,clock=clock,
        model=models['initial'],path=tmp_path,monkeypatch=monkeypatch)
    yield value
    wrapped.close()
    assert env.closed
    assert env.robot.get_observation==wrapped.original_observe
    assert env.robot.expert_publish_callback==wrapped.original_callback
    evidence_root=os.environ.get('PHASE6H1_FRESHNESS_EVIDENCE_DIR')
    if evidence_root:
        directory=Path(evidence_root)/re.sub(r'[^a-zA-Z0-9_.-]','_',request.node.name)
        directory.mkdir(parents=True,exist_ok=False)
        for path in tmp_path.glob('*.jsonl'):
            shutil.copyfile(path,directory/path.name)
        (directory/'case.json').write_text(json.dumps(dict(scripted_environment_only=True,
            raw_control_calls=env.control_calls,closed=env.closed,hooks_restored=True,
            pending_cleared=wrapped.pending is None,rejected=wrapped.rejected,
            source_observation_sha256=d.observation_identity(observation),training_steps=0),indent=2))


def rows(rig,name='action_intents.jsonl'):
    file=rig.path/name
    return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []


def delay(rig,which,seconds):
    original=getattr(d,which)
    def slow(*args,**kwargs):
        result=original(*args,**kwargs);rig.advance(seconds);return result
    rig.monkeypatch.setattr(d,which,slow)


def rejected(rig):
    intents=rows(rig)
    assert intents[-1]['status']=='rejected'
    assert intents[-1]['raw_control_call_occurred'] is False
    assert rig.env.control_calls==0 and not rig.env.dispatched_actions
    assert all(i['status']!='raw_control_called' for i in intents)
    assert rig.w.pending is None and rig.w.rejected and rig.env.robot.stops>0
    assert all(c['action_dispatch_state']!='raw_control_call' for c in rows(rig,'published_commands.jsonl'))
    assert all(c['published_velocity']==dict(linear_m_s=0.,angular_rad_s=0.) for c in rows(rig,'published_commands.jsonl'))
    with pytest.raises(RuntimeError,match='cannot be reused'):rig.w.step(np.zeros(2,dtype=np.float32))
    with pytest.raises(RuntimeError,match='cannot be reused'):rig.w.predict(rig.model,rig.observation)
    return intents[-1]


def test_fresh_through_dispatch_once_and_unchanged_inference(rig):
    original_obs=rig.observation.copy()
    before={k:v.clone() for k,v in rig.model.policy.state_dict().items()}
    expected,expected_details=d.inference(rig.model,rig.observation)
    delay(rig,'inference',.1);delay(rig,'publisher_check',.1)
    action=rig.w.predict(rig.model,rig.observation)
    post,*_=rig.w.step(action)
    assert rig.env.control_calls==1 and np.array_equal(action,expected)
    assert np.array_equal(rig.env.dispatched_actions[0],action)
    assert np.array_equal(rig.observation,original_obs)
    assert rig.w.rows[0]['normalizer_output']==expected_details['normalizer_output']
    assert all(np.array_equal(v.cpu(),rig.model.policy.state_dict()[k].cpu()) for k,v in before.items())
    checks=rows(rig)[-1]['freshness_checks']
    assert checks[-1]['stage']=='immediately_before_dispatch'
    assert checks[-1]['current_sensor_ages_s']==pytest.approx([.2,.2])
    assert rows(rig)[-1]['raw_control_call_occurred'] is True
    assert rig.w.rows[0]['post_action_observation']==post.tolist()


@pytest.mark.parametrize('which',['inference','publisher_check'])
def test_expires_during_work_with_zero_calls(rig,which):
    delay(rig,which,.6)
    with pytest.raises(RuntimeError,match='Stale'):
        action=rig.w.predict(rig.model,rig.observation);rig.w.step(action)
    intent=rejected(rig)
    assert intent['freshness_checks'][-1]['current_sensor_ages_s']==pytest.approx([.6,.6])
    assert intent['pre']['scan_stamp_ns']==10**9 and intent['pre']['odom_stamp_ns']==10**9
    assert intent['observation_identity'] and intent['snapshot_identity']
    assert intent['deterministic_selected_action'] and intent['sampled_action'] is None
    assert intent[('inference' if which=='inference' else 'publisher_check')+'_elapsed_s']==pytest.approx(.6)


def test_fresh_latest_cache_cannot_refresh_pending_input(rig):
    action=rig.w.predict(rig.model,rig.observation)
    rig.advance(.6)
    assert rig.env.robot._sensor_ages()==[0.,0.]
    rig.env.robot.scan.header.stamp.sec=10;rig.env.robot.odom.header.stamp.sec=10
    with pytest.raises(RuntimeError,match='Stale'):rig.w.step(action)
    intent=rejected(rig)
    assert intent['pre']['scan_stamp_ns']==10**9
    assert intent['freshness_checks'][-1]['elapsed_since_capture_s']==pytest.approx(.6)


@pytest.mark.parametrize('field,value',[
    ('scan_stamp_ns',None),('odom_stamp_ns',-1),('scan_stamp_ns',0),('odom_stamp_ns',float('nan')),
    ('scan_stamp_ns',1.5),('odom_stamp_ns',True),('capture_monotonic_ns',None),
    ('sensor_ages_s',[0.]),('sensor_ages_s',[-.1,0.]),('sensor_ages_s',[0.,float('inf')]),
    ('sensor_age_limit_s',float('nan')),('sensor_age_limit_s',.6)])
def test_invalid_original_provenance_rejected_before_inference(rig,field,value):
    rig.w.returned_snapshot[field]=value
    with pytest.raises(RuntimeError,match='timestamp|ages|limit'):
        action=rig.w.predict(rig.model,rig.observation);rig.w.step(action)
    intent=rejected(rig)
    assert intent['inference_started_monotonic_ns'] is None
    assert intent['rejection_reason']


@pytest.mark.parametrize('tamper',['observation','snapshot','pending_observation','pending_snapshot'])
def test_association_cannot_change_after_inference(rig,tamper):
    action=rig.w.predict(rig.model,rig.observation)
    if tamper=='observation':rig.w.returned_observation[36]+=.01
    if tamper=='snapshot':rig.w.returned_snapshot['odom_stamp_ns']+=1
    if tamper=='pending_observation':rig.w.pending['pre_action_observation'][36]+=.01
    if tamper=='pending_snapshot':rig.w.pending['pre']['odom_stamp_ns']+=1
    with pytest.raises(RuntimeError,match='association'):rig.w.step(action)
    rejected(rig)


def test_expiry_during_intent_write_rechecked_after_io(rig):
    original=d.append
    def slow(path,value):
        original(path,value)
        if value.get('status')=='dispatch_candidate':rig.advance(.6)
    rig.monkeypatch.setattr(d,'append',slow)
    action=rig.w.predict(rig.model,rig.observation)
    with pytest.raises(RuntimeError,match='Stale'):rig.w.step(action)
    intent=rejected(rig)
    assert intent['freshness_checks'][-1]['stage']=='immediately_before_dispatch'


def test_initial_receive_age_is_added_to_elapsed_time(rig):
    rig.w.returned_snapshot['sensor_ages_s']=[.35,.4]
    action=rig.w.predict(rig.model,rig.observation);rig.advance(.11)
    with pytest.raises(RuntimeError,match='Stale'):rig.w.step(action)
    intent=rejected(rig)
    assert intent['freshness_checks'][-1]['current_sensor_ages_s']==pytest.approx([.46,.51])


def test_exact_age_limit_is_inclusive(rig):
    action=rig.w.predict(rig.model,rig.observation);rig.advance(.5)
    rig.w.step(action)
    assert rig.env.control_calls==1


def test_backward_monotonic_clock_is_rejected(rig):
    action=rig.w.predict(rig.model,rig.observation);rig.advance(-.01)
    with pytest.raises(RuntimeError,match='monotonic'):rig.w.step(action)
    rejected(rig)


def test_intent_write_failure_stops_without_dispatch(rig):
    action=rig.w.predict(rig.model,rig.observation)
    original=d.append
    def broken(path,value):
        if path.name=='action_intents.jsonl':raise OSError('intent disk unavailable')
        return original(path,value)
    rig.monkeypatch.setattr(d,'append',broken)
    with pytest.raises(OSError,match='disk unavailable'):rig.w.step(action)
    assert rig.env.control_calls==0 and rig.env.robot.stops>0 and rig.w.rejected
    assert rig.w.pending is None


def test_missing_pending_action_stops_without_control(rig):
    with pytest.raises(RuntimeError,match='Missing pending'):rig.w.step(np.zeros(2))
    assert rig.env.control_calls==0 and rig.env.robot.stops>0
    assert rows(rig)[-1]['status']=='rejected'
    assert rows(rig)[-1]['raw_control_call_occurred'] is False


def test_full_scripted_episode_failure_stops_and_restores_hooks(tmp_path,models,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(PPO,'learn',lambda *a,**k:pytest.fail('No training'))
    env=CountingEnv(); original_observe=env.robot.get_observation
    clock=[time.monotonic_ns()];monkeypatch.setattr(d.time,'monotonic_ns',lambda:clock[0])
    original=d.inference
    def slow(*args):
        result=original(*args);clock[0]+=600000000;return result
    monkeypatch.setattr(d,'inference',slow)
    result=d.pilot(tmp_path/'scripted_failure',environment_factory=lambda:env,models=models,readiness=lambda:None)
    assert result['status']=='failed' and len(result['episodes'])==1
    assert result['training_steps']==0 and result['model_state_unchanged']
    assert result['shutdown']['closed'] and result['stop_verified']
    assert env.closed and env.control_calls==0
    assert env.robot.get_observation==original_observe and env.robot.expert_publish_callback is None


def test_missing_entire_source_snapshot_is_logged(rig):
    del rig.w.returned_snapshot
    with pytest.raises(RuntimeError,match='association'):
        rig.w.predict(rig.model,rig.observation)
    intent=rejected(rig)
    assert intent['pre'] is None and intent['inference_started_monotonic_ns'] is None


def test_invalid_observation_is_logged_without_dispatch(rig):
    rig.w.returned_observation[36]=np.nan
    with pytest.raises(RuntimeError,match='observation'):
        rig.w.predict(rig.model,rig.observation)
    intent=rejected(rig)
    assert intent['raw_control_call_occurred'] is False
