"""Scripted telemetry tests: no simulator and no model learning."""
import copy
from types import SimpleNamespace as NS
import json
import math
import time
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from agents.jackal_pretrained_ppo_policy import JackalPolicyContractEnv
from navigation import jackal_generalization_diagnostic as d


def msg(t):
    return NS(header=NS(stamp=NS(sec=t,nanosec=0),frame_id='odom'),
        pose=NS(pose=NS(position=NS(x=0.,y=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))),
        twist=NS(twist=NS(linear=NS(x=0.),angular=NS(z=0.))))


class Robot:
    def __init__(self):
        self.command_topic='/j100_0000/platform/cmd_vel_unstamped'
        self.scan=msg(1);self.odom=msg(1);self.distance=.75
        self.goal_x=.75;self.goal_y=0.;self.action_mapper=NS(previous=NS(linear=0.,angular=0.))
        self.expert_publish_callback=None;self.last_safety_reason='';self.stops=0;self.stale=False;self.competitor=False
    def get_name(self):return 'jackal_rl_interface'
    def get_namespace(self):return '/'
    def get_publishers_info_by_topic(self,topic):
        own=NS(node_name=self.get_name(),node_namespace='/',endpoint_gid=[1])
        return [own,own] if self.competitor else [own]
    def count_subscribers(self,topic):return 1
    def get_parameter(self,name):return NS(value=.5)
    def _sensor_ages(self):return [1.,1.] if self.stale else [0.,0.]
    def _goal_relative(self):return self.distance,0.
    def get_observation(self):
        x=np.zeros(40,dtype=np.float32);x[36]=self.distance/20
        x[38]=self.odom.twist.twist.linear.x/.3
        return x
    def stop(self):
        self.stops+=1;self.action_mapper.previous=NS(linear=0.,angular=0.)
        if self.expert_publish_callback:self.expert_publish_callback(0.,0.,self.odom.header.stamp.sec*10**9)


class Env(JackalPolicyContractEnv):
    def __init__(self, yaw=0.,stale_step=False):
        super().__init__();self.robot=Robot();self.max_episode_steps=300;self.goal_tolerance=.3
        self.closed=False;self.yaw=yaw;self.stale_step=stale_step
    def reset(self,*,seed=None,options=None):
        self.robot.distance=.75;self.robot.scan=msg(1);self.robot.odom=msg(1)
        return self.robot.get_observation(),dict(robot_world_pose=[0,-1.8,.3,self.yaw],goal_odom=[.75,0])
    def step(self,action):
        r=self.robot;r.action_mapper.previous=NS(linear=.12,angular=-.03)
        r.expert_publish_callback(.12,-.03,r.odom.header.stamp.sec*10**9)
        before=r.distance;r.distance-=.25
        t=r.odom.header.stamp.sec if self.stale_step else r.odom.header.stamp.sec+1
        x=r.odom.pose.pose.position.x+.25;r.odom=msg(t);r.scan=msg(t)
        r.odom.pose.pose.position.x=x;r.odom.twist.twist.linear.x=.1
        done=r.distance<.3
        info=dict(scan_stamp=float(t),odom_stamp=float(t),previous_distance=before,current_distance=r.distance,
            progress=.25,progress_reward=2.5,step_penalty=.01,action_accepted=True,
            goal_reached=done,collision_detected=False,time_limit_reached=False,termination_reason='goal_reached' if done else None)
        return r.get_observation(),2.49+(20 if done else 0),done,False,info
    def close(self):self.robot.stop();self.closed=True


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1)
    return d.load_models()


def wrapper(tmp_path,env):
    w=d.TelemetryEnv(env,tmp_path,time.monotonic()+30)
    w.context=dict(episode_id='test',policy='initial',seed=642,scenario='anchor')
    return w


def test_geometry_and_checkpoint_identity(monkeypatch):
    manifest=d.validate_matrix()
    assert len(manifest['scenarios'])==13 and manifest['controlled_routes_clear']
    assert all(p['reachable'] for p in manifest['plans'].values())
    assert d.verify_checkpoints()['final']['sha256']==d.CHECKPOINTS['final'][1]
    bad=copy.copy(d.CHECKPOINTS);bad['final']=(bad['final'][0],'0'*64)
    monkeypatch.setattr(d,'CHECKPOINTS',bad)
    with pytest.raises(ValueError,match='identity'):d.load_models()


def test_yaw_position_and_wrap():
    assert d.verify_pose((0,0,math.pi),(0,0,.3,-math.pi))['yaw_error_rad']<1e-10
    for pose in ((0,0,.3,.051),(.051,0,.3,0),(0,0,.3,float('nan'))):
        with pytest.raises(ValueError):d.verify_pose((0,0,0),pose)


def test_alignment_action_stages_and_velocity(tmp_path,models):
    w=wrapper(tmp_path,Env());obs,_=w.reset(options=dict(start=(0,-1.8,0),goal=(.75,-1.8)))
    action=w.predict(models['initial'],obs)
    post,*_=w.step(action)
    row=w.rows[0]
    assert row['pre_action_observation']==obs.tolist()
    assert row['post_action_observation']==post.tolist() and row['sampled_action'] is None
    assert row['pre']['odom_stamp_ns']<row['post']['odom_stamp_ns']
    assert row['published_commands'][0]['published_velocity']['linear_m_s']==.12
    assert row['post']['measured_velocity']['linear_m_s']==.1
    assert row['distance_progress_m']==.25
    assert np.allclose(row['actor_mean'],row['deterministic_selected_action'])


@pytest.mark.parametrize('failure',['yaw','stale_age','stale_stamp','publisher','alignment','action'])
def test_fail_closed(tmp_path,models,failure):
    env=Env(yaw=.1 if failure=='yaw' else 0,stale_step=failure=='stale_stamp');w=wrapper(tmp_path,env)
    try:
        if failure=='yaw':
            with pytest.raises(ValueError):w.reset(options=dict(start=(0,-1.8,0),goal=(.75,-1.8)))
            return
        obs,_=w.reset(options=dict(start=(0,-1.8,0),goal=(.75,-1.8)))
        if failure=='stale_age':env.robot.stale=True
        if failure=='publisher':env.robot.competitor=True
        if failure=='alignment':obs=obs+1
        with pytest.raises((ValueError,RuntimeError)):
            action=w.predict(models['initial'],obs)
            w.step(-action if failure=='action' else action)
    finally:
        assert d.confirm_stop(w)[0]
        w.close();assert env.closed and env.robot.stops>0


def test_pilot_evaluation_only_and_exact_order(tmp_path,models,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(PPO,'learn',lambda *a,**k:pytest.fail('Training is forbidden'))
    env=Env()
    result=d.pilot(tmp_path/'pilot',environment_factory=lambda:env,models=models,readiness=lambda:None)
    assert result['status']=='completed',result.get('failure')
    assert [(e['policy'],e['seed']) for e in result['episodes']]==list(d.PILOT)
    assert result['training_steps']==0 and result['model_state_unchanged']
    assert env.closed and result['shutdown']['closed'] and result['stop_verified']
    with pytest.raises(FileExistsError):d.pilot(tmp_path/'pilot')


def test_pilot_failure_stops_remaining_episodes(tmp_path,models,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    env=Env(yaw=.2)
    result=d.pilot(tmp_path/'failure',environment_factory=lambda:env,models=models,readiness=lambda:None)
    assert result['status']=='failed' and len(result['episodes'])==1
    assert env.closed and result['stop_verified'] and result['model_state_unchanged']
    assert (tmp_path/'failure/failures.jsonl').exists()


def test_preview_and_missing_output_never_execute(monkeypatch):
    monkeypatch.setattr(d,'pilot',lambda *a,**k:pytest.fail('Preview cannot execute'))
    monkeypatch.setattr(d,'run_diagnostic',lambda *a,**k:pytest.fail('Missing output cannot execute'))
    assert d.main(['preview'])==0
    for mode in ('train','matrix','full'):
        with pytest.raises(SystemExit):d.main([mode])


def test_checkpoint_failure_preserves_report_and_never_constructs_env(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'OUTPUT_ROOT',tmp_path)
    def fail():raise ValueError('Checkpoint identity mismatch')
    monkeypatch.setattr(d,'verify_checkpoints',fail)
    result=d.pilot(tmp_path/'bad_identity',environment_factory=lambda:pytest.fail('No environment after loading failure'))
    assert result['status']=='failed' and result['episodes']==[]
    assert json.loads((tmp_path/'bad_identity/summary.json').read_text())['training_steps']==0


def test_callback_recording_failure_is_not_silently_swallowed(tmp_path,models,monkeypatch):
    w=wrapper(tmp_path,Env());obs,_=w.reset(options=dict(start=(0,-1.8,0),goal=(.75,-1.8)))
    action=w.predict(models['initial'],obs)
    original_append=d.append
    def fail_publish_log(path,value):
        if path.name=='published_commands.jsonl':
            raise OSError('scripted disk failure')
        return original_append(path,value)
    monkeypatch.setattr(d,'append',fail_publish_log)
    with pytest.raises(RuntimeError,match='Publish telemetry failure'):w.step(action)
    w.close()


def test_supervisor_cleanup_is_scoped_to_owned_descendants(tmp_path,monkeypatch):
    from navigation import jackal_diagnostic_supervisor as s
    monkeypatch.setattr(s,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s.os,'environ',{**s.SETTINGS})
    monkeypatch.setattr(s.sys,'argv',['supervisor','--output',str(tmp_path/'run')])
    states=iter([{}, {900001:(1,'ros2 launch'),900002:(900001,'owned bridge'),900003:(1,'unrelated')}])
    monkeypatch.setattr(s,'processes',lambda:next(states))
    monkeypatch.setattr(s.time,'sleep',lambda _:None)
    class Process:
        def __init__(self,pid):self.pid=pid;self.done=False
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):self.done=True;return 1 if self.pid==900004 else 0
    children=iter([Process(900001),Process(900004)])
    launched=[]
    def popen(command,**kw):launched.append(command);return next(children)
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    killed=[]
    monkeypatch.setattr(s.os,'kill',lambda pid,sig:killed.append(pid))
    monkeypatch.setattr(s.os,'killpg',lambda pid,sig:killed.append(pid))
    assert s.main()==1  # Failed pilot is preserved; cleanup still occurs.
    assert 900001 in killed and 900002 in killed and 900003 not in killed
    assert 'pilot' in launched[1] and 'train' not in launched[1]
    report=json.loads((tmp_path/'run_supervisor/cleanup.json').read_text())
    assert report['training_steps']==0 and not report['full_matrix_launched']


def test_late_sensor_callbacks_do_not_relabel_transition_observation(tmp_path,models):
    class LateCacheEnv(Env):
        def step(self,action):
            observation,reward,term,trunc,info=super().step(action)
            # Models queued callbacks processed during collision monitoring AFTER
            # the environment captured its returned observation/info stamps.
            self.robot.scan=msg(self.robot.scan.header.stamp.sec+1)
            self.robot.odom.header.stamp.sec+=1
            self.robot.odom.twist.twist.linear.x=.2
            return observation,reward,term,trunc,info
    w=wrapper(tmp_path,LateCacheEnv())
    obs,_=w.reset(options=dict(start=(0,-1.8,0),goal=(.75,-1.8)))
    action=w.predict(models['initial'],obs);returned,*_=w.step(action)
    row=w.rows[0]
    assert row['post']['odom_stamp_ns']==2*10**9
    assert row['latest_cache_after_transition']['odom_stamp_ns']==3*10**9
    assert row['post']['measured_velocity']['linear_m_s']==.1
    assert row['latest_cache_after_transition']['measured_velocity']['linear_m_s']==.2
    w.predict(models['initial'],returned)
    assert w.pending['pre']['odom_stamp_ns']==2*10**9
    assert w.robot.get_observation()[38]!=returned[38]
    w.close()
