"""Evaluation-only Phase 6H.1 adapter. Explicit modes select four anchors or the 52-episode matrix.

Reuses Jackal reset/step, the closed-loop evaluator, publish callback and stop
verification. No learning or checkpoint writing occurs.
"""
import argparse
from collections import deque
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from navigation import jackal_ppo_diagnostic as prior
from navigation.jackal_ppo_train import _make_jackal_env, _shutdown, audit_loaded_policy, preflight_simulator
from navigation.jackal_bc_evaluation import (EvalScenario, evaluation_scenarios, plan_scenarios,
    run_closed_loop_episode, confirm_stop)
from navigation.live_expert_episode import LiveExpertEpisode
from curriculum.jackal_curriculum import validate_scenario

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/'artifacts/ppo/jackal_6h/diagnostics'
WORLD = ROOT/'jackal_rl_ws/worlds/jackal_obstacle_test_collision.sdf'
CHECKPOINTS = {
    'initial': (ROOT/'artifacts/ppo/jackal_6g1/run_20261022_230534/initialized_ppo.zip',
                'dff19be924f920490915d8f625d44f32da79452caaf3f7698160d2b8fc287750'),
    'final': (ROOT/'artifacts/ppo/jackal_6h/curriculum/run_20261026_experiment/checkpoint_00010240.zip',
              '8542b7e27aa51d064b46834b58f77badd2a98c8ec374af33f0ea0b38c638f23c')}
SETTINGS = dict(ROS_DOMAIN_ID='63', IGN_PARTITION='jackal_rl', GZ_PARTITION='jackal_rl',
                IGN_IP='127.0.0.1', GZ_IP='127.0.0.1')
BUDGETS = {'pilot':600, 'matrix':6000}
OUTER_TIMEOUTS = {'pilot':660, 'matrix':6060}
PILOT = (('initial',642),('final',642),('final',643),('initial',643))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, allow_nan=False)+'\n')


def matrix():
    def scenario(name, start, goal):
        return EvalScenario(name,'controlled',tuple(start),tuple(goal),1,90.,'6H.1 controlled diagnostic')
    result=[scenario('anchor',(0,-1.8,0),(.75,-1.8)),
        scenario('distance_1p5',(0,-1.8,0),(1.5,-1.8)),
        scenario('distance_2p5',(0,-1.8,0),(2.5,-1.8)),
        scenario('translation_south',(-.5,-1.5,0),(.25,-1.5)),
        scenario('translation_north',(0,1.8,0),(.75,1.8))]
    result += [scenario('heading_'+str(yaw),(0,-1.8,yaw),(.75,-1.8)) for yaw in (-2.7,-.4,.4,2.7)]
    result += [s for s in evaluation_scenarios(5) if s.scenario_id in prior.VALIDATION_SCENARIOS]
    return result


def episode_plan(mode, scenario_id=None):
    if mode not in BUDGETS:
        raise ValueError('Unknown diagnostic mode')
    scenarios=matrix() if mode=='matrix' else matrix()[:1]
    if scenario_id is not None:
        scenarios=[s for s in matrix() if s.scenario_id==scenario_id]
        if len(scenarios)!=1:
            raise ValueError('Unknown or ambiguous scenario: '+str(scenario_id))
    return [dict(episode_id=f'{s.scenario_id}_{name}_{seed}',scenario=s.scenario_id,
                 policy=name,seed=seed,start=list(s.start),goal=list(s.goal))
            for s in scenarios for name,seed in PILOT]


def validate_matrix():
    scenarios=matrix()
    for scenario in scenarios:
        validate_scenario(scenario)
    # Inspect the actual SDF, then reuse the existing inflated occupancy planner.
    world=ET.parse(WORLD).getroot().find('world')
    models=[m.get('name') for m in world.findall('model')]
    if set(models)!={'ground_plane','safety_test_box'}:
        raise ValueError('Unexpected world geometry')
    box=world.find("model[@name='safety_test_box']")
    if box.findtext('pose').split()!=['1.5','0','0.5','0','0','0'] or box.findtext('.//box/size').split()!=['0.5','0.5','1.0']:
        raise ValueError('Obstacle geometry differs from inspected world')
    plans=plan_scenarios(scenarios,WORLD,.45)
    if not all(p['reachable'] for p in plans.values()):
        raise ValueError('Unreachable diagnostic condition')
    # For the controlled cases, a conservative rectangle intersection is sufficient:
    # all routes are horizontal and lie outside the inflated box's y range.
    for s in scenarios[:9]:
        if not (s.start[1]==s.goal[1] and abs(s.start[1])>.7):
            raise ValueError('Controlled route not verified clear')
    return dict(world=str(WORLD),world_sha256=prior.file_sha256(WORLD),models=models,
                inflated_box=[.8,2.2,-.7,.7],scenarios=[asdict(s) for s in scenarios],
                plans=plans,rejected_or_modified=[],controlled_routes_clear=True)


def verify_checkpoints():
    result={}
    for name,(path,expected) in CHECKPOINTS.items():
        actual=prior.file_sha256(path)
        if actual!=expected:
            raise ValueError(f'Checkpoint identity mismatch: {name}')
        result[name]=dict(path=str(path),sha256=actual)
    return result


def load_models():
    verify_checkpoints()  # Both identities pass before either policy is used.
    models={name:PPO.load(path,device='cpu') for name,(path,_) in CHECKPOINTS.items()}
    audit_loaded_policy(models['initial'])  # This helper intentionally requires an untrained model.
    reference=models['initial'].policy
    for m in models.values():
        p=m.policy
        if (type(p) is not type(reference) or p.network_config!=reference.network_config
                or p.squash_output or p.use_sde or p.normalize_images or p.share_features_extractor
                or type(p.action_dist) is not type(reference.action_dist)
                or any(not torch.isfinite(t).all() for t in p.state_dict().values())):
            raise ValueError('Loaded PPO evaluation contract mismatch')
        for key in ('obs_mean','obs_std','normalizer_count'):
            if not torch.equal(getattr(p.pi_features_extractor.trunk,key),getattr(reference.pi_features_extractor.trunk,key)):
                raise ValueError('Frozen normalizer differs between checkpoints')
        m.policy.set_training_mode(False)
    return models


def wrap_angle(value):
    return math.atan2(math.sin(value),math.cos(value))


def verify_pose(requested, measured):
    # 5 cm is stricter than the environment's 15 cm positional reset check;
    # 0.05 rad (~2.9 degrees) is small relative to the smallest 0.4 rad contrast.
    if len(measured)!=4 or not np.isfinite(measured).all():
        raise ValueError('Missing finite measured world pose/yaw')
    position_error=math.dist(requested[:2],measured[:2])
    yaw_error=abs(wrap_angle(measured[3]-requested[2]))
    if position_error>.05 or yaw_error>.05:
        raise ValueError(f'Incorrect reset pose: position={position_error}, yaw={yaw_error}')
    return dict(position_error_m=position_error,yaw_error_rad=yaw_error,
                position_tolerance_m=.05,yaw_tolerance_rad=.05)


def publisher_check(robot):
    publishers=robot.get_publishers_info_by_topic(robot.command_topic)
    rows=[dict(node_name=p.node_name,node_namespace=p.node_namespace,
               gid=list(p.endpoint_gid)) for p in publishers]
    if (len(rows)!=1 or rows[0]['node_name']!=robot.get_name()
            or rows[0]['node_namespace']!=robot.get_namespace()
            or robot.count_subscribers(robot.command_topic)<1):
        raise RuntimeError('Unexpected command publisher or missing controller subscriber: '+str(rows))
    return dict(topic=robot.command_topic,publishers=rows,subscriber_count=robot.count_subscribers(robot.command_topic))


def intent_value(value):
    """Keep invalid provenance readable without writing non-standard JSON NaN."""
    if isinstance(value, np.ndarray):
        return intent_value(value.tolist())
    if isinstance(value, np.generic):
        return intent_value(value.item())
    if isinstance(value, dict):
        return {key:intent_value(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [intent_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'invalid_numeric':repr(value)}
    return value


def identity(value):
    return hashlib.sha256(json.dumps(intent_value(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def observation_identity(observation):
    array=np.asarray(observation,dtype=np.float32)
    if array.shape!=(40,) or not np.isfinite(array).all():
        raise RuntimeError('Invalid pending observation')
    return hashlib.sha256(array.dtype.str.encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def input_freshness(source, now_ns):
    """Age only the original snapshot, never a newer cache or another clock domain.

    ROS header stamps identify the input; receive ages measured at capture plus
    elapsed host-monotonic time establish age. ROS and host time are not subtracted.
    """
    result=dict(source=copy.deepcopy(source),checked_monotonic_ns=now_ns)
    try:
        for name in ('scan_stamp_ns','odom_stamp_ns','capture_monotonic_ns'):
            value=source[name]
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<=0:
                raise ValueError('Missing or invalid source timestamp: '+name)
        if isinstance(now_ns,bool) or not isinstance(now_ns,(int,np.integer)) or now_ns<source['capture_monotonic_ns']:
            raise ValueError('Invalid or reversed monotonic clock')
        ages=source['sensor_ages_s']
        limit=source['sensor_age_limit_s']
        if (not isinstance(ages,(list,tuple)) or len(ages)!=2
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in ages)
                or isinstance(limit,bool) or not isinstance(limit,(int,float)) or limit!=.5):
            raise ValueError('Invalid original sensor ages or unexpected age limit')
        elapsed=(now_ns-source['capture_monotonic_ns'])/1e9
        current=[v+elapsed for v in ages]
        result.update(original_sensor_ages_s=list(ages),elapsed_since_capture_s=elapsed,
                      current_sensor_ages_s=current,sensor_age_limit_s=limit)
        if max(current)>limit:
            raise ValueError('Stale pending policy-input observation')
        result['valid']=True
    except (KeyError,TypeError,ValueError) as exc:
        result.update(valid=False,rejection_reason=str(exc))
    return result


def snapshot(robot):
    if robot.scan is None or robot.odom is None:
        raise RuntimeError('Missing sensors')
    def stamp(msg):
        sec=msg.header.stamp.sec; nano=msg.header.stamp.nanosec
        if (isinstance(sec,bool) or isinstance(nano,bool) or not isinstance(sec,(int,np.integer))
                or not isinstance(nano,(int,np.integer)) or sec<0 or not 0<=nano<1000000000
                or sec*1000000000+nano<=0):
            raise RuntimeError('Missing or invalid source sensor timestamp')
        return int(sec)*1000000000+int(nano)
    captured_ns=time.monotonic_ns()  # Before age sampling: elapsed accounting is conservative.
    ages=robot._sensor_ages()
    limit=float(robot.get_parameter('sensor_timeout').value)
    if ages is None or len(ages)!=2 or limit!=.5 or not np.isfinite(ages).all() or min(ages)<0 or max(ages)>limit:
        raise RuntimeError('Stale sensor data: '+str(ages))
    pose=LiveExpertEpisode.pose_from_odom(robot.odom)
    relative=robot._goal_relative()
    values=[*pose,*relative,robot.odom.twist.twist.linear.x,robot.odom.twist.twist.angular.z]
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite measured motion or goal state')
    return dict(capture_monotonic_ns=captured_ns,scan_stamp_ns=stamp(robot.scan),
        odom_stamp_ns=stamp(robot.odom),sensor_ages_s=list(ages),sensor_age_limit_s=limit,
        pose_odom=list(pose),goal_distance_m=float(relative[0]),goal_bearing_rad=float(relative[1]),
        measured_velocity=dict(linear_m_s=float(values[-2]),angular_rad_s=float(values[-1])))


def inference(model, observation):
    obs=np.asarray(observation,dtype=np.float32)
    if obs.shape!=(40,) or not np.isfinite(obs).all():
        raise ValueError('Invalid pre-action observation')
    tensor=torch.as_tensor(obs[None],device=model.device)
    with torch.no_grad():
        raw,mean=model.policy.raw_and_mean(tensor)
        selected,_=model.predict(obs,deterministic=True)
        std=model.policy.log_std.exp().cpu().numpy()
        trunk=model.policy.pi_features_extractor.trunk
        z=((tensor-trunk.obs_mean)/trunk.obs_std).cpu().numpy()[0]
    raw=raw.cpu().numpy()[0]; mean=mean.cpu().numpy()[0]
    clipped=np.clip(selected,-1,1).astype(np.float32)
    if not np.allclose(selected,mean,atol=1e-6,rtol=0) or not all(np.isfinite(a).all() for a in (raw,mean,std,z,selected)):
        raise RuntimeError('Deterministic telemetry mismatch')
    return clipped,dict(pre_action_observation=obs.tolist(),normalizer_output=z.tolist(),
        actor_pre_tanh=raw.tolist(),actor_mean=mean.tolist(),policy_std=std.tolist(),
        deterministic_selected_action=np.asarray(selected).tolist(),sampled_action=None,
        clipped_normalized_action=clipped.tolist())


class TelemetryEnv(gym.Wrapper):
    def __init__(self, env, output, deadline):
        super().__init__(env)
        self.robot=env.unwrapped.robot
        self.max_episode_steps=env.unwrapped.max_episode_steps
        self.output=Path(output); self.deadline=deadline
        self.original_observe=self.robot.get_observation
        self.observations=deque(maxlen=32)
        self.robot.get_observation=self.capture_observation
        self.original_callback=getattr(self.robot,'expert_publish_callback',None)
        self.robot.expert_publish_callback=self.capture_command
        self.commands=[]; self.pending=None; self.rows=[]; self.reset_record=None
        self.context={}; self.step_id=0
        self.callback_error=None
        self.intent_sequence=0; self.dispatch_state='idle'; self.rejected=False

    def capture_observation(self):
        observation=self.original_observe()
        if observation is not None:
            self.observations.append((np.asarray(observation).copy(),snapshot(self.robot)))
        return observation

    def observation_snapshot(self, observation, stamps=None):
        for candidate,captured in reversed(self.observations):
            if np.array_equal(candidate,observation) and (stamps is None or
                    all(abs(captured[k]/1e9-v)<=1e-6 for k,v in stamps.items())):
                return captured
        raise RuntimeError('No sensor snapshot aligned to returned observation')

    def guard(self):
        if time.monotonic()>self.deadline:
            raise TimeoutError('Diagnostic wall-time budget exceeded')
        if self.callback_error:
            raise RuntimeError('Publish telemetry failure: '+self.callback_error)

    def capture_command(self, linear, angular, ros_ns):
        try:
            event=dict(**self.context,step=self.step_id,monotonic_ns=time.monotonic_ns(),ros_stamp_ns=int(ros_ns),
                published_velocity=dict(linear_m_s=float(linear),angular_rad_s=float(angular)),
                action_intent_id=self.pending.get('intent_id') if self.pending else None,
                action_dispatch_state=self.dispatch_state,mapper_velocity_at_publish=dict(linear_m_s=float(self.robot.action_mapper.previous.linear),
                    angular_rad_s=float(self.robot.action_mapper.previous.angular)))
            self.commands.append(event)
            append(self.output/'published_commands.jsonl',event)
            if self.original_callback:
                self.original_callback(linear,angular,ros_ns)
        except Exception as exc:
            self.callback_error=str(exc)  # Interface swallows callback exceptions; guard must fail closed.

    def reset(self, *, seed=None, options=None):
        self.guard(); self.rows=[]; self.step_id=0; self.pending=None
        self.rejected=False; self.dispatch_state='idle'
        start=time.monotonic_ns()
        obs,info=self.env.reset(seed=seed,options=options)
        measured=info.get('robot_world_pose',())
        record=dict(**self.context,requested_start=list(options['start']),measured_reset_world_pose=list(measured),
            reset_started_monotonic_ns=start,reset_finished_monotonic_ns=time.monotonic_ns())
        append(self.output/'reset_attempts.jsonl',record)
        record.update(verify_pose(options['start'],measured))
        self.pre=self.observation_snapshot(obs)
        self.returned_observation=np.asarray(obs).copy()
        self.returned_snapshot=self.pre
        record.update(sensor_snapshot=self.pre,publisher=publisher_check(self.robot),goal_world=list(options['goal']),goal_odom=list(info['goal_odom']))
        self.reset_record=record
        append(self.output/'verified_resets.jsonl',record)
        self.initial_distance=self.pre['goal_distance_m']; self.path=0.; self.heading_change=0.; self.abs_heading_change=0.
        return obs,info

    def record_intent(self, status, reason=None):
        record=dict(**self.context,**self.pending,status=status,rejection_reason=reason,
                    recorded_monotonic_ns=time.monotonic_ns())
        append(self.output/'action_intents.jsonl',intent_value(record))

    def reject_intent(self, reason):
        self.rejected=True; self.dispatch_state='rejected_stop'
        try:
            self.record_intent('rejected',reason)
        finally:
            try:
                self.robot.stop()  # Stop even if rejection telemetry cannot be persisted.
            finally:
                self.pending=None
        raise RuntimeError(reason)

    def check_pending_input(self, stage):
        source=self.pending.get('pre')
        try:
            if (source is None or identity(source)!=self.pending['snapshot_identity']
                    or identity(self.returned_snapshot)!=self.pending['snapshot_identity']
                    or observation_identity(self.pending['pre_action_observation'])!=self.pending['observation_identity']
                    or observation_identity(self.returned_observation)!=self.pending['observation_identity']):
                raise RuntimeError('Pending observation/source association mismatch')
            result=input_freshness(source,time.monotonic_ns())
        except (KeyError,TypeError,ValueError,RuntimeError) as exc:
            result=dict(valid=False,rejection_reason=str(exc),source=copy.deepcopy(source),
                        checked_monotonic_ns=time.monotonic_ns())
        result['stage']=stage
        self.pending['freshness_checks'].append(result)
        if not result['valid']:
            raise RuntimeError(result['rejection_reason'])

    def predict(self, model, observation, action_space=None):
        if self.rejected:
            raise RuntimeError('Rejected action cannot be reused; an explicit reset is required')
        if self.pending is not None:
            self.reject_intent('Previous pending action has not been consumed')
        self.intent_sequence+=1
        try:
            returned_identity=observation_identity(getattr(self,'returned_observation',None))
        except (TypeError,ValueError,RuntimeError):
            returned_identity=None
        self.pending=dict(intent_id=f"{self.context.get('episode_id','unknown')}:{self.intent_sequence}",
            pre=copy.deepcopy(getattr(self,'returned_snapshot',None)),pre_action_observation=np.asarray(observation).tolist(),
            observation_identity=returned_identity,
            snapshot_identity=identity(getattr(self,'returned_snapshot',None)),freshness_checks=[],
            raw_control_call_occurred=False,inference_started_monotonic_ns=None,
            inference_finished_monotonic_ns=None,publisher_check_started_monotonic_ns=None,
            publisher_check_finished_monotonic_ns=None)
        try:
            self.guard()
            snapshot(self.robot)  # Current-cache health does not refresh the pending input.
            self.pre=self.pending['pre']
            self.check_pending_input('before_inference')
            self.pending['inference_started_monotonic_ns']=time.monotonic_ns()
            action,details=inference(model,observation)
            self.pending.update(details)
            self.pending['inference_finished_monotonic_ns']=time.monotonic_ns()
            self.pending['inference_elapsed_s']=(self.pending['inference_finished_monotonic_ns']-self.pending['inference_started_monotonic_ns'])/1e9
            self.check_pending_input('after_inference')
            self.record_intent('prepared')
            return action
        except Exception as exc:
            self.reject_intent(str(exc))

    def step(self, action):
        if self.rejected:
            raise RuntimeError('Rejected action cannot be reused; an explicit reset is required')
        if self.pending is None:
            self.intent_sequence+=1
            self.pending=dict(intent_id=f"{self.context.get('episode_id','unknown')}:{self.intent_sequence}",
                pre=None,observation_identity=None,snapshot_identity=None,freshness_checks=[],
                unassociated_requested_action=np.asarray(action).tolist(),raw_control_call_occurred=False)
            self.reject_intent('Missing pending action; no raw control call occurred')
        self.step_id+=1
        self.pending['requested_dispatch_action']=np.asarray(action).tolist()
        try:
            self.guard()
            if not np.array_equal(action,np.array(self.pending['clipped_normalized_action'],dtype=np.float32)):
                raise RuntimeError('Selected/stepped action mismatch')
            self.pending['publisher_check_started_monotonic_ns']=time.monotonic_ns()
            graph=publisher_check(self.robot)
            self.pending['publisher_check_finished_monotonic_ns']=time.monotonic_ns()
            self.pending['publisher_check_elapsed_s']=(self.pending['publisher_check_finished_monotonic_ns']-self.pending['publisher_check_started_monotonic_ns'])/1e9
            self.guard()
            snapshot(self.robot)  # Independent cache-validity gate; never replaces the source snapshot.
            self.check_pending_input('after_publisher_verification')
            self.record_intent('dispatch_candidate')  # Durable intent, not a claim of dispatch.
            first=len(self.commands); started=time.monotonic_ns()
            self.check_pending_input('immediately_before_dispatch')  # Includes logging/check latency.
        except Exception as exc:
            self.reject_intent(str(exc))
        self.pending['raw_control_call_occurred']=True
        self.dispatch_state='raw_control_call'
        try:
            obs,reward,terminated,truncated,info=self.env.step(action)
        finally:
            self.record_intent('raw_control_called')
            self.dispatch_state='idle'
        self.guard()
        latest=snapshot(self.robot)
        append(self.output/'transition_attempts.jsonl',dict(**self.context,step=self.step_id,
            environment_info=prior._plain(info),latest_cache=latest,pre=self.pre,
            returned_observation=np.asarray(obs).tolist()))
        post=self.observation_snapshot(obs,dict(scan_stamp_ns=info['scan_stamp'],odom_stamp_ns=info['odom_stamp']))
        for field in ('scan_stamp_ns','odom_stamp_ns'):
            if post[field]<=self.pre[field]:
                raise RuntimeError('Stale sensor timestamp after step')
        if (abs(post['scan_stamp_ns']/1e9-info['scan_stamp'])>1e-6
                or abs(post['odom_stamp_ns']/1e9-info['odom_stamp'])>1e-6):
            raise RuntimeError('Post-action sensor/info mismatch')
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise RuntimeError('Nonfinite transition')
        commands=self.commands[first:]
        if not commands:
            raise RuntimeError('No captured published command during control step')
        self.path+=math.dist(self.pre['pose_odom'][:2],post['pose_odom'][:2])
        turn=wrap_angle(post['pose_odom'][2]-self.pre['pose_odom'][2])
        self.heading_change+=turn; self.abs_heading_change+=abs(turn)
        progress=self.pre['goal_distance_m']-post['goal_distance_m']
        if abs(progress-info['progress'])>1e-5 or abs(post['goal_distance_m']-info['current_distance'])>1e-5:
            raise RuntimeError('Reward/progress telemetry mismatch')
        row=dict(**self.context,step=self.step_id,control_started_monotonic_ns=started,
            control_finished_monotonic_ns=time.monotonic_ns(),**self.pending,post=post,
            post_action_observation=np.asarray(obs).tolist(),latest_cache_after_transition=latest,published_commands=commands,
            publisher_verification=graph,action_accepted=info.get('action_accepted'),
            safety_reason=getattr(self.robot,'last_safety_reason',''),
            distance_progress_m=progress,net_progress_m=self.initial_distance-post['goal_distance_m'],
            cumulative_path_length_m=self.path,cumulative_heading_change_rad=self.heading_change,
            cumulative_absolute_heading_change_rad=self.abs_heading_change,reward=float(reward),
            reward_components={k:info.get(k) for k in ('previous_distance','current_distance','progress','progress_reward','step_penalty')},
            termination_reward_adjustment=float(reward)-float(info['progress_reward'])+float(info['step_penalty']),
            terminated=bool(terminated),truncated=bool(truncated),termination_reason=info.get('termination_reason'),
            success=bool(info.get('goal_reached')),collision=bool(info.get('collision_detected')),timeout=bool(info.get('time_limit_reached')))
        append(self.output/'telemetry.jsonl',row); self.rows.append(row); self.pending=None
        self.returned_observation=np.asarray(obs).copy(); self.returned_snapshot=post
        return obs,reward,terminated,truncated,info

    def close(self):
        self.robot.expert_publish_callback=self.original_callback
        self.robot.get_observation=self.original_observe
        self.hooks_restored=(self.robot.get_observation==self.original_observe
            and self.robot.expert_publish_callback==self.original_callback)
        if not self.hooks_restored:
            raise RuntimeError('Diagnostic hooks not restored')
        return self.env.close()


def pilot(output, environment_factory=None, models=None, readiness=preflight_simulator):
    return run_diagnostic(output,environment_factory,models,readiness,mode='pilot')


def run_diagnostic(output, environment_factory=None, models=None, readiness=preflight_simulator, *, mode, scenario_id=None):
    budget=BUDGETS[mode]
    plan=episode_plan(mode, scenario_id)
    scenarios={s.scenario_id:s for s in matrix()}

    output=Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output==OUTPUT_ROOT:
        raise ValueError('Output must be a new directory under diagnostics')
    output.mkdir(parents=True,exist_ok=False)
    summary=dict(status='failed',mode=mode,training_steps=0,episodes=[],budget_s=budget,planned_episodes=len(plan),
                 execution_environment='scripted' if environment_factory is not None else 'live')
    started=time.monotonic(); env=None; identity=None
    previous_alarm=None
    if environment_factory is None:
        def alarm_handler(signum, frame):
            raise TimeoutError(f'{mode} hard {budget}-second work deadline exceeded')
        previous_alarm=signal.signal(signal.SIGALRM,alarm_handler)
        signal.setitimer(signal.ITIMER_REAL,budget)
    try:
        summary['checkpoints']=verify_checkpoints()
        summary['scenario_validation']=validate_matrix()
        write(output/'manifest.json',dict(command=sys.argv,settings=SETTINGS,pilot_order=PILOT,
            episode_order=plan,max_steps=len(plan)*300,deterministic=True,goal_tolerance_m=.30,
            horizon=300,input_age_limit_s=.5,source_sha256={str(path.relative_to(ROOT)):prior.file_sha256(path)
                for path in (Path(__file__),ROOT/'navigation/jackal_diagnostic_supervisor.py',
                    ROOT/'navigation/jackal_generalization_report.py',ROOT/'configs/jackal_ppo_diagnostic_6g3_bc.yaml')},**summary))
        if environment_factory is None:
            if any(os.environ.get(k)!=v for k,v in SETTINGS.items()):
                raise RuntimeError('Runtime settings differ from documented ROS/transport settings')
        models=load_models() if models is None else models
        identity={name:(m.num_timesteps,m._n_updates,{k:t.detach().cpu().clone() for k,t in m.policy.state_dict().items()}) for name,m in models.items()}
        write(output/'model_metadata.json',{name:dict(timesteps=m.num_timesteps,n_updates=m._n_updates,
            normalizer_mean=m.policy.pi_features_extractor.trunk.obs_mean.cpu().tolist(),
            normalizer_std=m.policy.pi_features_extractor.trunk.obs_std.cpu().tolist(),
            normalizer_count=int(m.policy.pi_features_extractor.trunk.normalizer_count.item()),
            action_scaling='Existing ActionMapper only; actual mapper values captured at publication') for name,m in models.items()})
        readiness()
        base=prior.load_config(ROOT/'configs/jackal_ppo_diagnostic_6g3_bc.yaml')
        raw=(environment_factory or (lambda: _make_jackal_env(base)))()
        env=TelemetryEnv(raw,output,started+budget)
        if raw.unwrapped.max_episode_steps!=300 or raw.unwrapped.goal_tolerance!=.30:
            raise RuntimeError('Environment termination contract differs')
        for item in plan:
            name,seed=item['policy'],item['seed']
            env.reset_record=None
            env.context=dict(episode_id=item['episode_id'],policy=name,checkpoint=str(CHECKPOINTS[name][0]),
                             checkpoint_sha256=CHECKPOINTS[name][1],scenario=item['scenario'],seed=seed)
            append(output/'episode_attempts.jsonl',dict(**item,started_monotonic_ns=time.monotonic_ns()))
            record,_=run_closed_loop_episode(env,models[name],scenarios[item['scenario']],seed,seed=seed,predict=env.predict)
            record['policy']=name
            record['episode_id']=env.context['episode_id']
            record['completed']=bool(not record['error'] and record['stop_verified'] and (record['success'] or record['collision'] or record['timeout']))
            result=dict(**env.context,evaluator=prior._plain(record),reset=env.reset_record,
                telemetry_rows=len(env.rows),final_distance_m=env.rows[-1]['post']['goal_distance_m'] if env.rows else None,
                path_length_m=env.path if env.reset_record else None,
                net_progress_m=env.initial_distance-env.rows[-1]['post']['goal_distance_m'] if env.rows else None)
            if env.rows:
                result['path_efficiency']=max(0.,result['net_progress_m'])/result['path_length_m'] if result['path_length_m'] else 0.
            summary['episodes'].append(result); append(output/'episodes.jsonl',result)
            if (record['error'] or not record['completed'] or not record['stop_verified'] or record['collision']
                    or not env.rows or len(env.rows)!=record['control_steps']):
                raise RuntimeError('Diagnostic episode failed or telemetry incomplete: '+str(record.get('error')))
            env.guard()
        for name,m in models.items():
            steps,updates,before=identity[name]
            if (m.num_timesteps!=steps or m._n_updates!=updates
                    or any(not torch.equal(t,m.policy.state_dict()[k].cpu()) for k,t in before.items())):
                raise RuntimeError('Evaluation mutated checkpoint model state')
        summary['model_state_unchanged']=True
        summary['status']='completed'
    except BaseException as exc:
        summary['failure']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        append(output/'failures.jsonl',summary['failure'])
    finally:
        if identity is not None:
            summary['model_state_unchanged']=all(m.num_timesteps==identity[name][0] and m._n_updates==identity[name][1]
                and all(torch.equal(t,m.policy.state_dict()[k].cpu()) for k,t in identity[name][2].items())
                for name,m in models.items())
            if not summary['model_state_unchanged']:
                summary['status']='failed'
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous_alarm)
        if env is not None:
            summary['stop_verified'],summary['stop_error']=confirm_stop(env)
        summary['shutdown']=_shutdown(None,env)
        summary['hooks_restored']=bool(getattr(env,'hooks_restored',False)) if env is not None else None
        summary['not_started_episode_ids']=[item['episode_id'] for item in plan if item['episode_id'] not in {e['episode_id'] for e in summary['episodes']}]
        summary['runtime_s']=time.monotonic()-started
        try:
            summary['checkpoint_hashes_after']=verify_checkpoints()
        except Exception as exc:
            summary['checkpoint_verification_error']=str(exc)
            summary['status']='failed'
        if environment_factory is None and env is not None:
            try:
                from navigation.jackal_ppo_train import _topic_counts
                summary['command_topic_counts_after_close']=list(_topic_counts(env.robot.command_topic))
                if summary['command_topic_counts_after_close'][0]!=0:
                    raise RuntimeError('Command publisher remains after close')
            except Exception as exc:
                summary['cleanup_graph_error']=str(exc)
                summary['status']='failed'
        if env is not None and (not summary['stop_verified'] or summary['shutdown']['errors'] or env.callback_error or not summary['hooks_restored']):
            summary['status']='failed'
        write(output/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preview','preview-matrix','pilot','matrix'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--scenario', choices=[s.scenario_id for s in matrix()])
    args=parser.parse_args(argv)
    if args.mode in ('preview','preview-matrix'):
        mode='matrix' if args.mode=='preview-matrix' else 'pilot'
        plan=episode_plan(mode, args.scenario)
        print(json.dumps(dict(checkpoints=verify_checkpoints(),validation=validate_matrix(),
            mode=mode,episode_order=plan,pilot_order=PILOT,max_episodes=len(plan),max_steps=len(plan)*300,
            wall_time_cap_s=BUDGETS[mode],outer_timeout_s=OUTER_TIMEOUTS[mode]),indent=2));return 0
    if args.output is None:
        parser.error('--output required')
    if args.scenario is not None:
        result=run_diagnostic(args.output,mode=args.mode,scenario_id=args.scenario)
    else:
        result=(pilot(args.output) if args.mode=='pilot' else run_diagnostic(args.output,mode='matrix'))
    print(json.dumps(prior._plain(result),indent=2))
    return 0 if result['status']=='completed' else 1

if __name__=='__main__':
    raise SystemExit(main())
