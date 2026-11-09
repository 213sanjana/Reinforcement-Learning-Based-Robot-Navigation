"""Report newly measured Phase 6H.1 anchor pilot telemetry; never edits history."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def distribution(values):
    if not values:return None
    a=np.array(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),
                quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist())


def metrics(rows):
    if not rows:return dict(telemetry_rows=0)
    commands=[c['published_velocity'] for r in rows for c in r['published_commands']]
    angular=[r['actor_mean'][1] for r in rows]
    measured=[r['post']['measured_velocity']['angular_rad_s'] for r in rows]
    dt=[(r['post']['odom_stamp_ns']-r['pre']['odom_stamp_ns'])/1e9 for r in rows]
    return dict(telemetry_rows=len(rows),actor_angular=distribution(angular),
        selected_angular=distribution([r['deterministic_selected_action'][1] for r in rows]),
        published_angular_rad_s=distribution([c['angular_rad_s'] for c in commands]),
        measured_angular_rad_s=distribution(measured),
        published_linear_m_s=distribution([c['linear_m_s'] for c in commands]),
        measured_linear_m_s=distribution([r['post']['measured_velocity']['linear_m_s'] for r in rows]),
        left_action_fraction=float(np.mean(np.array(angular)>.01)),
        right_action_fraction=float(np.mean(np.array(angular)<-.01)),
        left_measured_fraction=float(np.mean(np.array(measured)>.01)),
        right_measured_fraction=float(np.mean(np.array(measured)<-.01)),
        turning_without_progress_s=float(sum(t for r,t,w in zip(rows,dt,measured) if abs(w)>.1 and r['distance_progress_m']<=.001)),
        cumulative_heading_change_rad=rows[-1]['cumulative_heading_change_rad'],
        cumulative_absolute_heading_change_rad=rows[-1]['cumulative_absolute_heading_change_rad'],
        fresh_advanced_stamps=all(r['post']['scan_stamp_ns']>r['pre']['scan_stamp_ns'] and r['post']['odom_stamp_ns']>r['pre']['odom_stamp_ns'] for r in rows),
        sampled_actions_null=all(r['sampled_action'] is None for r in rows),
        all_finite=all(np.isfinite(r['pre_action_observation']).all() and np.isfinite(r['post_action_observation']).all()
                       and np.isfinite(r['normalizer_output']).all() and np.isfinite(r['reward']) for r in rows),
        control_monotonic_ordered=all(r['control_finished_monotonic_ns']>=r['control_started_monotonic_ns'] for r in rows))


def plot_episode(path,rows,name):
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=rows[0]['pre']['odom_stamp_ns']
    t=np.array([(r['post']['odom_stamp_ns']-start)/1e9 for r in rows])
    figure,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    axes[0,0].plot(t,[r['post']['goal_distance_m'] for r in rows],label='Measured goal distance')
    axes[0,0].axhline(.3,color='gray',ls='--',label='Success tolerance')
    axes[0,0].set(title='Goal distance',ylabel='Distance (m)')
    axes[0,1].plot(t,[r['post']['goal_bearing_rad'] for r in rows],label='Measured goal bearing')
    axes[0,1].plot(t,[r['cumulative_heading_change_rad'] for r in rows],label='Accumulated measured yaw change')
    axes[0,1].set(title='Heading and goal bearing',ylabel='Angle (rad)')
    axes[1,0].plot(t,[r['actor_mean'][1] for r in rows],label='Actor angular mean')
    axes[1,0].plot(t,[r['clipped_normalized_action'][1] for r in rows],ls='--',label='Clipped deterministic action')
    axes[1,0].set(title='Policy angular action',ylabel='Normalized action')
    events=[c for r in rows for c in r['published_commands']]
    ct=[(c['ros_stamp_ns']-start)/1e9 for c in events]
    for ax,key,label in ((axes[1,1],'angular_rad_s','Angular velocity'),(axes[2,0],'linear_m_s','Linear velocity')):
        ax.scatter(ct,[c['published_velocity'][key] for c in events],s=5,label='All published commands, including stops')
        ax.plot(t,[r['post']['measured_velocity'][key] for r in rows],lw=1,label='Measured odometry')
        ax.set(title=label,ylabel='Velocity ('+('rad/s' if key=='angular_rad_s' else 'm/s')+')')
    axes[2,1].plot(t,[max(abs(x) for x in r['normalizer_output']) for r in rows],label='Maximum absolute z-score')
    axes[2,1].set(title='Frozen normalizer output',ylabel='Absolute standard deviations')
    for ax in axes.flat:
        ax.set_xlabel('Simulation time since first pre-action observation (s)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    figure.suptitle(name+' · source: new pilot telemetry only',fontsize=12)
    figure.savefig(path,dpi=140);plt.close(figure)


def report(run,validation):
    """Audit the manifest's episode set, then report only recorded evidence."""
    import hashlib
    from collections import Counter
    p=Path(run).resolve()
    if (p/'report.md').exists() or (p/'analysis.json').exists():
        raise FileExistsError('Refusing to overwrite diagnostic analysis/report')
    summary=json.loads((p/'summary.json').read_text())
    manifest=json.loads((p/'manifest.json').read_text())
    plan=manifest.get('episode_order')
    if plan is None:  # Read-only compatibility with historical pilot manifests.
        plan=[dict(episode_id=f'anchor_{name}_{seed}',scenario='anchor',policy=name,seed=seed)
              for name,seed in manifest['pilot_order']]
    supervisor=p.with_name(p.name+'_supervisor')
    cleanup=json.loads((supervisor/'cleanup.json').read_text()) if (supervisor/'cleanup.json').exists() else None
    telemetry=read_rows(p/'telemetry.jsonl');intents=read_rows(p/'action_intents.jsonl')
    attempts=read_rows(p/'transition_attempts.jsonl');commands=read_rows(p/'published_commands.jsonl')
    grouped=defaultdict(list);ig=defaultdict(list);ag=defaultdict(list)
    for r in telemetry:grouped[r['episode_id']].append(r)
    for r in intents:ig[r['intent_id']].append(r)
    for r in attempts:ag[(r['episode_id'],r['step'])].append(r)
    expected_ids=[e['episode_id'] for e in plan]
    recorded=summary['episodes'];records={e['episode_id']:e for e in recorded}
    checks=dict(unique_plan=len(set(expected_ids))==len(expected_ids),unique_records=len(records)==len(recorded),
        no_unplanned_records=set(records).issubset(expected_ids),
        no_orphan_telemetry=set(grouped).issubset(records),
        prefix_order=[e['episode_id'] for e in recorded]==expected_ids[:len(recorded)])
    episodes=[]
    def oid(v):
        a=np.asarray(v,dtype=np.float32)
        return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
    def sid(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
    for item in plan:
        key=item['episode_id'];e=records.get(key);rr=grouped[key]
        if e is None:
            episodes.append(dict(**item,status='not_started',valid=False));continue
        ev=e['evaluator'];reset=e.get('reset') or {};errors=[]
        if not ev.get('completed') or ev.get('error') or not ev.get('stop_verified'):errors.append('incomplete episode or failed stop')
        if len(rr)!=ev.get('control_steps') or not rr:errors.append('incomplete telemetry')
        if reset.get('position_error_m',float('inf'))>.05 or reset.get('yaw_error_rad',float('inf'))>.05:errors.append('reset pose/yaw invalid')
        max_age=None
        try:
            for r in rr:
                fc=r['freshness_checks']
                if [c['stage'] for c in fc]!=['before_inference','after_inference','after_publisher_verification','immediately_before_dispatch']:
                    raise ValueError('missing freshness stage')
                for c in fc:
                    if c['source']!=r['pre']:raise ValueError('freshness source mismatch')
                    if any(isinstance(c['source'][k],bool) or not isinstance(c['source'][k],int) or c['source'][k]<=0 for k in ['scan_stamp_ns','odom_stamp_ns','capture_monotonic_ns']):raise ValueError('invalid timestamp provenance')
                    ages=[v+(c['checked_monotonic_ns']-c['source']['capture_monotonic_ns'])/1e9 for v in c['source']['sensor_ages_s']]
                    if not c['valid'] or c['sensor_age_limit_s']!=.5 or not np.allclose(ages,c['current_sensor_ages_s'],atol=1e-12,rtol=0) or min(ages)<0 or max(ages)>.5:raise ValueError('freshness invalid')
                max_age=max(max_age or 0,max(fc[-1]['current_sensor_ages_s']))
                if sid(r['pre'])!=r['snapshot_identity'] or oid(r['pre_action_observation'])!=r['observation_identity']:raise ValueError('source identity mismatch')
                lifecycle=ig[r['intent_id']]
                if Counter(i['status'] for i in lifecycle)!=Counter(prepared=1,dispatch_candidate=1,raw_control_called=1):raise ValueError('intent lifecycle incomplete')
                if not r['raw_control_call_occurred'] or r['sampled_action'] is not None:raise ValueError('dispatch/determinism mismatch')
                if not np.array_equal(np.clip(r['deterministic_selected_action'],-1,1).astype(np.float32),np.asarray(r['clipped_normalized_action'],dtype=np.float32)):raise ValueError('action mismatch')
                at=ag[(key,r['step'])]
                if len(at)!=1 or not np.array_equal(at[0]['returned_observation'],r['post_action_observation']):raise ValueError('transition observation mismatch')
                for sensor in ['scan','odom']:
                    stamp=sensor+'_stamp_ns'
                    if r['post'][stamp]<=r['pre'][stamp] or abs(r['post'][stamp]/1e9-at[0]['environment_info'][sensor+'_stamp'])>1e-6:raise ValueError('timestamp mismatch')
                graph=r['publisher_verification'];pub=graph['publishers']
                if len(pub)!=1 or pub[0]['node_name']!='jackal_rl_interface' or pub[0]['node_namespace']!='/' or graph['subscriber_count']<1:raise ValueError('publisher mismatch')
            diag=metrics(rr)
            if rr and not diag['all_finite']:raise ValueError('nonfinite telemetry')
        except (KeyError,TypeError,ValueError) as exc:errors.append(str(exc));diag={}
        flags=[bool(ev.get(k)) for k in ['success','collision','timeout']]
        if sum(flags)!=1 or (ev.get('success') and ev.get('termination_reason')!='goal_reached'):errors.append('outcome flags inconsistent')
        episodes.append(dict(**item,status='valid' if not errors else 'invalid',valid=not errors,errors=errors,
            success=bool(ev.get('success')),collision=bool(ev.get('collision')),timeout=bool(ev.get('timeout')),
            completed=bool(ev.get('completed')),termination_reason=ev.get('termination_reason'),
            control_steps=ev.get('control_steps'),wall_time_s=ev.get('wall_time_s'),
            initial_distance_m=ev.get('initial_goal_distance_m'),final_distance_m=e.get('final_distance_m'),
            net_progress_m=e.get('net_progress_m'),aligned_snapshot_path_m=e.get('path_length_m'),
            evaluator_pose_path_m=ev.get('path_length_m'),progress_path_ratio=e.get('path_efficiency'),
            max_dispatch_input_age_s=max_age,reset=reset,stop_verified=ev.get('stop_verified'),steering=diag))
    telemetry_intent_ids={r['intent_id'] for r in telemetry}
    checks['no_orphan_raw_calls']=all(k in telemetry_intent_ids for k,v in ig.items() if any(i['status']=='raw_control_called' for i in v))
    checks['publication_associations']=all(c.get('action_intent_id') in ig for c in commands if c.get('action_dispatch_state')=='raw_control_call')
    checks['all_planned_valid']=all(e['valid'] for e in episodes)
    checks['cleanup']=bool(cleanup is not None and not cleanup['remaining_owned_pids'] and summary.get('stop_verified') and summary['shutdown']['closed'] and not summary['shutdown']['errors'] and summary.get('command_topic_counts_after_close',[1])[0]==0)
    checks['frozen_model']=summary.get('model_state_unchanged') is True and summary.get('training_steps')==0
    checks['hooks']=summary.get('hooks_restored') is True if 'hooks_restored' in summary else None
    checks['checkpoint_hashes']=bool(summary.get('checkpoint_hashes_after'))
    for name,c in summary.get('checkpoints',{}).items():
        checks['checkpoint_hashes'] &= (hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']==summary.get('checkpoint_hashes_after',{}).get(name,{}).get('sha256'))
    complete=summary['status']=='completed' and all(v for v in checks.values() if v is not None)
    comparisons=[]
    for scenario in dict.fromkeys(e['scenario'] for e in plan):
        for seed in [642,643]:
            pair={e['policy']:e for e in episodes if e['scenario']==scenario and e['seed']==seed}
            delta=None
            if all(pair.get(k,{}).get('valid') for k in ['initial','final']):
                fields=['control_steps','final_distance_m','net_progress_m','aligned_snapshot_path_m','evaluator_pose_path_m']
                delta={k:pair['final'][k]-pair['initial'][k] for k in fields}
                delta['success']=int(pair['final']['success'])-int(pair['initial']['success'])
            comparisons.append(dict(scenario=scenario,seed=seed,final_minus_initial=delta))
    aggregates=[]
    for scenario in ['ALL']+list(dict.fromkeys(e['scenario'] for e in plan)):
        for policy in ['initial','final']:
            group=[e for e in episodes if e['policy']==policy and (scenario=='ALL' or e['scenario']==scenario)]
            valid=[e for e in group if e['valid']]
            aggregates.append(dict(scenario=scenario,policy=policy,planned=len(group),attempted=sum(e['status']!='not_started' for e in group),valid=len(valid),invalid=sum(e['status']=='invalid' for e in group),not_started=sum(e['status']=='not_started' for e in group),
                **{k+'_rate_valid':sum(e[k] for e in valid)/len(valid) if valid else None for k in ['success','collision','timeout']},
                **{k+'_valid':distribution([e[k] for e in valid if e[k] is not None]) for k in ['aligned_snapshot_path_m','evaluator_pose_path_m','net_progress_m','control_steps','wall_time_s']}))
    analysis=dict(instrumentation_gate_passed=bool(complete),checks=checks,episodes=episodes,paired_comparisons=comparisons,aggregates=aggregates,
        intent_status_counts=dict(Counter(i['status'] for i in intents)),publication_states=dict(Counter(c.get('action_dispatch_state') for c in commands)),
        summary=summary,cleanup=cleanup,manifest=manifest,validation_reference=str(Path(validation).resolve()))
    lines=[f"# Diagnostic {summary.get('mode','pilot')} report",'',f"Run status: {summary['status']}. Instrumentation gate passed: {complete}. Planned: {len(plan)}; recorded: {len(recorded)}.",'',
        'All metrics are recorded evidence. Clean termination is distinct from reaching the goal. No training is performed. Full per-episode, aggregate and paired policy comparisons are in analysis.json. Rates use valid-episode denominators and retain invalid/not-started counts.','',
        f"Checkpoint identities: `{summary.get('checkpoint_hashes_after',summary.get('checkpoints'))}`.",f"Runtime: {summary['runtime_s']:.3f} s; worker budget: {summary.get('budget_s')} s. Cleanup: `{cleanup}`.",f"Integrity checks: `{checks}`.",'']
    for e in episodes:
        lines += [f"## {e['episode_id']}",'',json.dumps(e,sort_keys=True),'']
        if grouped[e['episode_id']]:plot_episode(p/(e['episode_id']+'.png'),grouped[e['episode_id']],e['episode_id'])
    lines += ['## Interpretation and limitations','',
        'Compare both policies per scenario and matched seed using final-minus-initial differences in analysis.json. Aligned-snapshot path sums observation-bound odometry displacements; evaluator path uses its separately sampled pose sequence. Do not merge them. Progress/path ratio is net goal-distance reduction divided by recorded path, not an optimal-path ratio.',
        'Actor actions, actual ROS publications and measured odometry are separate. A publish is not actuator acknowledgement. Translations change position and LiDAR context together. Nonzero-yaw resets are supported only for episodes with verified measured yaw; do not extrapolate beyond recorded conditions. Two seeds provide descriptive evidence, not statistical significance.',
        'Publisher discovery is sampled. Python final freshness-check/call entry is not atomic against scheduler suspension. Age depends on original receive provenance. A failed telemetry sink cannot guarantee durable rejection evidence. Historical hook identity was not independently asserted; current runs record hook restoration.',
        f"Recorded failure: `{summary.get('failure')}`. Invalid and not-started episodes are not successes or navigation timeouts.",
        'Decision: review the measured policy comparisons before proposing further work. No automatic additional execution or training.' if complete else 'Decision: resolve recorded integrity, completeness or cleanup failures before another separately authorized experiment. Do not retry automatically.','']
    (p/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
    (p/'report.md').write_text('\n'.join(lines))
    return bool(complete)


def plot_commands(path,commands):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    start=commands[0]['monotonic_ns']
    t=[(c['monotonic_ns']-start)/1e9 for c in commands]
    for ax,key,unit in ((axes[0],'linear_m_s','m/s'),(axes[1],'angular_rad_s','rad/s')):
        ax.scatter(t,[c['published_velocity'][key] for c in commands],s=18,label='Published Twist events, including stops')
        ax.set(title=key,xlabel='Monotonic time since first captured publish (s)',ylabel='Published velocity ('+unit+')')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    figure.suptitle('Aborted first episode · measured publications only · source: published_commands.jsonl',fontsize=11)
    figure.savefig(path,dpi=140);plt.close(figure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--validation',required=True)
    args=parser.parse_args();report(args.run,args.validation)

if __name__=='__main__':main()
