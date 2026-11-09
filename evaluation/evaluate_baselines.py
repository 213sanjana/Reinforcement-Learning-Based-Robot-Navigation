"""Phase 5B: paired four-baseline evaluation with separately measured latency."""
import argparse
import csv
import inspect
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
from environments.fixed_map_env import FixedMapNavigationEnv
from environments.rewards import RewardConfig
from environments.map_generator import shortest_path
from evaluation.scenario_manifest import validate, contract, file_hash, digest, save_json
from evaluation.result_schema import EpisodeResult, summarize
from evaluation.baseline_adapters import load_baseline, action_observation

NAMES = ('random','astar','dqn','bc')
EXTRA = ('path_length','invalid_actions','repeated_invalid_actions','max_invalid_streak',
         'inference_seconds','environment_step_seconds','mean_inference_ms','mean_environment_step_ms')


def synchronize(policy):
    device = getattr(policy,'device',None)
    if device is not None and device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def rollout(policy, name, scenario, random_seed):
    s = scenario
    env = FixedMapNavigationEnv(np.asarray(s['obstacles']),tuple(s['goal']),
        max_steps=s['horizon'], bfs_shaping=False, reward_config=RewardConfig(**contract()['reward']))
    try:
        obs,_ = env.reset(seed=s['map_seed'],options={'start':s['start']})
        seed = np.random.SeedSequence([random_seed,s['map_seed']])
        # No scenario object or metric labels reach reset or act.
        policy.reset(seed)
        reward_sum=0.; collisions=0; moves=0; repeats=0; streak=0; longest=0
        previous_invalid=None; inference_ns=0; step_ns=0
        for steps in range(1,s['horizon']+1):
            current = tuple(obs['agent'])
            observation = action_observation(obs)
            synchronize(policy)
            before=time.perf_counter_ns()
            action=policy.act(observation)
            synchronize(policy)
            inference_ns += time.perf_counter_ns()-before
            if not env.action_space.contains(action):
                raise ValueError('Policy returned out-of-range action')
            before=time.perf_counter_ns()
            obs,reward,term,trunc,info=env.step(action)
            step_ns += time.perf_counter_ns()-before
            collision=bool(info['collision']); collisions+=int(collision)
            moves+=int(tuple(obs['agent']) != current)
            if collision:
                key=(current,int(action))
                repeats+=int(key == previous_invalid)
                previous_invalid=key; streak+=1; longest=max(longest,streak)
            else:
                previous_invalid=None; streak=0
            reward_sum+=reward
            if term or trunc:
                efficiency=s['optimal_steps']/steps if term else None
                row=EpisodeResult(name,s['scenario_id'],s['map_id'],s['map_seed'],s['density'],
                    s['horizon'],bool(term),bool(trunc),steps,collisions,float(reward_sum),
                    s['optimal_steps'],efficiency,efficiency or 0.).to_dict()
                row.update(path_length=moves,invalid_actions=collisions,repeated_invalid_actions=repeats,
                    max_invalid_streak=longest,inference_seconds=inference_ns/1e9,
                    environment_step_seconds=step_ns/1e9,
                    mean_inference_ms=inference_ns/1e6/steps,
                    mean_environment_step_ms=step_ns/1e6/steps)
                check_row(row)
                return row
        raise RuntimeError('Environment did not terminate at horizon')
    finally:
        env.close()


def check_row(row):
    fields=EpisodeResult.__dataclass_fields__
    EpisodeResult(**{k:row[k] for k in fields}).to_dict()
    if row['path_length']+row['collisions'] != row['episode_length'] or row['invalid_actions']!=row['collisions']:
        raise ValueError('Movement/collision accounting mismatch')
    if not 0 <= row['repeated_invalid_actions'] <= row['collisions'] or not 0 <= row['max_invalid_streak'] <= row['collisions']:
        raise ValueError('Invalid repetition metrics')
    for key in EXTRA[4:]:
        if not math.isfinite(row[key]) or row[key]<0:
            raise ValueError('Invalid timing')
    for total,mean in [('inference_seconds','mean_inference_ms'),('environment_step_seconds','mean_environment_step_ms')]:
        if not math.isclose(row[total]*1000/row['episode_length'],row[mean],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Timing total/mean mismatch')
    if row['policy']=='astar' and (not row['success'] or row['collisions'] or row['episode_length']!=row['optimal_steps'] or row['spl']!=1):
        raise ValueError('A* is not collision-free and optimal')


def summaries(rows):
    result={}
    for name in NAMES:
        result[name]={}
        for density in (.1,.3,.5):
            group=[r for r in rows if r['policy']==name and r['density']==density]
            base=summarize(group); n=len(group); steps=sum(r['episode_length'] for r in group)
            base.update(timeout_rate=sum(r['timeout'] for r in group)/n,
                total_collisions=sum(r['collisions'] for r in group),
                mean_path_length=sum(r['path_length'] for r in group)/n,
                total_repeated_invalid_actions=sum(r['repeated_invalid_actions'] for r in group),
                inference_seconds=sum(r['inference_seconds'] for r in group),
                environment_step_seconds=sum(r['environment_step_seconds'] for r in group))
            base['mean_inference_ms_per_action']=base['inference_seconds']*1000/steps
            base['mean_environment_step_ms_per_action']=base['environment_step_seconds']*1000/steps
            result[name][str(density)]=base
    return result


def check_pairing(rows,manifest):
    scenarios=manifest['scenarios']
    if len(rows)!=len(scenarios)*len(NAMES): raise ValueError('Incomplete baseline results')
    for name in NAMES:
        group=[r for r in rows if r['policy']==name]
        if len(group)!=len(scenarios): raise ValueError('Missing policy episodes')
        for r,s in zip(group,scenarios):
            check_row(r)
            for k in ('scenario_id','map_id','map_seed','density','horizon','optimal_steps'):
                if r[k]!=s[k]: raise ValueError('Unpaired scenario metadata')


def read_csv(path):
    ints={'map_seed','horizon','episode_length','collisions','optimal_steps',*EXTRA[:4]}
    strings={'policy','scenario_id','map_id'}
    rows=[]
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            parsed={}
            for k,v in r.items():
                if k in strings: parsed[k]=v
                elif k in ('success','timeout'):
                    if v not in ('True','False'): raise ValueError('Invalid CSV boolean')
                    parsed[k]=v=='True'
                elif k in ints: parsed[k]=int(v)
                else: parsed[k]=None if v=='' else float(v)
            rows.append(parsed)
    return rows


def validate_results(manifest,output):
    output=Path(output)
    if (output/'INCOMPLETE.txt').exists(): raise ValueError('Incomplete benchmark run')
    report=json.loads((output/'summary.json').read_text())
    if report['schema_version']!=1 or report['manifest_id']!=manifest['manifest_id']:
        raise ValueError('Report/manifest mismatch')
    if file_hash(output/'episodes.csv')!=report['episodes_csv_sha256']:
        raise ValueError('Episode CSV checksum mismatch')
    if report['policies']!=list(NAMES) or report['scenarios_per_policy']!=len(manifest['scenarios']):
        raise ValueError('Report policy/scenario count mismatch')
    for name in NAMES:
        entry=report['provenance'][name]
        if not entry['weights_unchanged'] or entry['state_before']!=entry['state_after']:
            raise ValueError('Policy mutation recorded in report')
        if name in ('dqn','bc'):
            expected=manifest['config']['policies'][name]['checkpoint']
            meta=entry['checkpoint']
            if meta['path']!=expected or file_hash(expected)!=meta['sha256']:
                raise ValueError('Checkpoint provenance mismatch')
    rows=read_csv(output/'episodes.csv'); check_pairing(rows,manifest)
    if summaries(rows)!=report['summary']: raise ValueError('Summary does not match CSV')
    return dict(valid=True,episodes=len(rows),scenarios_per_policy=len(manifest['scenarios']),
                astar_optimal=True)


def evaluate(manifest,output,device='cpu',random_seed=5102026,torch_threads=1):
    output=Path(output)
    if output.exists(): raise FileExistsError('Choose a new output directory: '+str(output))
    if random_seed<0: raise ValueError('Random seed must be nonnegative')
    if torch_threads<1: raise ValueError('Torch threads must be positive')
    import torch
    torch.set_num_threads(torch_threads)
    validate(manifest)
    # Independently establish scoring reference; never passed to policy interfaces.
    for s in manifest['scenarios']:
        route=shortest_path(np.asarray(s['obstacles']),tuple(s['start']),tuple(s['goal']))
        if route is None or len(route)-1!=s['optimal_steps']: raise ValueError('BFS reference mismatch')
    for name in ('dqn','bc'):
        path=Path(manifest['config']['policies'][name]['checkpoint'])
        if not path.is_file(): raise FileNotFoundError('Required checkpoint missing: '+str(path))
    output.mkdir(parents=True,exist_ok=False)
    rows=[]; provenance={}
    try:
        for name in NAMES:
            print('Evaluating '+name,flush=True)
            before_load=time.perf_counter()
            policy,meta=load_baseline(name,manifest,device)
            load_seconds=time.perf_counter()-before_load
            before=policy.fingerprint()
            # One untimed neural warmup on current observation only; no stepping.
            if name in ('dqn','bc'):
                s=manifest['scenarios'][0]
                obs=dict(obstacles=np.asarray(s['obstacles']),agent=np.asarray(s['start']),goal=np.asarray(s['goal']))
                policy.act(action_observation(obs)); synchronize(policy)
            for s in manifest['scenarios']:
                rows.append(rollout(policy,name,s,random_seed))
            if policy.fingerprint()!=before: raise RuntimeError('Policy state changed')
            if hasattr(policy,'verify'): policy.verify()
            provenance[name]=dict(checkpoint=meta,device=str(getattr(policy,'device','cpu')),load_seconds=load_seconds,
                state_before=before,state_after=policy.fingerprint(),weights_unchanged=True)
        check_pairing(rows,manifest)
        with (output/'episodes.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        import evaluation.baseline_adapters as adapters
        import evaluation.evaluate_baselines as runner
        report=dict(schema_version=1,manifest_id=manifest['manifest_id'],
            scenarios_per_policy=len(manifest['scenarios']),policies=list(NAMES),random_seed=random_seed,
            contract=contract(),provenance=provenance,leakage_scope=manifest['leakage_scope'],
            source_fingerprints={m.__name__:digest(inspect.getsource(m)) for m in (adapters,runner)},
            runtime=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads(),
                         requested_device=device,actual_devices={n:provenance[n].get('device') for n in NAMES}),
            timing='perf_counter_ns; CUDA synchronized; load, reset, observation-copy and warmup excluded; neural encoding included; A* replans each action',
            collision_semantics='Each blocked boundary/obstacle move is one collision and one invalid action. Repeated invalid actions count consecutive identical (state, action) blocked attempts after the first; not an additional collision category.',
            path_semantics='path_length counts successful cell moves; episode_length counts all attempted moves; efficiency/SPL denominator uses episode_length to penalize collisions',
            summary=summaries(rows),episodes_csv_sha256=file_hash(output/'episodes.csv'))
        save_json(output/'summary.json',report)
        return validate_results(manifest,output)
    except BaseException:
        with (output/'INCOMPLETE.txt').open('x') as f: f.write('Run failed; retain for diagnosis and use a fresh directory.\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--random-seed',type=int,default=5102026)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--torch-threads',type=int,default=1)
    a=p.parse_args();m=json.loads(Path(a.manifest).read_text())
    if a.validate_only:
        validate(m);result=validate_results(m,a.output)
    else: result=evaluate(m,a.output,a.device,a.random_seed,a.torch_threads)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
