"""Paired held-out BC/A*/random benchmark. New scenarios; weights never updated."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from environments.fixed_map_env import FixedMapNavigationEnv
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path
from imitation.bc_network import load_bc_checkpoint
from imitation.train_bc import seed_everything
from imitation.dataset import encode_observation, load_dataset, map_identity
from imitation.expert_generator import generate_dataset, historical_map_ids
from planners.astar import ACTIONS, astar, path_to_actions
from visualization.bc_routes import plot_paired_routes, plot_benchmark_summary


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def reserved_layout_ids(paths):
    """Read only map identity arrays and metadata, not held-out test actions."""
    seen=set()
    for path in paths:
        with np.load(path,allow_pickle=False) as source:
            grids=source['obstacles']
            ids=source['map_ids'].tolist()
            computed=[map_identity(grid) for grid in grids]
            if computed != ids:
                raise ValueError(f'Invalid reserved map identity: {path}')
            seen.update(ids)
    return seen


def detect_loops(route,actions,min_repeats=3,max_period=12):
    """Flag repeated state/action transition blocks; straight-line movement is not a loop."""
    if len(route) != len(actions)+1:
        raise ValueError('route/action length mismatch')
    transitions=[(tuple(route[i]),int(a),tuple(route[i+1])) for i,a in enumerate(actions)]
    first,period=None,None
    for end in range(min_repeats,len(transitions)+1):
        for size in range(1,min(max_period,end//min_repeats)+1):
            block=transitions[end-size:end]
            if all(transitions[end-(r+1)*size:end-r*size] == block for r in range(1,min_repeats)):
                first,period=end,size
                break
        if first is not None:
            break
    longest=run=0
    previous=None
    for action in actions:
        run=run+1 if action == previous else 1
        longest=max(longest,run); previous=action
    counts=Counter(tuple(p) for p in route)
    return dict(loop_detected=first is not None,loop_first_step=first,loop_period=period,
                loop_kind=None if period is None else ('stationary_action' if period == 1 else 'cycle'),
                max_consecutive_identical_actions=longest,max_state_visits=max(counts.values()),
                unique_positions=len(counts),revisited_steps=len(route)-len(counts))


def run_episode(env,start,policy,optimal_steps,scenario_index,map_seed,map_id,policy_name):
    obs,_=env.reset(options={'start':start})
    route=[obs['agent'].tolist()]
    actions=[]; reward_sum=0.; collisions=0
    while True:
        action=int(policy(obs))
        if not env.action_space.contains(action):
            raise ValueError('Policy emitted invalid action index')
        obs,reward,terminated,truncated,info=env.step(action)
        actions.append(action); route.append(obs['agent'].tolist())
        reward_sum+=reward; collisions+=int(info['collision'])
        if terminated or truncated:
            break
    steps=len(actions)
    loops=detect_loops(route,actions)
    return dict(scenario_index=int(scenario_index),map_seed=int(map_seed),map_id=str(map_id),policy=policy_name,
                start=list(map(int,start)),goal=obs['goal'].tolist(),
                start_row=int(start[0]),start_column=int(start[1]),
                goal_row=int(obs['goal'][0]),goal_column=int(obs['goal'][1]),success=bool(terminated),
                timeout=bool(truncated),steps=steps,collisions=collisions,reward=float(reward_sum),
                optimal_steps=int(optimal_steps),
                successful_path_efficiency=float(optimal_steps/steps) if terminated else None,
                spl=float(terminated)*optimal_steps/max(optimal_steps,steps),
                route=route,actions=actions,**loops)


def summarize(rows):
    successful=[r for r in rows if r['success']]
    count=len(rows)
    return dict(episodes=count,successes=len(successful),success_rate=len(successful)/count,
                timeout_rate=sum(r['timeout'] for r in rows)/count,
                total_collisions=sum(r['collisions'] for r in rows),
                mean_collisions=float(np.mean([r['collisions'] for r in rows])),
                mean_steps=float(np.mean([r['steps'] for r in rows])),
                mean_successful_steps=float(np.mean([r['steps'] for r in successful])) if successful else None,
                mean_reward=float(np.mean([r['reward'] for r in rows])),
                mean_successful_path_efficiency=float(np.mean([r['successful_path_efficiency'] for r in successful])) if successful else None,
                mean_spl=float(np.mean([r['spl'] for r in rows])),
                loop_episode_rate=sum(r['loop_detected'] for r in rows)/count)


def check_scenarios(arrays,metadata,density,episodes,excluded,height,width,max_steps):
    if metadata.get('split') != 'benchmark':
        raise ValueError('Only separate benchmark scenarios are accepted, not train/validation/test assets')
    if metadata.get('channels') != ['obstacles','agent','goal'] or metadata.get('actions') != [list(a) for a in ACTIONS]:
        raise ValueError('Scenario encoding/action contract mismatch')
    environment=metadata['environment']
    for key,expected in [('height',height),('width',width),('obstacle_density',density),('max_steps',max_steps),('bfs_shaping',False)]:
        if environment.get(key) != expected:
            raise ValueError(f'Scenario environment mismatch: {key}')
    if len(arrays['map_ids']) != episodes:
        raise ValueError('Scenario episode count mismatch')
    seen=set(excluded)
    regeneration=RobotNavigationEnv(**environment)
    try:
        for i,grid in enumerate(arrays['obstacles']):
            identity=map_identity(grid)
            if identity != arrays['map_ids'][i] or identity in seen:
                raise ValueError('Duplicate, reserved, or invalid benchmark map identity')
            seen.add(identity)
            regenerated,_=regeneration.reset(seed=int(arrays['map_seeds'][i]))
            if not np.array_equal(regenerated['obstacles'],grid) or not np.array_equal(regenerated['agent'],arrays['starts'][i]) or not np.array_equal(regenerated['goal'],arrays['goals'][i]):
                raise ValueError('Scenario map seed does not reproduce the saved task')
            if int(np.asarray(grid).sum()) != int(round(height*width*density)):
                raise ValueError('Actual obstacle density mismatch')
            start,goal=tuple(arrays['starts'][i]),tuple(arrays['goals'][i])
            path=shortest_path(grid,start,goal)
            if path is None or len(path)-1 != arrays['path_lengths'][i] or not 1 <= len(path)-1 <= max_steps:
                raise ValueError('Invalid benchmark task or shortest length')
    finally:
        regeneration.close()


def benchmark(checkpoint,train_dataset,validation_dataset,reserved_test_dataset,
              density=.1,episodes=100,seed=None,output_directory=None,device='auto',
              scenario_file=None,random_seed=90211,route_count=6):
    if not 0 < density < 1 or episodes < 1 or route_count < 0 or random_seed < 0:
        raise ValueError('Invalid density, episode, route count, or random seed')
    seed=9000000+int(round(density*100))*100000 if seed is None else int(seed)
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    label=f'{density:.2f}'.replace('.','p')
    output=Path(output_directory or f'artifacts/imitation/bc_benchmark/density_{label}')
    if output.exists():
        raise FileExistsError(f'{output} exists; select a new output directory')
    seed_everything(random_seed,deterministic=True)
    checkpoint_hash=file_sha256(checkpoint)
    model,payload=load_bc_checkpoint(checkpoint,device)
    model.eval()
    device=next(model.parameters()).device
    before=model_sha256(model)
    if payload.get('channels') != ['obstacles','agent','goal'] or payload.get('actions') != [list(a) for a in ACTIONS]:
        raise ValueError('Checkpoint observation/action contract mismatch')
    baseline_environment=payload['environment']
    if baseline_environment.get('bfs_shaping',False):
        raise ValueError('Benchmark currently requires the audited Manhattan reward baseline')
    height,width=model.height,model.width
    max_steps=int(baseline_environment['max_steps'])
    for name,path in [('train',train_dataset),('validation',validation_dataset)]:
        record=payload['dataset_provenance'][name]
        if file_sha256(path) != record['sha256']:
            raise ValueError(f'{name} dataset differs from checkpoint provenance')
        with np.load(path,allow_pickle=False) as source:
            meta=json.loads(str(source['metadata']))
            if meta.get('split') != name:
                raise ValueError(f'Incorrect reserved {name} dataset split')
    with np.load(reserved_test_dataset,allow_pickle=False) as source:
        if json.loads(str(source['metadata'])).get('split') != 'test':
            raise ValueError('Incorrect reserved test dataset split')
    excluded=reserved_layout_ids([train_dataset,validation_dataset,reserved_test_dataset])
    excluded |= historical_map_ids(dict(height=height,width=width,max_steps=max_steps,bfs_shaping=False))
    output.mkdir(parents=True,exist_ok=False)
    if scenario_file is None:
        scenario_path=output/'scenarios.npz'
        generate_dataset(scenario_path,episodes=episodes,density=density,seed=seed,split='benchmark',
                         height=height,width=width,max_steps=max_steps,excluded_ids=excluded)
    else:
        scenario_path=Path(scenario_file)
    arrays,metadata=load_dataset(scenario_path)
    check_scenarios(arrays,metadata,density,episodes,excluded,height,width,max_steps)
    records=[]
    scenario_records=[]
    with torch.inference_mode():
        for i,grid in enumerate(arrays['obstacles']):
            start,goal=tuple(arrays['starts'][i]),tuple(arrays['goals'][i])
            expert=astar(grid,start,goal)
            if expert is None or len(expert)-1 != int(arrays['path_lengths'][i]):
                raise RuntimeError('A* and BFS reference mismatch')
            expert_actions=iter(path_to_actions(expert))
            rng=np.random.default_rng(np.random.SeedSequence([random_seed,int(arrays['map_seeds'][i])]))
            def bc(obs):
                x=torch.from_numpy(encode_observation(obs)).unsqueeze(0).to(device)
                return int(model(x).argmax(1).item())
            policies={'bc':bc,'astar':lambda obs:next(expert_actions),'random':lambda obs:int(rng.integers(4))}
            paired={}
            for name,policy in policies.items():
                env=FixedMapNavigationEnv(grid,goal,max_steps=max_steps,bfs_shaping=False)
                try:
                    row=run_episode(env,start,policy,int(arrays['path_lengths'][i]),i,
                                    arrays['map_seeds'][i],arrays['map_ids'][i],name)
                finally:
                    env.close()
                row['density']=density
                if name == 'astar' and (not row['success'] or row['collisions'] or row['steps'] != row['optimal_steps']):
                    raise RuntimeError('A* baseline replay failed')
                records.append(row); paired[name]=row
            scenario_records.append(paired)
    after=model_sha256(model)
    if before != after or checkpoint_hash != file_sha256(checkpoint):
        raise RuntimeError('Model/checkpoint changed during evaluation')
    summary={name:summarize([r for r in records if r['policy']==name]) for name in ('bc','astar','random')}
    scalar_keys=[key for key in records[0] if key not in ('route','actions','start','goal')]
    with (output/'episodes.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=scalar_keys)
        writer.writeheader(); writer.writerows({k:r[k] for k in scalar_keys} for r in records)
    with (output/'summary.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['policy',*summary['bc']])
        writer.writeheader(); writer.writerows(dict(policy=name,**values) for name,values in summary.items())
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,checkpoint_epoch=int(payload['epoch']),
                model_sha256_before=before,model_sha256_after=after,weights_unchanged=True,
                scenarios=str(scenario_path),scenario_sha256=file_sha256(scenario_path),
                scenario_seed_start=metadata['seed_start'],random_action_seed=random_seed,
                density=density,episodes=episodes,environment=metadata['environment'],
                evaluation_role='Development benchmark on fresh held-out maps; reserved test dataset excluded',
                scenario_metadata=metadata,excluded_layout_count=len(excluded),reserved_layout_overlap=0,
                reward_protocol='Unchanged default Manhattan shaping; no action masking or loop intervention',
                distribution='40% optimal lengths 6-9, 40% 10-14, 20% 15+; reachable random tasks',
                distribution_shift=density != baseline_environment['obstacle_density'],
                loop_protocol='Three repeats of a state/action transition block, periods 1-12; report only',
                summary=summary,results=records)
    selected=[i for i,paired in enumerate(scenario_records) if not paired['bc']['success']]
    selected += [i for i in range(episodes) if i not in selected]
    for i in selected[:route_count]:
        plot_paired_routes(arrays['obstacles'][i],arrays['starts'][i],arrays['goals'][i],scenario_records[i],
                           output/'routes'/f'scenario_{i:04d}.png',
                           title=f"Density {density:.2f}; scenario {i}; map seed {int(arrays['map_seeds'][i])}")
    plot_benchmark_summary(summary,output/'comparison.png')
    report['route_scenario_indices']=selected[:route_count]
    with (output/'results.json').open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    return {k:v for k,v in report.items() if k not in ('results','scenario_metadata')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='artifacts/imitation/behavior_cloning/best.pt')
    parser.add_argument('--train-dataset',default='artifacts/imitation/expert_density10_v1/train.npz')
    parser.add_argument('--validation-dataset',default='artifacts/imitation/expert_density10_v1/validation.npz')
    parser.add_argument('--reserved-test-dataset',default='artifacts/imitation/expert_density10_v1/test.npz')
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--density',type=float,default=.1)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--random-seed',type=int,default=90211)
    parser.add_argument('--output-directory')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--scenario-file',help='Reuse frozen benchmark scenarios with a new output directory')
    parser.add_argument('--route-count',type=int,default=6)
    args=parser.parse_args()
    print(json.dumps(benchmark(args.checkpoint,args.train_dataset,args.validation_dataset,args.reserved_test_dataset,
        density=args.density,episodes=args.episodes,seed=args.seed,output_directory=args.output_directory,
        device=args.device,scenario_file=args.scenario_file,random_seed=args.random_seed,route_count=args.route_count),indent=2))


if __name__ == '__main__':
    main()
