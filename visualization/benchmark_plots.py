"""Phase 5E: evidence-backed figures and verified deterministic route replays."""
import argparse
import csv
import json
from pathlib import Path
import math
import inspect
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from evaluation.scenario_manifest import file_hash,save_json,digest,contract
from evaluation.statistical_analysis import validate_analysis,load_sources
from evaluation.compare_ppo_variants import read_json


def evidence(root):
    root=Path(root);validate_analysis(root,root/'statistics')
    manifest,data,provenance,sources=load_sources(root)
    report=read_json(root/'statistics/analysis.json')
    sources.update({str(root/'statistics'/n):file_hash(root/'statistics'/n) for n in
        ('analysis.json','summary.csv','paired_comparisons.csv','validation.json')})
    return manifest,data,report,sources


def density_rows(report):
    return [r for r in report['descriptive'] if r['axis']=='density']


def interval(r,metric):
    s=r['statistics']
    if metric=='success':return s['success_rate'],s['success_wilson95']
    if metric=='successful_efficiency':
        d=s['successful_efficiency'];return d['mean'],d['bootstrap95']
    d=s['metrics'][metric];return d['mean'],d['bootstrap95']


def metric_plot(report,metric,path):
    rows=density_rows(report);names=sorted({r['policy'] for r in rows})
    fig,axes=plt.subplots(1,3,figsize=(16,7),sharey=True)
    labels=dict(success='Success rate (Wilson 95% CI)',collisions='Collisions / episode (bootstrap 95% CI)',
                spl='All-episode SPL (bootstrap 95% CI)',successful_efficiency='Successful efficiency (bootstrap 95% CI)')
    for ax,density in zip(axes,(.1,.3,.5)):
        for i,name in enumerate(names):
            r=next(r for r in rows if r['policy']==name and r['density']==density)
            value,ci=interval(r,metric)
            if value is None:
                ax.text(.02,i,'No successful episodes',fontsize=7,va='center',transform=ax.get_yaxis_transform());continue
            # Draw endpoints directly: percentile intervals need not contain the estimate.
            ax.hlines(i,ci[0],ci[1],color='tab:blue',linewidth=2)
            ax.plot(value,i,'o',color='tab:blue')
            ax.annotate(f"n={r['statistics']['successful_efficiency']['n'] if metric=='successful_efficiency' else r['statistics']['n']}",
                        (value,i),xytext=(4,5),textcoords='offset points',fontsize=6)
        ax.set(title=f'Density {density}',xlabel=labels[metric],yticks=range(len(names)),yticklabels=names)
        ax.grid(axis='x',alpha=.25);ax.invert_yaxis()
        if metric in ('success','spl','successful_efficiency'):ax.set_xlim(-.02,1.08)
        else:ax.set_xlim(left=0)
    fig.suptitle('Fixed checkpoint outcomes; best/final selections are separate series')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def history_plot(root,path):
    history=[]
    with (Path(root)/'ppo_variants/historical_validation.csv').open() as f:
        for r in csv.DictReader(f):history.append(r)
    if not history:return dict(generated=False,reason='No historical validation rows')
    groups=sorted({(float(r['density']),r['validation_signature']) for r in history})
    cols=min(3,len(groups));rows=math.ceil(len(groups)/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(5.5*cols,3.7*rows),squeeze=False)
    for ax,(density,signature) in zip(axes.flat,groups):
        for name in sorted({r['policy'] for r in history}):
            selected=[r for r in history if r['policy']==name and float(r['density'])==density and r['validation_signature']==signature]
            if not selected:continue
            selected=sorted(selected,key=lambda r:int(r['consumed_online_steps']))
            # No synthetic intermediate points or inferred intervals.
            ax.plot([int(r['consumed_online_steps']) for r in selected],[float(r['success_rate']) for r in selected],'-o',markersize=2,label=name)
        ax.set(title=f'Density {density}; validation {signature[:10]}',xlabel='Consumed online transitions (includes inherited search)',ylabel='Logged validation success',ylim=(-.02,1.02))
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
    fig.suptitle('Historical validation only; separate panels for different scenario signatures\nLines join logged points; no test learning curves or unsupported confidence bands')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(path,dpi=160);plt.close(fig)
    return dict(generated=True,groups=[dict(density=d,validation_signature=s) for d,s in groups],
        uncertainty='Historical aggregate logs do not support per-point paired/bootstrap uncertainty; no bands invented')


def replay(policy,name,saved,scenario,seed):
    from environments.fixed_map_env import FixedMapNavigationEnv
    from environments.rewards import RewardConfig
    from evaluation.baseline_adapters import action_observation
    from evaluation.result_schema import EpisodeResult
    env=FixedMapNavigationEnv(np.asarray(scenario['obstacles']),tuple(scenario['goal']),
        max_steps=scenario['horizon'],bfs_shaping=False,reward_config=RewardConfig(**contract()['reward']))
    try:
        obs,_=env.reset(seed=scenario['map_seed'],options={'start':scenario['start']})
        policy.reset(np.random.SeedSequence([seed,scenario['map_seed']]))
        route=[obs['agent'].tolist()];actions=[];blocked=[];total=0.;collisions=0
        for step in range(1,scenario['horizon']+1):
            action=policy.act(action_observation(obs))
            obs,reward,term,trunc,info=env.step(action)
            route.append(obs['agent'].tolist());actions.append(int(action));total+=reward
            blocked.append(bool(info['collision']));collisions+=int(info['collision'])
            if term or trunc:break
        expected=dict(success=bool(term),timeout=bool(trunc),episode_length=step,collisions=collisions,
            path_length=step-collisions,reward=total,spl=scenario['optimal_steps']/step if term else 0.,
            path_efficiency=scenario['optimal_steps']/step if term else None)
        for key,value in expected.items():
            original=saved[key]
            if value is None:
                if original is not None:raise ValueError('Replay mismatch: '+key)
            elif isinstance(value,float):
                if not np.isclose(value,original,rtol=0,atol=1e-7):raise ValueError('Replay mismatch: '+key)
            elif value!=original:raise ValueError('Replay mismatch: '+key)
        return dict(policy_series=name,scenario_id=scenario['scenario_id'],map_id=scenario['map_id'],density=scenario['density'],
            start=scenario['start'],goal=scenario['goal'],route=route,actions=actions,blocked=blocked,
            saved_outcome=expected,replay_matches_saved=True)
    finally:env.close()


def route_plots(root,manifest,data,output,device):
    from evaluation.baseline_adapters import load_baseline
    from evaluation.policy_adapters import NeuralAdapter
    import torch
    torch.set_num_threads(1)
    scenarios={s['scenario_id']:s for s in manifest['scenarios']};records=[];availability=[]
    random_seed=read_json(Path(root)/'baselines/summary.json')['random_seed']
    for name in sorted(data):
        if '/' in name:
            policy_name,selection=name.split('/')
            report=read_json(Path(root)/'ppo_variants'/policy_name/selection/'summary.json')
            policy=NeuralAdapter('ppo',report['checkpoint']['path'],manifest['config']['environment'],device)
        else:policy,_=load_baseline(name,manifest,device)
        before=policy.fingerprint()
        fig,axes=plt.subplots(3,2,figsize=(10,13))
        for row,density in enumerate((.1,.3,.5)):
            for col,success in enumerate((True,False)):
                ax=axes[row,col]
                selected=next((r for r in data[name] if r['density']==density and r['success']==success),None)
                if selected is None:
                    ax.text(.5,.5,f'No {"successful" if success else "failed"} saved episodes\nat density {density}',ha='center',va='center');ax.set_axis_off()
                    availability.append(dict(policy_series=name,density=density,success=success,available=False));continue
                s=scenarios[selected['scenario_id']]
                trace=replay(policy,name,selected,s,random_seed);records.append(trace)
                availability.append(dict(policy_series=name,density=density,success=success,available=True,scenario_id=s['scenario_id']))
                grid=np.asarray(s['obstacles']);points=np.asarray(trace['route'])
                ax.imshow(grid,cmap='Greys',vmin=0,vmax=1,origin='upper')
                ax.plot(points[:,1],points[:,0],'-',color='tab:blue',alpha=.65,lw=2)
                if len(points)>1:ax.scatter(points[1:,1],points[1:,0],c=np.arange(len(points)-1),cmap='viridis',s=12)
                ax.scatter(s['start'][1],s['start'][0],marker='o',color='limegreen',s=70,label='Start')
                ax.scatter(s['goal'][1],s['goal'][0],marker='*',color='orange',s=120,label='Goal')
                collision_points=points[:-1][np.asarray(trace['blocked'])]
                if len(collision_points):ax.scatter(collision_points[:,1],collision_points[:,0],marker='x',color='red',s=30,label='Blocked attempt at current cell')
                ax.set(title=f"Density {density}: {'success' if success else 'timeout'}\n{s['scenario_id'][:10]}; steps={selected['episode_length']}, collisions={selected['collisions']}",
                    xticks=range(grid.shape[1]),yticks=range(grid.shape[0]))
        if policy.fingerprint()!=before:raise RuntimeError('Replay modified policy state')
        if hasattr(policy,'verify'):policy.verify()
        fig.suptitle(name+' — first saved success/failure per density; deterministic checked replay',fontsize=12)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color='limegreen',label='Start'),
            Line2D([],[],marker='*',linestyle='',color='orange',label='Goal'),
            Line2D([],[],marker='x',linestyle='',color='red',label='Blocked attempt at current cell')],
            loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,fontsize=8)
        fig.tight_layout(rect=(0,0,1,.935));fig.savefig(output/(name.replace('/','_')+'_routes.png'),dpi=140);plt.close(fig)
        del policy
    return dict(selection='First scenario in frozen order for each series/density/outcome; no rescue or cherry-picking by score',
        method='Deterministic replays; route/action logs were absent in original aggregate CSVs',records=records,availability=availability)


def generate(root,output,device='cpu'):
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a fresh figures directory: '+str(output))
    manifest,data,report,sources=evidence(root)
    sources[str(Path(__file__).resolve())]=file_hash(Path(__file__).resolve())
    output.mkdir(parents=True,exist_ok=False)
    try:
        for metric,filename in [('success','success_rates.png'),('collisions','collisions.png'),('spl','path_efficiency_spl.png'),('successful_efficiency','successful_path_efficiency.png')]:
            metric_plot(report,metric,output/filename)
        historical=history_plot(root,output/'historical_sample_efficiency.png')
        traces=route_plots(root,manifest,data,output,device)
        save_json(output/'trajectory_replays.json',traces)
        save_json(output/'plotted_statistics.json',dict(manifest_id=manifest['manifest_id'],rows=density_rows(report),historical=historical))
        for path,sha in sources.items():
            if file_hash(path)!=sha:raise RuntimeError('Source modified during plotting')
        save_json(output/'figures_manifest.json',dict(schema_version=1,input_root=str(Path(root).resolve()),manifest_id=manifest['manifest_id'],sources=sources,
            files={p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
            source_fingerprints={__name__:digest(inspect.getsource(__import__(__name__,fromlist=['_'])))},
            matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
            policy_series=list(sorted(data)),route_replays_verified=len(traces['records']),historical=historical))
        return dict(valid=True,policy_series=len(data),route_replays_verified=len(traces['records']),output=str(output))
    except BaseException:
        (output/'INCOMPLETE.txt').write_text('Figure generation incomplete; retain diagnosis and use a new output directory.\n');raise


def validate_figures(root,folder,manifest=None):
    folder=Path(folder)
    if (folder/'INCOMPLETE.txt').exists():raise ValueError('Incomplete figures')
    f=read_json(folder/'figures_manifest.json')
    if f['input_root']!=str(Path(root).resolve()):raise ValueError('Figure input directory mismatch')
    if manifest is None:manifest=read_json(Path(root)/'scenarios.json')
    if manifest is not None and f['manifest_id']!=manifest['manifest_id']:raise ValueError('Figure/scenario manifest mismatch')
    for path,sha in f['sources'].items():
        if file_hash(path)!=sha:raise ValueError('Figure source changed')
    for name,sha in f['files'].items():
        if file_hash(folder/name)!=sha:raise ValueError('Figure output changed')
    return f


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    result=dict(valid=True,files=len(validate_figures(a.input,a.output)['files'])) if a.validate_only else generate(a.input,a.output,a.device)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
