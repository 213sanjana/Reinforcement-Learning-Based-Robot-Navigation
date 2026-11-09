"""Paired validation across all densities, including the unchanged Phase 4B starting policy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO
from imitation.bc_network import resolve_device
from training.ppo_support import digest_json,evaluate_model,file_sha256,finite_policy
from training.train_curriculum_ppo import read_checkpoint
from curriculum.map_pools import load_pools,scenario


def evaluate(curriculum_run,output_directory,fixed_run=None,checkpoint_name='best',device='auto'):
    if checkpoint_name not in ('best','final'):
        raise ValueError('Choose best or final checkpoint')
    output=Path(output_directory)
    if output.exists():
        raise FileExistsError('Evaluation directory exists; choose a new one')
    run=Path(curriculum_run); manifest=json.loads((run/'run_manifest.json').read_text())
    summary=json.loads((run/'summary.json').read_text()); c=yaml.safe_load((run/'config.yaml').read_text())
    if manifest['mode']!='curriculum' or summary['status'].startswith('failed'):
        raise ValueError('Expected a curriculum experiment without numerical/exception failure')
    pools,pool_manifest=load_pools(manifest['pool_directory'])
    if pool_manifest['signature']!=manifest['pool_signature']:
        raise ValueError('Map pool specification changed')
    if file_sha256(manifest['source_checkpoint'])!=manifest['source_sha256']:
        raise ValueError('Starting policy changed')
    scenarios=[]
    for pair in pools:
        s=scenario(pair['validation']); count=c['validation']['episodes']
        scenarios.append(dict(s,arrays={k:v[:count] for k,v in s['arrays'].items()},
                              signature=digest_json(dict(pool_signature=s['signature'],episodes=count))))
    if [s['signature'] for s in scenarios]!=manifest['scenario_signatures']:
        raise ValueError('Evaluation scenario signatures changed')
    models=[('phase4b_baseline',Path(manifest['source_checkpoint'])),('curriculum_ppo',run/f'{checkpoint_name}.zip')]
    budgets=dict(curriculum_consumed_phase4c_steps=summary['cumulative_consumed_phase4c_steps'],budget_matched=None,
                 note='Additional Phase 4C environment transitions; both inherit the same Phase 4B/BC costs. Greedy validation only, no held-out test performance.')
    if fixed_run:
        other=Path(fixed_run); om=json.loads((other/'run_manifest.json').read_text()); os=json.loads((other/'summary.json').read_text())
        oc=yaml.safe_load((other/'config.yaml').read_text())
        if om['mode']!='fixed' or os['status'].startswith('failed'):
            raise ValueError('Expected a valid fixed-density control')
        for key in ('source_sha256','pool_signature','scenario_signatures','ppo','seed','densities','source_phase4b_consumed_steps'):
            if om[key]!=manifest[key]:
                raise ValueError(f'Unpaired comparison: {key}')
        if oc['validation']!=c['validation']:
            raise ValueError('Validation/selection protocol differs')
        models.append(('fixed_density_ppo',other/f'{checkpoint_name}.zip'))
        budgets.update(fixed_consumed_phase4c_steps=os['cumulative_consumed_phase4c_steps'],
                       fixed_density=om['fixed_density'],
                       budget_matched=os['cumulative_consumed_phase4c_steps']==summary['cumulative_consumed_phase4c_steps'])
    rows=[]; metrics=[]; checks={}; resolved=str(resolve_device(device))
    for name,path in models:
        checksum=file_sha256(path)
        side=read_checkpoint(path) if name!='phase4b_baseline' else None
        model=PPO.load(path,device=resolved); finite_policy(model)
        checks[name]=dict(path=str(path),sha256=checksum,model_step=int(model.num_timesteps),
                         selected_phase_step=side['phase_step'] if side else 0)
        for level,s in enumerate(scenarios):
            result,episodes=evaluate_model(model,s)
            metrics.append(dict(policy=name,level=level,density=manifest['densities'][level],**result))
            rows.extend(dict(policy=name,level=level,density=manifest['densities'][level],**r) for r in episodes)
        if file_sha256(path)!=checksum:
            raise RuntimeError('Evaluation changed a checkpoint')
        del model
    result=dict(checkpoint_selection=checkpoint_name,comparison=budgets,checkpoints=checks,
                pool_signature=manifest['pool_signature'],scenario_signatures=manifest['scenario_signatures'],
                test_performance_used=False,metrics=metrics,episodes=rows)
    output.mkdir(parents=True,exist_ok=False)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for filename,data in [('summary.csv',metrics),('episodes.csv',rows)]:
        keys=[k for k in data[0] if k not in ('actions','route')]
        with (output/filename).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows({k:r[k] for k in keys} for r in data)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for name,_ in models:
        selected=[r for r in metrics if r['policy']==name]
        for ax,key in zip(axes,('success_rate','mean_spl','mean_collisions')):
            ax.plot([r['density'] for r in selected],[r[key] for r in selected],'-o',label=name)
    for ax,title in zip(axes,('Validation success','SPL','Mean collisions')):
        ax.set(xlabel='Obstacle density',title=title); ax.grid(alpha=.2)
    axes[0].set_ylim(0,1.02); axes[-1].legend(fontsize=7)
    fig.savefig(output/'comparison.png',dpi=150); plt.close(fig)
    return {k:v for k,v in result.items() if k!='episodes'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curriculum-run',required=True); p.add_argument('--fixed-run'); p.add_argument('--output-directory',required=True)
    p.add_argument('--checkpoint-name',choices=['best','final'],default='best'); p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=p.parse_args()
    print(json.dumps(evaluate(args.curriculum_run,args.output_directory,args.fixed_run,args.checkpoint_name,args.device),indent=2))


if __name__=='__main__':
    main()
