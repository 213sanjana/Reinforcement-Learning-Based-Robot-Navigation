"""Paired held-out evaluation of selected fine-tuned, scratch and untouched initialization policies."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from environments.map_generator import shortest_path
from imitation.bc_network import resolve_device
from imitation.dataset import map_identity
from training.ppo_support import digest_json,evaluate_model,file_sha256,finite_policy


def load_scenarios(run):
    with np.load(Path(run)/'validation_scenarios.npz',allow_pickle=False) as source:
        metadata=json.loads(str(source['metadata']))
        arrays={key:source[key] for key in source.files if key!='metadata'}
    for i,grid in enumerate(arrays['obstacles']):
        if map_identity(grid) != arrays['map_ids'][i]:
            raise ValueError('Saved validation layout identity mismatch')
        route=shortest_path(grid,tuple(arrays['starts'][i]),tuple(arrays['goals'][i]))
        if route is None or len(route)-1 != arrays['path_lengths'][i]:
            raise ValueError('Saved validation reference length mismatch')
    signature=digest_json(dict(environment=metadata['environment'],indices=metadata['indices'],
        map_ids=arrays['map_ids'].tolist(),starts=arrays['starts'].tolist(),goals=arrays['goals'].tolist(),
        map_seeds=arrays['map_seeds'].tolist()))
    if signature != metadata['signature']:
        raise ValueError('Saved validation scenario signature mismatch')
    return dict(arrays=arrays,**metadata)


def evaluate(pretrained_run,output_directory,scratch_run=None,checkpoint_name='best',device='auto'):
    output=Path(output_directory)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new evaluation directory')
    if checkpoint_name not in ('best','final'):
        raise ValueError('checkpoint_name must be best or final')
    run=Path(pretrained_run)
    manifest=json.loads((run/'run_manifest.json').read_text()); run_summary=json.loads((run/'summary.json').read_text())
    config=yaml.safe_load((run/'config.yaml').read_text())
    if manifest['method']!='pretrained' or run_summary['status'].startswith('failed'):
        raise ValueError('Select a nonfailed pretrained experiment; inspect failure logs otherwise')
    scenarios=load_scenarios(run)
    if scenarios['signature'] != manifest['validation_signature']:
        raise ValueError('Validation signature differs from training manifest')
    baseline=Path(manifest['initialization'])
    if file_sha256(baseline) != manifest['initialization_sha256']:
        raise ValueError('Verified baseline initialization changed')
    models=[('bc_initialization',baseline),('pretrained_ppo',run/f'{checkpoint_name}.zip')]
    comparison=None
    if scratch_run is not None:
        scratch=Path(scratch_run)
        other=json.loads((scratch/'run_manifest.json').read_text())
        other_summary=json.loads((scratch/'summary.json').read_text())
        other_config=yaml.safe_load((scratch/'config.yaml').read_text())
        if other['method']!='scratch' or other_summary['status'].startswith('failed'):
            raise ValueError('Scratch comparison must use a nonfailed scratch experiment')
        for key in ('validation_signature','train_dataset_sha256','initialization_sha256','environment','seed'):
            if manifest[key] != other[key]:
                raise ValueError(f'Unpaired comparison: {key} differs')
        if config['ppo'] != other_config['ppo']:
            raise ValueError('PPO hyperparameters differ between compared methods')
        if load_scenarios(scratch)['signature'] != scenarios['signature']:
            raise ValueError('Scratch validation scenarios differ')
        models.append(('scratch_ppo',scratch/f'{checkpoint_name}.zip'))
        comparison=dict(budget_matched=run_summary['cumulative_consumed_env_timesteps']==other_summary['cumulative_consumed_env_timesteps'],
                        pretrained_total_env_timesteps=run_summary['cumulative_consumed_env_timesteps'],
                        scratch_total_env_timesteps=other_summary['cumulative_consumed_env_timesteps'],
                        pretrained_model_timesteps=run_summary['actual_env_timesteps'],
                        scratch_model_timesteps=other_summary['actual_env_timesteps'],
                        pretrained_status=run_summary['status'],scratch_status=other_summary['status'],
                        note='Match consumed environment transitions. Best checkpoints can occur at different steps within equal search budgets. BC demonstration/pretraining cost is additional.')
    rows=[]; summaries={}; checkpoints={}; resolved=str(resolve_device(device))
    for name,path in models:
        checksum=file_sha256(path)
        if name!='bc_initialization':
            sidecar=json.loads(path.with_suffix('.run.json').read_text())
            if sidecar['checkpoint_sha256'] != checksum:
                raise ValueError('Checkpoint fingerprint mismatch')
        model=PPO.load(path,device=resolved); finite_policy(model)
        summary,episodes=evaluate_model(model,scenarios)
        if file_sha256(path) != checksum:
            raise RuntimeError('Evaluation changed a checkpoint')
        checkpoints[name]=dict(path=str(path),sha256=checksum,checkpoint_step=int(model.num_timesteps),
                               ppo_updates=int(model._n_updates))
        summaries[name]=summary
        for episode in episodes:
            rows.append(dict(policy=name,**episode))
        del model
    output.mkdir(parents=True,exist_ok=False)
    fields=['policy',*summaries['bc_initialization']]
    with (output/'summary.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows(dict(policy=name,**values) for name,values in summaries.items())
    scalar_keys=[key for key in rows[0] if key not in ('route','actions')]
    with (output/'episodes.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=scalar_keys); writer.writeheader()
        writer.writerows({key:row[key] for key in scalar_keys} for row in rows)
    report=dict(validation_signature=scenarios['signature'],episodes_per_policy=len(scenarios['arrays']['map_ids']),
                environment=scenarios['environment'],test_dataset_used=False,checkpoint_selection=checkpoint_name,
                comparison=comparison,checkpoints=checkpoints,summary=summaries,results=rows)
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    names=list(summaries); labels=[name.replace('_','\n') for name in names]
    for axis,key,title in zip(axes,('success_rate','mean_spl','mean_collisions'),('Goal success','Step-based SPL','Collisions / episode')):
        axis.bar(labels,[summaries[name][key] for name in names]); axis.set_title(title); axis.grid(axis='y',alpha=.2)
        if key!='mean_collisions':
            axis.set_ylim(0,1)
    figure.savefig(output/'comparison.png',dpi=150); plt.close(figure)
    return {key:value for key,value in report.items() if key!='results'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-run',required=True)
    parser.add_argument('--scratch-run')
    parser.add_argument('--checkpoint-name',choices=['best','final'],default='best')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output-directory',default='artifacts/ppo/diagnostic_evaluation')
    args=parser.parse_args()
    print(json.dumps(evaluate(args.pretrained_run,args.output_directory,args.scratch_run,args.checkpoint_name,args.device),indent=2))


if __name__=='__main__':
    main()
