"""Phase 5C deterministic paired PPO evaluation, preserving frozen Phase 5A code."""
import argparse
import csv
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from evaluation.scenario_manifest import validate, save_json, file_hash, digest
from evaluation.policy_adapters import NeuralAdapter
from evaluation.evaluate_baselines import rollout, read_csv, check_row
from evaluation.compare_ppo_variants import (VARIANTS,run_records,expert_cost,historical_records,
    history_comparison,density_summary,read_json)


def inspect_checkpoint(path,name,selection,manifest):
    run=Path(path).parent;m,s,c=run_records(run)
    curriculum=name=='ppo_curriculum'
    sidepath=Path(path).with_suffix('.curriculum.json' if curriculum else '.run.json')
    side=read_json(sidepath)
    if side['checkpoint_sha256']!=file_hash(path):raise ValueError('Checkpoint sidecar hash mismatch')
    method='curriculum' if curriculum else ('scratch' if name=='ppo_scratch' else 'pretrained')
    if m.get('mode' if curriculum else 'method')!=method:raise ValueError('Checkpoint run method mismatch')
    fingerprint='fingerprint' if curriculum else 'config_fingerprint'
    if side[fingerprint]!=m[fingerprint]:raise ValueError('Checkpoint/run fingerprint mismatch')
    adapter=NeuralAdapter('ppo',str(path),manifest['config']['environment'],'cpu')
    from agents.pretrained_ppo_policy import PretrainedPPOPolicy, EncodedNavigationObservation
    from environments.fixed_map_env import FixedMapNavigationEnv
    if type(adapter.module) is not PretrainedPPOPolicy:
        raise ValueError('Expected audited feed-forward PPO architecture')
    # The wrapper actually used during training must encode identically.
    first=manifest['scenarios'][0]
    env=EncodedNavigationObservation(FixedMapNavigationEnv(np.asarray(first['obstacles']),tuple(first['goal']),max_steps=first['horizon']))
    try:
        encoded,_=env.reset(options={'start':first['start']})
        from imitation.dataset import encode_observation
        if not np.array_equal(encoded,encode_observation(env.unwrapped._observation())):
            raise ValueError('Training/evaluation preprocessing differs')
    finally:env.close()
    step=int(adapter.model.num_timesteps)
    if step!=int(side['model_step' if curriculum else 'checkpoint_step']):raise ValueError('Checkpoint timestep mismatch')
    architecture=dict(policy_class=type(adapter.module).__module__+'.'+type(adapter.module).__name__,
        hidden_size=adapter.module.hidden_size,observation_shape=list(adapter.model.observation_space.shape),
        normalize_images=adapter.module.normalize_images,
        state_shapes={k:list(v.shape) for k,v in adapter.module.state_dict().items()})
    inherited_files={}
    if curriculum:
        cost=int(s['total_inherited_plus_phase4c_steps'])
        breakdown=dict(inherited_phase4b=int(s['source_phase4b_consumed_steps']),additional_phase4c=int(s['cumulative_consumed_phase4c_steps']))
        if sum(breakdown.values())!=cost:raise ValueError('Curriculum cost ledger mismatch')
        parent=Path(m['source_checkpoint'])
        if file_hash(parent)!=m['source_sha256']:raise ValueError('Curriculum source checkpoint changed')
        parent_side=read_json(parent.with_suffix('.run.json'))
        pm,ps,pc=run_records(parent.parent)
        if (parent_side['checkpoint_sha256']!=file_hash(parent) or
                int(parent_side['checkpoint_step'])!=int(side['start_model_step']) or
                pm['method']!='pretrained' or int(ps['cumulative_consumed_env_timesteps'])!=int(s['source_phase4b_consumed_steps'])):
            raise ValueError('Curriculum inherited lineage mismatch')
        inherited_files={str(p):file_hash(p) for p in (parent,parent.with_suffix('.run.json'),
            parent.parent/'run_manifest.json',parent.parent/'summary.json',parent.parent/'config.yaml')}
        if file_hash(m['verified_initialization'])!=pm['initialization_sha256']:
            raise ValueError('Curriculum BC initialization lineage mismatch')
        if int(side['phase_step'])!=step-int(side['start_model_step']):raise ValueError('Phase/model timestep mismatch')
    else:
        cost=int(s['cumulative_consumed_env_timesteps']);breakdown=dict(online_phase4b_search=cost)
        if int(s['actual_env_timesteps'])+int(s['lineage_cost_before_segment'])-int(s['starting_step'])!=cost:
            raise ValueError('Phase4B cost ledger mismatch')
    if cost<step:raise ValueError('Search cost cannot be smaller than checkpoint model age')
    expert=expert_cost(manifest,m,name)
    return dict(path=str(path),sha256=file_hash(path),selection=selection,
        selection_role='validation_selected' if selection=='best' else 'final_training_checkpoint',
        checkpoint_model_timesteps=step,ppo_updates=int(adapter.model._n_updates),
        completed_search_online_interactions=cost,online_cost_breakdown=breakdown,
        expert_data=expert,architecture=architecture,seed=m['seed'],ppo_configuration=c['ppo'],
        training_status=s['status'],sidecar=side,run_manifest=m,run_summary=s,training_config=c,
        protected_files={str(path):file_hash(path),str(sidepath):file_hash(sidepath),**inherited_files}),adapter


def validate_saved(root,manifest=None):
    root=Path(root)
    if (root/'INCOMPLETE.txt').exists():raise ValueError('Incomplete PPO comparison')
    index=read_json(root/'index.json')
    if index['schema_version']!=1:raise ValueError('Unknown comparison schema')
    if manifest is None:manifest=read_json(index['manifest_path'])
    validate(manifest)
    if index['manifest_id']!=manifest['manifest_id']:raise ValueError('Manifest mismatch')
    expected=manifest['scenarios'];count=0
    for filename,sha in index['output_files'].items():
        if file_hash(root/filename)!=sha:raise ValueError('Output file checksum mismatch')
    for name in VARIANTS:
        for selection in index['selections']:
            folder=root/name/selection;report=read_json(folder/'summary.json')
            if report['manifest_id']!=index['manifest_id'] or report['policy']!=name or report['selection']!=selection:
                raise ValueError('Policy/selection report mismatch')
            rows=read_csv(folder/'episodes.csv')
            if len(rows)!=len(expected):raise ValueError('Dropped or duplicated PPO episode')
            for row,s in zip(rows,expected):
                check_row(row)
                if row['policy']!=name:raise ValueError('Wrong policy row')
                for k in ('scenario_id','map_id','map_seed','density','horizon','optimal_steps'):
                    if row[k]!=s[k]:raise ValueError('Scenario pairing mismatch')
            if density_summary(rows)!=report['summary']:raise ValueError('Summary differs from episodes')
            meta=report['checkpoint']
            if meta['sha256']!=file_hash(meta['path']):raise ValueError('Checkpoint changed')
            if report['state_before']!=report['state_after']:raise ValueError('PPO weights changed')
            for filename,sha in meta['protected_files'].items():
                if file_hash(filename)!=sha:raise ValueError('Checkpoint/sidecar source changed')
            count+=len(rows)
    for filename,sha in index['historical_source_files'].items():
        if file_hash(filename)!=sha:raise ValueError('Historical evidence changed')
    return dict(valid=True,variants=3,selections=index['selections'],scenarios_per_checkpoint=len(expected),episodes=count)


def evaluate(manifest_path,output,selection='both',device='cpu',torch_threads=1):
    root=Path(output)
    if root.exists():raise FileExistsError('Choose a fresh output directory: '+str(root))
    if selection not in ('both','best','final') or torch_threads<1:raise ValueError('Invalid selection/thread count')
    torch.set_num_threads(torch_threads)
    manifest=read_json(manifest_path);validate(manifest)
    selections=['best','final'] if selection=='both' else [selection]
    inspected={};histories={};history_sources={};protected={};architecture=None
    # Preflight ALL selected checkpoints before evaluating ANY test episode.
    for name in VARIANTS:
        run=Path(manifest['config']['policies'][name]['checkpoint']).parent
        histories[name],history_sources[name]=historical_records(run,name)
        for source in history_sources[name]:protected.update(source['files'])
        for choice in selections:
            meta,adapter=inspect_checkpoint(run/(choice+'.zip'),name,choice,manifest)
            if architecture is not None and meta['architecture']!=architecture:raise ValueError('PPO variant architectures differ')
            architecture=meta['architecture'];inspected[(name,choice)]=meta
            protected.update(meta['protected_files'])
            exp=meta['expert_data']
            protected.update(exp.get('provenance_files',{}))
            if exp.get('sha256'):protected[exp['source']]=exp['sha256']
            if exp.get('bc_sha256'):protected[exp['bc_checkpoint']]=exp['bc_sha256']
            del adapter
    history=history_comparison(histories)
    a=history_sources['ppo_scratch'][0];b=history_sources['ppo_bc'][0]
    matched=all(a['manifest'].get(k)==b['manifest'].get(k) for k in
        ('seed','train_dataset_sha256','validation_signature','initialization_sha256','environment')) and a['config']['ppo']==b['config']['ppo']
    history['scratch_bc_training_protocol_matched']=matched
    if not matched:history['matched_scratch_bc_validation_points']=[]
    root.mkdir(parents=True,exist_ok=False)
    outputs={}
    try:
        for name in VARIANTS:
            for choice in selections:
                print('Evaluating '+name+'/'+choice,flush=True)
                meta=inspected[(name,choice)]
                policy=NeuralAdapter('ppo',meta['path'],manifest['config']['environment'],device)
                before=policy.fingerprint()
                rows=[rollout(policy,name,s,0) for s in manifest['scenarios']]
                policy.verify()
                if before!=policy.fingerprint():raise RuntimeError('PPO parameter/counter mutation')
                folder=root/name/choice;folder.mkdir(parents=True,exist_ok=False)
                with (folder/'episodes.csv').open('x',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                report=dict(schema_version=1,policy=name,selection=choice,manifest_id=manifest['manifest_id'],
                    deterministic=True,episodes=len(rows),checkpoint=meta,state_before=before,state_after=policy.fingerprint(),
                    summary=density_summary(rows),actual_device=str(policy.device))
                save_json(folder/'summary.json',report)
                for filename in ('episodes.csv','summary.json'):
                    path=folder/filename;outputs[str(path.relative_to(root))]=file_hash(path)
                del policy
        flat=[r for name in VARIANTS for r in histories[name]]
        with (root/'historical_validation.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        outputs['historical_validation.csv']=file_hash(root/'historical_validation.csv')
        for path,sha in protected.items():
            if file_hash(path)!=sha:raise RuntimeError('Protected source modified during evaluation')
        import evaluation.compare_ppo_variants as comparison
        import evaluation.evaluate_ppo_variants as runner
        index=dict(schema_version=1,manifest_path=str(Path(manifest_path).resolve()),manifest_id=manifest['manifest_id'],
            variants=list(VARIANTS),selections=selections,output_files=outputs,historical_source_files=protected,
            historical_sample_efficiency=history,historical_sources=history_sources,
            source_fingerprints={m.__name__:digest(inspect.getsource(m)) for m in (comparison,runner)},
            runtime=dict(torch=str(torch.__version__),threads=torch.get_num_threads(),requested_device=device),
            leakage_scope=manifest['leakage_scope'],evaluation='Deterministic only; stochastic results not collected')
        save_json(root/'index.json',index)
        status=validate_saved(root,manifest)
        from evaluation.compare_ppo_variants import compare
        save_json(root/'comparison.json',compare(root))
        return status
    except BaseException:
        with (root/'INCOMPLETE.txt').open('x') as f:f.write('Incomplete run. Retain for diagnosis; choose a new directory.\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--selection',choices=['best','final','both'],default='both')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu');p.add_argument('--torch-threads',type=int,default=1)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    result=validate_saved(a.output,read_json(a.manifest)) if a.validate_only else evaluate(a.manifest,a.output,a.selection,a.device,a.torch_threads)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
