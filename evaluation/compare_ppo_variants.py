"""Checkpoint provenance and historical validation curves, separate from test scores."""
import argparse
import csv
import json
from pathlib import Path
import math
import yaml
import numpy as np
from evaluation.scenario_manifest import file_hash, save_json, digest
from evaluation.result_schema import summarize

VARIANTS=('ppo_scratch','ppo_bc','ppo_curriculum')


def read_json(path):
    return json.loads(Path(path).read_text())


def run_records(run):
    run=Path(run)
    return read_json(run/'run_manifest.json'),read_json(run/'summary.json'),yaml.safe_load((run/'config.yaml').read_text())


def expert_cost(manifest,run_manifest,name):
    if name=='ppo_scratch':
        return dict(trajectories=0,transitions=0,source='Scratch actor trained without BC demonstrations')
    import torch
    init=Path(run_manifest.get('initialization',run_manifest.get('verified_initialization','')))
    report=read_json(init.with_suffix('.transfer.json'))
    if report.get('status')!='passed' or report.get('training_started'):
        raise ValueError('Unverified BC-to-PPO initialization')
    if file_hash(init)!=report['initialization_sha256']:
        raise ValueError('Transfer initialization changed')
    expected=run_manifest.get('initialization_sha256')
    if expected is not None and file_hash(init)!=expected:
        raise ValueError('Initialization hash mismatch')
    bc=Path(report['source_checkpoint'])
    if file_hash(bc)!=report['source_checkpoint_sha256']:
        raise ValueError('Transferred BC source hash mismatch')
    payload=torch.load(bc,map_location='cpu',weights_only=True)
    train=Path(payload['dataset_provenance']['train']['path'])
    if file_hash(train)!=payload['dataset_provenance']['train']['sha256']:
        raise ValueError('Expert training dataset hash mismatch')
    with np.load(train,allow_pickle=False) as data:
        offsets=data['episode_offsets']; trajectories=len(data['map_ids'])
        if len(offsets)!=trajectories+1 or offsets[0]!=0 or np.any(np.diff(offsets)<=0):
            raise ValueError('Invalid expert trajectory offsets')
        transitions=int(offsets[-1])
    return dict(trajectories=trajectories,transitions=transitions,source=str(train),
        sha256=file_hash(train),bc_checkpoint=str(bc),bc_sha256=file_hash(bc),
        provenance_files={str(init):file_hash(init),str(init.with_suffix('.transfer.json')):file_hash(init.with_suffix('.transfer.json')),
                          str(bc):file_hash(bc),str(train):file_hash(train)},
        interpretation='Unique offline expert transitions; reused epochs are not additional environment interactions')


def historical_records(run,name):
    """Follow continuation lineage; retain duplicate baseline rows and failed outcomes."""
    rows=[];sources=[];seen=set();run=Path(run)
    while True:
        key=str(run.resolve())
        if key in seen:raise ValueError('Cyclic historical lineage')
        seen.add(key)
        m,s,c=run_records(run)
        curriculum=name=='ppo_curriculum'
        expected='curriculum' if curriculum else ('scratch' if name=='ppo_scratch' else 'pretrained')
        if m.get('mode' if curriculum else 'method')!=expected:
            raise ValueError('Historical method mismatch')
        path=run/'validation.csv'
        with path.open(newline='') as f: raw=list(csv.DictReader(f))
        if not raw:raise ValueError('Historical validation log is empty')
        if curriculum:
            offset=int(s['total_inherited_plus_phase4c_steps'])-int(s['observed_model_step'])
        else:
            offset=int(m['lineage_cost_before_segment'])-int(m['starting_step'])
        for r in raw:
            step=int(r['model_step' if curriculum else 'global_step'])
            density=float(r['density']) if curriculum else float(m['environment']['obstacle_density'])
            level=int(r['evaluated_level']) if curriculum else None
            signature=m['scenario_signatures'][level] if curriculum else m['validation_signature']
            success=float(r['success_rate']);spl=float(r['mean_spl'])
            if not 0<=success<=1 or not math.isfinite(spl) or not 0<=spl<=1:
                raise ValueError('Invalid historical validation metric')
            rows.append(dict(policy=name,run=str(run),source=str(path),validation_signature=signature,
                density=density,model_step=step,consumed_online_steps=offset+step,
                success_rate=success,mean_spl=spl,kind=r.get('kind','curriculum'),
                run_status=s['status']))
        sources.append(dict(run=str(run),manifest=m,summary=s,config=c,
            files={str(run/n):file_hash(run/n) for n in ('run_manifest.json','summary.json','config.yaml','validation.csv')}))
        parent=m.get('resume' if curriculum else 'resume_source')
        if not parent:break
        run=Path(parent).parent
    return sorted(rows,key=lambda r:(r['consumed_online_steps'],r['density'],r['run'])),sources


def history_comparison(histories,threshold=.8):
    curves={}
    for name,rows in histories.items():
        groups={}
        for r in rows:
            key=str(r['density'])+':'+r['validation_signature']
            groups.setdefault(key,[]).append(r)
        curves[name]={key:dict(density=rs[0]['density'],validation_signature=rs[0]['validation_signature'],
            evaluations=len(rs),first_logged_success_at_least_threshold=min(
                (r['consumed_online_steps'] for r in rs if r['success_rate']>=threshold),default=None))
            for key,rs in groups.items()}
    # Compare exact shared costs only on the same validation scenarios.
    a=histories['ppo_scratch'];b=histories['ppo_bc'];matches=[]
    for x in a:
        for y in b:
            if (x['validation_signature'],x['density'],x['consumed_online_steps'])==(y['validation_signature'],y['density'],y['consumed_online_steps']):
                matches.append(dict(consumed_online_steps=x['consumed_online_steps'],density=x['density'],
                    validation_signature=x['validation_signature'],scratch_success=x['success_rate'],
                    bc_initialized_success=y['success_rate'],difference=y['success_rate']-x['success_rate']))
    return dict(threshold=threshold,curves=curves,matched_scratch_bc_validation_points=matches,
        interpretation='Historical validation, not held-out test learning curves. Threshold is first logged crossing, not sustained mastery. No interpolation. Curriculum signatures/densities must match before curve comparisons. Offline expert-data cost remains additional.')


def density_summary(rows):
    result={}
    for d in (.1,.3,.5):
        group=[r for r in rows if r['density']==d];m=summarize(group)
        m['timeout_rate']=sum(r['timeout'] for r in group)/len(group)
        m['mean_path_length']=sum(r['path_length'] for r in group)/len(group)
        result[str(d)]=m
    return result


def compare(output):
    root=Path(output);index=read_json(root/'index.json')
    from evaluation.evaluate_ppo_variants import validate_saved
    validate_saved(root)
    entries={}
    for name in VARIANTS:
        entries[name]={selection:read_json(root/name/selection/'summary.json')
                       for selection in index['selections']}
    deltas={}
    for selection in index['selections']:
        deltas[selection]={}
        for density in ('.1','.3','.5'):
            key=str(float(density));scratch=entries['ppo_scratch'][selection]['summary'][key]
            bc=entries['ppo_bc'][selection]['summary'][key];curr=entries['ppo_curriculum'][selection]['summary'][key]
            deltas[selection][key]=dict(bc_minus_scratch_success=bc['success_rate']-scratch['success_rate'],
                curriculum_minus_bc_success=curr['success_rate']-bc['success_rate'],
                bc_minus_scratch_spl=bc['mean_spl']-scratch['mean_spl'],
                curriculum_minus_bc_spl=curr['mean_spl']-bc['mean_spl'])
    report=dict(manifest_id=index['manifest_id'],primary_inference='deterministic', differences=deltas,
        policies=entries,historical_sample_efficiency=index['historical_sample_efficiency'],
        limits=['Fixed checkpoints from one training seed do not establish multi-seed superiority.',
                'Best checkpoint age differs from completed search cost.',
                'Best and final are reported separately; no test-based checkpoint selection.',
                'Historical validation curves use their original maps, separate from this benchmark.'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    save_json(a.output,compare(a.input));print('Comparison saved: '+a.output)


if __name__=='__main__':main()
