"""Phase 5D: reproducible paired inference on actual Phase 5B/C saved episodes."""
import argparse
import csv
import inspect
import itertools
import json
import math
from pathlib import Path
import numpy as np
from evaluation.scenario_manifest import validate, file_hash, digest, save_json
from evaluation.result_schema import EpisodeResult,wilson
from evaluation.evaluate_baselines import read_csv, check_row, validate_results, EXTRA
from evaluation.evaluate_ppo_variants import validate_saved
from evaluation.reliability_analysis import strata

METRICS=('success','collisions','timeout','episode_length','reward','spl')


def rng_for(seed,key):
    value=int(digest(dict(seed=seed,key=key))[:16],16)
    return np.random.default_rng(value)


def mean_interval(values,reps,seed,key):
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(n=0,mean=None,bootstrap95=None,degenerate=False,median=None,sample_sd=None,p90=None)
    if not np.isfinite(x).all():raise ValueError('Nonfinite statistical observations')
    rng=rng_for(seed,key);means=[]
    # Chunk to bound memory for larger manifests.
    for begin in range(0,reps,256):
        idx=rng.integers(0,len(x),size=(min(256,reps-begin),len(x)))
        means.extend(x[idx].mean(axis=1).tolist())
    lo,hi=np.quantile(means,[.025,.975],method='linear')
    return dict(n=len(x),mean=float(x.mean()),bootstrap95=[float(lo),float(hi)],degenerate=bool(lo==hi),median=float(np.median(x)),
        sample_sd=float(np.std(x,ddof=1)) if len(x)>1 else None,p90=float(np.quantile(x,.9,method='linear')))


def describe(rows,reps,seed,key):
    n=len(rows);successes=sum(r['success'] for r in rows)
    result=dict(n=n,successes=successes,failures=n-successes,
        success_rate=successes/n if n else None,success_wilson95=wilson(successes,n) if n else None,
        metrics={m:mean_interval([float(r[m]) for r in rows],reps,seed,[key,m]) for m in METRICS},
        successful_efficiency=mean_interval([r['path_efficiency'] for r in rows if r['success']],reps,seed,[key,'efficiency']),
        timeout_wilson95=wilson(sum(r['timeout'] for r in rows),n) if n else None,
        missing_efficiency_due_to_failure=n-successes,unexpected_missing=0,
        sparse=n<30)
    return result


def exact_mcnemar(a_wins,b_wins):
    n=a_wins+b_wins
    if not n:return 1.
    k=min(a_wins,b_wins)
    # Exact two-sided conditional binomial test, evaluated in log space.
    terms=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)-n*math.log(2) for j in range(k+1)]
    largest=max(terms)
    return min(1.,2*math.exp(largest)*sum(math.exp(t-largest) for t in terms))


def paired(a,b,reps,seed,key):
    if [r['scenario_id'] for r in a]!=[r['scenario_id'] for r in b]:raise ValueError('Unaligned paired samples')
    n=len(a);a_wins=sum(x['success'] and not y['success'] for x,y in zip(a,b))
    b_wins=sum(y['success'] and not x['success'] for x,y in zip(a,b))
    jointly=[(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    metrics={m:mean_interval([float(x[m])-float(y[m]) for x,y in zip(a,b)],reps,seed,[key,m]) for m in METRICS}
    return dict(n_pairs=n,a_only_success=a_wins,b_only_success=b_wins,discordant_success_pairs=a_wins+b_wins,
        both_success=len(jointly),both_failure=sum(not x['success'] and not y['success'] for x,y in zip(a,b)),
        difference_direction='A minus B',metrics=metrics,
        jointly_successful_efficiency_difference=mean_interval([x['path_efficiency']-y['path_efficiency'] for x,y in jointly],reps,seed,[key,'joint_efficiency']),
        exact_mcnemar_p=exact_mcnemar(a_wins,b_wins) if n else None,
        sparse=n<30,uncertainty_scope='Evaluation scenarios, conditional on fixed checkpoints; not training-seed variability')


def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);result=[0.]*len(pvalues);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*pvalues[i]));result[i]=running
    return result


def load_sources(input_dir):
    root=Path(input_dir);manifest_path=root/'scenarios.json'
    manifest=json.loads(manifest_path.read_text());validate(manifest)
    base=root/'baselines';ppo=root/'ppo_variants'
    validate_results(manifest,base);validate_saved(ppo,manifest)
    index=json.loads((ppo/'index.json').read_text())
    expected_csvs={base/'episodes.csv'}|{ppo/name/selection/'episodes.csv' for name in index['variants'] for selection in index['selections']}
    discovered=set(base.rglob('episodes.csv'))|set(ppo.rglob('episodes.csv'))
    if discovered!=expected_csvs:raise ValueError('Unexpected or missing per-episode result files in Phase 5B/C result roots')
    sources={str(manifest_path):file_hash(manifest_path),str(base/'summary.json'):file_hash(base/'summary.json'),
             str(ppo/'index.json'):file_hash(ppo/'index.json')}
    data={};provenance={}
    expected=[s['scenario_id'] for s in manifest['scenarios']]
    required=set(EpisodeResult.__dataclass_fields__)|set(EXTRA)
    def ingest(path,name,policy,meta):
        with path.open(newline='') as f:
            reader=csv.reader(f);header=next(reader)
        if len(header)!=len(set(header)) or set(header)!=required:raise ValueError('Inconsistent episode CSV schema: '+str(path))
        rows=[r for r in read_csv(path) if r['policy']==policy]
        if [r['scenario_id'] for r in rows]!=expected:raise ValueError('Missing, duplicate or reordered scenarios: '+name)
        for r in rows:
            if any(r.get(k) is None for k in required if k!='path_efficiency'):
                raise ValueError('Unexpected missing episode field')
            check_row(r)
            for metric in METRICS:
                if r[metric] is None or not math.isfinite(float(r[metric])):raise ValueError('Missing/nonfinite required outcome')
            if r['success'] and r['path_efficiency'] is None:raise ValueError('Missing successful efficiency')
            if not r['success'] and r['path_efficiency'] is not None:raise ValueError('Failed efficiency must be null')
        data[name]=rows;provenance[name]=meta;sources[str(path)]=file_hash(path)
    base_report=json.loads((base/'summary.json').read_text())
    for name in ('random','astar','dqn','bc'):
        ingest(base/'episodes.csv',name,name,dict(record=base_report['provenance'][name],
            random_action_seed=base_report['random_seed'] if name=='random' else None))
    for name in index['variants']:
        for selection in index['selections']:
            folder=ppo/name/selection;report=json.loads((folder/'summary.json').read_text())
            ingest(folder/'episodes.csv',name+'/'+selection,name,report['checkpoint'])
            sources[str(folder/'summary.json')]=file_hash(folder/'summary.json')
    return manifest,data,provenance,sources


def analyze(manifest,data,provenance,sources,reps=5000,seed=5172026):
    if reps<100 or seed<0:raise ValueError('Use at least 100 bootstrap replicates and a nonnegative seed')
    expected=[s['scenario_id'] for s in manifest['scenarios']]
    if len(expected)!=len(set(expected)):raise ValueError('Duplicate manifest scenarios')
    for name,rows in data.items():
        if [r['scenario_id'] for r in rows]!=expected:raise ValueError('Scenario alignment mismatch: '+name)
    features,groups=strata(manifest['scenarios']);descriptive=[];comparisons=[]
    for group in groups:
        ids=set(group['scenario_ids']);tag={k:v for k,v in group.items() if k!='scenario_ids'}
        selected={name:[r for r in rows if r['scenario_id'] in ids] for name,rows in data.items()}
        for name,rows in selected.items():
            descriptive.append(dict(policy=name,**tag,statistics=describe(rows,reps,seed,[name,tag])))
        # Stratified comparisons are exploratory; density-only family is primary.
        for a,b in itertools.combinations(sorted(data),2):
            comparisons.append(dict(policy_a=a,policy_b=b,**tag,statistics=paired(selected[a],selected[b],reps,seed,[a,b,tag]),
                role='primary_density_success' if group['axis']=='density' else 'exploratory_stratum'))
    primary=[r for r in comparisons if r['role']=='primary_density_success' and r['statistics']['n_pairs']]
    adjusted=holm([r['statistics']['exact_mcnemar_p'] for r in primary])
    for r,p in zip(primary,adjusted):
        r['statistics']['success_mcnemar_holm_p']=p
        r['statistics']['multiplicity_family_size']=len(primary)
        r['statistics']['holm_reject_at_0_05']=p<.05
    import evaluation.statistical_analysis as stats_module
    import evaluation.reliability_analysis as reliability_module
    return dict(schema_version=1,manifest_id=manifest['manifest_id'],sources=sources,provenance=provenance,
        settings=dict(seed=seed,bootstrap_replicates=reps,confidence=.95,numpy_version=np.__version__,
            algorithm='percentile bootstrap of scenario means; paired difference vectors resampled jointly',
            quantile_method='linear',multiplicity='Holm over all density-only success comparisons in this report'),
        source_fingerprints={m.__name__:digest(inspect.getsource(m)) for m in (stats_module,reliability_module)},
        descriptive=descriptive,paired_comparisons=comparisons,scenario_features=features,
        reliability=dict(training_seed_variability_estimated=False,training_seed_variability=None,
            evaluation_unit='One unique canonical layout/scenario; fixed policies and frozen accepted-map distribution',
            bootstrap_limit='Constant/rare outcomes can yield degenerate bootstrap intervals; Wilson success intervals and exact discordant-pair tests are also supplied.',
            strata='Predefined observation-only distance/graph bins; empty groups retained, n<30 flagged sparse.',
            failure_handling='Failures retained for all primary outcomes; null successful efficiency is structural and not imputed. Joint-success efficiency comparisons use a restricted population.',
            significance='Only density success tests have global Holm adjustment. Other intervals are descriptive/exploratory, not simultaneous or causal evidence. Multiple selections share training histories and are not independent training seeds.',
            training_limit='No training-seed variance estimate or general algorithm-superiority claim from these fixed checkpoints.'),
        leakage_scope=manifest.get('leakage_scope'))


def flatten(report):
    rows=[]
    for r in report['descriptive']:
        s=r['statistics']
        rows.append(dict(policy=r['policy'],density=r['density'],axis=r['axis'],bin=r['bin'],n=s['n'],
            successes=s['successes'],failures=s['failures'],success_rate=s['success_rate'],
            wilson_lower=s['success_wilson95'][0] if s['n'] else None,wilson_upper=s['success_wilson95'][1] if s['n'] else None,
            mean_collisions=s['metrics']['collisions']['mean'],timeout_rate=s['metrics']['timeout']['mean'],
            mean_spl=s['metrics']['spl']['mean'],successful_efficiency_n=s['successful_efficiency']['n'],
            mean_successful_efficiency=s['successful_efficiency']['mean'],sparse=s['sparse']))
    return rows


def flatten_pairs(report):
    rows=[]
    for r in report['paired_comparisons']:
        s=r['statistics'];success=s['metrics']['success'];spl=s['metrics']['spl']
        rows.append(dict(policy_a=r['policy_a'],policy_b=r['policy_b'],density=r['density'],axis=r['axis'],bin=r['bin'],role=r['role'],
            n_pairs=s['n_pairs'],a_only_success=s['a_only_success'],b_only_success=s['b_only_success'],
            success_difference=success['mean'],success_lower=success['bootstrap95'][0] if s['n_pairs'] else None,
            success_upper=success['bootstrap95'][1] if s['n_pairs'] else None,exact_mcnemar_p=s['exact_mcnemar_p'],
            holm_p=s.get('success_mcnemar_holm_p'),spl_difference=spl['mean'],
            joint_success_n=s['jointly_successful_efficiency_difference']['n'],sparse=s['sparse']))
    return rows


def run(input_dir,output,reps=5000,seed=5172026):
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a fresh output directory: '+str(output))
    manifest,data,provenance,sources=load_sources(input_dir)
    report=analyze(manifest,data,provenance,sources,reps,seed)
    for path,sha in sources.items():
        if file_hash(path)!=sha:raise RuntimeError('Source changed during analysis')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'analysis.json',report)
    rows=flatten(report)
    with (output/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pair_rows=flatten_pairs(report)
    with (output/'paired_comparisons.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
    save_json(output/'validation.json',dict(valid=True,policies=len(data),scenarios_per_policy=len(manifest['scenarios']),
        input_episode_rows=sum(len(rows) for rows in data.values()),primary_success_comparisons=sum(r['role']=='primary_density_success' for r in report['paired_comparisons']),
        analysis_sha256=file_hash(output/'analysis.json'),summary_sha256=file_hash(output/'summary.csv'),
        paired_sha256=file_hash(output/'paired_comparisons.csv')))
    return dict(valid=True,policies=len(data),scenarios_per_policy=len(manifest['scenarios']),output=str(output))


def validate_analysis(input_dir,output):
    output=Path(output);stored=json.loads((output/'analysis.json').read_text())
    stamp=json.loads((output/'validation.json').read_text())
    if (file_hash(output/'analysis.json')!=stamp['analysis_sha256'] or file_hash(output/'summary.csv')!=stamp['summary_sha256']
            or file_hash(output/'paired_comparisons.csv')!=stamp['paired_sha256']):
        raise ValueError('Statistical output checksum mismatch')
    manifest,data,provenance,sources=load_sources(input_dir)
    settings=stored['settings']
    reproduced=analyze(manifest,data,provenance,sources,settings['bootstrap_replicates'],settings['seed'])
    if stored!=reproduced:raise ValueError('Statistical analysis does not reproduce')
    return dict(valid=True,reproduced=True,policies=len(data),scenarios_per_policy=len(manifest['scenarios']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap-replicates',type=int,default=5000);p.add_argument('--seed',type=int,default=5172026)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    result=validate_analysis(a.input,a.output) if a.validate_only else run(a.input,a.output,a.bootstrap_replicates,a.seed)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
