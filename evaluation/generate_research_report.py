"""Generate a GitHub-ready Markdown report using verified saved experiment evidence."""
import argparse
import os
import inspect
from pathlib import Path
from evaluation.compare_ppo_variants import read_json
from evaluation.scenario_manifest import file_hash,save_json,digest
from visualization.benchmark_plots import evidence,validate_figures,density_rows


def number(x,digits=3):
    return 'N/A' if x is None else f'{x:.{digits}f}'


def probability(x):
    return 'N/A' if x is None else f'{x:.3g}'


def ci_text(mean,ci,percent=False):
    if mean is None or ci is None:return 'N/A'
    scale=100 if percent else 1;unit='%' if percent else ''
    return f'{number(mean*scale)}{unit} [{number(ci[0]*scale)}, {number(ci[1]*scale)}]'


def render(root,output,manifest,data,stats,figures,figure_index):
    root=Path(root);output=Path(output);rows=density_rows(stats)
    base=read_json(root/'baselines/summary.json');ppo=read_json(root/'ppo_variants/index.json')
    n=len(manifest['scenarios']);series=len(data)
    lines=['# GridWorld Robot Navigation: Reproducible Benchmark',
        '', '## Scope and evidence', '',
        f'This report summarizes {series} fixed outcome series on {n} held-out scenarios each, comprising {sum(len(r) for r in data.values())} saved episode rows. All source validators and Phase 5D statistical recomputation passed before report generation. PPO best/final selections share training histories and are not independent training seeds.',
        '',f"Scenario manifest: `{manifest['manifest_id']}`. Numerical results below are observed saved outcomes, not estimates from smoke tests or invented training runs.",
        '', '## Problem and algorithm development', '',
        'The task is fully observed four-action navigation on obstacle grids. The original baseline is CNN Double DQN: an online Q-network selects greedy actions during evaluation; Double DQN separates next-action selection from target-value estimation during training. The revised approach first uses A* shortest-path demonstrations for supervised CNN Behavior Cloning, then initializes a PPO actor from BC and fine-tunes it. Curriculum PPO inherits a selected pretrained PPO checkpoint and changes the training-map difficulty over stages. Scratch PPO provides a control without BC actor pretraining. A* is an optimal planning reference; random actions provide an untrained baseline.',
        '', 'A* has the same map and endpoint information as the neural observation, but explicitly plans at inference time. Its optimality is a reference for this known, fully observed grid problem, not evidence of learned generalization. The DQN experiment YAML inspected before benchmarking specifies obstacle density **0.05** despite its `density05` name and earlier descriptions of 0.5. Checkpoint counters and hashes identify what was actually evaluated; the filename does not verify density-0.5 training.',
        '', '## Methodology', '',
        f"Environment settings: `{manifest['config']['environment']}`. Densities are evaluated separately at 0.1, 0.3 and 0.5; minimum accepted shortest-path length is {manifest['config']['min_path_length']}.",
        '', 'Every series receives the identical scenarios in the same order. Observations are binary float32 obstacle/agent/goal channels; actions 0/1/2/3 are up/down/left/right. Step reward is -1, blocked movement adds -10, Manhattan progress coefficient is 2, and reaching the goal adds 100. BFS shaping, masking, expert rescue and loop recovery are disabled. Goal arrival at the horizon counts as success; otherwise the episode times out. Neural evaluation is greedy/deterministic with no learning or checkpoint updates. Random actions use the recorded reproducible map-specific RNG stream.',
        '', 'A* optimal lengths are checked against BFS and actual environment replay. Canonical obstacle identity groups rotations/reflections and ignores endpoints. All finite supplied train/validation/old-test/development pools are excluded from the reserved test set. This benchmark is conditioned on the accepted, solvable-map distribution, minimum path filter and curriculum reserved hash bucket.',
        '',f"**Leakage scope:** {manifest.get('leakage_scope','Not recorded')}. Historical random DQN coverage was not exhaustively logged; declared seed exclusions do not prove disjointness from every past random experiment.",
        '', 'Metrics: success and timeout use all episodes. Collisions count blocked boundary/obstacle moves; consecutive identical blocked attempts are also recorded as a subset, not extra collisions. Path length counts successful cell moves, while episode length counts all action attempts. Successful path efficiency is optimal steps / episode length and is null on failure; SPL uses that ratio on success and zero on failure. This penalizes collisions and retains failure costs.',
        '', '## Observed held-out results', '',
        'Each bracket is a 95% interval. Success uses Wilson intervals; collisions and SPL use scenario bootstrap intervals. Collision frequency means collisions per episode, not a probability. All density rows retain failed episodes.',
        '', '| Outcome series | Density | Success / n | Success % [95% CI] | Collisions/episode [95% CI] | SPL [95% CI] | Timeout % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda x:(x['density'],x['policy'])):
        s=r['statistics'];c=s['metrics']['collisions'];spl=s['metrics']['spl']
        lines.append(f"| {r['policy']} | {r['density']} | {s['successes']}/{s['n']} | {ci_text(s['success_rate'],s['success_wilson95'],True)} | {ci_text(c['mean'],c['bootstrap95'])} | {ci_text(spl['mean'],spl['bootstrap95'])} | {number(s['metrics']['timeout']['mean']*100)} |")
    def figure(filename,alt):
        relative=os.path.relpath(Path(figures)/filename,output.parent).replace(os.sep,'/')
        return f'![{alt}]({relative})'
    lines+=['','## Figures','',figure('success_rates.png','Success rates with Wilson confidence intervals'),'',
            figure('collisions.png','Collision frequency with bootstrap confidence intervals'),'',
            figure('path_efficiency_spl.png','Failure-sensitive SPL with bootstrap confidence intervals'),'',
            figure('successful_path_efficiency.png','Successful-only efficiency and sample sizes'),'',
            'Successful-only efficiency conditions on success and can conceal failures; interpret it alongside all-episode SPL and the successful sample counts. A missing estimate means no successes, not zero efficiency.']
    if figure_index['historical']['generated']:
        lines+=['',figure('historical_sample_efficiency.png','Historical validation success by training interaction cost and scenario signature'),'',
            'Historical curves contain only logged validation observations. Different validation signatures appear in separate panels; lines merely connect observed logged points. No held-out test learning curve, intermediate measurement, or unsupported confidence band is invented.']
    lines+=['','## Paired comparisons and statistical reliability','',
        f"Bootstrap settings: {stats['settings']['bootstrap_replicates']} replicates, seed {stats['settings']['seed']}, percentile 95% intervals, NumPy {stats['settings']['numpy_version']}. Paired bootstrap resamples within-scenario difference vectors. All intervals describe variation across evaluation maps conditional on the saved checkpoints, not training-seed variability.",
        '', 'Success tests use exact two-sided McNemar discordant-pair tests with Holm correction over all density-only success comparisons. Other outcome intervals and distance/complexity subgroups are descriptive/exploratory and not simultaneous significance statements. A zero-width bootstrap interval for constant outcomes does not prove certainty or equivalence. The following rows preserve the stored A-minus-B direction.',
        '', '| A | B | Density | Paired n | Success difference [95% CI] | A-only/B-only successes | Holm-adjusted p | Holm rejects at 0.05 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    def selected_pair(a,b):
        if {a,b}=={'bc','dqn'}:return True
        if '/' not in a or '/' not in b:return False
        aa,sa=a.split('/');bb,sb=b.split('/')
        return sa==sb and {aa,bb} in ({'ppo_bc','ppo_scratch'},{'ppo_curriculum','ppo_bc'})
    for r in stats['paired_comparisons']:
        if r['role']!='primary_density_success' or not selected_pair(r['policy_a'],r['policy_b']):continue
        s=r['statistics'];d=s['metrics']['success']
        lines.append(f"| {r['policy_a']} | {r['policy_b']} | {r['density']} | {s['n_pairs']} | {ci_text(d['mean'],d['bootstrap95'])} | {s['a_only_success']}/{s['b_only_success']} | {probability(s.get('success_mcnemar_holm_p'))} | {s.get('holm_reject_at_0_05',False)} |")
    lines+=['','Adjusted p-values are evidence about fixed checkpoint outcomes in the defined success-test family; they do not establish causal improvement across training seeds. Effect sizes and uncertainty must accompany any rejection decision.',
        '', '## Training cost and historical sample efficiency', '',
        '| PPO series | Selected model steps | Completed online search transitions | Offline expert trajectories | Offline expert transitions | Training status |',
        '|---|---:|---:|---:|---:|---|']
    for name in ppo['variants']:
        for choice in ppo['selections']:
            meta=read_json(root/'ppo_variants'/name/choice/'summary.json')['checkpoint'];e=meta['expert_data']
            lines.append(f"| {name}/{choice} | {meta['checkpoint_model_timesteps']} | {meta['completed_search_online_interactions']} | {e['trajectories']} | {e['transitions']} | {meta['training_status']} |")
    dqn=base['provenance']['dqn']['checkpoint']
    lines+=['',f"DQN checkpoint model training-step counter: {dqn['training_steps']}. This selected-checkpoint counter is not automatically the complete experiment/search budget. BC uses offline expert data; its repeated supervised epochs are not additional environment transitions.",
        '', 'Selected model age differs from completed search cost: training after the best checkpoint, discarded branches and inherited search can add cost. Curriculum inherits both the BC data and earlier PPO search; the inherited expert dataset is not added twice. These three PPO approaches are not automatically budget-matched. No wall-clock training-cost or multi-seed sample-efficiency claim is made.',
        '', '| Historical series | Density | Validation signature | Logged evaluations | First logged success >= 0.8 at consumed online steps |',
        '|---|---:|---|---:|---:|']
    hist=ppo['historical_sample_efficiency']
    for name,groups in hist['curves'].items():
        for group in groups.values():
            crossing=group['first_logged_success_at_least_threshold']
            lines.append(f"| {name} | {group['density']} | `{group['validation_signature'][:12]}` | {group['evaluations']} | {crossing if crossing is not None else 'Not reached in available logs'} |")
    lines+=['','A first logged threshold crossing is not sustained mastery and is not interpolated between observations. Historical validation sets and the current held-out benchmark are distinct. Scratch/pretrained historical comparisons require matching configuration, signature, density and consumed cost; curriculum validation histories with different signatures are not directly paired.',
        '', '## Distance and map-complexity reliability', '',
        'Phase 5D separately stratifies each density by Manhattan distance (0–5, 6–9, 10–14, 15+), optimal path length (1–9, 10–14, 15+), and reachable dead-end fraction (<0.10, [0.10,0.25), >=0.25). Complexity uses the fraction of reachable free cells with one free four-connected neighbor. These are observation-only predefined descriptors, not a universal difficulty score or causal variables. Empty strata are retained and n<30 is marked sparse.',
        '', '<details><summary>Full distance and complexity table</summary>', '', '| Axis | Bin | Density | Series | n | Success % | Sparse |', '|---|---|---:|---|---:|---:|---|']
    for r in stats['descriptive']:
        if r['axis']=='density':continue
        s=r['statistics'];lines.append(f"| {r['axis']} | {r['bin']} | {r['density']} | {r['policy']} | {s['n']} | {number(s['success_rate']*100) if s['n'] else 'N/A'} | {s['sparse']} |")
    lines+=['','</details>','','## Verified example trajectories','',
        'Examples are selected as the first saved success and first saved failure in frozen scenario order for each density and series. No missing outcome category is fabricated. Because original CSVs contain no action trace, examples are explicitly deterministic replays; saved success, timeout, length, reward, collisions, movement count, efficiency and SPL must match before a figure is accepted. Blocked attempts are marked at the current cell; repeated loops may overlap. Start is green and goal is orange.']
    for name in sorted(data):lines+=['',figure(name.replace('/','_')+'_routes.png',name+' verified successful and failed trajectories')]
    lines+=['',f"Verified replay examples: {figure_index['route_replays_verified']}. Availability and full replay actions/routes are recorded in `trajectory_replays.json` in the figures directory.",
        '', '## Observations, hypotheses and conclusions','',
        'Observed: the tables and figures report exact saved outcome counts and interval estimates. Per-density observed success ordering is shown below; ordering by a point estimate alone does not establish a statistically reliable winner.']
    for density in (.1,.3,.5):
        group=sorted([r for r in rows if r['density']==density],key=lambda r:(-r['statistics']['success_rate'],r['policy']))
        lines+=['',f"- Density {density}: "+'; '.join(f"{r['policy']} {r['statistics']['successes']}/{r['statistics']['n']}" for r in group)+'.']
    lines+=['','Hypotheses, not established mechanisms: differences may reflect expert-data coverage, policy optimization stability, training-map distribution or curriculum promotion/budget settings. This benchmark does not isolate those causes. Repeated blocked actions or cycles visible in replay examples describe behavior; they do not establish why training produced it.',
        '', 'Conclusion: the study provides a reproducible comparison of these exact checkpoints on the frozen solvable-map distribution. Claims about BC pretraining or curriculum improvement must use the density-specific paired effects and cost table, with checkpoint selection held explicit. One training seed/lineage cannot establish a general algorithm advantage, and best/final selections are not independent replications.',
        '', '## Limitations','',
        '- Evaluation uncertainty and training-seed variability are different. The latter is not estimated here.',
        '- A* has full map observability and planning computation; this benchmark is not unknown-map robotic navigation or physical deployment.',
        '- Maps are small, static and filtered for solvability/minimum path length; results do not establish transfer to larger/dynamic maps.',
        '- Finite pool exclusions are exact, but exhaustive historical random DQN map coverage cannot be reconstructed from the supplied logs.',
        '- Conditional efficiency excludes failures by definition; all-episode SPL, success and collision costs must remain visible.',
        '- Sparse subgroup results, many exploratory comparisons and degenerate bootstrap intervals require caution.',
        '- Historical learning curves use their original validation maps and aggregate logs; missing confidence bands or histories are not filled in.',
        '', '## Reproducibility','',
        'Run from the repository root with the existing environment; preserve source artifacts. All commands refuse existing output paths.',
        '', '```bash', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_report_generation.py -v -p no:cacheprovider',
        'python -m visualization.benchmark_plots --input artifacts/benchmarks --output artifacts/benchmarks/figures',
        'python -m evaluation.generate_research_report --input artifacts/benchmarks --output reports/gridworld_benchmark.md',
        'python -m evaluation.generate_research_report --input artifacts/benchmarks --output reports/gridworld_benchmark.md --validate-only', '```',
        '',f"Phase 5B runtime: `{base['runtime']}`. Phase 5C runtime: `{ppo['runtime']}`. Bootstrap settings: `{stats['settings']}`.",
        '', '### Checkpoint fingerprints','', '| Series | Checkpoint | SHA256 |', '|---|---|---|']
    for name in ('dqn','bc'):
        cp=base['provenance'][name]['checkpoint'];lines.append(f"| {name} | `{cp['path']}` | `{cp['sha256']}` |")
    for name in ppo['variants']:
        for choice in ppo['selections']:
            cp=read_json(root/'ppo_variants'/name/choice/'summary.json')['checkpoint'];lines.append(f"| {name}/{choice} | `{cp['path']}` | `{cp['sha256']}` |")
    lines+=['','The report sidecar records source and figure fingerprints. Keep the repository-relative figure paths and evidence directories together when publishing to GitHub; omit large training checkpoints from the Git repository as appropriate without claiming independent artifact validation is possible when their bytes are unavailable.','']
    return '\n'.join(lines)


def generate(root,output,figures=None,validate_only=False):
    root=Path(root);output=Path(output);figures=Path(figures) if figures else root/'figures'
    sidecar=output.with_suffix(output.suffix+'.provenance.json')
    if not validate_only and (output.exists() or sidecar.exists()):raise FileExistsError('Report or provenance file exists: '+str(output))
    manifest,data,stats,sources=evidence(root);figure_index=validate_figures(root,figures,manifest)
    text=render(root,output,manifest,data,stats,figures,figure_index)
    if validate_only:
        saved=read_json(sidecar)
        if file_hash(output)!=saved['report_sha256'] or output.read_text()!=text:raise ValueError('Report does not reproduce')
        for path,sha in saved['sources'].items():
            if file_hash(path)!=sha:raise ValueError('Report source changed')
        return dict(valid=True,reproduced=True,report=str(output))
    for path,sha in sources.items():
        if file_hash(path)!=sha:raise RuntimeError('Report evidence changed during generation')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(text)
    sources[str(figures/'figures_manifest.json')]=file_hash(figures/'figures_manifest.json')
    sources[str(Path(__file__).resolve())]=file_hash(Path(__file__).resolve())
    save_json(sidecar,dict(schema_version=1,manifest_id=manifest['manifest_id'],report_sha256=file_hash(output),sources=sources,
        figures=str(figures),policy_series=len(data),source_files=[str(Path(__file__))]))
    return dict(valid=True,report=str(output),policy_series=len(data))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--figures');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    import json
    print(json.dumps(generate(a.input,a.output,a.figures,a.validate_only),indent=2))


if __name__=='__main__':main()
