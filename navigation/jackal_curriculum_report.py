"""Generate a Phase 6H evidence report from actual append-only run artifacts."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def wilson(k,n):
    if not n:
        return None
    z=1.96; p=k/n; denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def summarize(episodes):
    completed=[e for e in episodes if e.get('status')=='completed']
    n=len(completed); successes=sum(bool(e.get('success')) for e in completed)
    return dict(completed=n,partial_or_failed=len(episodes)-n,successes=successes,
        collisions=sum(bool(e.get('collision')) for e in completed),timeouts=sum(bool(e.get('timeout')) for e in completed),
        success_rate=successes/n if n else None,success_wilson_95=wilson(successes,n),
        mean_return=float(np.mean([e['episode_return'] for e in completed])) if n else None,
        mean_length=float(np.mean([e['length'] for e in completed])) if n else None,
        mean_path_length_odom_m=float(np.mean([e['path_length_odom_m'] for e in completed if e.get('path_length_odom_m') is not None]))
            if any(e.get('path_length_odom_m') is not None for e in completed) else None)


def write_report(directory, audit):
    p=Path(directory).resolve(); audit=Path(audit).resolve()
    summary=json.loads((p/'summary.json').read_text())
    cfg=json.loads(sorted(p.glob('config_*.json'))[0].read_text())
    baseline=json.loads((audit/'baseline_audit.json').read_text())
    episodes=records(p/'training_episodes.jsonl'); transitions=records(p/'transitions.jsonl')
    evals=records(p/'evaluation_episodes.jsonl'); updates=records(p/'updates.jsonl')
    training=defaultdict(list); development=defaultdict(list)
    for e in episodes: training[str(e['stage'])].append(e)
    for e in evals:
        if e['source']=='development': development[str(e['timesteps'])].append(e)
    fixed=[e for e in evals if e['source']=='fixed']
    # Only the latest fixed run belongs to summary; retries remain in raw logs.
    fixed=summary.get('evaluation',{}).get('episodes',fixed[-4:])
    metrics=dict(training={k:summarize(v) for k,v in training.items()},
        development={k:summarize(v) for k,v in development.items()},fixed=summarize(fixed))
    metrics['development_by_scenario']={name:summarize([e for e in evals if e['source']=='development' and e['scenario_id']==name])
        for name in sorted({e['scenario_id'] for e in evals if e['source']=='development'})}
    metrics['fixed_by_scenario']={e['scenario_id']:summarize([e]) for e in fixed}
    if transitions:
        obs=np.asarray([r['observation'] for r in transitions]); actions=np.asarray([r['action'] for r in transitions])
        metrics['transition_diagnostics']=dict(count=len(transitions),all_finite=bool(np.isfinite(obs).all()),
            observation_min=obs.min(0).tolist(),observation_max=obs.max(0).tolist(),observation_mean=obs.mean(0).tolist(),
            observation_std=obs.std(0).tolist(),action_mean=actions.mean(0).tolist(),action_std=actions.std(0).tolist(),
            saturated_fraction=float((np.max(np.abs(actions),axis=1)>=.99).mean()),
            observed_linear_velocity_mean_m_s=float(obs[:,-2].mean()*.3),
            observed_abs_angular_velocity_mean_rad_s=float(np.abs(obs[:,-1]).mean()*.6),
            reward_sum=float(sum(r['reward'] for r in transitions)))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('checkpoint_*.zip')}
    unchanged=all(Path(f).exists() and hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest
                  for f,digest in baseline['historical_hashes'].items())
    metrics['historical_6g3_artifacts_unchanged']=unchanged
    metrics['checkpoint_hashes']=hashes
    (p/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    fixed_summary=metrics['fixed']
    improvement=bool(fixed_summary['successes']) and not fixed_summary['collisions']
    mastery=(summary['curriculum']['level']==3 and summary['curriculum']['passes']>=2)
    verdict='complete' if summary['status']=='completed' and mastery and improvement else 'partial'
    lines=['# Phase 6H completion report', '',f'Verdict: **{verdict}**. Execution status: `{summary["status"]}`.', '',
        'The bounded implementation and experiment do not by themselves establish curriculum mastery or statistical superiority.', '',
        '## Objective and evidence', '',
        'Improve actual Jackal goal reaching using progressively harder tasks; preserve historical checkpoints, interface contracts and fixed comparison rules.', '',
        'The full Phase 6G.3 comparison report and completed runs were inspected. Both final fixed evaluations had 0/4 successes, 0 collisions and four 300-step timeouts. BC final mean return was 11.8742694321; scratch was 14.6290243861. Both trained 49,920 committed steps.', '',
        'Training did reach goals: BC 62/190 completed episodes; scratch 17/173. Thus global inability to move or reach goals is contradicted by the records. Final odometry path lengths were 1.43–2.21 m for BC and 1.78–2.77 m for scratch. The policies moved and reduced goal distance, but did not reach the 0.30 m tolerance.', '',
        'Inverting the unchanged timeout return formula gives approximate final open-south distances 1.402 m (BC) and 1.097 m (scratch); this is an inference from configured start distance, reward and 300-step penalty, not a directly logged final pose. All obstacle cases likewise remain roughly 2.67–3.05 m from the goal.', '',
        'BC final deterministic actions were saturated on 98–99% of steps; scratch final deterministic saturation was zero. Training angular magnitude exceeded 0.8 on 47.9% and 43.1% of steps respectively, and reverse linear actions occurred on 17.8% and 24.9%. Near-zero two-component actions were below 0.13%. These support movement and high stochastic turning, but do not prove oscillation, a wrong-direction policy, or safety blocking without per-step poses/commands.', '',
        'Final Gaussian standard deviations remained approximately one in both runs. Stochastic training successes alongside deterministic failures and loss of the initial BC open-route success are observed. Exploration/mean-policy mismatch, forgetting and restricted training distribution are hypotheses, not established root causes. No verified observation/action-scaling bug was found.', '',
        'A separately recorded Phase 6H initial-segment normalizer diagnostic found maximum absolute z-scores around 17.69 in LiDAR features (0.415% of feature values exceeded 10 standard deviations). This is measured distribution shift under the frozen BC normalizer, not proof of an incorrect normalizer or a cause of historical failures. The normalizer was preserved. Raw historical observation vectors and per-step reward component histories were not saved in steps.csv; only finiteness flags/scalar rewards are available. Missing diagnostics cannot be reconstructed as measurements. New Phase 6H transitions record observation vectors and reward components. baseline_details.json preserves loss/entropy/KL/value series ranges and validation records.', '',
        'BC discarded 2,233 steps; scratch discarded 10,463 steps. Resumption did not restore global RNG, simulator or rollout buffers. These infrastructure and stochastic differences weaken causal initialization comparisons, but do not invalidate the observed final four failures. No previous training was repeated.', '',
        '## Environment audit', '',
        'Reset pauses physics, stops commands, clears safety/action state, teleports, waits for fresh sensors, transforms world goals into the odometry frame, resets counters and contact state. Each step advances 100 physics iterations and requires newer LiDAR and odometry. Failures stop the robot.', '',
        'Observations retain 36 normalized LiDAR bins plus normalized goal distance/heading and velocities, followed by the frozen train-split BC normalizer once. Actions retain the existing Gaussian around tanh(actor raw output), SB3 clipping to [-1,1], asymmetric 0.30 m/s forward and 0.15 m/s reverse limits, 0.60 rad/s angular limit and acceleration limits. No second tanh was added.', '',
        'Reward remains 10*(previous_distance-current_distance)-0.01, plus the existing +20 goal bonus/collision adjustment. Collision has priority over goal; timeout is 300 steps. Reward is informative about net progress but a higher return does not establish navigation success. No reward/termination change was justified.', '',
        '## Curriculum and predefined criteria', '',
        'A: three short 0.75 m southern free routes with varied heading. B: four position/direction/heading variants in clear northern/southern regions. C: four routes near different sides of the existing isolated box. D: the four established comparison routes. Existing collision-enabled box world only; no new worlds. Endpoint checks reject nonfinite, out-of-bounds, occupied and already-reached goals.', '',
        'Promotion requires two distinct development evaluations, each with at least 12 completed episodes, >=80% success, <=5% collision and >=2,048 level steps. Maintain otherwise. Fallback requires two checks below 40% success or above 20% collisions after minimum dwell. Twenty percent of training samples revisit easier levels. Evaluation failures abort and cannot count as successes. Only development evaluation drives decisions; training and fixed comparison never do.', '',
        'The 10,240-step cap is an initial bounded diagnostic. Earliest D entry under these gates is 12,288 steps, so this cap cannot demonstrate full four-stage mastery. No gate was relaxed or stage forced in the experiment. Pilot A→B→A switches tested resets only.', '',
        'Predefined navigation-improvement criterion: at least one of four fixed goals reached without collisions and no integrity/finiteness/cleanup failure. Full mastery additionally requires Stage D competence. One seed and four deterministic benchmark episodes provide limited uncertainty; repeated development trials are not independent seeds.', '',
        '## Configuration, initialization and changes', '',
        f'Configuration: `{cfg}`', '',
        'PPO reuses Phase 6G.3 BC settings: n_steps=128, batch_size=64, n_epochs=4, learning_rate=3e-5, gamma=.99, gae_lambda=.95, clip_range=.1, entropy coefficient=0, value coefficient=.5, max_grad_norm=.5; one CPU environment and seed 642.', '',
        'Initialization is the original Phase 6G.1 initialized_ppo.zip, SHA-256 dff19be924f920490915d8f625d44f32da79452caaf3f7698160d2b8fc287750. Source checks also protect 6F.2 and 6G.2. Model, observation, action, reward and historical source files were not edited.', '',
        'New source files: curriculum/jackal_curriculum.py; navigation/jackal_curriculum_train.py; navigation/jackal_curriculum_report.py; configs/jackal_curriculum_6h.yaml; tests/test_jackal_curriculum.py; tests/test_jackal_curriculum_report.py; docs/phase6h_curriculum.md. All new experiment artifacts are under artifacts/ppo/jackal_6h/. Existing unrelated workspace modifications were retained.', '',
        '## Tests and pilot', '',
        'Final focused command: `.venv/bin/python -m pytest -q tests/test_jackal_curriculum.py tests/test_jackal_curriculum_report.py`; actual result 10 passed, 14 warnings, 5.29 s. The task-owned simulator and bridge PIDs were stopped after final evaluation; simulator_cleanup.json records no remaining active processes. The optional psutil cleanup attempt could not import psutil; standard Python process inspection and SIGTERM then completed cleanup. A read-only process inspection initially hit an automatic approval-review deadline; its allowed retry succeeded.', '',
        'Initial unsourced pytest collection failed because jackal_rl_navigation was not on PYTHONPATH. After sourcing ROS/workspace, the first focused run caught a completed-resume checkpoint-path bug (7 passed, 1 failed); it was corrected.', '',
        'Regression command after sourcing ROS/workspace and setting OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1:', '',
        '`.venv/bin/python -m pytest -q tests/test_jackal_curriculum.py tests/test_jackal_ppo_diagnostic.py tests/test_jackal_ppo_smoke.py tests/test_jackal_gazebo_env_contract.py tests/test_jackal_bc_ppo_transfer.py tests/test_curriculum.py`', '',
        'Actual result: 74 passed, 1 skipped (CUDA required), 14 warnings, 142.17 s. After added segment accounting/environment checks: focused test_jackal_curriculum.py result 8 passed, 14 warnings, 10.73 s. Report aggregation tests: 2 passed, 14 warnings, 3.62 s. compileall and side-effect-free preview passed. Earlier interrupted test logs remain preserved; they are not counted as successful suites.', '',
        'Additional transport regression command: `.venv/bin/python -m pytest -q tests/test_gazebo_persistent_transport.py tests/test_gazebo_controller_diagnostics.py tests/test_collision_launch_configuration.py`. Default invocation: 27 passed, 1 failed, 4.97 s. The failure was the existing warning-capture assertion: logger propagation was false in pytest, so the warning appeared on stderr but not in caplog. An isolated rerun reproduced it. With the audit-only phase6h_logging_capture plugin restoring propagation for that assertion, the unchanged tests passed: 28 passed, 14 warnings, 4.49 s. No production code or historical test was modified; the default-invocation capture issue remains a known test-environment limitation. Reproduce with `PYTHONPATH=artifacts/ppo/jackal_6h/curriculum/run_20261026_audit:$PYTHONPATH .venv/bin/python -m pytest -q -p phase6h_logging_capture tests/test_gazebo_persistent_transport.py tests/test_gazebo_controller_diagnostics.py tests/test_collision_launch_configuration.py`.', '',
        'First live pilot failed at zero steps because compiling the existing transport helper exceeded its 60-second deadline. The helper was then compiled separately successfully. A zero-step resume failed preflight because the simulator was absent; preserved. After relaunch, resume from the saved zero-step checkpoint completed 256 PPO steps in 80.1066 s, plus three zero-action reset checks. It reached two goals, had finite transitions and sensor timestamps, passed policy reload equality checks, and stopped/closed successfully. Pilot physical stage switches were A→B→A, not learned promotions. Pilot logs and each session summary are preserved separately.', '',
        '## Actual experiment', '',
        f'Committed training steps: {summary["trained_steps"]}. Runtime: {summary["runtime_s"]:.3f} s. Final checkpoint: `{summary["checkpoint"]}`.', '',
        f'Curriculum state: `{summary["curriculum"]}`', '',
        f'Shutdown: `{summary["shutdown"]}`', '',
        'Runtime includes training, development and fixed evaluation. There were no interrupted segments or discarded training steps in this experiment; the earlier pilot failures are separate. The 96.7-minute observed runtime exceeded the initial 45–70-minute estimate because evaluations required long looping episodes, but remained below the three-hour cap.', '',
        'Checkpoint hashes:', '',*[f'- `{f}`: `{h}`' for f,h in hashes.items()], '',
        '## Stage and fixed-scenario results', '',
        f'Training summaries by active curriculum level (easier replay included): `{metrics["training"]}`', '',
        f'Development evaluations by committed timestep: `{metrics["development"]}`', '',
        f'Final fixed comparison: `{fixed_summary}`', '',
        *[f'- {e["scenario_id"]}: success={e.get("success")}, collision={e.get("collision")}, timeout={e.get("timeout")}, length={e.get("length")}, return={e.get("episode_return")}' for e in fixed], '',
        'Historical BC: 0% success, 0% collision, 100% timeout, length 300, mean reward 11.8743. Historical scratch: 0% success, 0% collision, 100% timeout, length 300, mean reward 14.6290. Both used 49,920 steps versus this shorter budget; comparisons are descriptive, not matched-budget causal evidence. Historical 0/4 success Wilson 95% interval is [0, 0.490].', '',
        f'Navigation-improvement criterion met: {improvement}. Full curriculum mastery met: {mastery}. All hashed historical 6G.3 artifacts unchanged: {unchanged}.', '',
        'See metrics.json for observation/action statistics, reward totals, stage counts and uncertainty; updates.jsonl and sb3 CSV files contain policy-learning diagnostics. Partial training episodes are excluded from completed outcome rates.', '',
        '## Failure diagnosis and next bottleneck', '',
        'The experiment completed its budget but failed the predefined fixed-navigation improvement criterion. Stage A development succeeded in 36/36 episodes across three checks; Stage B failed in 0/24 successes across two checks, causing the planned fallback. C and D were never reached. This directly identifies failure to generalize beyond the short Stage A pose/goal distribution; the available experiment does not isolate position, heading, or distance as the sole cause.', '',
        'Final fixed odometry path lengths were 5.344, 5.735, 6.022 and 5.700 m, with four timeouts. Movement without sufficient net progress and heading-sensitive development path lengths support an inefficient steering/trajectory hypothesis, not stationary behavior. A separate offline final-policy diagnostic on recorded active-B training observations (including easier replay) predicted negative angular action on 97.39% of samples, versus 90.82% for the initial actor on the same inputs. Mean normalized angular action changed from -0.509 to -0.782. These are counterfactual offline predictions, not a recording of executed fixed-evaluation actions; obstacle-aware turning can legitimately point away from the local goal. See policy_diagnostic.json.', '',
        'The next concrete diagnostic should vary one factor at a time between A and B: distance, translation, then heading. Compare original BC-initialized and final policies, record per-step goal bearing, actor mean, sampled/clipped action, applied command, measured velocity and fresh sensor stamps. Test whether persistent turn bias and frozen-normalizer distribution shift explain failed generalization before adding reward terms or expanding the expensive budget. Curriculum learning did not solve navigation in this experiment.', '',
        '## Recovery, reproducibility and limitations', '',
        'Checkpoints are saved after complete PPO blocks and development decisions. Sidecars restore stage/history, model/optimizer counters and scenario sampler state and verify hashes/config. Global RNG, rollout buffer, simulator and partial episodes cannot be restored exactly. Raw failed/uncommitted transitions remain; segments.jsonl records discarded transition counts where resumed. No automatic restart is performed.', '',
        'Launch with the established `ros2 launch jackal_rl_bringup jackal_sim.launch.py`, ROS domain 63, Gazebo partition jackal_rl and loopback IPs; source ROS/workspace and activate .venv. Full shell setup is in docs/phase6h_curriculum.md.', '',
        f'Resume/evaluate: `python -m navigation.jackal_curriculum_train train --config configs/jackal_curriculum_6h.yaml --output {p} --resume`. At completed budget this performs zero training and reruns the fixed evaluation using a reloaded checkpoint. For an independent run, use a new output directory. Do not overwrite old checkpoints.', '',
        f'Rebuild report: `python -m navigation.jackal_curriculum_report --run {p} --audit {audit}`.', '',
        'The original initialized policy had historical fixed success 1/4 before Phase 6G.3 training. Matching 1/4 after curriculum would improve on the failed final baselines but would not demonstrate improvement over initialization. No contemporaneous untrained control or multiple-seed causal comparison was run. Short-route training successes alone may reflect BC competence already present. The budget cannot prove all-stage mastery; fixed training routes may eventually overlap Stage D, so they are a benchmark suite rather than an untouched generalization set. Scenario sampling is discrete and limited to one static obstacle world. Additional seeds, layouts and held-out pose distributions remain necessary.', '',
        'Phase 6I next action: benchmark verified checkpoints with a preregistered multi-seed protocol, preserve these four historical scenarios, add a separate held-out generalization suite, and compare deterministic/stochastic behavior and observed velocities. If fixed success remains zero, resolve the measured final-approach/mean-policy bottleneck before claiming curriculum solved navigation. Do not use higher reward alone as the success criterion.', '']
    if summary.get('error'):
        lines += ['## Execution failure','',json.dumps(summary['error'],indent=2),'']
    plot_results(p,episodes,development,updates)
    lines += ['## Figures', '', '![Training and development diagnostics](diagnostics.png)', '', 'Training outcomes and development evaluations remain separate. A/B annotations identify the development stage; the evaluated scenario distribution changes between stages. Development intervals describe repeated poses within one seed, not population confidence across seeds.', '']
    (p/'report.md').write_text('\n'.join(lines))
    return metrics


def plot_results(directory, episodes, development, updates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    completed=[e for e in episodes if e.get('status')=='completed']
    if completed:
        x=np.arange(1,len(completed)+1)
        axes[0,0].plot(x,[e['episode_return'] for e in completed],lw=1)
        axes[0,1].plot(x,[e['length'] for e in completed],lw=1)
    axes[0,0].set(title='Completed stochastic training episode return',xlabel='Completed episode index',ylabel='Return (reward units)')
    axes[0,1].set(title='Completed stochastic training episode length',xlabel='Completed episode index',ylabel='Length (environment steps)')
    if development:
        x=sorted(int(k) for k in development)
        axes[1,0].plot(x,[summarize(development[str(k)])['success_rate'] for k in x],marker='o',label='Current-stage development')
        for step in x:
            name=development[str(step)][0]['scenario_id'].split('_')[0]
            value=summarize(development[str(step)])['success_rate']
            axes[1,0].annotate(name,(step,value),xytext=(0,-15 if value>.5 else 10),textcoords='offset points',ha='center')
        axes[1,0].legend()
    axes[1,0].set(title='Deterministic development goal success',xlabel='Committed PPO environment steps',ylabel='Success fraction',ylim=(-.05,1.05))
    if updates:
        x=[u['timesteps'] for u in updates]
        axes[1,1].plot(x,[u['metrics'].get('train/value_loss',np.nan) for u in updates],label='PPO value loss')
        axes[1,1].legend()
    axes[1,1].set(title='PPO value-learning diagnostic',xlabel='Committed PPO environment steps',ylabel='Value loss (framework units)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Phase 6H · one seed · source: run JSONL logs, entire recorded run',fontsize=12)
    fig.savefig(directory/'diagnostics.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); parser.add_argument('--audit',required=True)
    args=parser.parse_args(); write_report(args.run,args.audit)

if __name__=='__main__': main()
