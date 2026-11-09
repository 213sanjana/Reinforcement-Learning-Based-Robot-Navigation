"""Phase 6H adapter using the audited Jackal PPO initialization and evaluator.

No simulator is launched here. Preview never constructs an environment. Pilot
is capped independently of the experiment. Checkpoints are committed only after
complete PPO updates and development decisions. Interrupted logs stay append-only.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from agents.jackal_bc_anchored_ppo import JackalBCAnchoredPPO, adopt_initialized_model
from curriculum.jackal_curriculum import JackalCurriculum, stages
from navigation import jackal_ppo_diagnostic as diagnostic
from navigation.jackal_ppo_train import (_make_jackal_env, _save_checkpoint, _shutdown,
    _stop_robot, _write_json, TimeLimitInfoWrapper, StopOnFailureWrapper)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')


def _mapping(config, key):
    value=config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    return value


def _positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a real number')
    if not math.isfinite(float(value)) or float(value)<=0:
        raise ValueError(f'{label} must be finite and positive')
    return float(value)


_REPO = Path(__file__).resolve().parents[1]
ANCHORED_OUTPUT_NAMESPACE = (
    _REPO / 'artifacts' / 'ppo' / 'jackal_6h_bc_anchored' / 'curriculum'
).resolve()


def _normalize_directory(path):
    """Resolve a configured path from the repository root without creating it."""

    path = Path(path).expanduser()
    if not path.is_absolute():
        path = _REPO / path
    return path.resolve()


def _historical_locations():
    """Protected historical artifact locations. Ancestors and descendants are included."""

    roots = (
        _REPO / 'artifacts' / 'ppo' / 'jackal_6h',
        _REPO / 'artifacts' / 'ppo' / 'jackal_6g1',
        _REPO / 'artifacts' / 'ppo' / 'jackal_6g2',
        _REPO / 'artifacts' / 'behavior_cloning',
    )
    resolved = []
    for path in (*roots, *diagnostic.FORBIDDEN_OUTPUT_ROOTS):
        candidate = Path(path).resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _historical_relation(candidate, protected):
    if candidate == protected:
        return 'historical root'
    try:
        candidate.relative_to(protected)
    except ValueError:
        pass
    else:
        return 'historical descendant'
    try:
        protected.relative_to(candidate)
    except ValueError:
        return None
    return 'historical ancestor'


def _reject_historical(path, label):
    for protected in _historical_locations():
        relation = _historical_relation(path, protected)
        if relation is not None:
            raise ValueError(
                f'{label} is a {relation} of protected historical artifact location {protected}'
            )


def validate_anchored_output(output_root, output):
    """Reject anchored paths outside the dedicated namespace or on historical artifacts.

    Both paths are normalized before the check. This function does not create
    or modify directories.
    """

    root = _normalize_directory(output_root)
    target = _normalize_directory(output)
    _reject_historical(root, 'Anchored output_root')
    _reject_historical(target, 'Anchored output')
    namespace = ANCHORED_OUTPUT_NAMESPACE
    if root != namespace:
        raise ValueError(
            'Anchored output_root resolves outside the anchored output namespace '
            f'{namespace}: {root}'
        )
    if target == namespace or namespace not in target.parents:
        raise ValueError(
            'Anchored output resolves outside the anchored output namespace '
            f'{namespace}: {target}'
        )


def _validate_anchored_config(config):
    if config.get('base_config')!='configs/jackal_ppo_diagnostic_6g3_bc.yaml':
        raise ValueError('BC-anchored curriculum must use the Phase 6G.3 BC initialization config')
    anchoring=_mapping(config, 'actor_anchoring')
    gate=_mapping(config, 'efficiency_gate')
    if anchoring.get('enabled') is not True or gate.get('enabled') is not True:
        raise ValueError('BC-anchored curriculum enables actor anchoring and the efficiency gate')
    if anchoring.get('expected_sha256')!=diagnostic.OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError('Anchor checkpoint hash does not match the verified Phase 6G.1 checkpoint')
    declared=Path(anchoring.get('checkpoint') or '')
    if not declared.is_absolute():
        declared=_REPO/declared
    if declared.resolve()!=diagnostic.OFFICIAL_CHECKPOINT.resolve():
        raise ValueError('Anchor checkpoint must be the verified Phase 6G.1 initialized_ppo.zip')
    _positive_finite(anchoring.get('coefficient'), 'Anchor coefficient')
    if gate.get('definition')!='net_goal_progress_over_path_length':
        raise ValueError('Efficiency definition must be net_goal_progress_over_path_length')
    _positive_finite(gate.get('threshold'), 'Efficiency threshold')
    output_root=_normalize_directory(config.get('output_root') or '')
    _reject_historical(output_root, 'Anchored output_root')
    if output_root != ANCHORED_OUTPUT_NAMESPACE:
        raise ValueError(
            'Anchored output_root resolves outside the anchored output namespace '
            f'{ANCHORED_OUTPUT_NAMESPACE}: {output_root}'
        )


def load_config(path):
    import yaml
    c=yaml.safe_load(Path(path).read_text())
    if c.get('phase') not in ('6h', '6h_bc_anchored'):
        raise ValueError('Expected phase 6h or 6h_bc_anchored')
    for key in ('seed','total_timesteps','interval','pilot_timesteps','development_episodes','min_level_steps','required_passes'):
        if type(c.get(key)) is not int or c[key]<1:
            raise ValueError(f'Invalid {key}')
    if (c['total_timesteps']%128 or c['interval']%128 or c['pilot_timesteps']%128
            or c['pilot_timesteps']>512 or c['development_episodes']<12
            or c['total_timesteps']>10240 or c['required_passes']<2
            or not 0<c['wall_clock_timeout_s']<=10800 or not 0<c['pilot_timeout_s']<=1200):
        raise ValueError('Invalid or unbounded curriculum budget')
    if c['phase']=='6h':
        if 'actor_anchoring' in c or 'efficiency_gate' in c:
            raise ValueError('Historical phase 6h cannot carry BC anchoring or efficiency-gate settings')
    else:
        _validate_anchored_config(c)
    stages()
    return c


def _anchoring(config):
    return config.get('actor_anchoring') or {'enabled': False, 'coefficient': 0.0}


def _efficiency(config):
    return config.get('efficiency_gate') or {'enabled': False, 'threshold': None}


def _curriculum(config):
    gate=_efficiency(config)
    kwargs=dict(min_steps=config['min_level_steps'], min_episodes=config['development_episodes'],
                required_passes=config['required_passes'])
    if gate.get('enabled'):
        kwargs.update(efficiency_enabled=True, efficiency_threshold=float(gate['threshold']))
    return JackalCurriculum(**kwargs)


def load_curriculum_model(path, config):
    """Load a committed curriculum model, restoring a saved BC reference when anchored."""

    anchoring=_anchoring(config)
    if not anchoring.get('enabled'):
        return PPO.load(path, device='cpu')
    model=JackalBCAnchoredPPO.load(path, device='cpu')
    provenance=model.anchor_provenance or {}
    if provenance.get('source_checkpoint_sha256')!=anchoring['expected_sha256']:
        raise ValueError('Loaded BC reference provenance does not match the configured checkpoint')
    if float(model.anchor_coefficient)!=float(anchoring['coefficient']):
        raise ValueError('Loaded anchor coefficient does not match the configuration')
    return model


def _same_state(model, restored):
    import torch
    restored_policy=restored.policy.state_dict()
    if any(not torch.equal(tensor.detach().cpu(), restored_policy[key].detach().cpu())
           for key, tensor in model.policy.state_dict().items()):
        return False
    left=getattr(model, 'reference_policy', None)
    right=getattr(restored, 'reference_policy', None)
    if (left is None)!=(right is None):
        return False
    if left is None:
        return True
    restored_reference=right.state_dict()
    return all(torch.equal(tensor.detach().cpu(), restored_reference[key].detach().cpu())
               for key, tensor in left.state_dict().items())


class CurriculumEnv(gym.Wrapper):
    def __init__(self, env, manager, rng, output, deadline):
        super().__init__(env)
        self.manager=manager; self.rng=rng; self.output=Path(output); self.deadline=deadline
        self.open_episode=None
        self.steps=0
        self.segment_id=str(time.time_ns())

    def check_deadline(self):
        if time.monotonic()>=self.deadline:
            raise TimeoutError('Phase 6H wall-clock cap exceeded')

    def partial(self, reason):
        if self.open_episode and self.open_episode['length']:
            append(self.output/'training_episodes.jsonl',dict(self.open_episode,status='partial',interruption=reason))
        self.open_episode=None

    def reset(self, *, seed=None, options=None):
        self.check_deadline()
        self.partial('reset_or_evaluation_boundary')
        scenario=self.manager.sample(self.rng)
        obs,info=self.env.reset(seed=seed,options={'start':scenario.start,'goal':scenario.goal})
        self._finite(obs)
        self.open_episode=dict(source='training',stage=self.manager.level,segment_id=self.segment_id,scenario_id=scenario.scenario_id,
            start=scenario.start,goal=scenario.goal,length=0,episode_return=0.,success=False,collision=False,timeout=False)
        return obs,info

    @staticmethod
    def _finite(obs):
        if np.shape(obs)!=(40,) or not np.isfinite(obs).all():
            raise ValueError('Invalid or nonfinite observation')

    def step(self,action):
        self.check_deadline()
        try:
            if np.shape(action)!=(2,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
                raise ValueError('Invalid action contract')
            obs,reward,term,trunc,info=self.env.step(action)
            self._finite(obs)
            if not np.isfinite(reward):
                raise ValueError('Nonfinite reward')
            self.steps+=1
            e=self.open_episode
            if e is None:
                raise RuntimeError('Step before reset')
            e['length']+=1; e['episode_return']+=float(reward)
            append(self.output/'transitions.jsonl',dict(source='training',stage=e['stage'],segment_id=self.segment_id,step=self.steps,
                action=np.asarray(action).tolist(),reward=float(reward),observation=np.asarray(obs).tolist(),
                diagnostics={k:info.get(k) for k in ('progress','progress_reward','step_penalty','current_distance',
                    'scan_stamp','odom_stamp','termination_reason')},terminated=bool(term),truncated=bool(trunc)))
            if term or trunc:
                e.update(success=bool(info.get('goal_reached')) and not bool(info.get('collision_detected')),
                         collision=bool(info.get('collision_detected')),timeout=bool(info.get('time_limit_reached')),
                         status='completed',termination_reason=info.get('termination_reason'))
                if sum((e['success'],e['collision'],e['timeout']))!=1:
                    raise RuntimeError('Ambiguous or missing episode outcome')
                append(self.output/'training_episodes.jsonl',e)
                self.open_episode=None
            return obs,reward,term,trunc,info
        except BaseException:
            _stop_robot(self)
            raise


class DeadlineEnv(gym.Wrapper):
    """Apply the same wall-time cap during training, reset and evaluation."""
    def __init__(self, env, deadline):
        super().__init__(env)
        self.deadline=deadline
    def check(self):
        if time.monotonic()>=self.deadline:
            _stop_robot(self)
            raise TimeoutError('Phase 6H wall-clock cap exceeded')
    def reset(self, **kwargs):
        self.check()
        return self.env.reset(**kwargs)
    def step(self, action):
        self.check()
        return self.env.step(action)


class DiagnosticsCallback(BaseCallback):
    def __init__(self, output):
        super().__init__(); self.output=output
    def _on_step(self):
        raw=np.asarray(self.locals['actions'])
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite raw PPO action')
        append(self.output/'policy_actions.jsonl',dict(timesteps=self.num_timesteps,raw=raw.tolist(),
            clipped=np.asarray(self.locals['clipped_actions']).tolist()))
        return True


def evaluate(model, env, scenarios, seed, count, output, kind, steps):
    # Reuse fixed evaluator; round-robin development set has exactly count episodes.
    episodes=[]
    for i in range(count):
        episode=diagnostic._evaluate_episode(model,env,scenarios[i%len(scenarios)],seed,i,True,
            episode_sink=lambda e: append(output/'evaluation_episodes.jsonl',dict(e,source=kind,timesteps=steps)))
        episodes.append(episode)
    return episodes


def commit(model, manager, rng, output, fingerprint, config):
    path=output/f'checkpoint_{model.num_timesteps:08d}.zip'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    _save_checkpoint(model,path)
    restored=load_curriculum_model(path, config)
    if restored.num_timesteps!=model.num_timesteps:
        raise RuntimeError('Reload timestep mismatch')
    if not _same_state(model, restored):
        raise RuntimeError('Reload parameter mismatch')
    provenance=getattr(model, 'anchor_provenance', None)
    meta=dict(phase=config['phase'],checkpoint=str(path.resolve()),sha256=diagnostic.file_sha256(path),
        timesteps=int(model.num_timesteps),n_updates=int(model._n_updates),curriculum=manager.state_dict(),
        sampler_rng=rng.bit_generator.state,config_sha256=fingerprint,
        anchor_reference_sha256=None if not provenance else provenance.get('reference_sha256'),
        recorded_transitions=sum(1 for _ in (output/'transitions.jsonl').open()) if (output/'transitions.jsonl').exists() else 0,
        resume_limitations=diagnostic.RESUME_LIMITATIONS)
    _write_json(path.with_suffix('.json'),meta)
    _write_json(output/'latest.json',meta)
    return meta


def _accept_resume(output, config, fingerprint):
    """Read resume metadata and load the checkpoint. This function does not write."""

    latest = output / 'latest.json'
    if not latest.is_file():
        raise ValueError('No committed checkpoint to resume')
    meta = json.loads(latest.read_text())
    if meta.get('config_sha256') != fingerprint:
        raise ValueError('Resume config or checkpoint hash mismatch')
    checkpoint = meta.get('checkpoint')
    if not isinstance(checkpoint, str) or diagnostic.file_sha256(checkpoint) != meta.get('sha256'):
        raise ValueError('Resume config or checkpoint hash mismatch')
    manager = JackalCurriculum(**meta['curriculum'])
    model = load_curriculum_model(checkpoint, config)
    if _anchoring(config).get('enabled'):
        if meta.get('anchor_reference_sha256') != model.anchor_provenance.get('reference_sha256'):
            raise ValueError(
                'Resume sidecar does not match the saved BC reference. '
                'The updated actor was not copied into the reference.'
            )
    if manager.steps != model.num_timesteps:
        raise ValueError('Curriculum/model timestep mismatch')
    return dict(meta=meta, model=model, manager=manager)


def _rejected_resume(config, mode, exc):
    """In-memory failure record. The target directory is left unchanged."""

    target = config['pilot_timesteps'] if mode == 'pilot' else config['total_timesteps']
    return dict(
        phase=config['phase'], mode=mode, status='failed', target_steps=target,
        trained_steps=0, checkpoint=None,
        error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()),
    )


def run(config, output, *, mode='pilot', resume=False, environment_factory=None):
    if mode not in ('pilot','train'):
        raise ValueError('Only explicit pilot/train may execute')
    c=copy.deepcopy(config)
    output=_normalize_directory(output)
    if _anchoring(c).get('enabled'):
        validate_anchored_output(c['output_root'], output)
    else:
        root=_normalize_directory(c['output_root'])
        if output == root or not output.is_relative_to(root):
            raise ValueError('Output must be a new run below the configured output root')
    fingerprint=diagnostic.config_sha256(c)
    resumed=None
    if resume:
        try:
            resumed=_accept_resume(output, c, fingerprint)
        except Exception as exc:
            return _rejected_resume(c, mode, exc)
    elif output.exists():
        raise FileExistsError(output)
    base=diagnostic.load_config(c['base_config'])
    diagnostic.verify_source_hashes(base)
    target=c['pilot_timesteps'] if mode=='pilot' else c['total_timesteps']
    deadline=time.monotonic()+(c['pilot_timeout_s'] if mode=='pilot' else c['wall_clock_timeout_s'])
    manager=resumed['manager'] if resume else _curriculum(c)
    rng=np.random.default_rng(c['seed'])
    summary=dict(phase=c['phase'],mode=mode,status='failed',target_steps=target,trained_steps=0,checkpoint=None)
    started=time.monotonic(); model=None; raw=None; wrapped=None
    writes_allowed=False
    try:
        if not resume:
            output.mkdir(parents=True,exist_ok=False)
        writes_allowed=True
        _write_json(output/f'config_{time.time_ns()}.json',c)
        if resume:
            meta=resumed['meta']
            model=resumed['model']
            rng.bit_generator.state=meta['sampler_rng']
            summary.update(checkpoint=meta['checkpoint'],trained_steps=int(model.num_timesteps))
            append(output/'segments.jsonl',dict(event='resume',timesteps=model.num_timesteps,
                limitations=diagnostic.RESUME_LIMITATIONS,
                discarded_transitions_since_checkpoint=max(0,(sum(1 for _ in (output/'transitions.jsonl').open())
                    if (output/'transitions.jsonl').exists() else 0)-meta.get('recorded_transitions',0))))
        else:
            model,initialization=diagnostic.create_model(base)
            if _anchoring(c).get('enabled'):
                model=adopt_initialized_model(
                    model,
                    coefficient=float(c['actor_anchoring']['coefficient']),
                    provenance=dict(
                        source_checkpoint=initialization['source_checkpoint'],
                        source_checkpoint_sha256=initialization['source_checkpoint_sha256'],
                        sb3_version='2.9.0',
                    ),
                )
                if model.anchor_provenance['source_checkpoint_sha256']!=c['actor_anchoring']['expected_sha256']:
                    raise ValueError('Initialized checkpoint hash does not match the anchored configuration')
                _write_json(output/'anchor_reference.json', model.anchor_provenance)
            _write_json(output/'initialization.json',initialization)
        if environment_factory is None:
            diagnostic.preflight_simulator()
            environment_factory=lambda: _make_jackal_env(base)
        raw=DeadlineEnv(environment_factory(),deadline)
        diagnostic._check_environment(raw,base)
        controller=getattr(raw.unwrapped,'controller',None)
        if controller is not None and hasattr(controller,'diagnostics_path'):
            controller.diagnostics_path=output/f'control_{time.time_ns()}.jsonl'
        wrapped=CurriculumEnv(StopOnFailureWrapper(TimeLimitInfoWrapper(raw)),manager,rng,output,deadline)
        model.set_env(wrapped)
        model.set_logger(configure(str(output/f'sb3_{time.time_ns()}'),['csv']))
        if not resume:
            summary['checkpoint']=commit(model,manager,rng,output,fingerprint,c)['checkpoint']
        if mode=='pilot' and model.num_timesteps<target:
            # Explicit smoke switches test physical resets only, never promotion competence.
            for level in (0,1,0):
                manager.level=level
                obs,info=wrapped.reset()
                wrapped.step(np.zeros(2,dtype=np.float32))
                append(output/'pilot_resets.jsonl',dict(stage=level,finite=bool(np.isfinite(obs).all()),info=diagnostic._plain(info)))
            manager.level=0
            wrapped.partial('pilot_reset_checks')
        while model.num_timesteps<target:
            wrapped.check_deadline()
            before=int(model.num_timesteps)
            block=min(c['interval'],target-before)
            # Reset after evaluation so no simulator state crosses the boundary.
            model._last_obs=None
            model.learn(total_timesteps=block,reset_num_timesteps=False,callback=DiagnosticsCallback(output))
            if model.num_timesteps-before!=block:
                raise RuntimeError('Unexpected PPO step budget')
            manager.charge(block)
            wrapped.partial('evaluation_boundary')
            if mode=='train':
                episodes=evaluate(model,raw,stages()[manager.level],c['seed'],c['development_episodes'],output,'development',int(model.num_timesteps))
                manager.observe(episodes)
            update=dict(timesteps=int(model.num_timesteps),n_updates=model._n_updates,
                metrics=diagnostic._train_metrics(model),curriculum=manager.state_dict())
            if getattr(model, 'anchor_metrics', None):
                update['anchor']=model.anchor_metrics
            append(output/'updates.jsonl',update)
            meta=commit(model,manager,rng,output,fingerprint,c)
            summary.update(trained_steps=int(model.num_timesteps),checkpoint=meta['checkpoint'])
            decision=manager.history[-1] if manager.history else {}
            printed=dict(timesteps=model.num_timesteps,stage=manager.level,checkpoint=meta['checkpoint'],
                decision=decision.get('decision'),promotion=decision.get('promotion'),
                efficiency_pass=decision.get('efficiency_pass'),
                mean_path_efficiency=decision.get('mean_path_efficiency'))
            if getattr(model, 'anchor_metrics', None):
                printed['anchor']=model.anchor_metrics
            print(json.dumps(printed),flush=True)
        # Reload final policy for fixed comparison, retaining historical rules.
        reloaded=load_curriculum_model(summary['checkpoint'], c)
        if mode=='train':
            fixed=diagnostic.evaluate_policy(reloaded,raw,diagnostic.resolve_validation_scenarios(base),c['seed'],
                episode_sink=lambda e: append(output/'evaluation_episodes.jsonl',dict(e,source='fixed',timesteps=int(model.num_timesteps))))
            _write_json(output/'fixed_evaluation.json',fixed)
            summary['evaluation']=fixed
        summary['status']='completed'
    except BaseException as exc:
        if not writes_allowed:
            raise
        summary['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        append(output/'failures.jsonl',summary['error'])
        if model is not None:
            summary['observed_model_steps']=int(model.num_timesteps)
        print(traceback.format_exc(),flush=True)
    finally:
        if writes_allowed:
            if wrapped is not None:
                wrapped.partial('session_end')
            summary['shutdown']=_shutdown(model,raw)
            summary['runtime_s']=time.monotonic()-started
            summary['curriculum']=manager.state_dict()
            _write_json(output/f'summary_{time.time_ns()}.json',summary)
            _write_json(output/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preview','pilot','train'))
    parser.add_argument('--config',default='configs/jackal_curriculum_6h.yaml')
    parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv); c=load_config(args.config)
    if args.mode=='preview':
        print(json.dumps(dict(config=c,stages=[[s.scenario_id for s in p] for p in stages()]),indent=2)); return 0
    if not args.output:
        parser.error('--output is required for execution')
    result=run(c,args.output,mode=args.mode,resume=args.resume)
    print(json.dumps(result,indent=2)); return 0 if result['status']=='completed' else 1

if __name__=='__main__':
    raise SystemExit(main())
