"""Bounded, validation-gated PPO curriculum; fixed-density control uses the same policy."""
import argparse
import copy
import csv
import json
import logging
import math
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from agents.pretrained_ppo_policy import EncodedNavigationObservation
from imitation.bc_network import resolve_device
from imitation.train_bc import seed_everything
from training.ppo_support import (actor_hash,digest_json,evaluate_model,file_sha256,finite_policy,
                                  read_verified_initialization,critic_snapshot)
from curriculum.curriculum_manager import CurriculumManager,wilson_lower
from curriculum.map_pools import build_pools,scenario,CurriculumMapEnv

LOG=logging.getLogger(__name__)


class EpisodeCallback(BaseCallback):
    def __init__(self,writer,stream,manager,start_step):
        super().__init__(); self.writer=writer; self.stream=stream; self.manager=manager; self.start_step=start_step

    def _on_step(self):
        for key in ('new_obs','rewards','values','log_probs'):
            value=self.locals.get(key)
            if value is not None:
                array=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
                if not np.isfinite(array).all():
                    raise FloatingPointError(f'Non-finite rollout {key}')
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                row={k:info[k] for k in ('curriculum_level','sampled_level','density','episode_reward',
                     'episode_collisions','episode_length','is_success','timeout','map_id','map_seed')}
                row.update(model_step=int(self.model.num_timesteps),phase_step=int(self.model.num_timesteps)-self.start_step)
                self.writer.writerow(row); self.stream.flush()
        return True


def validate_config(c):
    p=c['ppo']; v=c['validation']; q=c['curriculum']; m=c['maps']
    integers=[p[k] for k in ('n_steps','batch_size','n_epochs')]+[q[k] for k in ('level_budget','min_level_steps','required_passes')]+[v[k] for k in ('episodes','min_episodes','interval')]+[m[k] for k in ('train_maps_per_level','validation_maps_per_level','max_candidates_per_pool','min_path_length')]+[c['training']['total_timesteps'],c['training']['checkpoint_interval'],c['training']['seed'],m['seed']]
    if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in integers):
        raise ValueError('Counts/budgets/seeds must be nonnegative integers')
    if p['n_steps']<2 or p['batch_size']<2 or p['n_steps']%p['batch_size'] or p['n_epochs']<1:
        raise ValueError('Invalid PPO rollout/minibatch settings')
    if q['level_budget']<p['n_steps'] or q['level_budget']%p['n_steps'] or q['min_level_steps']%p['n_steps']:
        raise ValueError('Level budgets/minimum steps must align to whole PPO rollouts')
    if not 0 < p['learning_rate'] or not 0 < p['clip_range'] < 1 or not 0 < p['gamma'] <= 1:
        raise ValueError('Invalid PPO rates')
    if not all(np.isfinite(float(value)) for value in p.values()):
        raise ValueError('PPO settings must be finite numeric constants')
    if not 0<=p['gae_lambda']<=1 or p['vf_coef']<0 or p['ent_coef']<0 or p['max_grad_norm']<=0 or p['target_kl']<=0:
        raise ValueError('Invalid PPO optimization settings')
    if q['min_level_steps']>q['level_budget'] or q['required_passes']<2:
        raise ValueError('Invalid promotion duration/repetition')
    if v['episodes']<v['min_episodes'] or v['episodes']>m['validation_maps_per_level'] or v['min_episodes']<1 or v['interval']<1:
        raise ValueError('Validation sample count too small or interval invalid')
    if min(m['train_maps_per_level'],m['max_candidates_per_pool'],m['min_path_length'],c['training']['total_timesteps'],c['training']['checkpoint_interval'])<1:
        raise ValueError('Positive map counts, budgets and intervals required')
    if not m['exclude_manifests']:
        raise ValueError('Existing map identities must be excluded')
    manager=make_manager(c,'curriculum'); manager.weights(q['easier_weight'],q['harder_weight'])


def make_manager(c,mode):
    q=c['curriculum']
    return CurriculumManager(densities=q['densities'],level_budget=q['level_budget'],threshold=q['success_threshold'],
        required_passes=q['required_passes'],min_episodes=c['validation']['min_episodes'],min_level_steps=q['min_level_steps'],
        forgetting_drop=q['forgetting_drop'],mode=mode,fixed_level=q['fixed_level'])


def load_start(c,device):
    initial,transfer=read_verified_initialization(c['verified_initialization'],device)
    path=Path(c['starting_checkpoint'])
    side=json.loads(path.with_suffix('.run.json').read_text())
    if side['checkpoint_sha256']!=file_sha256(path) or side['method']!='pretrained':
        raise ValueError('Starting checkpoint must be a hash-verified Phase 4B pretrained policy')
    parent=Path(side['run_directory'])
    manifest=json.loads((parent/'run_manifest.json').read_text()); summary=json.loads((parent/'summary.json').read_text())
    if summary['status']!='completed' or manifest['initialization_sha256']!=file_sha256(c['verified_initialization']):
        raise ValueError('Starting policy must come from a completed run with the verified initialization')
    if manifest['environment']!=transfer['source_environment'] or transfer['source_environment'].get('bfs_shaping',False):
        raise ValueError('Preserve Phase 4B environment/reward semantics')
    # Exclude all layouts the inherited model used for training or model selection.
    excluded=set(c['maps']['exclude_manifests'])
    for name in (manifest['train_dataset'],manifest['validation_source']):
        if name not in excluded:
            raise ValueError(f'Add inherited data to maps.exclude_manifests: {name}')
    model=PPO.load(path,device=device,**c['ppo'])
    if int(model.num_timesteps)!=side['checkpoint_step']:
        raise ValueError('Starting timestep does not match sidecar')
    if model.observation_space!=initial.observation_space or model.action_space!=initial.action_space:
        raise ValueError('Starting policy spaces differ from verified policy')
    finite_policy(model)
    return model,transfer,int(summary['cumulative_consumed_env_timesteps'])


def checkpoint(model,path,state,raw):
    finite_policy(model)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    state=copy.deepcopy(state)
    state['model_step']=int(model.num_timesteps)
    state['phase_step']=int(model.num_timesteps)-state['start_model_step']
    state['map_rng']=copy.deepcopy(raw.pending_rng if raw.pending_rng is not None else raw.np_random.bit_generator.state)
    temporary=path.with_name(path.stem+'.tmp.zip'); model.save(temporary); temporary.replace(path)
    state['checkpoint_sha256']=file_sha256(path)
    side=path.with_suffix('.curriculum.json'); temp=side.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(side)
    return state


def read_checkpoint(path,fingerprint=None,mode=None):
    path=Path(path); state=json.loads(path.with_suffix('.curriculum.json').read_text())
    if state['checkpoint_sha256']!=file_sha256(path):
        raise ValueError('Curriculum checkpoint hash mismatch')
    if fingerprint is not None and state['fingerprint']!=fingerprint or mode is not None and state['mode']!=mode:
        raise ValueError('Resume must retain mode, source, map pools and all training/validation settings')
    for key in ('best_checkpoint',):
        if state.get(key) and file_sha256(state[key])!=state['best_checkpoint_sha256']:
            raise ValueError('Retain parent run and immutable best checkpoint')
    return state


def open_csv(run,name,fields):
    stream=(run/name).open('x',newline=''); writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader(); stream.flush()
    return stream,writer


def train(config,device=None,steps=None,mode='curriculum',output_directory=None,resume=None):
    c=copy.deepcopy(config); validate_config(c)
    if mode not in ('curriculum','fixed'):
        raise ValueError('Choose curriculum or fixed mode')
    requested=c['training']['total_timesteps'] if steps is None else steps
    if not isinstance(requested,int) or requested<1:
        raise ValueError('steps is a positive cumulative Phase 4C target')
    n=c['ppo']['n_steps']; target=math.ceil(requested/n)*n
    run=Path(output_directory or Path(c['training']['output_root'])/f'{mode}_steps_{requested}_seed{c["training"]["seed"]}')
    if run.exists():
        raise FileExistsError(f'{run} exists; choose a new --output-directory')
    resolved=str(resolve_device(device or c['training']['device']))
    source,transfer,prior_cost=load_start(c,resolved)
    pools,pool_manifest=build_pools(c,transfer['source_environment'])
    scenarios=[]
    for pair in pools:
        s=scenario(pair['validation']); size=c['validation']['episodes']
        arrays={k:v[:size] for k,v in s['arrays'].items()}
        s=dict(s,arrays=arrays,signature=digest_json(dict(pool_signature=s['signature'],episodes=size)))
        scenarios.append(s)
    protected={c['starting_checkpoint']:file_sha256(c['starting_checkpoint']),c['verified_initialization']:file_sha256(c['verified_initialization']),
               transfer['source_checkpoint']:file_sha256(transfer['source_checkpoint'])}
    specification=dict(config={k:v for k,v in c.items() if k!='training'},seed=c['training']['seed'],
                       checkpoint_interval=c['training']['checkpoint_interval'],pool_signature=pool_manifest['signature'],
                       starting_checkpoint_sha256=protected[c['starting_checkpoint']],mode=mode)
    fingerprint=digest_json(specification)
    state=read_checkpoint(resume,fingerprint,mode) if resume else dict(format_version=1,mode=mode,fingerprint=fingerprint,
        start_model_step=int(source.num_timesteps),phase_step=0,total_prior_phase4b_cost=prior_cost,
        best_checkpoint=None,best_score=None,last_evaluation=-1,last_periodic=0,completed_episodes=0,
        consumed_before_segment=0)
    manager=CurriculumManager.from_state(state['manager']) if resume else make_manager(c,mode)
    if manager.status not in ('running','total_budget_exhausted'):
        raise ValueError('Cannot resume mastered or exhausted level; preserve outcome and design a new experiment')
    manager.status='running'
    if resume:
        protected[str(resume)]=file_sha256(resume)
        parent=Path(state['run_directory'])/'summary.json'
        previous=json.loads(parent.read_text())
        if previous['status'].startswith('failed'):
            raise ValueError('Inspect failed run before designing a new experiment; automatic resume rejected')
        if previous.get('final_saved_phase_step')!=state['phase_step']:
            raise ValueError('Resume the latest finite final checkpoint; rewinding a curriculum stage would reset its budget')
        state['consumed_before_segment']=previous['cumulative_consumed_phase4c_steps']
        lost=state['consumed_before_segment']-state.get('cumulative_consumed_at_checkpoint',state['phase_step'])
        if lost<0:
            raise ValueError('Invalid checkpoint cost ledger')
        manager.level_steps+=lost
        model=PPO.load(resume,device=resolved)
        if model.num_timesteps!=state['model_step'] or manager.phase_steps!=state['phase_step']:
            raise ValueError('Manager/checkpoint step mismatch')
    else:
        model=source
    start_phase=manager.phase_steps; start_model=int(model.num_timesteps)
    if target<=state['consumed_before_segment']:
        raise ValueError('Target must exceed consumed cumulative Phase 4C steps')
    effective_target=state['consumed_before_segment']+math.ceil((target-state['consumed_before_segment'])/n)*n
    seed_everything(c['training']['seed'],deterministic=True)
    q=c['curriculum']; raw=CurriculumMapEnv(pools,manager,q['easier_weight'],q['harder_weight'],state.get('map_rng'))
    raw.np_random=np.random.default_rng(c['training']['seed'])
    env=Monitor(EncodedNavigationObservation(raw)); model.set_env(env,force_reset=True)
    model.set_random_seed(c['training']['seed'])
    if model.n_steps!=n:
        raise ValueError('Loaded rollout size differs from config')
    run.mkdir(parents=True,exist_ok=False)
    model.set_logger(configure(str(run/'sb3_logs'),['csv']))
    (run/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    state.update(run_directory=str(run.resolve()),segment_start_phase=start_phase)
    manifest=dict(format_version=1,mode=mode,fingerprint=fingerprint,source_checkpoint=c['starting_checkpoint'],
        source_sha256=protected[c['starting_checkpoint']],verified_initialization=c['verified_initialization'],
        starting_actor_sha256=actor_hash(model.policy),starting_critic_sha256=digest_json({k:__import__('hashlib').sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in critic_snapshot(model.policy).items()}),
        source_model_step=state['start_model_step'],source_phase4b_consumed_steps=prior_cost,
        pool_directory=str(Path(c['maps']['directory']).resolve()),pool_signature=pool_manifest['signature'],
        scenario_signatures=[s['signature'] for s in scenarios],ppo=c['ppo'],seed=c['training']['seed'],
        densities=q['densities'],fixed_density=q['densities'][q['fixed_level']],
        requested_phase4c_target=requested,rounded_phase4c_target=effective_target,resume=str(resume) if resume else None,
        reward_protocol='Original Manhattan reward, no BFS shaping, masking, reward normalization or loop intervention',
        validation_protocol='Greedy fixed maps; macro SPL/success/reward select best across all levels',
        continuation_protocol='Optimizer and counters retained; map stream restored; fresh episode and reseeded torch; not bitwise uninterrupted',
        test_protocol='Existing test identities excluded only; no test observations/actions/performance used; future hash test bucket reserved')
    (run/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    episode_fields=['model_step','phase_step','curriculum_level','sampled_level','density','episode_reward','episode_collisions',
                    'episode_length','is_success','timeout','map_id','map_seed']
    es,ew=open_csv(run,'episodes.csv',episode_fields)
    vs,vw=open_csv(run,'validation.csv',['phase_step','model_step','active_level','evaluated_level','density','episodes','successes',
        'success_rate','wilson_lower','mean_collisions','mean_steps','mean_reward','mean_spl','timeout_rate','loop_episode_rate','streak','decision'])
    us,uw=open_csv(run,'updates.csv',['phase_step','model_step','active_level','level_steps','loss','policy_gradient_loss','value_loss',
        'entropy_loss','approx_kl','clip_fraction','explained_variance','learning_rate'])
    callback=EpisodeCallback(ew,es,manager,state['start_model_step'])
    def save(path):
        state['manager']=manager.state_dict(); state['phase_step']=manager.phase_steps
        state['cumulative_consumed_at_checkpoint']=state['consumed_before_segment']+manager.phase_steps-start_phase
        return checkpoint(model,path,state,raw)
    def evaluate_all(allow):
        active=manager.level; summaries=[]; details=[]
        for level,s in enumerate(scenarios):
            summary,rows=evaluate_model(model,s); summaries.append(summary)
            details.append(dict(level=level,density=q['densities'][level],signature=s['signature'],summary=summary,results=rows))
        decision=manager.observe(summaries) if allow else 'none'
        score=[float(np.mean([s[k] for s in summaries])) for k in ('mean_spl','success_rate','mean_reward')]
        state['last_evaluation']=manager.phase_steps
        if state['best_score'] is None or tuple(score)>tuple(state['best_score']):
            path=run/'checkpoints'/f'best_phase_{manager.phase_steps:09d}.zip'
            state['best_score']=score; state['best_checkpoint']=None
            saved=save(path)
            state['best_checkpoint']=str(path.resolve()); state['best_checkpoint_sha256']=saved['checkpoint_sha256']
            # Rewrite the immutable checkpoint sidecar once to include its own best reference.
            saved.update(best_checkpoint=state['best_checkpoint'],best_checkpoint_sha256=saved['checkpoint_sha256'])
            path.with_suffix('.curriculum.json').write_text(json.dumps(saved,indent=2)+'\n')
        source_best=Path(state['best_checkpoint'])
        shutil.copyfile(source_best,run/'best.zip'); shutil.copyfile(source_best.with_suffix('.curriculum.json'),run/'best.curriculum.json')
        for level,s in enumerate(summaries):
            vw.writerow(dict(phase_step=manager.phase_steps,model_step=int(model.num_timesteps),active_level=active,evaluated_level=level,
                density=q['densities'][level],wilson_lower=wilson_lower(s['successes'],s['episodes']),
                **{k:s[k] for k in ('episodes','successes','success_rate','mean_collisions','mean_steps','mean_reward','mean_spl','timeout_rate','loop_episode_rate')},
                streak=manager.streak,decision=decision))
        vs.flush()
        (run/'validation_latest.json').write_text(json.dumps(dict(phase_step=manager.phase_steps,active_level=active,
            decision=decision,manager=manager.state_dict(),levels=details),indent=2)+'\n')
        # Each evaluated level gets a checkpoint; names reflect validation decisions, not mastery claims.
        save(run/'levels'/f'level_{active}_latest.zip')
        if decision in ('promoted','mastered'):
            save(run/'levels'/f'level_{active}_passed.zip')
        LOG.info('%s phase_step=%d level=%d density=%.2f success=%.3f lower=%.3f streak=%d decision=%s',
            mode,manager.phase_steps,active,q['densities'][active],summaries[active]['success_rate'],
            wilson_lower(summaries[active]['successes'],summaries[active]['episodes']),manager.streak,decision)
        return summaries
    status='running'; reason=None; observed=start_model
    try:
        # Baseline evaluation cannot promote or count a duplicate pass on recovery.
        if resume:
            saved_last=state['last_evaluation']
            evaluate_all(False)
            state['last_evaluation']=saved_last
        else:
            evaluate_all(False)
        save(run/'initial.zip'); save(run/'last_finite.zip')
        while state['consumed_before_segment']+manager.phase_steps-start_phase<effective_target and manager.status=='running':
            if mode=='curriculum' and manager.level_budget-manager.level_steps<n:
                manager.status='level_budget_exhausted'
                save(run/'levels'/f'level_{manager.level}_budget_exhausted.zip')
                break
            active=manager.level
            model.learn(total_timesteps=n,callback=callback,reset_num_timesteps=False,progress_bar=False)
            observed=int(model.num_timesteps)
            finite_policy(model)
            values=model.logger.name_to_value
            row=dict(phase_step=manager.phase_steps+n,model_step=observed,active_level=active,level_steps=manager.level_steps+n)
            for k in ('loss','policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance','learning_rate'):
                value=values.get('train/'+k)
                if value is not None and not np.isfinite(float(value)):
                    if k!='explained_variance':
                        raise FloatingPointError(f'Non-finite optimization metric {k}')
                    value=None
                row[k]=float(value) if value is not None else None
            uw.writerow(row); us.flush()
            manager.charge(n)
            due=(manager.phase_steps-state['last_evaluation']>=c['validation']['interval'] or
                 manager.level_steps>=manager.level_budget and mode=='curriculum' or state['consumed_before_segment']+manager.phase_steps-start_phase>=effective_target)
            if due:
                evaluate_all(True)
            manager.enforce_budget()
            if manager.status=='level_budget_exhausted':
                save(run/'levels'/f'level_{manager.level}_budget_exhausted.zip')
            if manager.phase_steps-state['last_periodic']>=c['training']['checkpoint_interval']:
                state['last_periodic']=manager.phase_steps
                save(run/'checkpoints'/f'phase_{manager.phase_steps:09d}.zip')
            save(run/'last_finite.zip')
        if manager.status=='running':
            manager.status='total_budget_exhausted'
        status=manager.status
        save(run/'final.zip')
    except KeyboardInterrupt:
        status='interrupted'; reason='Interrupted; recover only the last completed finite update'
    except Exception as exc:
        status='failed_numerical' if isinstance(exc,FloatingPointError) else 'failed_exception'
        reason=repr(exc); (run/'failure_traceback.txt').write_text(traceback.format_exc())
        LOG.exception('Curriculum run stopped')
    finally:
        observed=int(model.num_timesteps)
        if not (run/'final.zip').exists() and (run/'last_finite.zip').exists():
            shutil.copyfile(run/'last_finite.zip',run/'final.zip')
            shutil.copyfile(run/'last_finite.curriculum.json',run/'final.curriculum.json')
        for stream in (es,vs,us):
            stream.close()
        env.close()
    for path,checksum in protected.items():
        if file_sha256(path)!=checksum:
            raise RuntimeError(f'Protected source changed: {path}')
    final=read_checkpoint(run/'final.zip') if (run/'final.zip').exists() else None
    summary=dict(status=status,reason=reason,mode=mode,current_level=manager.level,current_density=q['densities'][manager.level],
        requested_phase4c_target=requested,rounded_phase4c_target=effective_target,
        completed_phase4c_steps=manager.phase_steps,observed_model_step=observed,segment_consumed_steps=observed-start_model,
        cumulative_consumed_phase4c_steps=state['consumed_before_segment']+observed-start_model,
        source_phase4b_consumed_steps=prior_cost,total_inherited_plus_phase4c_steps=prior_cost+state['consumed_before_segment']+observed-start_model,
        final_saved_phase_step=final['phase_step'] if final else None,best_macro_score=state['best_score'],
        best_checkpoint=str(run/'best.zip'),pool_signature=pool_manifest['signature'],source_artifacts_preserved=True,
        test_performance_used=False,manager=manager.state_dict(),run_directory=str(run))
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    training_statistics(run)
    plot(run)
    return summary


def training_statistics(run):
    with (run/'episodes.csv').open() as f:
        rows=list(csv.DictReader(f))
    fields=['level','density','episodes','success_rate','mean_collisions','mean_steps','mean_reward']
    with (run/'training_by_level.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for level in sorted(set(int(r['sampled_level']) for r in rows)):
            selected=[r for r in rows if int(r['sampled_level'])==level]
            writer.writerow(dict(level=level,density=selected[0]['density'],episodes=len(selected),
                success_rate=float(np.mean([r['is_success']=='True' for r in selected])),
                mean_collisions=float(np.mean([int(r['episode_collisions']) for r in selected])),
                mean_steps=float(np.mean([int(r['episode_length']) for r in selected])),
                mean_reward=float(np.mean([float(r['episode_reward']) for r in selected]))))


def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with (run/'validation.csv').open() as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for level in sorted(set(r['evaluated_level'] for r in rows),key=int):
        selected=[r for r in rows if r['evaluated_level']==level]
        for ax,key in zip(axes,('success_rate','mean_collisions','mean_steps')):
            ax.plot([int(r['phase_step']) for r in selected],[float(r[key]) for r in selected],label=f'd={selected[0]["density"]}')
    for ax,title in zip(axes,('Validation success','Mean collisions','Mean episode length')):
        ax.set(title=title,xlabel='Phase 4C transitions'); ax.grid(alpha=.2)
    axes[0].set_ylim(0,1.02)
    if rows: axes[-1].legend(fontsize=8)
    fig.savefig(run/'curriculum.png',dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/curriculum_ppo.yaml'); p.add_argument('--device',choices=['auto','cpu','cuda'])
    p.add_argument('--steps',type=int,help='Cumulative consumed Phase 4C budget, excluding inherited Phase 4B steps; full rollouts can round up')
    p.add_argument('--mode',choices=['curriculum','fixed'],default='curriculum'); p.add_argument('--output-directory'); p.add_argument('--resume')
    args=p.parse_args(); config=yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    result=train(config,args.device,args.steps,args.mode,args.output_directory,args.resume)
    print(json.dumps(result,indent=2))
    if result['status'].startswith('failed'):
        raise SystemExit(2)


if __name__=='__main__':
    main()
