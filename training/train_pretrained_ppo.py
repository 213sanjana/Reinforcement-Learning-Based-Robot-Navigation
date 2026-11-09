"""PPO fine-tuning or matched scratch control; new run directories and held-out validation."""
import argparse
from collections import deque
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
from agents.pretrained_ppo_policy import EncodedNavigationObservation,PretrainedPPOPolicy
from imitation.bc_network import resolve_device
from imitation.train_bc import seed_everything
from training.ppo_support import (TrainingMapPoolEnv,actor_hash,critic_snapshot,digest_json,evaluate_model,
    file_sha256,finite_policy,load_continuation,load_data,read_verified_initialization,save_checkpoint)

LOG=logging.getLogger(__name__)


class StopTraining(RuntimeError):
    def __init__(self,status,reason):
        super().__init__(reason); self.status=status


def validate_config(config,target):
    if target < 1 or int(target) != target:
        raise ValueError('Target total timesteps must be a positive integer')
    p=config['ppo']; v=config['validation']; t=config['training']
    if p['n_steps'] < 2 or p['batch_size'] < 2 or p['n_steps'] % p['batch_size']:
        raise ValueError('n_steps must be >=2 and divisible by batch_size for matched budgets')
    if p['n_epochs'] < 1 or p['learning_rate'] <= 0 or not 0 <= p['gamma'] <= 1:
        raise ValueError('Invalid PPO settings')
    if t['seed'] < 0 or t['checkpoint_interval'] < 1 or v['interval'] < 1 or v['regression_patience'] < 1:
        raise ValueError('Invalid seed or monitoring intervals')
    if not 0 <= v['regression_drop'] <= 1:
        raise ValueError('regression_drop is an absolute success-rate fraction')


class GuardedCallback(BaseCallback):
    def __init__(self,run,config,scenarios,pool_env,state,resumed=False):
        super().__init__()
        self.run=Path(run); self.config=config; self.scenarios=scenarios; self.pool_env=pool_env
        self.state=state; self.resumed=resumed
        self.recent=deque(state.get('recent_episodes',[]),maxlen=100)
        self.total_episodes=state.get('completed_episodes',0)
        self.last_processed=-1; self.last_evaluation=int(state.get('checkpoint_step',-1)) if resumed else -1
        self.last_periodic=int(state.get('checkpoint_step',0))
        self.episode_fields=['episode','global_step','reward','steps','collisions','success','timeout','map_id','map_seed','task_index']
        self.validation_fields=['global_step','ppo_updates','kind','success_rate','mean_reward','mean_steps',
                                'mean_collisions','mean_spl','timeout_rate','loop_episode_rate','best_step','regression_streak']
        self.update_fields=['global_step','ppo_updates','completed_episodes','train_success_100','train_reward_100',
                            'train_collisions_100','train_steps_100','loss','policy_gradient_loss','value_loss',
                            'entropy_loss','approx_kl','clip_fraction','explained_variance','learning_rate']
        self.streams=[]
        for name,fields in [('episodes',self.episode_fields),('validation',self.validation_fields),('updates',self.update_fields)]:
            stream=(self.run/f'{name}.csv').open('x',newline='',encoding='utf8')
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); stream.flush()
            setattr(self,name+'_writer',writer); setattr(self,name+'_stream',stream); self.streams.append(stream)

    def close(self):
        for stream in self.streams:
            stream.close()

    def refresh_state(self):
        self.state.update(completed_episodes=self.total_episodes,recent_episodes=list(self.recent),
                          train_map_rng=self.pool_env.stream_state(),
                          cumulative_consumed_at_checkpoint=self.state['lineage_cost_before_segment']+int(self.model.num_timesteps)-self.state['segment_start_step'])

    def checkpoint(self,path,is_best=False):
        self.refresh_state()
        save_checkpoint(self.model,path,self.state,is_best=is_best)

    def copy_best_alias(self):
        source=Path(self.state['best_checkpoint'])
        shutil.copyfile(source,self.run/'best.zip')
        shutil.copyfile(source.with_suffix('.run.json'),self.run/'best.run.json')

    def _on_training_start(self):
        finite_policy(self.model)
        # Start/resume evaluation does not increment a previously recorded regression streak.
        self.evaluate('resume_start' if self.resumed else 'initial',update_guard=False)
        self.checkpoint(self.run/'initial.zip')
        self.checkpoint(self.run/'last_finite.zip')
        self.last_processed=int(self.model.num_timesteps)

    def _on_step(self):
        for name in ('rewards','values','log_probs','new_obs'):
            value=self.locals.get(name)
            if value is None:
                continue
            valid=torch.isfinite(value).all().item() if isinstance(value,torch.Tensor) else np.isfinite(value).all()
            if not valid:
                raise StopTraining('failed_numerical',f'Non-finite training rollout {name}')
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                self.total_episodes+=1
                row=dict(episode=self.total_episodes,global_step=int(self.model.num_timesteps),
                         reward=info['episode_reward'],steps=info['episode_length'],collisions=info['episode_collisions'],
                         success=int(info['is_success']),timeout=int(info['timeout']),map_id=info['map_id'],
                         map_seed=info['map_seed'],task_index=info['task_index'])
                self.episodes_writer.writerow(row); self.episodes_stream.flush(); self.recent.append(row)
        return True

    def evaluate(self,kind,update_guard=True):
        finite_policy(self.model)
        summary,rows=evaluate_model(self.model,self.scenarios)
        step=int(self.model.num_timesteps)
        score=[summary['success_rate'],summary['mean_spl'],summary['mean_reward']]
        if self.state.get('baseline_success_rate') is None:
            self.state['baseline_success_rate']=summary['success_rate']
        if self.state.get('best_score') is None or tuple(score)>tuple(self.state['best_score']):
            self.state['best_score']=score
            self.checkpoint(self.run/'checkpoints'/f'best_step_{step:09d}.zip',is_best=True)
        self.copy_best_alias()
        if update_guard:
            low=summary['success_rate'] < self.state['baseline_success_rate']-self.config['validation']['regression_drop']
            self.state['regression_streak']=self.state.get('regression_streak',0)+1 if low else 0
        row=dict(global_step=step,ppo_updates=int(self.model._n_updates),kind=kind,
                 **{key:summary[key] for key in ('success_rate','mean_reward','mean_steps','mean_collisions','mean_spl','timeout_rate','loop_episode_rate')},
                 best_step=self.state['best_step'],regression_streak=self.state.get('regression_streak',0))
        self.validation_writer.writerow(row); self.validation_stream.flush()
        (self.run/'validation_latest.json').write_text(json.dumps(dict(global_step=step,summary=summary,
              scenario_signature=self.scenarios['signature'],results=rows),indent=2)+'\n')
        self.last_evaluation=step
        self.checkpoint(self.run/'last_finite.zip')
        LOG.info('%s step=%d val_success=%.3f val_SPL=%.3f best_step=%d train_success_100=%s',
                 self.state['method'],step,summary['success_rate'],summary['mean_spl'],self.state['best_step'],
                 f'{np.mean([r["success"] for r in self.recent]):.3f}' if self.recent else 'not-yet-available')
        v=self.config['validation']
        if update_guard and step >= v['regression_min_steps'] and self.state.get('regression_streak',0) >= v['regression_patience']:
            raise StopTraining('stopped_regression',f'Validation success dropped more than {v["regression_drop"]:.0%} below the starting baseline for {v["regression_patience"]} checks')

    def process_completed_update(self,force=False):
        step=int(self.model.num_timesteps)
        if step==self.last_processed:
            return
        finite_policy(self.model)
        values=self.model.logger.name_to_value
        row=dict(global_step=step,ppo_updates=int(self.model._n_updates),completed_episodes=self.total_episodes)
        for output,key in [('train_success_100','success'),('train_reward_100','reward'),
                           ('train_collisions_100','collisions'),('train_steps_100','steps')]:
            row[output]=float(np.mean([r[key] for r in self.recent])) if self.recent else None
        for key in ('loss','policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance','learning_rate'):
            number=values.get('train/'+key)
            if number is not None and not np.isfinite(float(number)):
                if key != 'explained_variance':
                    raise StopTraining('failed_numerical',f'Non-finite optimization metric {key}')
                number=None
            row[key]=float(number) if number is not None else None
        self.updates_writer.writerow(row); self.updates_stream.flush(); self.last_processed=step
        v=self.config['validation']
        due=force or step//v['interval'] > max(0,self.last_evaluation)//v['interval'] or any(self.last_evaluation < early <= step for early in v['early_steps'])
        if due:
            self.evaluate('final' if force else 'periodic')
        interval=self.config['training']['checkpoint_interval']
        if step//interval > self.last_periodic//interval:
            self.checkpoint(self.run/'checkpoints'/f'step_{step:09d}.zip'); self.last_periodic=step

    def _on_rollout_start(self):
        # SB3 calls this after the preceding rollout's train() has finished.
        self.process_completed_update()

    def _on_training_end(self):
        self.process_completed_update(force=True)


def plot_logs(run):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with (run/'episodes.csv').open() as stream:
        episodes=list(csv.DictReader(stream))
    with (run/'validation.csv').open() as stream:
        validation=list(csv.DictReader(stream))
    figure,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for axis,key,title in zip(axes.flat,('reward','success','collisions','steps'),
                              ('Training reward','Success rate','Training collisions','Episode length')):
        if episodes:
            y=[float(row[key]) for row in episodes]
            axis.plot([int(row['global_step']) for row in episodes],
                      [float(np.mean(y[max(0,i-99):i+1])) for i in range(len(y))],label='Training trailing 100')
        if key=='success' and validation:
            axis.plot([int(row['global_step']) for row in validation],
                      [float(row['success_rate']) for row in validation],'-o',label='Greedy validation')
            axis.set_ylim(0,1)
        axis.set(title=title,xlabel='Environment timesteps'); axis.grid(alpha=.2)
        if axis.lines:
            axis.legend()
    figure.savefig(run/'training.png',dpi=150); plt.close(figure)


def train(config,steps=None,device=None,mode='pretrained',output_directory=None,resume=None):
    target=int(steps if steps is not None else config['training']['total_timesteps'])
    validate_config(config,target)
    diagnostic_ceiling=math.ceil(config['training'].get('diagnostic_timesteps',50000)/config['ppo']['n_steps'])*config['ppo']['n_steps']
    if target>diagnostic_ceiling and resume is None:
        raise ValueError('Extended budgets require --resume from a completed diagnostic checkpoint; inspect diagnostic evaluation first')
    if mode not in ('pretrained','scratch'):
        raise ValueError('mode must be pretrained or scratch')
    seed=int(config['training']['seed']); resolved=str(resolve_device(device or config['training']['device']))
    label=f'{mode}_'+('resume_to_' if resume else 'steps_')+f'{target}_seed{seed}'
    run=Path(output_directory or Path(config['training']['output_root'])/label)
    if run.exists():
        raise FileExistsError(f'{run} exists; choose a new --output-directory')
    initialization=Path(config['initialization'])
    initial,transfer=read_verified_initialization(initialization,resolved)
    if transfer['source_environment'].get('bfs_shaping',False):
        raise ValueError('Phase 4B comparison uses unchanged Manhattan rewards with BFS shaping disabled')
    pool,scenarios=load_data(config,transfer['source_environment'],transfer)
    signature=digest_json(dict(ppo=config['ppo'],seed=seed,initialization_sha256=file_sha256(initialization),
                               train_sha256=pool['sha256'],validation_signature=scenarios['signature'],
                               validation_settings=config['validation'],method=mode))
    state=load_continuation(resume,signature,mode) if resume else dict(
        format_version=1,method=mode,config_fingerprint=signature,baseline_success_rate=None,
        best_score=None,regression_streak=0,completed_episodes=0,recent_episodes=[])
    lineage_before=0
    if resume:
        parent_summary=Path(state['run_directory'])/'summary.json'
        parent_data=json.loads(parent_summary.read_text()) if parent_summary.is_file() else None
        if target>diagnostic_ceiling and (parent_data is None or parent_data['status']!='completed' or state['checkpoint_step']<diagnostic_ceiling):
            raise ValueError('Complete and inspect the diagnostic before extending its final checkpoint')
        lineage_before=parent_data['cumulative_consumed_env_timesteps'] if parent_data else state['cumulative_consumed_at_checkpoint']
    source_hashes={str(initialization):file_sha256(initialization),transfer['source_checkpoint']:transfer['source_checkpoint_sha256']}
    if resume:
        source_hashes[str(resume)]=file_sha256(resume)
    raw=TrainingMapPoolEnv(pool,pending_rng=state.get('train_map_rng'))
    env=Monitor(EncodedNavigationObservation(raw))
    seed_everything(seed,deterministic=True)
    if resume:
        model=PPO.load(resume,env=env,device=resolved,seed=seed,force_reset=True)
    elif mode=='pretrained':
        model=PPO.load(initialization,env=env,device=resolved,seed=seed,**config['ppo'])
    else:
        model=PPO(PretrainedPPOPolicy,env,policy_kwargs={'hidden_size':initial.policy.hidden_size},
                  device=resolved,seed=seed,verbose=0,**config['ppo'])
        # Same independently initialized critic in both methods; only actor pretraining differs.
        model.policy.vf_features_extractor.load_state_dict(initial.policy.vf_features_extractor.state_dict(),strict=True)
        model.policy.value_net.load_state_dict(initial.policy.value_net.state_dict(),strict=True)
    start=int(model.num_timesteps)
    if target <= start:
        env.close(); raise ValueError(f'Target {target} must exceed continuation timestep {start}')
    if model.n_steps != config['ppo']['n_steps']:
        env.close(); raise ValueError('Loaded rollout size differs from configuration')
    initial_actor=actor_hash(initial.policy); starting_actor=actor_hash(model.policy)
    if not resume and mode=='pretrained' and starting_actor != initial_actor:
        env.close(); raise ValueError('Pretrained actor changed before learning')
    if not resume and mode=='scratch' and starting_actor == initial_actor:
        env.close(); raise ValueError('Scratch actor was accidentally initialized from pretrained weights')
    if not resume:
        initial_value=critic_snapshot(initial.policy); starting_value=critic_snapshot(model.policy)
        if initial_value.keys()!=starting_value.keys() or any(not torch.equal(initial_value[key],starting_value[key]) for key in initial_value):
            raise ValueError('Fresh critic weights differ between initialization and selected method')
        if model.policy.optimizer.state:
            raise ValueError('Fresh experiment unexpectedly has optimizer state')
    for group in model.policy.optimizer.param_groups:
        group['lr']=config['ppo']['learning_rate']
    del initial
    state.update(run_directory=str(run.resolve()),segment_start_step=start,lineage_cost_before_segment=lineage_before)
    effective_target=start+math.ceil((target-start)/model.n_steps)*model.n_steps
    run.mkdir(parents=True,exist_ok=False)
    model.set_logger(configure(str(run/'sb3_logs'),['csv']))
    config_copy=copy.deepcopy(config); config_copy['training']['total_timesteps']=target
    (run/'config.yaml').write_text(yaml.safe_dump(config_copy,sort_keys=False))
    with (run/'validation_scenarios.npz').open('xb') as stream:
        np.savez_compressed(stream,**scenarios['arrays'],metadata=np.asarray(json.dumps({key:value for key,value in scenarios.items() if key!='arrays'})))
    manifest=dict(format_version=1,method=mode,seed=seed,config_fingerprint=signature,
                  initialization=str(initialization),initialization_sha256=file_sha256(initialization),
                  environment=pool['environment'],train_dataset=pool['path'],train_dataset_sha256=pool['sha256'],
                  validation_signature=scenarios['signature'],validation_source=scenarios['source_path'],
                  requested_total_timesteps=target,rounded_total_timesteps=effective_target,
                  rollout_steps=model.n_steps,resume_source=str(resume) if resume else None,
                  starting_step=start,lineage_cost_before_segment=lineage_before,starting_actor_sha256=starting_actor,verified_actor_sha256=initial_actor,
                  critic_protocol='Same fresh Phase 4A critic for both fresh methods; optimizer state retained on continuation',
                  reward_protocol='Unchanged original Manhattan rewards; no masking, loop interventions or reward normalization',
                  continuation_protocol='Weights, optimizer, counters and next-map RNG continue; starts a new episode; not bitwise uninterrupted replay')
    (run/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    callback=GuardedCallback(run,config,scenarios,raw,state,resumed=bool(resume))
    status='completed'; reason=None; final_saved=False
    try:
        model.learn(total_timesteps=target-start,callback=callback,reset_num_timesteps=not bool(resume),progress_bar=False)
    except StopTraining as exc:
        status=exc.status; reason=str(exc)
        LOG.warning('%s: %s',status,reason)
    except FloatingPointError as exc:
        status='failed_numerical'; reason=str(exc)
        (run/'failure_traceback.txt').write_text(traceback.format_exc())
        LOG.warning('%s: %s',status,reason)
    except KeyboardInterrupt:
        status='interrupted'; reason='KeyboardInterrupt; continuation starts a fresh episode'
    except Exception as exc:
        status='failed_exception'; reason=repr(exc)
        (run/'failure_traceback.txt').write_text(traceback.format_exc())
        LOG.exception('PPO experiment failed')
    finally:
        observed=int(model.num_timesteps)
        try:
            if not status.startswith('failed') and status!='interrupted':
                finite_policy(model)
                callback.checkpoint(run/'final.zip'); final_saved=True
        except Exception as exc:
            status='failed_numerical'; reason=str(exc)
        if not final_saved and (run/'last_finite.zip').exists():
            shutil.copyfile(run/'last_finite.zip',run/'final.zip')
            shutil.copyfile(run/'last_finite.run.json',run/'final.run.json')
            final_saved=True
        callback.close(); env.close()
    for path,checksum in source_hashes.items():
        if file_sha256(path) != checksum:
            raise RuntimeError(f'Protected source artifact changed: {path}')
    summary=dict(status=status,reason=reason,method=mode,requested_total_timesteps=target,
                 rounded_total_timesteps=effective_target,actual_env_timesteps=observed,
                 segment_env_timesteps=observed-start,starting_step=start,source_artifacts_preserved=True,
                 cumulative_consumed_env_timesteps=lineage_before+observed-start,
                 lineage_cost_before_segment=lineage_before,
                 best_step=state.get('best_step'),best_score=state.get('best_score'),
                 final_checkpoint_saved=final_saved,
                 final_checkpoint_step=json.loads((run/'final.run.json').read_text())['checkpoint_step'] if final_saved else None,
                 validation_signature=scenarios['signature'],config_fingerprint=signature,
                 best_can_be_unmodified_initial_policy=state.get('best_step')==0,
                 selected_checkpoint=str(run/'best.zip'),run_directory=str(run))
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if status.startswith('failed'):
        (run/'failure.json').write_text(json.dumps(dict(status=status,reason=reason),indent=2)+'\n')
    plot_logs(run)
    LOG.info('Run %s: %s; actual timesteps=%d best_step=%s',run,status,observed,state.get('best_step'))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ppo_finetuning.yaml')
    parser.add_argument('--steps',type=int,help='Target cumulative environment timesteps, including previous checkpoint steps')
    parser.add_argument('--device',choices=['auto','cpu','cuda'])
    parser.add_argument('--mode',choices=['pretrained','scratch'],default='pretrained')
    parser.add_argument('--output-directory')
    parser.add_argument('--resume',help='Checkpoint ZIP plus matching .run.json; keep parent experiment')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    summary=train(config,args.steps,args.device,args.mode,args.output_directory,args.resume)
    print(json.dumps(summary,indent=2))
    if summary['status'].startswith('failed'):
        raise SystemExit(2)


if __name__=='__main__':
    main()
