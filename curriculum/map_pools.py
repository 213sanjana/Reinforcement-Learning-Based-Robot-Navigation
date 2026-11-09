"""Deterministic layout-only pools. Canonical identities own a stable split."""
from collections import Counter
import copy
import glob
import json
import logging
from pathlib import Path
import gymnasium as gym
import numpy as np
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path
from imitation.dataset import map_identity
from training.ppo_support import digest_json,file_sha256

LOG=logging.getLogger(__name__)

KEYS=('obstacles','starts','goals','map_ids','map_seeds','path_lengths')


def split_for(identity, salt):
    bucket=int(digest_json(dict(salt=salt,identity=identity))[:16],16)%100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'test'


def exclusion_ids(paths):
    result=set(); fingerprints={}
    for name in paths:
        path=Path(name)
        with np.load(path,allow_pickle=False) as data:
            # Identity-only inspection: no test observations, actions, outcomes or scores.
            result.update(map(str,data['map_ids']))
        fingerprints[str(path)]=file_sha256(path)
    return result,fingerprints


def scenario(pool):
    return dict(arrays=pool['arrays'],environment=pool['environment'],signature=pool['signature'])


def build_pools(config,source_environment):
    cfg=config['maps']; path=Path(cfg['directory'])
    paths=set(cfg['exclude_manifests'])
    for pattern in cfg.get('exclude_globs',[]):
        paths.update(glob.glob(pattern,recursive=True))
    excluded,fingerprints=exclusion_ids(sorted(paths))
    spec=dict(format_version=1,densities=config['curriculum']['densities'],environment=source_environment,
              maps={k:v for k,v in cfg.items() if k!='directory'},excluded_sources=fingerprints)
    signature=digest_json(spec)
    if path.exists():
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['signature']!=signature:
            raise ValueError('Existing map pools differ from requested specification; choose a new directory')
        return load_pools(path,excluded,signature)
    path.mkdir(parents=True,exist_ok=False)
    files={}; stats={}; seen=set(excluded)
    try:
        for level,density in enumerate(config['curriculum']['densities']):
            environment=dict(source_environment,obstacle_density=density)
            env=RobotNavigationEnv(**environment)
            try:
                for split in ('train','validation'):
                    count=cfg['train_maps_per_level'] if split=='train' else cfg['validation_maps_per_level']
                    rows=[]; rejected=Counter()
                    for candidate in range(cfg['max_candidates_per_pool']):
                        seed=int(np.random.SeedSequence([cfg['seed'],level,0 if split=='train' else 1,candidate]).generate_state(1)[0])
                        try:
                            obs,_=env.reset(seed=seed)
                        except RuntimeError:
                            rejected['map_generation_failed']+=1; continue
                        identity=map_identity(obs['obstacles'])
                        if identity in seen:
                            rejected['reserved_or_duplicate_layout']+=1; continue
                        if split_for(identity,cfg['split_salt'])!=split:
                            rejected['different_split_bucket']+=1; continue
                        route=shortest_path(obs['obstacles'],tuple(obs['agent']),tuple(obs['goal']))
                        length=len(route)-1 if route else -1
                        if not cfg['min_path_length']<=length<=environment['max_steps']:
                            rejected['path_length']+=1; continue
                        rows.append(dict(obstacles=obs['obstacles'],starts=obs['agent'],goals=obs['goal'],
                                         map_ids=identity,map_seeds=seed,path_lengths=length))
                        seen.add(identity)
                        if len(rows)==count:
                            break
                    if len(rows)!=count:
                        raise RuntimeError(f'Pool {level}/{split}: accepted {len(rows)}/{count}; bounded attempts exhausted')
                    arrays={k:np.asarray([r[k] for r in rows]) for k in KEYS}
                    key=f'level_{level}_{split}'; name=key+'.npz'
                    metadata=dict(environment=environment,split=split,level=level,specification_signature=signature)
                    metadata['signature']=digest_json(dict(metadata=metadata,arrays={k:v.tolist() for k,v in arrays.items()}))
                    with (path/name).open('xb') as stream:
                        np.savez_compressed(stream,**arrays,metadata=np.asarray(json.dumps(metadata)))
                    files[name]=file_sha256(path/name)
                    lengths=arrays['path_lengths']
                    LOG.info('Map pool level=%d density=%.2f split=%s maps=%d',level,density,split,count)
                    stats[key]=dict(maps=count,path_min=int(lengths.min()),path_max=int(lengths.max()),
                                    path_mean=float(lengths.mean()),realized_density=float(arrays['obstacles'].mean()),
                                    rejected=dict(rejected))
            finally:
                env.close()
        manifest=dict(signature=signature,specification=spec,files=files,statistics=stats,
                      test_policy='Hash buckets 90..99 reserved; no test tasks generated or evaluated',
                      existing_test_access='map identities only for exclusion; never observations/actions/outcomes')
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    except BaseException:
        (path/'INCOMPLETE.txt').write_text('Pool generation did not complete. Retain for diagnosis and choose a new map directory.\n')
        raise
    return load_pools(path,excluded,signature)


def load_pools(path,excluded=None,expected_signature=None):
    path=Path(path); manifest=json.loads((path/'manifest.json').read_text())
    if expected_signature is not None and manifest['signature']!=expected_signature:
        raise ValueError('Map specification signature mismatch')
    seen=set(excluded or ()); pools=[]
    densities=manifest['specification']['densities']
    for level,density in enumerate(densities):
        pair={}
        for split in ('train','validation'):
            name=f'level_{level}_{split}.npz'
            if file_sha256(path/name)!=manifest['files'][name]:
                raise ValueError('Map pool checksum mismatch')
            with np.load(path/name,allow_pickle=False) as data:
                metadata=json.loads(str(data['metadata'])); a={k:data[k] for k in KEYS}
            base={k:v for k,v in metadata.items() if k!='signature'}
            if digest_json(dict(metadata=base,arrays={k:v.tolist() for k,v in a.items()}))!=metadata['signature']:
                raise ValueError('Map scenario signature mismatch')
            if (metadata['level']!=level or metadata['split']!=split or
                    metadata['environment']!=dict(manifest['specification']['environment'],obstacle_density=density)):
                raise ValueError('Invalid pool environment or split')
            expected_count=manifest['specification']['maps']['train_maps_per_level' if split=='train' else 'validation_maps_per_level']
            if len(a['map_ids'])!=expected_count or any(len(v)!=expected_count for v in a.values()):
                raise ValueError('Pool cardinality mismatch')
            for i,grid in enumerate(a['obstacles']):
                identity=map_identity(grid)
                if identity!=a['map_ids'][i] or identity in seen or split_for(identity,manifest['specification']['maps']['split_salt'])!=split:
                    raise ValueError('Duplicate, reserved or incorrectly split layout')
                expected_shape=(metadata['environment']['height'],metadata['environment']['width'])
                if grid.shape!=expected_shape or int(grid.sum())!=round(grid.size*density):
                    raise ValueError('Pool grid size/density mismatch')
                route=shortest_path(grid,tuple(a['starts'][i]),tuple(a['goals'][i]))
                if route is None or len(route)-1!=a['path_lengths'][i] or not manifest['specification']['maps']['min_path_length']<=len(route)-1<=metadata['environment']['max_steps']:
                    raise ValueError('Invalid reference route')
                seen.add(identity)
            pair[split]=dict(arrays=a,**metadata)
        pools.append(pair)
    return pools,manifest


class CurriculumMapEnv(RobotNavigationEnv):
    """Curriculum weights are pending settings; maps and densities change only in reset."""
    def __init__(self,pools,manager,easier=.2,harder=.1,pending_rng=None):
        self.pools=pools; self.manager=manager; self.easier=easier; self.harder=harder; self.pending_rng=pending_rng
        super().__init__(**pools[0]['train']['environment'])

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        if options:
            raise ValueError('No training reset overrides')
        if self.pending_rng is not None:
            self.np_random.bit_generator.state=copy.deepcopy(self.pending_rng); self.pending_rng=None
        self.episode_stage=self.manager.level
        self.sampled_level=int(self.np_random.choice(len(self.pools),p=self.manager.weights(self.easier,self.harder)))
        pool=self.pools[self.sampled_level]['train']; a=pool['arrays']
        self.task_index=int(self.np_random.integers(len(a['map_ids'])))
        i=self.task_index
        self.obstacles=a['obstacles'][i].copy(); self.agent=tuple(map(int,a['starts'][i])); self.goal=tuple(map(int,a['goals'][i]))
        self.obstacle_density=pool['environment']['obstacle_density']
        self.steps,self._finished,self._initialized=0,False,True
        self.episode_reward=0.; self.episode_collisions=0
        return self._observation(),self._info()

    def step(self,action):
        obs,reward,term,trunc,info=super().step(action)
        self.episode_reward+=reward; self.episode_collisions+=int(info['collision'])
        a=self.pools[self.sampled_level]['train']['arrays']; i=self.task_index
        info.update(is_success=bool(term),timeout=bool(trunc),episode_reward=float(self.episode_reward),
                    episode_collisions=int(self.episode_collisions),episode_length=self.steps,
                    sampled_level=self.sampled_level,curriculum_level=self.episode_stage,density=self.obstacle_density,
                    map_id=str(a['map_ids'][i]),map_seed=int(a['map_seeds'][i]))
        return obs,reward,term,trunc,info
