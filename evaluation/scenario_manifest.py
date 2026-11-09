"""Frozen, reproducible benchmark tasks; reads identity-only NPZ fields."""
import argparse
import glob
import inspect
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path
from imitation.dataset import map_identity
from planners.astar import astar


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def contract():
    return dict(channels=['obstacles', 'agent', 'goal'], dtype='float32',
                actions=[[-1, 0], [1, 0], [0, -1], [0, 1]],
                reward=dict(step=-1., collision=-10., progress=2., goal=100.),
                bfs_shaping=False, success='agent_equals_goal',
                timeout='horizon_without_goal', interventions=False)


def exclusions(cfg):
    ids = set(); sources = []; split_ids = {}
    for entry in cfg.get('identity_sources', []):
        paths = sorted(glob.glob(entry['pattern'], recursive=True))
        if entry.get('required', True) and len(paths) < entry.get('min_matches', 1):
            raise FileNotFoundError('Required identity source missing: ' + entry['pattern'])
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                grids = data['obstacles']; saved = list(map(str, data['map_ids']))
                computed = [map_identity(g) for g in grids]
            if saved != computed:
                raise ValueError('Identity mismatch: ' + path)
            role = entry['role']
            split_ids.setdefault(role, set()).update(saved)
            ids.update(saved)
            sources.append(dict(path=path, sha256=file_hash(path), role=role, count=len(saved)))
    roles = sorted(split_ids)
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            if split_ids[a] & split_ids[b]:
                raise ValueError(f'Existing cross-split leakage: {a}/{b}')
    # Explicit historical ranges, never advertised as exhaustive training coverage.
    for entry in cfg.get('seed_sources', []):
        env = RobotNavigationEnv(**entry['environment'])
        try:
            for start, stop in entry['ranges']:
                for seed in range(start, stop):
                    obs, _ = env.reset(seed=seed)
                    ids.add(map_identity(obs['obstacles']))
        finally:
            env.close()
    return ids, sources


def source_fingerprints():
    from environments import gridworld_env, fixed_map_env, map_generator, rewards
    from imitation import dataset
    from planners import astar as planner_module
    from evaluation import scenario_manifest, policy_adapters, benchmark_protocol, result_schema
    return {m.__name__: hashlib.sha256(inspect.getsource(m).encode()).hexdigest()
            for m in (gridworld_env, fixed_map_env, map_generator, rewards, dataset, planner_module, scenario_manifest, policy_adapters, benchmark_protocol, result_schema)}


def is_reserved_test(identity, salt):
    # Same JSON/hash contract as curriculum.map_pools.split_for, without SB3 import.
    return int(digest(dict(salt=salt, identity=identity))[:16], 16) % 100 >= 90


def generate(cfg):
    if cfg['contract'] != contract():
        raise ValueError('Unsupported observation/action/reward contract')
    if cfg['environment'].get('bfs_shaping') is not False or cfg['environment']['max_steps'] < cfg['min_path_length']:
        raise ValueError('Invalid common horizon/reward settings')
    if cfg['densities'] != [.1, .3, .5]:
        raise ValueError('Expected densities 0.1, 0.3, 0.5')
    if cfg['episodes_per_density'] < 1 or cfg['min_path_length'] < 1:
        raise ValueError('Invalid task counts/path limit')
    excluded, sources = exclusions(cfg)
    seen = set(excluded); rows = []
    for level, density in enumerate(cfg['densities']):
        env = RobotNavigationEnv(**cfg['environment'], obstacle_density=density)
        count = 0
        try:
            for candidate in range(cfg['max_candidates_per_density']):
                seed = int(np.random.SeedSequence([cfg['seed'], level, candidate]).generate_state(1)[0])
                obs, _ = env.reset(seed=seed)
                identity = map_identity(obs['obstacles'])
                path = shortest_path(obs['obstacles'], tuple(obs['agent']), tuple(obs['goal']))
                length = len(path)-1 if path else -1
                if (identity in seen or not is_reserved_test(identity, cfg['curriculum_split_salt'])
                        or not cfg['min_path_length'] <= length <= env.max_steps):
                    continue
                row = dict(map_id=identity, map_seed=seed, density=density,
                           start=obs['agent'].tolist(), goal=obs['goal'].tolist(),
                           obstacles=obs['obstacles'].tolist(),
                           obstacle_layout_hash=digest(obs['obstacles'].tolist()),
                           horizon=env.max_steps, optimal_steps=length)
                row['scenario_id'] = digest(row)
                rows.append(row); seen.add(identity); count += 1
                if count == cfg['episodes_per_density']:
                    break
            if count != cfg['episodes_per_density']:
                raise RuntimeError('Candidate budget exhausted')
        finally:
            env.close()
    result = dict(schema_version=1, config=cfg, source_fingerprints=source_fingerprints(),
                  numpy_version=np.__version__, identity_sources=sources,
                  excluded_layout_count=len(excluded), scenarios=rows,
                  leakage_scope='Verified supplied finite pools and declared historical seed ranges; random training coverage is not exhaustively logged')
    result['manifest_id'] = digest(result)
    return result


def validate(manifest):
    if manifest['schema_version'] != 1 or manifest['manifest_id'] != digest({k:v for k,v in manifest.items() if k != 'manifest_id'}):
        raise ValueError('Manifest checksum/schema mismatch')
    # Full regeneration also re-verifies exclusions, source fingerprints, unique
    # identities, environment settings and every saved seed/start/goal/layout.
    if generate(manifest['config']) != manifest:
        raise ValueError('Manifest does not reproduce from current source pools/code')
    for row in manifest['scenarios']:
        path = astar(np.asarray(row['obstacles']), tuple(row['start']), tuple(row['goal']))
        if path is None or len(path)-1 != row['optimal_steps']:
            raise ValueError('A*/BFS disagreement')
    return dict(valid=True, scenarios=len(manifest['scenarios']), manifest_id=manifest['manifest_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    m = generate(yaml.safe_load(Path(a.config).read_text()))
    status = validate(m); save_json(a.output, m)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
