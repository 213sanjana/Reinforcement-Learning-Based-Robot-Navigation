"""Phase 5A validation and small paired smoke rollouts; no full benchmark CLI."""
import argparse
import json
from pathlib import Path
import numpy as np
from environments.fixed_map_env import FixedMapNavigationEnv
from environments.gridworld_env import RobotNavigationEnv
from environments.rewards import RewardConfig
from imitation.dataset import encode_observation
from evaluation.scenario_manifest import contract, file_hash, save_json, validate
from evaluation.policy_adapters import load_policy
from evaluation.result_schema import EpisodeResult, SCHEMA_VERSION, summarize


def smoke_subset(manifest, count=10):
    if count != 10:
        raise ValueError('Phase 5A smoke uses exactly 10 scenarios per policy')
    groups = [[s for s in manifest['scenarios'] if s['density'] == d] for d in [.1,.3,.5]]
    selected = []
    for i in range(4):
        for group in groups:
            if i < len(group) and len(selected) < count:
                selected.append(group[i])
    if len(selected) != count:
        raise ValueError('Manifest needs at least four scenarios per density for smoke')
    return selected


def run_episode(policy, name, row):
    if [list(a) for a in RobotNavigationEnv.ACTIONS] != contract()['actions']:
        raise ValueError('Environment action contract changed')
    env = FixedMapNavigationEnv(np.asarray(row['obstacles']), tuple(row['goal']),
                               max_steps=row['horizon'], bfs_shaping=False,
                               reward_config=RewardConfig(**contract()['reward']))
    try:
        obs, _ = env.reset(seed=row['map_seed'], options={'start':row['start']})
        policy.reset(row); total = 0.; collisions = 0
        for step in range(1,row['horizon']+1):
            encoded = encode_observation(obs)
            if encoded.dtype != np.float32 or encoded.shape != (3,env.height,env.width):
                raise ValueError('Observation contract mismatch')
            action = policy.act(obs)
            if not env.action_space.contains(action):
                raise ValueError('Invalid policy action')
            obs, reward, term, trunc, info = env.step(action)
            total += reward; collisions += int(info['collision'])
            if term or trunc:
                efficiency = row['optimal_steps']/step if term else None
                result = EpisodeResult(name,row['scenario_id'],row['map_id'],row['map_seed'],row['density'],
                    row['horizon'],bool(term),bool(trunc),step,collisions,float(total),
                    row['optimal_steps'],efficiency,efficiency or 0.).to_dict()
                if name == 'astar' and (not term or collisions or step != row['optimal_steps']):
                    raise RuntimeError('A* replay failed')
                return result
        raise RuntimeError('Environment failed to end at common horizon')
    finally:
        env.close()


def run_smoke(manifest, names, device='cpu'):
    validate(manifest)
    scenarios = smoke_subset(manifest)
    records = []; provenance = {}
    for name in names:
        spec = manifest['config']['policies'][name]
        policy = load_policy(name,spec,manifest['config']['environment'],device)
        before = policy.fingerprint()
        records.extend(run_episode(policy,name,s) for s in scenarios)
        if before != policy.fingerprint():
            raise RuntimeError('Policy changed during inference')
        if hasattr(policy,'verify'):
            policy.verify()
        provenance[name] = dict(specification=spec, state_fingerprint=before,
            checkpoint_sha256=file_hash(spec['checkpoint']) if 'checkpoint' in spec else None,
            weights_unchanged=True)
    summary = {name:{str(d):summarize([r for r in records if r['policy']==name and r['density']==d])
                    for d in [.1,.3,.5]} for name in names}
    return dict(schema_version=SCHEMA_VERSION, role='infrastructure_smoke_only',
                manifest_id=manifest['manifest_id'], scenarios_per_policy=10,
                scenario_ids=[s['scenario_id'] for s in scenarios],
                provenance=provenance, results=records, summary=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--policies', nargs='+', default=['astar'])
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='cpu')
    p.add_argument('--output')
    a = p.parse_args()
    m = json.loads(Path(a.manifest).read_text())
    if a.validate_only == a.smoke:
        p.error('Choose exactly one of --validate-only or --smoke')
    if a.validate_only:
        print(json.dumps(validate(m),indent=2)); return
    if not a.output:
        p.error('--smoke requires --output')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    if len(set(a.policies)) != len(a.policies):
        p.error('Duplicate policy names')
    report = run_smoke(m,a.policies,a.device)
    save_json(a.output,report)
    print(json.dumps(dict(status='passed',scenarios_per_policy=10,rollouts=len(report['results']),output=a.output),indent=2))


if __name__ == '__main__':
    main()
