"""Independently reload and replay every saved trajectory; also check split identities."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path
from imitation.dataset import assert_disjoint, encode_observation, load_dataset, map_identity


def verify(paths):
    assert_disjoint(paths)
    report = {}
    for path in paths:
        a, meta = load_dataset(path)
        count = len(a['map_seeds'])
        offsets = a['episode_offsets']
        if len(offsets) != count + 1 or offsets[0] != 0 or not np.all(np.diff(offsets) > 0):
            raise ValueError('invalid episode offsets')
        if offsets[-1] != len(a['actions']) or len(a['observations']) != len(a['actions']):
            raise ValueError('transition count mismatch')
        env = RobotNavigationEnv(**meta['environment'])
        try:
            for i in range(count):
                obs, _ = env.reset(seed=int(a['map_seeds'][i]))
                np.testing.assert_array_equal(obs['obstacles'], a['obstacles'][i])
                np.testing.assert_array_equal(obs['agent'], a['starts'][i])
                np.testing.assert_array_equal(obs['goal'], a['goals'][i])
                if map_identity(obs['obstacles']) != a['map_ids'][i]:
                    raise ValueError('map identity mismatch')
                lo, hi = map(int, offsets[i:i+2])
                reference = shortest_path(obs['obstacles'], tuple(obs['agent']), tuple(obs['goal']))
                if reference is None or len(reference)-1 != hi-lo or a['path_lengths'][i] != hi-lo:
                    raise ValueError('not a shortest path')
                total = 0.
                for t in range(lo, hi):
                    if a['observations'].dtype != np.float32 or a['actions'].dtype != np.int64:
                        raise ValueError('incorrect CNN/action dtype')
                    np.testing.assert_array_equal(encode_observation(obs), a['observations'][t])
                    obs, reward, term, trunc, info = env.step(int(a['actions'][t]))
                    if info['collision'] or trunc or term != (t == hi-1):
                        raise ValueError('invalid trajectory ending or collision')
                    np.testing.assert_allclose(reward, a['rewards'][t])
                    total += reward
                if tuple(obs['agent']) != tuple(obs['goal']) or not a['terminated'][i] or a['truncated'][i]:
                    raise ValueError('unsuccessful episode')
                np.testing.assert_allclose(total, a['episode_returns'][i])
        finally:
            env.close()
        report[Path(path).name] = {'episodes_verified':count, 'transitions_verified':len(a['actions'])}
    report['layout_overlap'] = 0
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    args = parser.parse_args()
    print(json.dumps(verify(args.datasets), indent=2, sort_keys=True))
