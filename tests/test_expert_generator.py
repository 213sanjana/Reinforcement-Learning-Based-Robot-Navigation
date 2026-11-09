import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from environments.fixed_map_env import FixedMapNavigationEnv
from environments.gridworld_env import RobotNavigationEnv
from imitation.dataset import assert_disjoint, encode_observation, load_dataset, map_identity
from imitation.expert_generator import InvalidTrajectory, generate_dataset, replay_expert


def test_encoding_contract():
    obs = {'obstacles': np.zeros((10,10), dtype=np.int8),
           'agent': np.array([2,3], dtype=np.int32), 'goal': np.array([8,8], dtype=np.int32)}
    x = encode_observation(obs)
    assert x.shape == (3,10,10) and x.dtype == np.float32
    assert x[1,2,3] == x[2,8,8] == 1 and x.sum() == 2


def test_identity_groups_endpoint_changes_rotations_and_reflections():
    grid = np.zeros((10,10), dtype=np.int8)
    grid[1,2] = grid[3,6] = 1
    assert map_identity(grid) == map_identity(np.rot90(grid)) == map_identity(np.fliplr(grid))
    other = grid.copy(); other[9,9] = 1
    assert map_identity(grid) != map_identity(other)


def test_generation_determinism_and_every_transition(tmp_path):
    paths = [tmp_path/'a.npz', tmp_path/'b.npz']
    for p in paths:
        generate_dataset(p, episodes=25, seed=42)
    a, metadata = load_dataset(paths[0]); b, other = load_dataset(paths[1])
    assert metadata == other
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert metadata['stats']['length_bins'] == {'6-9':10, '10-14':10, '15+':5}
    assert a['episode_offsets'][-1] == len(a['actions']) == len(a['observations'])
    assert len(set(a['map_ids'])) == 25
    assert metadata['stats']['rejected_candidates'] > 0
    for i in range(25):
        env = FixedMapNavigationEnv(a['obstacles'][i], tuple(a['goals'][i]), max_steps=150)
        try:
            obs, _ = env.reset(options={'start': a['starts'][i]})
            lo, hi = a['episode_offsets'][i:i+2]
            for t in range(lo, hi):
                np.testing.assert_array_equal(encode_observation(obs), a['observations'][t])
                obs, reward, term, trunc, info = env.step(int(a['actions'][t]))
                assert reward == pytest.approx(a['rewards'][t])
                assert not info['collision'] and not trunc
                assert term == (t == hi - 1)
            assert tuple(obs['agent']) == tuple(a['goals'][i])
        finally:
            env.close()


def test_disjointness_exclusion_and_overwrite(tmp_path):
    first, second = tmp_path/'train.npz', tmp_path/'validation.npz'
    generate_dataset(first, episodes=10, seed=42)
    a, _ = load_dataset(first)
    # Deliberately reuse seed range: identity exclusion must still prevent leakage.
    generate_dataset(second, episodes=10, seed=42, excluded_ids=set(a['map_ids']))
    assert len(assert_disjoint([first,second])) == 20
    with pytest.raises(ValueError, match='duplicate'):
        assert_disjoint([first, first])
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        generate_dataset(first, episodes=1)
    assert first.read_bytes() == original


def test_rejects_bad_trajectory_and_short_budget(tmp_path):
    env = FixedMapNavigationEnv(np.zeros((4,4), dtype=np.int8), (0,2))
    try:
        obs, _ = env.reset(options={'start': (0,0)})
        with pytest.raises(InvalidTrajectory, match='not_shortest'):
            replay_expert(env, obs, [(0,0), (1,0), (1,1), (0,1), (0,2)])
        obs, _ = env.reset(options={'start': (0,0)})
        with pytest.raises(InvalidTrajectory, match='endpoint'):
            replay_expert(env, obs, [(0,1), (0,2)])
    finally:
        env.close()
    path = tmp_path/'failed.npz'
    with pytest.raises(RuntimeError, match='budget exhausted'):
        generate_dataset(path, episodes=10, seed_budget=10, min_path_length=149, balanced_lengths=False)
    assert not path.exists()


def test_truncation_is_rejected():
    env = FixedMapNavigationEnv(np.zeros((4,4), dtype=np.int8), (0,2), max_steps=1)
    try:
        obs, _ = env.reset(options={'start': (0,0)})
        with pytest.raises(InvalidTrajectory, match='episode_ending'):
            replay_expert(env, obs, [(0,0), (0,1), (0,2)])
    finally:
        env.close()


def test_cli_config_and_direct_script(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/generate_expert_dataset.py'
    result = subprocess.run([sys.executable, str(script), '--episodes', '5', '--output', str(tmp_path/'smoke.npz')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cfg = {'environment':dict(height=10,width=10,obstacle_density=.1,max_steps=150),
           'min_path_length':6, 'balanced_lengths':True, 'seed_budget':1000,
           'exclude_historical_dqn':False, 'output_directory':str(tmp_path/'full'),
           'splits':{name:{'episodes':5,'seed':10000+i*1000} for i,name in enumerate(['train','validation','test'])}}
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    result = subprocess.run([sys.executable, str(script), '--config', str(config)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(assert_disjoint([tmp_path/'full'/f'{s}.npz' for s in cfg['splits']])) == 15
    result = subprocess.run([sys.executable, str(script), '--config', str(config)], capture_output=True, text=True)
    assert result.returncode != 0 and 'already exists' in result.stderr


def test_dqn_encoder_exact_parity():
    pytest.importorskip('torch')
    from agents.dqn_agent import encode_observation as dqn_encode
    env = RobotNavigationEnv()
    try:
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(encode_observation(obs), dqn_encode(obs))
    finally:
        env.close()


def test_saved_dataset_verifier(tmp_path):
    from scripts.verify_expert_dataset import verify
    path = tmp_path/'verified.npz'
    generate_dataset(path, episodes=5)
    result = verify([path])
    assert result['verified.npz']['episodes_verified'] == 5
