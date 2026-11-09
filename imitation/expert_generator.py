"""A* trajectories verified by BFS and by replay through the real environment."""
from collections import Counter
import hashlib
import inspect
import numpy as np
from environments.gridworld_env import RobotNavigationEnv
from environments.map_generator import shortest_path
from imitation.dataset import SCHEMA_VERSION, encode_observation, map_identity, save_dataset
from planners.astar import ACTIONS, astar, path_to_actions


class InvalidTrajectory(ValueError):
    pass


def replay_expert(env, initial, path):
    """Replay from an already-reset environment; reject all inconsistencies."""
    if tuple(env.ACTIONS) != ACTIONS:
        raise InvalidTrajectory('action_mapping_mismatch')
    if not path or tuple(initial['agent']) != tuple(path[0]) or tuple(initial['goal']) != tuple(path[-1]):
        raise InvalidTrajectory('endpoint_mismatch')
    reference = shortest_path(initial['obstacles'], tuple(initial['agent']), tuple(initial['goal']))
    if reference is None or len(reference) != len(path):
        raise InvalidTrajectory('not_shortest')
    actions = path_to_actions(path)
    if not actions:
        raise InvalidTrajectory('empty_trajectory')
    observations, rewards = [], []
    obs = initial
    for index, action in enumerate(actions):
        target = path[index + 1]
        if not (0 <= target[0] < env.height and 0 <= target[1] < env.width) or obs['obstacles'][target]:
            raise InvalidTrajectory('illegal_planned_move')
        if not env.action_space.contains(action):
            raise InvalidTrajectory('invalid_action')
        observations.append(encode_observation(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info['collision'] or tuple(obs['agent']) != tuple(target):
            raise InvalidTrajectory('replay_mismatch')
        last = index == len(actions) - 1
        if truncated or terminated != last:
            raise InvalidTrajectory('incorrect_episode_ending')
    if tuple(obs['agent']) != tuple(obs['goal']):
        raise InvalidTrajectory('goal_not_reached')
    return np.stack(observations), np.asarray(actions, dtype=np.int64), np.asarray(rewards, dtype=np.float32)


def historical_map_ids(environment):
    """Reserve known 5% DQN layouts, including its validation and configured evaluation."""
    params = dict(environment, obstacle_density=0.05)
    env = RobotNavigationEnv(**params)
    ids = set()
    try:
        for seed in list(range(2027, 4464)) + list(range(50211, 50241)) + list(range(90211, 90311)):
            obs, _ = env.reset(seed=seed)
            ids.add(map_identity(obs['obstacles']))
    finally:
        env.close()
    return ids


def generate_dataset(output, episodes=100, density=0.1, seed=42, split='smoke',
                     height=10, width=10, max_steps=150, min_path_length=6,
                     seed_budget=1000000, excluded_ids=None, balanced_lengths=True):
    if episodes < 1 or seed < 0 or seed_budget < episodes:
        raise ValueError('invalid episode count, seed, or seed budget')
    if min_path_length < 1 or max_steps < min_path_length:
        raise ValueError('invalid path length limits')
    if balanced_lengths and min_path_length != 6:
        raise ValueError('balanced lengths require min_path_length=6')
    environment = dict(height=height, width=width, obstacle_density=density,
                       max_steps=max_steps, bfs_shaping=False)
    env = RobotNavigationEnv(**environment)
    seen = set(excluded_ids or ())
    observations, actions, rewards = [], [], []
    seeds, starts, goals, grids, ids, lengths, returns = [], [], [], [], [], [], []
    offsets = [0]
    rejected = []
    # 40% length 6-9, 40% length 10-14, 20% length >=15 (rounding goes to final bin).
    targets = [int(episodes * .4), int(episodes * .4)]
    targets.append(episodes - sum(targets))
    bins = [0, 0, 0]
    try:
        for candidate in range(seed, seed + seed_budget):
            if len(seeds) == episodes:
                break
            try:
                obs, _ = env.reset(seed=candidate)
                identity = map_identity(obs['obstacles'])
                if identity in seen:
                    raise InvalidTrajectory('duplicate_or_excluded_layout')
                path = astar(obs['obstacles'], obs['agent'], obs['goal'])
                if path is None:
                    raise InvalidTrajectory('no_path')
                length = len(path) - 1
                if length < min_path_length:
                    raise InvalidTrajectory('path_too_short')
                if length > max_steps:
                    raise InvalidTrajectory('path_exceeds_time_limit')
                bucket = 0 if length < 10 else (1 if length < 15 else 2)
                if balanced_lengths and bins[bucket] >= targets[bucket]:
                    raise InvalidTrajectory('length_bin_full')
                x, a, r = replay_expert(env, obs, path)
            except (InvalidTrajectory, ValueError, RuntimeError) as exc:
                rejected.append({'seed': candidate, 'reason': str(exc)})
                continue
            seen.add(identity)
            observations.append(x); actions.append(a); rewards.append(r)
            seeds.append(candidate); starts.append(obs['agent']); goals.append(obs['goal'])
            grids.append(obs['obstacles']); ids.append(identity); lengths.append(length)
            returns.append(float(r.sum())); offsets.append(offsets[-1] + length)
            bins[bucket] += 1
        if len(seeds) != episodes:
            raise RuntimeError(f'Accepted {len(seeds)}/{episodes}; seed budget exhausted. No dataset saved.')
    finally:
        env.close()
    values = np.asarray(lengths, dtype=np.int32)
    stats = dict(episodes=episodes, transitions=int(values.sum()), success_rate=1.0,
                 collisions=0, shortest_paths_verified=episodes,
                 path_min=int(values.min()), path_max=int(values.max()),
                 path_mean=float(values.mean()), path_median=float(np.median(values)),
                 path_percentiles=dict(zip(('p10', 'p25', 'p75', 'p90'), map(float, np.percentile(values, [10,25,75,90])))),
                 length_bins={'6-9': bins[0], '10-14': bins[1], '15+': bins[2]},
                 path_histogram={str(k): int(v) for k, v in sorted(Counter(lengths).items())},
                 rejected_candidates=len(rejected), rejection_counts=dict(Counter(r['reason'] for r in rejected)))
    arrays = dict(observations=np.concatenate(observations), actions=np.concatenate(actions),
                  rewards=np.concatenate(rewards), episode_offsets=np.asarray(offsets, dtype=np.int64),
                  map_seeds=np.asarray(seeds, dtype=np.int64), starts=np.asarray(starts, dtype=np.int32),
                  goals=np.asarray(goals, dtype=np.int32), obstacles=np.stack(grids),
                  map_ids=np.asarray(ids), path_lengths=values,
                  episode_returns=np.asarray(returns, dtype=np.float32),
                  terminated=np.ones(episodes, dtype=np.bool_), truncated=np.zeros(episodes, dtype=np.bool_))
    source_hashes = {}
    for cls in (RobotNavigationEnv,):
        source_hashes[cls.__name__] = hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
    metadata = dict(schema_version=SCHEMA_VERSION, split=split, environment=environment,
                    seed_start=seed, seed_budget=seed_budget, min_path_length=min_path_length,
                    balanced_lengths=balanced_lengths, planner='astar_manhattan_unit_cost_v1',
                    map_identity='sha256_canonical_D4_obstacles_and_dimensions_v1',
                    channels=['obstacles', 'agent', 'goal'], actions=[list(a) for a in ACTIONS],
                    reward='default Manhattan shaping; BFS shaping disabled', source_hashes=source_hashes,
                    stats=stats, rejections=rejected)
    save_dataset(output, arrays, metadata)
    return stats
