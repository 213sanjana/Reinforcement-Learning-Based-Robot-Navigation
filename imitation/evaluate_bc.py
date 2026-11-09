"""Separate greedy closed-loop validation of a fixed BC checkpoint; no model updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environments.fixed_map_env import FixedMapNavigationEnv
from imitation.bc_dataset import BCDataset
from imitation.bc_network import load_bc_checkpoint
from imitation.dataset import encode_observation, load_dataset


def evaluate(checkpoint, dataset_path, output, device='auto', episodes=100):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new report path')
    validated = BCDataset(dataset_path, 'validation')
    arrays, metadata = load_dataset(dataset_path)
    network, payload = load_bc_checkpoint(checkpoint, device)
    device = next(network.parameters()).device
    env_config = metadata['environment']
    if payload.get('channels') != metadata['channels'] or payload.get('actions') != metadata['actions']:
        raise ValueError('Checkpoint observation/action contract mismatch')
    for name in ('height','width','obstacle_density','max_steps','bfs_shaping'):
        if payload['environment'].get(name) != env_config.get(name):
            raise ValueError(f'Checkpoint/validation environment mismatch: {name}')
    if episodes < 1 or episodes > validated.episodes:
        raise ValueError(f'episodes must be in [1, {validated.episodes}]')
    # Fixed seeded subset avoids taking a path-length-biased prefix of the quota-generated file.
    indices = np.random.default_rng(90211).choice(validated.episodes, size=episodes, replace=False)
    rows = []
    for index in indices:
        i = int(index)
        env = FixedMapNavigationEnv(arrays['obstacles'][i], tuple(arrays['goals'][i]),
                                    max_steps=env_config['max_steps'],
                                    bfs_shaping=env_config.get('bfs_shaping',False),
                                    shaping_gamma=env_config.get('shaping_gamma',.99))
        try:
            obs, _ = env.reset(options={'start': arrays['starts'][i]})
            reward_sum, collisions, steps = 0., 0, 0
            route = [obs['agent'].tolist()]
            with torch.inference_mode():
                while True:
                    tensor = torch.from_numpy(encode_observation(obs)).unsqueeze(0).to(device)
                    action = int(network(tensor).argmax(1).item())
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1; reward_sum += reward; collisions += int(info['collision'])
                    route.append(obs['agent'].tolist())
                    if terminated or truncated:
                        break
            optimal = int(arrays['path_lengths'][i])
            row = dict(episode_index=i, map_seed=int(arrays['map_seeds'][i]), map_id=str(arrays['map_ids'][i]),
                       start=arrays['starts'][i].tolist(), goal=arrays['goals'][i].tolist(),
                       success=bool(terminated), truncated=bool(truncated), steps=steps,
                       reward=float(reward_sum), collisions=collisions, optimal_steps=optimal,
                       spl=float(terminated)*optimal/max(optimal,steps))
            if len(rows) < 3:
                row['route'] = route
            rows.append(row)
        finally:
            env.close()
    successes = [r for r in rows if r['success']]
    report = dict(checkpoint=str(checkpoint), checkpoint_epoch=int(payload['epoch']),
                  selection_metric=payload['selection_metric'], dataset=str(dataset_path),
                  dataset_sha256=validated.sha256, split='validation', device=str(device), episodes=episodes,
                  subset_seed=90211, policy='greedy argmax, no action masking',
                  success_rate=len(successes)/episodes, mean_reward=float(np.mean([r['reward'] for r in rows])),
                  mean_steps=float(np.mean([r['steps'] for r in rows])),
                  mean_successful_steps=float(np.mean([r['steps'] for r in successes])) if successes else None,
                  mean_collisions=float(np.mean([r['collisions'] for r in rows])),
                  mean_spl=float(np.mean([r['spl'] for r in rows])), results=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='artifacts/imitation/behavior_cloning/best.pt')
    parser.add_argument('--dataset', default='artifacts/imitation/expert_density10_v1/validation.npz')
    parser.add_argument('--output', default='artifacts/imitation/behavior_cloning/closed_loop_validation.json')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    report = evaluate(args.checkpoint,args.dataset,args.output,args.device,args.episodes)
    print(json.dumps({k:v for k,v in report.items() if k != 'results'},indent=2))


if __name__ == '__main__':
    main()
