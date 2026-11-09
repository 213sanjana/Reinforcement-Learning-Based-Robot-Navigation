"""Run from repository root, as a script or module. Never overwrites output."""
import argparse
import json
from pathlib import Path
import sys

# Direct script execution otherwise places only scripts/ on sys.path.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from imitation.dataset import assert_disjoint, load_dataset
from imitation.expert_generator import generate_dataset, historical_map_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--density', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='artifacts/imitation/expert_smoke.npz')
    parser.add_argument('--exclude-dataset', action='append', default=[])
    args = parser.parse_args()
    excluded = assert_disjoint(args.exclude_dataset)
    if args.config:
        with open(args.config, encoding='utf8') as stream:
            cfg = yaml.safe_load(stream)
        destination = Path(cfg['output_directory'])
        if destination.exists():
            raise FileExistsError(f'{destination} already exists; choose a new output_directory')
        env = cfg['environment']
        if cfg.get('exclude_historical_dqn', True):
            excluded |= historical_map_ids(env)
        # Check complete seed ranges, not just the accepted episode count.
        budget = cfg['seed_budget']
        ranges = []
        for spec in cfg['splits'].values():
            interval = (spec['seed'], spec['seed'] + budget)
            if any(interval[0] < b and a < interval[1] for a, b in ranges):
                raise ValueError('split seed budgets overlap')
            ranges.append(interval)
        destination.mkdir(parents=True, exist_ok=False)
        paths, reports = [], {}
        for split, spec in cfg['splits'].items():
            path = destination / f'{split}.npz'
            reports[split] = generate_dataset(path, episodes=spec['episodes'], seed=spec['seed'],
                split=split, density=env['obstacle_density'], height=env['height'], width=env['width'],
                max_steps=env['max_steps'], min_path_length=cfg['min_path_length'],
                balanced_lengths=cfg['balanced_lengths'], seed_budget=budget, excluded_ids=excluded)
            arrays, _ = load_dataset(path)
            excluded.update(arrays['map_ids'].tolist())
            paths.append(path)
            print(json.dumps({split: reports[split]}, sort_keys=True), flush=True)
        assert_disjoint(paths)
        reports['cross_split_layout_overlap'] = 0
        with (destination / 'statistics.json').open('x') as stream:
            json.dump(reports, stream, indent=2, sort_keys=True)
    else:
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        print(json.dumps(generate_dataset(args.output, episodes=args.episodes, density=args.density,
                                         seed=args.seed, excluded_ids=excluded), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
