"""Pickle-free compressed NPZ schema and canonical obstacle-layout identity."""
import hashlib
import json
from pathlib import Path
import numpy as np
from planners.astar import validate_grid

SCHEMA_VERSION = 1


def map_identity(obstacles):
    """Group all rotations/reflections; endpoints intentionally excluded."""
    grid = validate_grid(obstacles).astype(np.uint8)
    variants = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        for a in (rotated, np.fliplr(rotated)):
            variants.append(np.asarray(a.shape, dtype='<i4').tobytes() + a.tobytes())
    return hashlib.sha256(min(variants)).hexdigest()


def encode_observation(obs):
    """Same float32 obstacle/agent/goal channel convention as DQN, without importing Torch."""
    grid = validate_grid(obs['obstacles'])
    h, w = grid.shape
    result = np.zeros((3, h, w), dtype=np.float32)
    result[0] = grid
    for channel, name in ((1, 'agent'), (2, 'goal')):
        p = np.asarray(obs[name])
        if p.shape != (2,) or not np.issubdtype(p.dtype, np.integer):
            raise ValueError('invalid coordinate')
        r, c = map(int, p)
        if not (0 <= r < h and 0 <= c < w) or grid[r, c]:
            raise ValueError('coordinate must be a free in-bounds cell')
        result[channel, r, c] = 1
    return result


def save_dataset(path, arrays, metadata):
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('output must end in .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting even when another process races.
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_dataset(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files if k != 'metadata'}
        metadata = json.loads(str(source['metadata']))
    if metadata['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported dataset schema')
    return arrays, metadata


def assert_disjoint(paths):
    seen = set()
    for path in paths:
        arrays, _ = load_dataset(path)
        identities = arrays['map_ids'].tolist()
        if len(set(identities)) != len(identities) or seen.intersection(identities):
            raise ValueError('duplicate obstacle layout within/across datasets')
        seen.update(identities)
    return seen
