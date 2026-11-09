"""Validated tensor Dataset for the Phase 3B pickle-free NPZ schema."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from imitation.dataset import load_dataset, map_identity


class BCDataset(Dataset):
    def __init__(self, path, expected_split):
        if expected_split not in ('train', 'validation'):
            raise ValueError('BC training supports only train/validation assets')
        self.path = Path(path)
        arrays, self.metadata = load_dataset(path)
        if self.metadata.get('split') != expected_split:
            raise ValueError(f'Expected split {expected_split}; got {self.metadata.get("split")}')
        if self.metadata.get('channels') != ['obstacles', 'agent', 'goal']:
            raise ValueError('CNN channel order mismatch')
        if self.metadata.get('actions') != [[-1,0],[1,0],[0,-1],[0,1]]:
            raise ValueError('Action mapping mismatch')
        self.environment = self.metadata['environment']
        h, w = self.environment['height'], self.environment['width']
        x, y = arrays['observations'], arrays['actions']
        n = len(y)
        if n < 1 or x.shape != (n,3,h,w) or x.dtype != np.float32:
            raise ValueError('Invalid float32 CNN observation shape')
        if y.dtype != np.int64 or y.ndim != 1 or not ((0 <= y) & (y < 4)).all():
            raise ValueError('Invalid int64 expert actions')
        if not np.isfinite(x).all() or not np.isin(x, (0,1)).all():
            raise ValueError('Observations must be finite binary planes')
        for channel in (1,2):
            if not (x[:,channel].sum(axis=(1,2)) == 1).all():
                raise ValueError('Agent and goal planes must be one-hot')
            if (x[:,0] * x[:,channel]).any():
                raise ValueError('Agent or goal occupies an obstacle')
        offsets = arrays['episode_offsets']
        self.map_ids = arrays['map_ids'].tolist()
        e = len(self.map_ids)
        if e < 1 or len(offsets) != e+1 or offsets[0] != 0 or offsets[-1] != n or not (np.diff(offsets)>0).all():
            raise ValueError('Invalid episode offsets')
        if len(set(self.map_ids)) != e:
            raise ValueError('Duplicate layout within dataset')
        if not arrays['terminated'].all() or arrays['truncated'].any():
            raise ValueError('Unsuccessful expert episode')
        for i, identity in enumerate(self.map_ids):
            grid = arrays['obstacles'][i]
            if map_identity(grid) != identity:
                raise ValueError('Stored layout identity mismatch')
            lo, hi = map(int, offsets[i:i+2])
            if not np.all(x[lo:hi,0] == grid):
                raise ValueError('Observation/layout mismatch')
        # Contiguous host tensors; Dataset never moves data to CUDA.
        self.observations = torch.from_numpy(np.ascontiguousarray(x))
        self.actions = torch.from_numpy(np.ascontiguousarray(y))
        self.episodes = e
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]


def load_training_pair(train_path, validation_path):
    train = BCDataset(train_path, 'train')
    validation = BCDataset(validation_path, 'validation')
    for key in ('height','width','obstacle_density','max_steps','bfs_shaping'):
        if train.environment.get(key) != validation.environment.get(key):
            raise ValueError(f'Train/validation environment mismatch: {key}')
    if set(train.map_ids).intersection(validation.map_ids):
        raise ValueError('Train/validation map leakage')
    return train, validation
