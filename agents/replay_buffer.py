"""Bounded, seeded replay memory; observations are immutable snapshots."""
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.data = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, terminated, truncated):
        s = np.asarray(state, dtype=np.float32).copy()
        ns = np.asarray(next_state, dtype=np.float32).copy()
        if s.shape != ns.shape:
            raise ValueError('state shapes must match')
        self.data.append((s, int(action), float(reward), ns, bool(terminated), bool(truncated)))

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        if batch_size < 1 or batch_size > len(self):
            raise ValueError('invalid batch size')
        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        rows = [self.data[int(i)] for i in indices]
        return Batch(
            torch.as_tensor(np.stack([r[0] for r in rows]), device=device),
            torch.as_tensor([r[1] for r in rows], dtype=torch.long, device=device),
            torch.as_tensor([r[2] for r in rows], dtype=torch.float32, device=device),
            torch.as_tensor(np.stack([r[3] for r in rows]), device=device),
            torch.as_tensor([r[4] for r in rows], dtype=torch.bool, device=device),
            torch.as_tensor([r[5] for r in rows], dtype=torch.bool, device=device),
        )
