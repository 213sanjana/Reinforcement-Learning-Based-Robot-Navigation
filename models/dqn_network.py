"""MLP DQN over three spatial planes: obstacles, robot, goal."""
import torch
from torch import nn

class DQNNetwork(nn.Module):
    def __init__(self, height: int, width: int, hidden_size: int = 256, n_actions: int = 4):
        super().__init__()
        self.height, self.width = height, width
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * height * width, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4 or tuple(observations.shape[1:]) != (3, self.height, self.width):
            raise ValueError('Expected batch of (3, H, W) observations')
        return self.net(observations)
