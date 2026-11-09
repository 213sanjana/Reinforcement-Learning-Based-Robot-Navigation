
"""Convolutional Q-network for spatial GridWorld observations."""

import torch
from torch import nn


class CNNDQNNetwork(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        hidden_size: int = 256,
        n_actions: int = 4,
    ):
        super().__init__()

        if height < 3 or width < 3:
            raise ValueError("CNN requires height and width >= 3")

        self.height = height
        self.width = width

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width)
            feature_size = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        expected_shape = (3, self.height, self.width)

        if observations.ndim != 4:
            raise ValueError("Expected observations with shape (N, 3, H, W)")

        if tuple(observations.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}"
            )

        return self.head(self.features(observations))
