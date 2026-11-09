"""Fresh supervised CNN actor built from the existing DQN feature architecture."""
import torch
from torch import nn
from models.cnn_dqn_network import CNNDQNNetwork

ARCHITECTURE = 'cnn_bc_v1'


class BCNetwork(nn.Module):
    def __init__(self, height=10, width=10, hidden_size=256, n_actions=4):
        super().__init__()
        if n_actions != 4:
            raise ValueError('navigation requires four actions')
        template = CNNDQNNetwork(height, width, hidden_size, n_actions)
        self.height, self.width = int(height), int(width)
        self.hidden_size, self.n_actions = int(hidden_size), int(n_actions)
        self.features = template.features
        self.shared = nn.Sequential(*list(template.head.children())[:-1])
        # New classification head; no DQN checkpoint is loaded.
        self.actor = nn.Linear(hidden_size, n_actions)

    def forward(self, observations):
        if observations.ndim != 4 or tuple(observations.shape[1:]) != (3, self.height, self.width):
            raise ValueError(f'Expected (N, 3, {self.height}, {self.width})')
        return self.actor(self.shared(self.features(observations)))

    @torch.no_grad()
    def predict(self, observations):
        return self(observations).argmax(dim=-1)


def resolve_device(request='auto'):
    if request not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    chosen = 'cuda' if request == 'auto' and torch.cuda.is_available() else ('cpu' if request == 'auto' else request)
    if chosen == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(chosen)


def load_bc_checkpoint(path, device='cpu'):
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported BC checkpoint format')
    network = BCNetwork(**checkpoint['network_config']).to(device)
    network.load_state_dict(checkpoint['model_state_dict'], strict=True)
    network.eval()
    return network, checkpoint
