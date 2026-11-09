"""DQN with replay, target network and optional Double DQN."""
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from agents.replay_buffer import ReplayBuffer
from models.dqn_network import DQNNetwork
from models.cnn_dqn_network import CNNDQNNetwork

def encode_observation(obs: dict) -> np.ndarray:
    obstacles = np.asarray(obs['obstacles'], dtype=np.float32)
    if obstacles.ndim != 2 or not np.isin(obstacles, (0, 1)).all():
        raise ValueError('Invalid obstacle grid')
    h, w = obstacles.shape
    planes = np.zeros((3, h, w), dtype=np.float32)
    planes[0] = obstacles
    for plane, name in ((1, 'agent'), (2, 'goal')):
        pos = np.asarray(obs[name], dtype=int)
        if pos.shape != (2,) or not (0 <= pos[0] < h and 0 <= pos[1] < w):
            raise ValueError(f'Invalid {name} coordinate')
        planes[plane, pos[0], pos[1]] = 1
    return planes

@dataclass
class DQNConfig:
    learning_rate: float = 0.0001
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50000
    warmup_steps: int = 2000
    train_frequency: int = 4
    target_update_frequency: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150000
    hidden_size: int = 256
    architecture: str = "mlp"
    double_dqn: bool = False
    bootstrap_on_truncation: bool = True
    gradient_clip: float = 10.0

    def __post_init__(self):
        if self.architecture not in ("mlp", "cnn"):
            raise ValueError("architecture must be 'mlp' or 'cnn'")

        if not (
            0 < self.learning_rate
            and 0 <= self.gamma <= 1
            and 0 <= self.epsilon_end <= self.epsilon_start <= 1
        ):
            raise ValueError("Invalid learning parameters")

        for name in (
            "batch_size",
            "replay_capacity",
            "warmup_steps",
            "train_frequency",
            "target_update_frequency",
            "epsilon_decay_steps",
            "hidden_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")

class DQNAgent:
    def __init__(self, height: int, width: int, config: DQNConfig, seed: int = 0, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        self.config = config
        self.height, self.width = height, width
        self.rng = np.random.default_rng(seed)
            
        network_class = (
            CNNDQNNetwork
            if config.architecture == "cnn"
            else DQNNetwork
        )

        self.online = network_class(
            height, width, config.hidden_size
        ).to(self.device)

        self.target = network_class(
            height, width, config.hidden_size
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.replay_capacity, seed)
        self.steps = 0
        self.updates = 0

    @property
    def epsilon(self) -> float:
        fraction = min(1., self.steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, obs: dict, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        with torch.no_grad():
            x = torch.as_tensor(encode_observation(obs)[None], device=self.device)
            return int(self.online(x).argmax(dim=1).item())

    def remember(self, obs, action, reward, next_obs, terminated, truncated):
        self.buffer.add(encode_observation(obs), action, reward, encode_observation(next_obs), terminated, truncated)

    def td_targets(self, batch):
        with torch.no_grad():
            if self.config.double_dqn:
                actions = self.online(batch.next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(batch.next_states).gather(1, actions).squeeze(1)
            else:
                next_q = self.target(batch.next_states).max(dim=1).values
            done = batch.terminated | (batch.truncated if not self.config.bootstrap_on_truncation else torch.zeros_like(batch.truncated))
            return batch.rewards + self.config.gamma * next_q * (~done).float()

    def optimize(self):
        if len(self.buffer) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        batch = self.buffer.sample(self.config.batch_size, self.device)
        predicted = self.online(batch.states).gather(1, batch.actions[:, None]).squeeze(1)
        target = self.td_targets(batch)
        loss = nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())

    def observe_step(self, obs, action, reward, next_obs, terminated, truncated):
        self.remember(obs, action, reward, next_obs, terminated, truncated)
        self.steps += 1
        loss = self.optimize() if self.steps % self.config.train_frequency == 0 else None
        if self.steps % self.config.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'online': self.online.state_dict(), 'target': self.target.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'config': asdict(self.config),
                    'height': self.height, 'width': self.width, 'steps': self.steps,
                    'updates': self.updates}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu'):
        # Only load checkpoints you trust. weights_only avoids unrestricted pickle objects.
        data = torch.load(path, map_location='cpu', weights_only=True)
        agent = cls(int(data['height']), int(data['width']), DQNConfig(**data['config']), device=device)
        agent.online.load_state_dict(data['online'])
        agent.target.load_state_dict(data['target'])
        agent.optimizer.load_state_dict(data['optimizer'])
        # Move optimizer states when loading a CPU checkpoint onto CUDA.
        for state in agent.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(agent.device)
        agent.steps, agent.updates = int(data['steps']), int(data['updates'])
        agent.target.eval()
        return agent
