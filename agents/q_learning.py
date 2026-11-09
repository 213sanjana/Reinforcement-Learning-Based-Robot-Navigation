"""NumPy tabular Q-learning restricted to one known occupancy map and goal."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import numpy as np


def map_signature(obstacles: np.ndarray, goal: tuple[int, int]) -> str:
    grid = np.asarray(obstacles, dtype=np.uint8)
    if grid.ndim != 2 or not np.isin(grid, [0, 1]).all():
        raise ValueError('Expected a binary 2D obstacle map')
    metadata = json.dumps({'shape': list(grid.shape), 'goal': list(goal)}, sort_keys=True)
    return sha256(metadata.encode() + grid.tobytes(order='C')).hexdigest()


@dataclass(frozen=True)
class QLearningConfig:
    learning_rate: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    bootstrap_on_truncation: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError('learning_rate must be (0,1], gamma must be [0,1]')
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError('Require 0 <= epsilon_min <= epsilon_start <= 1')
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError('epsilon_decay must be (0,1]')


class QLearningAgent:
    def __init__(self, obstacles: np.ndarray, goal: tuple[int, int], config: QLearningConfig,
                 seed: int = 0) -> None:
        self.obstacles = np.asarray(obstacles, dtype=np.int8).copy()
        self.goal = tuple(map(int, goal))
        self.signature = map_signature(self.obstacles, self.goal)
        h, w = self.obstacles.shape
        if not (0 <= self.goal[0] < h and 0 <= self.goal[1] < w) or self.obstacles[self.goal]:
            raise ValueError('Goal must be a free cell')
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((h * w, 4), dtype=np.float64)
        self.epsilon = config.epsilon_start

    def state(self, observation: dict[str, np.ndarray]) -> int:
        if map_signature(observation['obstacles'], tuple(map(int, observation['goal']))) != self.signature:
            raise ValueError('Map/goal mismatch: this Q-table is only valid on its training map')
        row, col = map(int, observation['agent'])
        h, w = self.obstacles.shape
        if not (0 <= row < h and 0 <= col < w) or self.obstacles[row, col]:
            raise ValueError('Agent position is not a free cell')
        return row * w + col

    def select_action(self, state: int, explore: bool = True) -> int:
        if not 0 <= state < len(self.q_table):
            raise ValueError('State out of range')
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        # Deterministic tie-breaking for reproducible greedy evaluation.
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int,
               terminated: bool, truncated: bool) -> float:
        if not 0 <= state < len(self.q_table) or not 0 <= next_state < len(self.q_table):
            raise ValueError('State out of range')
        if not 0 <= action < 4:
            raise ValueError('Invalid action')
        terminal_target = terminated or (truncated and not self.config.bootstrap_on_truncation)
        bootstrap = 0.0 if terminal_target else self.config.gamma * float(np.max(self.q_table[next_state]))
        target = float(reward) + bootstrap
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.config.learning_rate * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q_table=self.q_table, obstacles=self.obstacles,
                            goal=np.array(self.goal, dtype=np.int64),
                            signature=np.array(self.signature), epsilon=np.array(self.epsilon),
                            config=np.array(json.dumps(self.config.__dict__, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> 'QLearningAgent':
        with np.load(path, allow_pickle=False) as data:
            obstacles = data['obstacles'].copy()
            goal = tuple(map(int, data['goal']))
            config = QLearningConfig(**json.loads(str(data['config'].item())))
            agent = cls(obstacles, goal, config, seed=seed)
            if agent.signature != str(data['signature'].item()):
                raise ValueError('Checkpoint map signature mismatch')
            q = data['q_table']
            if q.shape != agent.q_table.shape or not np.isfinite(q).all():
                raise ValueError('Invalid checkpoint Q-table')
            agent.q_table[:] = q
            agent.epsilon = float(data['epsilon'].item())
            if not 0 <= agent.epsilon <= 1:
                raise ValueError('Invalid checkpoint epsilon')
            return agent
