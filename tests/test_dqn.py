"""Focused DQN tests; smoke training/evaluation uses tiny configs."""
import numpy as np
import pytest
import torch
from agents.dqn_agent import DQNAgent, DQNConfig, encode_observation
from agents.replay_buffer import ReplayBuffer
from models.dqn_network import DQNNetwork
from environments.gridworld_env import RobotNavigationEnv
from training.train_dqn import load_config, train
from evaluation.evaluate_dqn import evaluate


def observation():
    return {'obstacles': np.zeros((10, 10), dtype=np.int8),
            'agent': np.array([2, 3]), 'goal': np.array([8, 8])}


def test_encoding_changes_with_map_robot_goal():
    a = observation()
    x = encode_observation(a)
    assert x.shape == (3, 10, 10) and x.dtype == np.float32
    assert x[1, 2, 3] == 1 and x[2, 8, 8] == 1
    b = observation()
    b['obstacles'][1, 1] = 1
    assert not np.array_equal(x, encode_observation(b))
    b['agent'][0] = 1
    assert not np.array_equal(x, encode_observation(b))
    b['goal'][0] = 7
    assert not np.array_equal(x, encode_observation(b))


def test_network_outputs_four_values():
    model = DQNNetwork(10, 10)
    assert model(torch.zeros((5, 3, 10, 10))).shape == (5, 4)


def test_replay_capacity_and_copy():
    buf = ReplayBuffer(2, seed=4)
    x = np.zeros((3, 10, 10), dtype=np.float32)
    buf.add(x, 0, 1, x, False, False)
    x[0, 0, 0] = 1
    assert buf.data[0][0][0, 0, 0] == 0
    buf.add(x, 1, 2, x, False, True)
    buf.add(x, 2, 3, x, True, False)
    assert len(buf) == 2
    batch = buf.sample(2, torch.device('cpu'))
    assert batch.states.shape == (2, 3, 10, 10)
    assert batch.actions.dtype == torch.long


def test_terminal_vs_truncation_bootstrap():
    cfg = DQNConfig(batch_size=2, warmup_steps=2, gamma=.9)
    agent = DQNAgent(10, 10, cfg)
    # Constant next-state Q=2 by setting final layer bias.
    with torch.no_grad():
        for p in agent.target.parameters():
            p.zero_()
        agent.target.net[-1].bias.fill_(2.)
    x = np.zeros((3, 10, 10), dtype=np.float32)
    agent.buffer.add(x, 0, 1, x, True, False)
    agent.buffer.add(x, 0, 1, x, False, True)
    batch = agent.buffer.sample(2, agent.device)
    targets = agent.td_targets(batch)
    for i in range(2):
        assert targets[i].item() == pytest.approx(1. if batch.terminated[i] else 2.8)
    agent.config.bootstrap_on_truncation = False
    assert torch.allclose(agent.td_targets(batch), torch.ones(2, device=agent.device))


def test_epsilon_schedule_and_greedy_action():
    agent = DQNAgent(10, 10, DQNConfig(epsilon_decay_steps=10))
    assert agent.epsilon == 1.
    agent.steps = 10
    assert agent.epsilon == pytest.approx(.05)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        agent.online.net[-1].bias[3] = 5
    assert agent.act(observation(), explore=False) == 3


def test_optimizer_target_sync_and_checkpoint(tmp_path):
    cfg = DQNConfig(batch_size=2, warmup_steps=2, train_frequency=1, target_update_frequency=2)
    agent = DQNAgent(10, 10, cfg, seed=1)
    obs = observation()
    agent.observe_step(obs, 0, 1, obs, False, False)
    loss = agent.observe_step(obs, 1, 1, obs, False, True)
    assert np.isfinite(loss) and agent.updates == 1
    assert all(torch.equal(x, y) for x, y in zip(agent.online.parameters(), agent.target.parameters()))
    path = tmp_path / 'model.pt'
    agent.save(path)
    loaded = DQNAgent.load(path)
    assert loaded.steps == 2 and loaded.updates == 1
    assert loaded.act(obs, explore=False) == agent.act(obs, explore=False)


def test_double_dqn_uses_online_selection():
    agent = DQNAgent(10, 10, DQNConfig(batch_size=1, warmup_steps=1, double_dqn=True, gamma=1))
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        for p in agent.target.parameters():
            p.zero_()
        agent.online.net[-1].bias.copy_(torch.tensor([5., 0., 0., 0.]))
        agent.target.net[-1].bias.copy_(torch.tensor([2., 9., 0., 0.]))
    x = np.zeros((3, 10, 10), dtype=np.float32)
    agent.buffer.add(x, 0, 1, x, False, False)
    batch = agent.buffer.sample(1, agent.device)
    assert agent.td_targets(batch).item() == pytest.approx(3.)
    agent.config.double_dqn = False
    assert agent.td_targets(batch).item() == pytest.approx(10.)


def test_gymnasium_integration():
    env = RobotNavigationEnv(height=10, width=10)
    try:
        obs, _ = env.reset(seed=1)
        agent = DQNAgent(10, 10, DQNConfig())
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated, truncated)
        assert len(agent.buffer) == 1
    finally:
        env.close()


def test_smoke_train_and_evaluate(tmp_path):
    cfg = load_config('configs/dqn.yaml')
    cfg['training'].update(total_steps=12, seed=5, log_interval=6,
                           checkpoint=str(tmp_path/'model.pt'), metrics=str(tmp_path/'metrics.csv'),
                           figure=str(tmp_path/'curve.png'), device='cpu')
    cfg['agent'].update(batch_size=2, warmup_steps=2, train_frequency=1, target_update_frequency=3)
    cfg['environment']['max_steps'] = 5
    cfg['evaluation'].update(seed=90000, episodes=2, device='cpu',
                             results=str(tmp_path/'eval.json'), route_plot=str(tmp_path/'route.png'))
    agent, metrics = train(cfg)
    assert agent.steps == 12 and metrics
    assert (tmp_path/'model.pt').exists()
    summary = evaluate(cfg)
    assert summary['episodes'] == 2
    assert (tmp_path/'eval.json').exists() and (tmp_path/'route.png').exists()
