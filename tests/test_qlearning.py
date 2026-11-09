"""Algorithm, fixed-map, checkpoint and evaluation regression tests."""
from dataclasses import replace
import numpy as np
import pytest
from agents.q_learning import QLearningAgent, QLearningConfig
from environments.fixed_map_env import FixedMapNavigationEnv
from evaluation.evaluate_qlearning import evaluate
from training.train_qlearning import build_env, load_config, train


def test_td_nonterminal():
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8), (2,2), QLearningConfig(learning_rate=.5,gamma=.9))
    a.q_table[1, :] = [1,2,3,4]
    a.update(0, 2, 2, 1, False, False)
    assert a.q_table[0,2] == pytest.approx(.5 * (2 + .9*4))


def test_terminated_has_no_bootstrap():
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8), (2,2), QLearningConfig(learning_rate=1))
    a.q_table[1,:] = 100
    a.update(0, 0, 5, 1, True, False)
    assert a.q_table[0,0] == 5


def test_truncation_bootstraps_by_default():
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8), (2,2), QLearningConfig(learning_rate=1,gamma=.5))
    a.q_table[1,0] = 10
    a.update(0, 0, 2, 1, False, True)
    assert a.q_table[0,0] == 7


def test_truncation_no_bootstrap_when_configured():
    cfg = QLearningConfig(learning_rate=1, bootstrap_on_truncation=False)
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8), (2,2), cfg)
    a.q_table[1,0] = 10
    a.update(0, 0, 2, 1, False, True)
    assert a.q_table[0,0] == 2


def test_map_and_goal_identity():
    grid = np.zeros((3,3),dtype=np.int8)
    a = QLearningAgent(grid,(2,2),QLearningConfig())
    obs = {'agent': np.array([0,0]), 'goal': np.array([2,2]), 'obstacles': grid}
    assert a.state(obs) == 0
    changed = {**obs, 'obstacles': grid.copy()}
    changed['obstacles'][1,1] = 1
    with pytest.raises(ValueError, match='mismatch'):
        a.state(changed)
    with pytest.raises(ValueError, match='mismatch'):
        a.state({**obs, 'goal': np.array([2,1])})


def test_epsilon_and_decay():
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8),(2,2),QLearningConfig(epsilon_start=.5,epsilon_min=.2,epsilon_decay=.5))
    a.decay_epsilon()
    assert a.epsilon == .25
    a.decay_epsilon()
    assert a.epsilon == .2


def test_greedy_action():
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8),(2,2),QLearningConfig())
    a.q_table[0] = [0,2,5,1]
    assert a.select_action(0,explore=False) == 2


def test_save_load(tmp_path):
    a = QLearningAgent(np.zeros((3,3),dtype=np.int8),(2,2),QLearningConfig())
    a.q_table[0,2] = 7.5
    path = tmp_path / 'agent.npz'
    a.save(path)
    loaded = QLearningAgent.load(path)
    np.testing.assert_array_equal(loaded.q_table,a.q_table)
    assert loaded.signature == a.signature
    assert loaded.epsilon == a.epsilon


def test_fixed_map_seeded_starts():
    env = FixedMapNavigationEnv(np.zeros((4,4),dtype=np.int8),(3,3))
    try:
        x,_ = env.reset(seed=22)
        y,_ = env.reset(seed=22)
        np.testing.assert_array_equal(x['agent'], y['agent'])
        assert tuple(x['agent']) != (3,3)
        assert env.observation_space.contains(x)
    finally:
        env.close()


def test_fixed_map_preserves_phase1_api():
    env = FixedMapNavigationEnv(np.zeros((4,4),dtype=np.int8),(3,3),max_steps=2)
    try:
        obs,_ = env.reset(options={'start': (0,0)})
        assert tuple(obs['agent']) == (0,0)
        _,_,term,trunc,_ = env.step(0)
        assert not term and not trunc
        _,_,term,trunc,_ = env.step(0)
        assert not term and trunc
    finally:
        env.close()


def test_short_train_and_evaluation(tmp_path):
    cfg = load_config('configs/qlearning.yaml')
    cfg['training']['episodes'] = 12
    cfg['evaluation']['episodes'] = 5
    cfg['evaluation']['route_plot'] = str(tmp_path / 'route.png')
    agent, metrics = train(cfg)
    assert len(metrics) == 12
    assert set(metrics[0]) == {'episode','reward','steps','collisions','success','epsilon'}
    report = evaluate(cfg,agent)
    assert report['episodes'] == 5
    assert 0 <= report['q_learning']['success_rate'] <= 1
    assert (tmp_path / 'route.png').exists()
