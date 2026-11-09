
"""Tests for CNN Double DQN validation and checkpoint loading."""

import numpy as np
import pytest
import torch

from agents.dqn_agent import DQNAgent, DQNConfig
from evaluation.validate_dqn import validate_agent
from models.cnn_dqn_network import CNNDQNNetwork


ENV_CONFIG = {
    "height": 10,
    "width": 10,
    "obstacle_density": 0.18,
    "max_steps": 15,
}


def make_agent(architecture="cnn"):
    return DQNAgent(
        height=10,
        width=10,
        config=DQNConfig(
            architecture=architecture,
            double_dqn=True,
        ),
        seed=2027,
        device="cpu",
    )


def test_cnn_output_shape():
    model = CNNDQNNetwork(10, 10)
    observations = torch.zeros(3, 3, 10, 10)
    output = model(observations)

    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()


def test_validation_metrics():
    agent = make_agent()
    results = validate_agent(
        agent,
        ENV_CONFIG,
        seeds=[50211, 50212],
    )

    assert results["episodes"] == 2
    assert 0 <= results["success_rate"] <= 1
    assert np.isfinite(results["mean_reward"])
    assert 0 < results["mean_steps"] <= 15
    assert results["mean_collisions"] >= 0


def test_validation_rejects_empty_seeds():
    agent = make_agent()

    with pytest.raises(ValueError):
        validate_agent(agent, ENV_CONFIG, seeds=[])


def test_cnn_checkpoint_roundtrip(tmp_path):
    agent = make_agent()
    checkpoint = tmp_path / "cnn_test.pt"

    agent.save(str(checkpoint))
    loaded = DQNAgent.load(str(checkpoint), device="cpu")

    assert loaded.config.architecture == "cnn"
    assert loaded.config.double_dqn is True

    observations = torch.zeros(1, 3, 10, 10)

    with torch.no_grad():
        original = agent.online(observations)
        restored = loaded.online(observations)

    torch.testing.assert_close(original, restored)


def test_mlp_backward_compatibility(tmp_path):
    agent = make_agent(architecture="mlp")
    checkpoint = tmp_path / "mlp_test.pt"

    agent.save(str(checkpoint))
    loaded = DQNAgent.load(str(checkpoint), device="cpu")

    assert loaded.config.architecture == "mlp"
