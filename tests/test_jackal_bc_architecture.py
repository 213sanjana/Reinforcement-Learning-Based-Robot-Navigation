"""Phase 6F.1 Jackal behavior-cloning architecture and data pipeline."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from jackal_rl_navigation.action_mapper import ActionMapper
from jackal_rl_navigation.observation_builder import ObservationBuilder
from navigation.expert_demonstration_recorder import ExpertDemonstrationRecorder
from navigation.expert_demonstration_schema import (
    DemonstrationEpisode,
    jackal_policy_spaces,
)
from navigation.jackal_bc_data import (
    JackalBCDataLoader,
    JackalDemonstrationCorpus,
    ObservationNormalizer,
    assert_no_split_leakage,
    load_dataset,
    make_behavior_cloning_loaders,
)
from navigation.jackal_bc_inference import load_config, run_initialized_inference
from navigation.jackal_bc_policy import (
    ARCHITECTURE,
    JackalBCPolicy,
    LidarEncoder,
    RobotStateEncoder,
    build_policy,
    initialize_deterministically,
    load_checkpoint,
    predict_numpy,
    resolve_device,
    save_checkpoint,
)
from navigation.jackal_bc_schema import (
    describe_contract,
    require_env_actions,
    split_observation,
    validate_recorded_schema,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = ROOT / "artifacts" / "expert_dataset" / "v6e.2.0"
CONFIG_PATH = ROOT / "configs" / "jackal_behavior_cloning_6f1.yaml"
GAZEBO_ENV = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/gazebo_env.py"
)

TRAIN_ROWS = (
    (0, "train_0000", 4, 0.0, 0.2, True),
    (1, "train_0001", 2, 1.0, 0.4, True),
    (2, "train_0002", 0, 0.0, 0.0, False),
    (3, "train_0003", 3, 0.5, -0.1, False),
)
VALIDATION_ROWS = (
    (0, "validation_0000", 3, 10.0, 0.1, True),
    (1, "validation_0001", 5, -0.25, -0.3, True),
)


def make_episode(episode_id, scenario_id, steps, fill, action, success):
    goal = np.array([1.0, -1.0], dtype=np.float64)
    if steps == 0:
        zeros = {
            "observations": np.zeros((0, 40), dtype=np.float32),
            "actions": np.zeros((0, 2), dtype=np.float32),
            "requested_actions": np.zeros((0, 2), dtype=np.float32),
            "applied_velocities": np.zeros((0, 2), dtype=np.float32),
            "poses": np.zeros((0, 3), dtype=np.float64),
            "simulation_timestamps_ns": np.zeros((0,), dtype=np.int64),
        }
        return DemonstrationEpisode(
            episode_id=episode_id,
            scenario_id=scenario_id,
            termination_reason="PLANNING_FAILED",
            goal=goal,
            **zeros,
        )

    observations = np.full((steps, 40), 0.25, dtype=np.float32)
    observations[:, 0] = np.float32(fill)
    actions = np.zeros((steps, 2), dtype=np.float32)
    actions[:, 0] = np.float32(action)
    actions[:, 1] = np.float32(-0.2)
    velocities = np.zeros((steps, 2), dtype=np.float32)
    velocities[:, 0] = np.float32(0.05)
    poses = np.zeros((steps, 3), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, 0.1, steps)
    timestamps = np.arange(steps, dtype=np.int64) * 100_000_000 + 1_000
    return DemonstrationEpisode(
        episode_id=episode_id,
        scenario_id=scenario_id,
        termination_reason="GOAL_REACHED" if success else "SAFETY_STOP",
        goal=goal,
        observations=observations,
        actions=actions,
        requested_actions=np.array(actions, copy=True),
        applied_velocities=velocities,
        poses=poses,
        simulation_timestamps_ns=timestamps,
    )


def record_split(directory, rows):
    observation_space, action_space = jackal_policy_spaces()
    with ExpertDemonstrationRecorder(
        directory,
        observation_space,
        action_space,
    ) as recorder:
        for row in rows:
            recorder.record(make_episode(*row))


def write_splits(root):
    record_split(root / "train", TRAIN_ROWS)
    record_split(root / "validation", VALIDATION_ROWS)
    return root


def fit_unit_normalizer(seed=0, rows=8):
    values = np.random.default_rng(seed).normal(size=(rows, 40))
    normalizer = ObservationNormalizer.from_observations(
        values,
        split="train",
        epsilon=1e-6,
        scenario_ids=["train_unit"],
        success_only=False,
        excluded_empty=0,
        excluded_failures=0,
    )
    return normalizer, values.astype(np.float32)


def ready_policy(seed=5):
    normalizer, values = fit_unit_normalizer()
    policy = JackalBCPolicy(seed=seed)
    policy.install_normalizer(normalizer)
    return policy, normalizer, values


def synthetic_config(root, output):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["dataset"]["root"] = str(root)
    config["normalization"]["output"] = str(output / "observation_stats.json")
    config["checkpoint"]["output"] = str(output / "initialized_policy.pt")
    config["inference"]["output"] = str(output / "example_inference.json")
    config["data"]["batch_size"] = 2
    config["runtime"]["device"] = "cpu"
    return config


def test_observation_and_action_contract_matches_the_environment():
    contract = describe_contract()
    source = GAZEBO_ENV.read_text(encoding="utf-8")
    builder = ObservationBuilder()
    result = builder.build(
        lidar_ranges=np.full(72, 6.0, dtype=np.float32),
        goal_distance=4.0,
        goal_heading=math.pi / 2,
        linear_velocity=0.15,
        angular_velocity=-0.3,
    )

    assert result.valid
    assert result.vector.shape == (40,)
    assert np.allclose(result.vector[:36], 0.5)
    assert np.isclose(result.vector[36], 0.2)
    assert np.isclose(result.vector[37], 0.5)
    assert np.isclose(result.vector[38], 0.5)
    assert np.isclose(result.vector[39], -0.5)
    lidar, state = split_observation(result.vector)
    assert lidar.shape == (36,)
    assert state.shape == (4,)
    assert np.allclose(state, result.vector[36:])

    assert contract["lidar_slice"] == [0, 36]
    assert contract["robot_state_slice"] == [36, 40]
    assert contract["robot_state_names"] == [
        "goal_distance",
        "goal_heading",
        "linear_velocity",
        "angular_velocity",
    ]
    assert contract["action_names"] == ["linear", "angular"]
    assert contract["action_low"] == [-1.0, -1.0]
    assert contract["action_high"] == [1.0, 1.0]
    assert contract["training_action_key"] == "actions"
    mapper = ActionMapper()
    assert contract["velocity_limits"]["max_forward_m_s"] == mapper.max_forward
    assert contract["velocity_limits"]["max_reverse_m_s"] == mapper.max_reverse
    assert contract["velocity_limits"]["max_angular_rad_s"] == mapper.max_angular
    assert "shape=(40,)" in source
    assert "shape=(2,)" in source
    assert "low=-1.0" in source
    assert "high=1.0" in source

    recorded = json.loads((REAL_ROOT / "train" / "schema.json").read_text())
    validate_recorded_schema(recorded)
    assert recorded["robot_configuration"]["action_mapper"]["max_forward_m_s"] == 0.3
    with pytest.raises(ValueError):
        require_env_actions(np.array([1.1, 0.0], dtype=np.float32))


def test_forward_dimensions_separate_encoders_and_env_actions():
    policy, _normalizer, _values = ready_policy()
    assert isinstance(policy.lidar_encoder, LidarEncoder)
    assert isinstance(policy.state_encoder, RobotStateEncoder)
    assert policy.lidar_encoder is not policy.state_encoder
    assert "obs_mean" not in dict(policy.named_parameters())
    assert "obs_mean" in dict(policy.named_buffers())
    assert sum(parameter.numel() for parameter in policy.parameters()) == 98498

    batch = policy(torch.zeros(4, 40))
    sequence = policy(torch.zeros(2, 5, 40))
    assert batch.shape == (4, 2)
    assert sequence.shape == (2, 5, 2)
    assert batch.dtype == torch.float32
    assert bool(torch.all((batch > -1.0) & (batch < 1.0)))
    assert bool(torch.isfinite(sequence).all())
    steps = torch.randn(5, 40)
    torch.testing.assert_close(policy(steps.unsqueeze(0))[0], policy(steps))

    captured = {}

    def _grab_lidar(_module, inputs, _output):
        captured["lidar"] = inputs[0].detach()

    def _grab_state(_module, inputs, _output):
        captured["state"] = inputs[0].detach()

    policy.lidar_encoder.register_forward_hook(_grab_lidar)
    policy.state_encoder.register_forward_hook(_grab_state)
    raw = torch.randn(3, 40)
    policy(raw)
    manual = (raw - policy.obs_mean) / policy.obs_std
    torch.testing.assert_close(captured["lidar"], manual[:, :36])
    torch.testing.assert_close(captured["state"], manual[:, 36:])

    action = predict_numpy(policy, raw[0].numpy())
    _observation_space, action_space = jackal_policy_spaces()
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert action_space.contains(action)
    with pytest.raises(ValueError, match="40"):
        policy(torch.zeros(4, 39))
    with pytest.raises(RuntimeError, match="not installed"):
        JackalBCPolicy(seed=1)(torch.zeros(2, 40))


def test_initialization_is_deterministic_and_independent_of_global_rng():
    torch.manual_seed(1)
    first = JackalBCPolicy(seed=9)
    torch.randn(8)
    second = JackalBCPolicy(seed=9)
    third = JackalBCPolicy(seed=10)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
    assert not torch.equal(
        next(iter(first.parameters())),
        next(iter(third.parameters())),
    )
    initialize_deterministically(first, 9)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)

    config = load_config(CONFIG_PATH)
    built = build_policy(config["model"], seed=config["runtime"]["seed"])
    again = build_policy(config["model"], seed=config["runtime"]["seed"])
    for left, right in zip(built.parameters(), again.parameters()):
        assert torch.equal(left, right)


def test_gradients_reach_both_encoders_and_skip_padding():
    policy, _normalizer, values = ready_policy(seed=11)
    observations = torch.from_numpy(values[:8])
    loss = torch.nn.functional.mse_loss(policy(observations), torch.full((8, 2), 0.3))
    loss.backward()

    lidar_grad = policy.lidar_encoder.projection.weight.grad
    state_grad = policy.state_encoder.network[0].weight.grad
    fusion_grad = policy.fusion[-1].weight.grad
    assert lidar_grad is not None and torch.isfinite(lidar_grad).all()
    assert state_grad is not None and torch.isfinite(state_grad).all()
    assert fusion_grad is not None and torch.isfinite(fusion_grad).all()
    assert lidar_grad.abs().sum() > 0
    assert state_grad.abs().sum() > 0
    assert fusion_grad.abs().sum() > 0
    assert policy.obs_mean.requires_grad is False
    assert policy.obs_mean.grad is None
    for parameter in policy.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    padded = torch.randn(2, 4, 40, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    policy.forward_valid(padded, mask).sum().backward()
    assert torch.count_nonzero(padded.grad[1, 2]) == 0
    assert torch.count_nonzero(padded.grad[1, 3]) == 0
    assert torch.count_nonzero(padded.grad[1, 0]) > 0
    assert torch.isfinite(padded.grad).all()


def test_checkpoint_roundtrip(tmp_path):
    from imitation.bc_network import ARCHITECTURE as CNN_ARCHITECTURE
    from imitation.bc_network import BCNetwork, load_bc_checkpoint
    from models.cnn_dqn_network import CNNDQNNetwork
    from models.dqn_network import DQNNetwork

    policy, normalizer, values = ready_policy(seed=4)
    observation = torch.from_numpy(values[:3])
    expected = policy(observation).detach().clone()
    path = tmp_path / "initialized_policy.pt"
    save_checkpoint(path, policy, normalizer)
    with torch.no_grad():
        policy.fusion[-1].weight.add_(1.0)

    loaded, payload = load_checkpoint(path, device="cpu")
    torch.testing.assert_close(loaded(observation), expected, rtol=0, atol=0)
    assert payload["architecture"] == ARCHITECTURE == "jackal_bc_v1"
    assert payload["trained"] is False
    assert payload["normalization"]["source_split"] == "train"
    assert loaded.normalizer_is_ready
    with pytest.raises(FileExistsError, match="overwrite"):
        save_checkpoint(path, policy, normalizer)

    foreign = tmp_path / "cnn.pt"
    torch.save({"format_version": 1, "architecture": "cnn_bc_v1", "phase": "3"}, foreign)
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(foreign)
    with pytest.raises(ValueError, match="Unsupported"):
        load_bc_checkpoint(path)

    assert CNN_ARCHITECTURE == "cnn_bc_v1"
    assert DQNNetwork(3, 3)(torch.zeros(1, 3, 3, 3)).shape == (1, 4)
    assert CNNDQNNetwork(4, 4)(torch.zeros(1, 3, 4, 4)).shape == (1, 4)
    assert BCNetwork(4, 4, 8)(torch.zeros(1, 3, 4, 4)).shape == (1, 4)


def test_normalizer_is_fit_on_train_rows_and_saved(tmp_path):
    values = np.full((6, 40), 0.25, dtype=np.float64)
    values[4:, 0] = 1.0
    values[:4, 0] = 0.0
    normalizer = ObservationNormalizer.from_observations(
        values,
        split="train",
        epsilon=1e-6,
        scenario_ids=["train_known"],
        success_only=False,
        excluded_empty=0,
        excluded_failures=0,
    )
    before = normalizer.mean.copy()
    transformed = normalizer.transform(values.astype(np.float32))
    assert normalizer.count == 6
    assert normalizer.mean[0] == pytest.approx(1.0 / 3.0)
    assert normalizer.std[0] == pytest.approx(math.sqrt(2.0 / 9.0))
    assert normalizer.std[1] == 1e-6
    assert "lidar_00" not in normalizer.floored_features
    assert "lidar_01" in normalizer.floored_features
    assert transformed[0, 0] == pytest.approx((0.0 - 1.0 / 3.0) / math.sqrt(2.0 / 9.0))
    assert transformed[0, 1] == pytest.approx(0.0)
    assert np.array_equal(before, normalizer.mean)

    path = tmp_path / "observation_stats.json"
    normalizer.save(path)
    loaded = ObservationNormalizer.load(path)
    assert np.array_equal(loaded.mean, normalizer.mean)
    assert np.array_equal(loaded.std, normalizer.std)
    assert np.array_equal(loaded.unclipped_std, normalizer.unclipped_std)
    assert loaded.source_split == "train"
    with pytest.raises(FileExistsError, match="overwrite"):
        normalizer.save(path)
    with pytest.raises(ValueError, match="train split only"):
        ObservationNormalizer.from_observations(
            values,
            split="validation",
            epsilon=1e-6,
            scenario_ids=["validation_0000"],
            success_only=False,
            excluded_empty=0,
            excluded_failures=0,
        )
    with pytest.raises(ValueError, match="train scenarios"):
        ObservationNormalizer.from_observations(
            values,
            split="train",
            epsilon=1e-6,
            scenario_ids=["validation_0000"],
            success_only=False,
            excluded_empty=0,
            excluded_failures=0,
        )
    with pytest.raises(ValueError, match="at least two"):
        ObservationNormalizer.from_observations(
            values[:1],
            split="train",
            epsilon=1e-6,
            scenario_ids=["train_known"],
            success_only=False,
            excluded_empty=0,
            excluded_failures=0,
        )


def test_loaders_keep_episode_boundaries_and_exclude_test(tmp_path, monkeypatch):
    root = write_splits(tmp_path)
    sentinel = root / "test" / "manifest.json"
    sentinel.parent.mkdir()
    sentinel.write_text("unread", encoding="utf-8")
    opened = []
    original = load_dataset

    def wrapped(path):
        opened.append(Path(path).name)
        return original(path)

    monkeypatch.setattr("navigation.jackal_bc_data.load_dataset", wrapped)
    loaders = make_behavior_cloning_loaders(
        root,
        batch_size=2,
        seed=7,
        success_only=False,
        num_workers=0,
        epsilon=1e-6,
    )

    assert opened == ["train", "validation"]
    assert sentinel.read_text(encoding="utf-8") == "unread"
    with pytest.raises(ValueError, match="test split"):
        JackalDemonstrationCorpus(root / "test", "test")
    with pytest.raises(ValueError, match="test split"):
        JackalBCDataLoader(
            type("Fake", (), {"split": "test"})(),
            batch_size=1,
            seed=0,
            shuffle=False,
            mode="transition",
        )

    train = loaders.train_corpus
    assert train.excluded_empty == 1
    assert train.excluded_failures == 0
    assert len(loaders.train.dataset) == 9
    assert loaders.normalizer.count == 9
    assert loaders.normalizer.source_split == "train"
    assert loaders.normalizer.mean[0] == pytest.approx(
        train.stacked_observations()[:, 0].mean()
    )
    combined = np.concatenate([
        train.stacked_observations(),
        loaders.validation_corpus.stacked_observations(),
    ])
    assert loaders.normalizer.mean[0] != pytest.approx(combined[:, 0].mean())
    assert set(train.scenario_ids).isdisjoint(loaders.validation_corpus.scenario_ids)
    assert 0 in {record["episode_id"] for record in train.records}
    assert 0 in {record["episode_id"] for record in loaders.validation_corpus.records}

    filtered = JackalDemonstrationCorpus(root / "train", "train", success_only=True)
    assert filtered.excluded_failures == 1
    assert len(filtered.transitions()) == 6

    before = loaders.normalizer.mean.copy()
    seen = []
    for batch in loaders.train:
        assert batch["observation"].shape[1:] == (40,)
        assert batch["action"].shape[1:] == (2,)
        assert torch.all(batch["timestep"] < batch["episode_length"])
        for row in range(batch["observation"].shape[0]):
            episode_index = int(batch["episode_index"][row])
            timestep = int(batch["timestep"][row])
            torch.testing.assert_close(
                batch["observation"][row],
                train.observations[episode_index][timestep],
            )
            seen.append((int(batch["episode_id"][row]), timestep))
    assert np.array_equal(before, loaders.normalizer.mean)
    assert len(seen) == 9

    repeated = []
    loader = JackalBCDataLoader(
        train.transitions(),
        batch_size=4,
        seed=7,
        shuffle=True,
        mode="transition",
    )
    for batch in loader:
        repeated.extend(zip(batch["episode_id"].tolist(), batch["timestep"].tolist()))
    assert repeated == seen

    dataset = train.transitions()
    original_row = train.observations[0][0].clone()
    dataset[0]["observation"].add_(1.0)
    torch.testing.assert_close(train.observations[0][0], original_row)

    episode_batch = next(iter(loaders.train_episodes))
    assert episode_batch["observations"].shape == (2, 4, 40)
    assert episode_batch["lengths"].tolist() == [4, 2]
    assert episode_batch["mask"].sum(dim=1).tolist() == [4, 2]
    assert torch.equal(episode_batch["actions"][0, :4, 0], torch.full((4,), 0.2))
    assert torch.equal(episode_batch["actions"][1, :2, 0], torch.full((2,), 0.4))
    assert torch.equal(episode_batch["actions"][1, 2:], torch.zeros(2, 2))

    garbage = episode_batch["observations"].clone()
    garbage[~episode_batch["mask"]] = 50.0
    policy, _normalizer, _values = ready_policy(seed=2)
    # The synthetic observations use the loader normalizer, not the unit one.
    policy.install_normalizer(loaders.normalizer)
    original_actions = policy.forward_valid(
        episode_batch["observations"],
        episode_batch["mask"],
    )
    replaced_actions = policy.forward_valid(garbage, episode_batch["mask"])
    torch.testing.assert_close(original_actions, replaced_actions)
    assert torch.equal(
        original_actions[~episode_batch["mask"]],
        torch.zeros_like(original_actions[~episode_batch["mask"]]),
    )

    with pytest.raises(IndexError, match="neighboring episode"):
        train.causal_window(1, 0, 2)
    window, window_actions = train.causal_window(1, 1, 2)
    torch.testing.assert_close(window, train.observations[1])
    torch.testing.assert_close(window_actions, train.actions[1])
    assert not torch.equal(window[0], train.observations[0][-1])


def test_shared_trajectory_is_leakage_and_episode_ids_may_restart(tmp_path):
    record_split(tmp_path / "train", ((0, "train_same", 3, 0.3, 0.2, True),))
    record_split(
        tmp_path / "validation",
        ((0, "validation_same", 3, 0.3, 0.2, True),),
    )
    train = JackalDemonstrationCorpus(tmp_path / "train", "train")
    validation = JackalDemonstrationCorpus(tmp_path / "validation", "validation")
    assert train.records[0]["episode_id"] == validation.records[0]["episode_id"]
    with pytest.raises(ValueError, match="trajectory leakage"):
        assert_no_split_leakage(train, validation)

    record_split(tmp_path / "foreign", ((0, "train_0000", 2, 0.2, 0.2, True),))
    with pytest.raises(ValueError, match="does not match split"):
        JackalDemonstrationCorpus(tmp_path / "foreign", "validation")


def test_example_inference_writes_new_artifacts_only(tmp_path):
    root = write_splits(tmp_path / "data")
    output = tmp_path / "out"
    report = run_initialized_inference(synthetic_config(root, output))
    action = np.asarray(report["action"], dtype=np.float32)
    _observation_space, action_space = jackal_policy_spaces()

    assert report["phase"] == "6f.1"
    assert report["architecture"] == "jackal_bc_v1"
    assert report["trained"] is False
    assert report["test_split_read"] is False
    assert report["normalizer_source_split"] == "train"
    assert report["normalizer_count"] == 9
    assert report["parameter_count"] == 98498
    assert report["transition_batch_shape"] == [2, 40]
    assert report["episode_batch_shape"][0] == 2
    assert report["episode_batch_valid_steps"] == 6
    assert report["episode_batch_padded_steps"] == 2
    assert action.shape == (2,)
    assert action_space.contains(action)
    assert Path(report["checkpoint"]).is_file()
    assert Path(report["statistics"]).is_file()
    assert Path(report["statistics"]).with_suffix(".npz").is_file()
    loaded, _payload = load_checkpoint(report["checkpoint"], device="cpu")
    assert loaded.normalizer_is_ready

    with pytest.raises(FileExistsError, match="overwrite"):
        run_initialized_inference(synthetic_config(root, output))
    rejected = synthetic_config(root, tmp_path / "unused")
    rejected["dataset"]["read_test"] = True
    with pytest.raises(ValueError, match="cannot read the test"):
        run_initialized_inference(rejected)
    assert not (tmp_path / "unused").exists()


def test_phase_does_not_define_a_training_loop():
    sources = [
        ROOT / "navigation" / name
        for name in (
            "jackal_bc_schema.py",
            "jackal_bc_policy.py",
            "jackal_bc_data.py",
            "jackal_bc_inference.py",
        )
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "torch.optim" not in text
        assert "loss.backward" not in text


def test_device_resolution_supports_cpu_and_cuda():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    policy, _normalizer, values = ready_policy()
    policy.to(torch.device("cpu"))
    action = policy(torch.from_numpy(values[:2]))
    assert action.shape == (2, 2)
    assert action.device.type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_and_backward():
    normalizer, values = fit_unit_normalizer()
    policy = JackalBCPolicy(seed=3).cuda()
    policy.install_normalizer(normalizer)
    observations = torch.from_numpy(values[:4]).cuda()
    loss = policy(observations).sum()
    loss.backward()
    assert torch.isfinite(loss).item()
    assert policy.obs_mean.device.type == "cuda"
    assert policy.lidar_encoder.projection.weight.grad.device.type == "cuda"


def test_real_expert_dataset_statistics_use_train_only(monkeypatch):
    if not (REAL_ROOT / "train" / "manifest.json").exists():
        pytest.skip("expert dataset is not present")

    opened = []
    original = load_dataset

    def wrapped(path):
        opened.append(Path(path).name)
        return original(path)

    monkeypatch.setattr("navigation.jackal_bc_data.load_dataset", wrapped)
    loaders = make_behavior_cloning_loaders(
        REAL_ROOT,
        batch_size=32,
        seed=642,
        success_only=False,
        num_workers=0,
        epsilon=1e-6,
    )
    manifest = json.loads((REAL_ROOT / "train" / "manifest.json").read_text())
    expected = sum(int(item["num_transitions"]) for item in manifest["episodes"])

    assert opened == ["train", "validation"]
    assert loaders.normalizer.count == expected == len(loaders.train.dataset)
    assert loaders.normalizer.count > len(loaders.validation.dataset)
    assert set(loaders.train_corpus.scenario_ids).isdisjoint(
        set(loaders.validation_corpus.scenario_ids)
    )
    assert loaders.normalizer.mean.shape == (40,)
    assert np.all(loaders.normalizer.std > 0)
    sample = next(iter(loaders.validation))
    assert sample["observation"].shape[1:] == (40,)
    assert sample["action"].shape[1:] == (2,)


def test_config_locks_the_phase_contract():
    config = load_config(CONFIG_PATH)
    assert config["phase"] == "6f.1"
    assert config["model"]["architecture"] == ARCHITECTURE
    assert config["dataset"]["read_test"] is False
    assert config["data"]["context_length"] == 1
    assert config["runtime"]["deterministic"] is True
    assert config["model"]["lidar_bins"] == 36
    assert config["model"]["action_dim"] == 2
    broken = copy.deepcopy(config)
    broken["model"]["lidar_bins"] = 8
    with pytest.raises(ValueError, match="36 LiDAR"):
        build_policy(broken["model"], seed=1)
