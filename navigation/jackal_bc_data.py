"""Train and validation data for Jackal behavior cloning.

Each stored episode keeps its own length. Transition samples point at one
episode and one timestep. Episode batches are padded inside the batch, and
the padding mask is the only signal that a step is real. Windows that would
cross into another episode are rejected.

Observation mean and standard deviation are estimated from the train split.
Validation uses that frozen transform. The test split is not opened.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from navigation.expert_demonstration_recorder import load_dataset
from navigation.jackal_bc_schema import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PHASE,
    TRAINING_ACTION_KEY,
    create_new_file,
    feature_names,
    validate_recorded_schema,
)


class ObservationNormalizer:
    """Frozen per-feature z-score estimated on train observations."""

    def __init__(
        self,
        mean,
        std,
        unclipped_std,
        count,
        epsilon,
        scenario_ids,
        observation_sha256,
        success_only,
        excluded_empty,
        excluded_failures,
    ):
        self.mean = np.array(mean, dtype=np.float64, copy=True)
        self.std = np.array(std, dtype=np.float64, copy=True)
        self.unclipped_std = np.array(unclipped_std, dtype=np.float64, copy=True)
        self.count = int(count)
        self.epsilon = float(epsilon)
        self.source_split = "train"
        self.feature_names = feature_names()
        self.scenario_ids = tuple(scenario_ids)
        self.observation_sha256 = str(observation_sha256)
        self.success_only = bool(success_only)
        self.excluded_empty = int(excluded_empty)
        self.excluded_failures = int(excluded_failures)
        self.floored_features = tuple(
            name
            for name, scale in zip(self.feature_names, self.unclipped_std)
            if float(scale) < self.epsilon
        )
        self._validate()

    @classmethod
    def from_observations(
        cls,
        observations,
        *,
        split,
        epsilon,
        scenario_ids,
        success_only,
        excluded_empty,
        excluded_failures,
    ):
        """Fit population statistics. ``split`` must be the train split."""

        if split != "train":
            raise ValueError(
                "Observation normalization statistics are fit on the train split only"
            )
        identifiers = _train_scenario_ids(scenario_ids)
        values = np.array(observations, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[1] != OBSERVATION_DIM:
            raise ValueError("Training observations must be shaped (N, 40)")
        if values.shape[0] < 2:
            raise ValueError("Need at least two training observations")
        if not np.all(np.isfinite(values)):
            raise ValueError("Training observations must be finite")

        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1e-2:
            raise ValueError("Normalization epsilon must be in (0, 1e-2]")

        mean = values.mean(axis=0)
        unclipped = np.sqrt(np.mean((values - mean) ** 2, axis=0))
        std = np.maximum(unclipped, epsilon)
        return cls(
            mean=mean,
            std=std,
            unclipped_std=unclipped,
            count=int(values.shape[0]),
            epsilon=epsilon,
            scenario_ids=identifiers,
            observation_sha256=array_sha256(values.astype(np.float32)),
            success_only=success_only,
            excluded_empty=excluded_empty,
            excluded_failures=excluded_failures,
        )

    def transform(self, observations):
        """Apply the frozen train statistics without updating them."""

        values = np.asarray(observations, dtype=np.float32)
        mean = self.mean.astype(np.float32)
        std = self.std.astype(np.float32)
        return ((values - mean) / std).astype(np.float32, copy=False)

    def scenario_fingerprint(self):
        hasher = hashlib.sha256()
        for scenario_id in sorted(self.scenario_ids):
            hasher.update(scenario_id.encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()

    def save(self, json_path):
        """Write JSON metadata and an exact float64 NPZ beside it."""

        json_path = Path(json_path)
        npz_path = json_path.with_suffix(".npz")
        if json_path.exists() or npz_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite normalization statistics at {json_path}"
            )
        payload = self.as_dict()
        arrays = {
            "mean": self.mean,
            "std": self.std,
            "unclipped_std": self.unclipped_std,
        }
        written = []
        try:
            create_new_file(npz_path, lambda path: _write_npz(path, arrays))
            written.append(npz_path)
            create_new_file(json_path, lambda path: _write_json(path, payload))
            written.append(json_path)
        except Exception:
            if json_path not in written:
                for path in written:
                    path.unlink(missing_ok=True)
            raise
        return json_path

    @classmethod
    def load(cls, json_path):
        json_path = Path(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("phase") != PHASE or payload.get("source_split") != "train":
            raise ValueError("Normalization file is not a Phase 6F.1 train fit")
        if payload.get("method") != "per_feature_population_zscore":
            raise ValueError("Unsupported normalization method")
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            with np.load(npz_path) as data:
                mean = np.asarray(data["mean"], dtype=np.float64)
                std = np.asarray(data["std"], dtype=np.float64)
                unclipped = np.asarray(data["unclipped_std"], dtype=np.float64)
        else:
            mean = np.asarray(payload["mean"], dtype=np.float64)
            std = np.asarray(payload["std"], dtype=np.float64)
            unclipped = np.asarray(payload["unclipped_std"], dtype=np.float64)
        return cls(
            mean=mean,
            std=std,
            unclipped_std=unclipped,
            count=payload["count"],
            epsilon=payload["epsilon"],
            scenario_ids=payload["scenario_ids"],
            observation_sha256=payload["observation_sha256"],
            success_only=payload["success_only"],
            excluded_empty=payload["excluded_empty"],
            excluded_failures=payload["excluded_failures"],
        )

    def as_dict(self):
        return {
            "phase": PHASE,
            "method": "per_feature_population_zscore",
            "source_split": self.source_split,
            "epsilon": self.epsilon,
            "count": self.count,
            "observation_dim": OBSERVATION_DIM,
            "feature_names": list(self.feature_names),
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "unclipped_std": [float(value) for value in self.unclipped_std],
            "floored_features": list(self.floored_features),
            "scenario_ids": list(self.scenario_ids),
            "scenario_fingerprint": self.scenario_fingerprint(),
            "observation_sha256": self.observation_sha256,
            "success_only": self.success_only,
            "excluded_empty": self.excluded_empty,
            "excluded_failures": self.excluded_failures,
            "layout": {
                "lidar": [0, 36],
                "robot_state": [36, 40],
            },
        }

    def _validate(self):
        if self.mean.shape != (OBSERVATION_DIM,) or self.std.shape != (OBSERVATION_DIM,):
            raise ValueError("Normalizer statistics must have 40 features")
        if self.unclipped_std.shape != (OBSERVATION_DIM,):
            raise ValueError("Unclipped standard deviations must have 40 features")
        if self.count < 2:
            raise ValueError("Normalizer does not contain training observations")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise ValueError("Normalizer statistics must be finite")
        if np.any(self.std <= 0):
            raise ValueError("Normalizer standard deviations must be positive")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Normalizer scenario ids must be unique")
        for scenario_id in self.scenario_ids:
            if not isinstance(scenario_id, str) or not scenario_id.startswith("train_"):
                raise ValueError("Normalizer scenario ids must belong to the train split")
            if scenario_id == "train_":
                raise ValueError("Normalizer scenario ids must belong to the train split")


class JackalDemonstrationCorpus:
    """One expert split, with episode boundaries retained."""

    def __init__(self, dataset_dir, split, success_only=False):
        if split == "test":
            raise ValueError(
                "The test split is excluded from the behavior-cloning data pipeline"
            )
        if split not in ("train", "validation"):
            raise ValueError("Behavior cloning accepts the train and validation splits")
        if not isinstance(success_only, bool):
            raise TypeError("success_only must be a boolean")

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.success_only = success_only
        if self.dataset_dir.name != split:
            raise ValueError(
                f"Dataset directory {self.dataset_dir.name} does not match split {split}"
            )

        schema_path = self.dataset_dir / "schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validate_recorded_schema(schema)
        if schema.get("training_action_key") != TRAINING_ACTION_KEY:
            raise ValueError("Training action key must be 'actions'")

        loaded = load_dataset(self.dataset_dir)
        self.records = []
        self.included_records = []
        self.observations = []
        self.actions = []
        self.transition_index = []
        self.scenario_ids = []
        self.integrity_sha256 = []
        self.trajectory_digests = []
        self.excluded_empty = 0
        self.excluded_failures = 0
        seen_ids = set()
        seen_digests = set()

        for episode in loaded:
            scenario_id = str(episode["scenario_id"])
            prefix = f"{split}_"
            if not scenario_id.startswith(prefix) or scenario_id == prefix:
                raise ValueError(f"{scenario_id} does not belong to the {split} split")
            if scenario_id in seen_ids:
                raise ValueError(f"Duplicate scenario id {scenario_id}")
            seen_ids.add(scenario_id)

            observations = np.asarray(episode["observations"], dtype=np.float32)
            actions = np.asarray(episode["actions"], dtype=np.float32)
            length = int(observations.shape[0])
            if actions.shape[0] != length or actions.shape[1:] != (ACTION_DIM,):
                raise ValueError(f"{scenario_id} observations and actions are misaligned")
            if length and observations.shape[1:] != (OBSERVATION_DIM,):
                raise ValueError(f"{scenario_id} observations are not 40-dimensional")
            if length and not np.all(np.isfinite(observations)):
                raise ValueError(f"{scenario_id} contains nonfinite observations")
            if length and not np.all(np.isfinite(actions)):
                raise ValueError(f"{scenario_id} contains nonfinite actions")
            if length and np.any((actions < -1.0) | (actions > 1.0)):
                raise ValueError(f"{scenario_id} actions leave [-1, 1]")

            digest = None
            if length:
                digest = trajectory_digest(observations, actions)
                if digest in seen_digests:
                    raise ValueError(f"Duplicate trajectory in {scenario_id}")
                seen_digests.add(digest)
                self.trajectory_digests.append(digest)

            integrity = str(episode["integrity_sha256"])
            if integrity in self.integrity_sha256:
                raise ValueError(f"Duplicate integrity hash in {scenario_id}")
            self.scenario_ids.append(scenario_id)
            self.integrity_sha256.append(integrity)
            self.records.append({
                "episode_id": int(episode["episode_id"]),
                "scenario_id": scenario_id,
                "success": bool(episode["success"]),
                "termination_reason": str(episode["termination_reason"]),
                "integrity_sha256": integrity,
                "trajectory_digest": digest,
                "path": str((
                    self.dataset_dir / f"episode_{int(episode['episode_id']):04d}.npz"
                ).resolve()),
                "length": length,
            })

            if length == 0:
                self.excluded_empty += 1
                continue
            if success_only and not bool(episode["success"]):
                self.excluded_failures += 1
                continue

            position = len(self.observations)
            self.observations.append(torch.from_numpy(np.ascontiguousarray(observations)))
            self.actions.append(torch.from_numpy(np.ascontiguousarray(actions)))
            self.records[-1]["included_index"] = position
            self.included_records.append(self.records[-1])
            for timestep in range(length):
                self.transition_index.append((position, timestep))

        if not self.transition_index:
            raise ValueError(f"{split} split has no transitions")
        self._stacked = np.concatenate(
            [item.numpy() for item in self.observations],
            axis=0,
        )
        self.observation_sha256 = array_sha256(self._stacked)

    def stacked_observations(self):
        return np.array(self._stacked, dtype=np.float32, copy=True)

    def included_scenario_ids(self):
        return tuple(
            record["scenario_id"]
            for record in self.records
            if "included_index" in record
        )

    def transitions(self):
        return JackalTransitionDataset(self)

    def episodes(self):
        return JackalEpisodeDataset(self)

    def causal_window(self, episode_index, timestep, length):
        """Return a window that stays inside one episode."""

        length = int(length)
        timestep = int(timestep)
        if length < 1:
            raise ValueError("Window length must be positive")
        observations = self.observations[episode_index]
        actions = self.actions[episode_index]
        start = timestep - length + 1
        if start < 0 or timestep >= int(observations.shape[0]):
            raise IndexError(
                "Window is outside this episode and cannot use a neighboring episode"
            )
        return (
            observations[start:timestep + 1].clone(),
            actions[start:timestep + 1].clone(),
        )


class JackalTransitionDataset(Dataset):
    """One item is one expert transition."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.split = corpus.split

    def __len__(self):
        return len(self.corpus.transition_index)

    def __getitem__(self, index):
        episode_index, timestep = self.corpus.transition_index[int(index)]
        record = self.corpus.included_records[episode_index]
        return {
            "observation": self.corpus.observations[episode_index][timestep].clone(),
            "action": self.corpus.actions[episode_index][timestep].clone(),
            "episode_index": episode_index,
            "episode_id": record["episode_id"],
            "scenario_id": record["scenario_id"],
            "timestep": timestep,
            "episode_length": record["length"],
            "success": record["success"],
            "termination_reason": record["termination_reason"],
        }


class JackalEpisodeDataset(Dataset):
    """One item is one variable-length episode."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.split = corpus.split

    def __len__(self):
        return len(self.corpus.observations)

    def __getitem__(self, index):
        index = int(index)
        record = self.corpus.included_records[index]
        return {
            "observations": self.corpus.observations[index].clone(),
            "actions": self.corpus.actions[index].clone(),
            "episode_index": index,
            "episode_id": record["episode_id"],
            "scenario_id": record["scenario_id"],
            "length": record["length"],
            "success": record["success"],
            "termination_reason": record["termination_reason"],
        }


class JackalBCDataLoader(DataLoader):
    """Loader for train or validation transitions, or padded episodes."""

    def __init__(
        self,
        dataset,
        *,
        batch_size,
        seed,
        shuffle,
        num_workers=0,
        mode,
    ):
        split = getattr(dataset, "split", None)
        if split == "test":
            raise ValueError(
                "The test split is excluded from behavior-cloning loaders"
            )
        if split not in ("train", "validation"):
            raise ValueError("Behavior cloning accepts the train and validation splits")
        if mode == "transition" and not isinstance(dataset, JackalTransitionDataset):
            raise TypeError("Transition mode requires JackalTransitionDataset")
        if mode == "episode" and not isinstance(dataset, JackalEpisodeDataset):
            raise TypeError("Episode mode requires JackalEpisodeDataset")
        if mode not in ("transition", "episode"):
            raise ValueError("mode must be transition or episode")
        if len(dataset) < 1:
            raise ValueError("Dataset has no samples")

        batch_size = _positive_int(batch_size, "batch_size")
        seed = _nonnegative_int(seed, "seed")
        num_workers = _nonnegative_int(num_workers, "num_workers")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")

        generator = torch.Generator()
        generator.manual_seed(seed)
        collate = collate_episodes if mode == "episode" else collate_transitions
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
            collate_fn=collate,
            worker_init_fn=_seed_worker,
            drop_last=False,
        )
        self.split = split
        self.seed = seed
        self.mode = mode


class BehaviorCloningLoaders:
    """Train and validation loaders sharing one frozen normalizer."""

    def __init__(
        self,
        train_corpus,
        validation_corpus,
        normalizer,
        train,
        validation,
        train_episodes,
    ):
        self.train_corpus = train_corpus
        self.validation_corpus = validation_corpus
        self.normalizer = normalizer
        self.train = train
        self.validation = validation
        self.train_episodes = train_episodes


def make_behavior_cloning_loaders(
    dataset_root,
    *,
    batch_size,
    seed,
    success_only,
    num_workers,
    epsilon,
):
    """Load train and validation. The test directory is not referenced."""

    root = Path(dataset_root)
    train = JackalDemonstrationCorpus(root / "train", "train", success_only)
    validation = JackalDemonstrationCorpus(
        root / "validation",
        "validation",
        success_only,
    )
    assert_no_split_leakage(train, validation)
    normalizer = ObservationNormalizer.from_observations(
        train.stacked_observations(),
        split=train.split,
        epsilon=epsilon,
        scenario_ids=train.included_scenario_ids(),
        success_only=success_only,
        excluded_empty=train.excluded_empty,
        excluded_failures=train.excluded_failures,
    )
    if normalizer.observation_sha256 != train.observation_sha256:
        raise RuntimeError("Normalizer observations do not match the train split")
    if set(normalizer.scenario_ids) != set(train.included_scenario_ids()):
        raise RuntimeError("Normalizer scenarios do not match the train split")
    if set(normalizer.scenario_ids) & set(validation.scenario_ids):
        raise ValueError("Train normalization scenarios overlap validation")

    return BehaviorCloningLoaders(
        train_corpus=train,
        validation_corpus=validation,
        normalizer=normalizer,
        train=JackalBCDataLoader(
            train.transitions(),
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
            mode="transition",
        ),
        validation=JackalBCDataLoader(
            validation.transitions(),
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            num_workers=num_workers,
            mode="transition",
        ),
        train_episodes=JackalBCDataLoader(
            train.episodes(),
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            num_workers=num_workers,
            mode="episode",
        ),
    )


def assert_no_split_leakage(train, validation):
    """Raise when train and validation share scenarios or trajectories.

    Episode ids restart at zero inside each split, so they are not identities.
    """

    problems = []
    shared_scenarios = set(train.scenario_ids) & set(validation.scenario_ids)
    if shared_scenarios:
        problems.append("scenario leakage: " + ", ".join(sorted(shared_scenarios)))
    shared_hashes = set(train.integrity_sha256) & set(validation.integrity_sha256)
    if shared_hashes:
        problems.append("integrity leakage")
    shared_paths = {record["path"] for record in train.records} & {
        record["path"] for record in validation.records
    }
    if shared_paths:
        problems.append("episode path leakage")
    shared_trajectories = set(train.trajectory_digests) & set(validation.trajectory_digests)
    if shared_trajectories:
        problems.append("trajectory leakage")
    if problems:
        raise ValueError("; ".join(problems))


def collate_transitions(batch):
    """Stack independent transitions. Time is not concatenated."""

    if not batch:
        raise ValueError("Cannot collate an empty transition batch")
    return {
        "observation": torch.stack([item["observation"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "episode_index": _long([item["episode_index"] for item in batch]),
        "episode_id": _long([item["episode_id"] for item in batch]),
        "scenario_id": [item["scenario_id"] for item in batch],
        "timestep": _long([item["timestep"] for item in batch]),
        "episode_length": _long([item["episode_length"] for item in batch]),
        "success": torch.tensor([item["success"] for item in batch], dtype=torch.bool),
        "termination_reason": [item["termination_reason"] for item in batch],
    }


def collate_episodes(batch):
    """Pad episodes in this batch only. Padding is not another episode."""

    if not batch:
        raise ValueError("Cannot collate an empty episode batch")
    lengths = [int(item["length"]) for item in batch]
    width = int(batch[0]["observations"].shape[-1])
    action_dim = int(batch[0]["actions"].shape[-1])
    limit = max(lengths)
    observations = torch.zeros(len(batch), limit, width, dtype=torch.float32)
    actions = torch.zeros(len(batch), limit, action_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), limit, dtype=torch.bool)
    for index, item in enumerate(batch):
        length = lengths[index]
        observations[index, :length] = item["observations"]
        actions[index, :length] = item["actions"]
        mask[index, :length] = True
    return {
        "observations": observations,
        "actions": actions,
        "mask": mask,
        "lengths": _long(lengths),
        "episode_index": _long([item["episode_index"] for item in batch]),
        "episode_id": _long([item["episode_id"] for item in batch]),
        "scenario_id": [item["scenario_id"] for item in batch],
        "success": torch.tensor([item["success"] for item in batch], dtype=torch.bool),
        "termination_reason": [item["termination_reason"] for item in batch],
    }


def array_sha256(values):
    array = np.ascontiguousarray(values, dtype=np.float32)
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def trajectory_digest(observations, actions):
    hasher = hashlib.sha256()
    for name, value in (("observations", observations), ("actions", actions)):
        array = np.ascontiguousarray(value)
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _train_scenario_ids(scenario_ids):
    identifiers = tuple(scenario_ids)
    if not identifiers:
        raise ValueError("Train normalizer requires scenario ids")
    cleaned = []
    for scenario_id in identifiers:
        if not isinstance(scenario_id, str) or not scenario_id.startswith("train_"):
            raise ValueError(
                "Normalization statistics can only be fit on train scenarios"
            )
        if scenario_id == "train_" or scenario_id in cleaned:
            raise ValueError(f"Invalid train scenario id {scenario_id}")
        cleaned.append(scenario_id)
    return tuple(cleaned)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_npz(path, arrays):
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _long(values):
    return torch.tensor(values, dtype=torch.long)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
