"""Record Jackal expert trajectories without overwriting existing data.

Successful and failed episodes are both kept. A crash leaves completed
episodes readable, and the next recorder continues from that directory.
"""

import json
import os
from pathlib import Path

import numpy as np

from navigation.expert_demonstration_schema import (
    SCHEMA_VERSION,
    DemonstrationEpisode,
    build_schema,
    content_digest,
    normalization_metadata,
    robot_configuration_metadata,
    validate_episode,
    validate_jackal_policy_spaces,
)


__all__ = [
    "DemonstrationEpisode",
    "ExpertDemonstrationRecorder",
    "load_dataset",
    "load_episode",
]

_OPEN_RECORDERS = set()


class ExpertDemonstrationRecorder:
    """Append-only dataset writer for NPZ or HDF5 episodes."""

    def __init__(
        self,
        output_dir,
        observation_space,
        action_space,
        storage="npz",
        robot=None,
    ):
        validate_jackal_policy_spaces(observation_space, action_space)

        if storage not in ("npz", "hdf5"):
            raise ValueError("storage must be 'npz' or 'hdf5'")

        self.output_dir = Path(output_dir)
        self.observation_space = observation_space
        self.action_space = action_space
        self.storage = storage
        self.extension = "npz" if storage == "npz" else "h5"
        self.robot_configuration = robot_configuration_metadata(robot)
        self.normalization = normalization_metadata(
            None if robot is None else robot.observation_builder
        )
        self.schema = build_schema(
            observation_space,
            action_space,
            storage,
            self.normalization,
            self.robot_configuration,
        )
        self.recovered_interrupt = False
        self._closed = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()

        try:
            self._open_dataset()
        except BaseException:
            self._release_lock()
            raise

        _OPEN_RECORDERS.add(id(self))

    def record(self, episode):
        """Validate and store one episode. Existing files are left unchanged."""

        if self._closed:
            raise RuntimeError("Recorder is closed")

        payload = validate_episode(
            episode,
            self.action_space,
            self.robot_configuration,
        )
        digest = content_digest(payload)
        payload = dict(payload)
        payload["integrity_sha256"] = digest

        filename = f"episode_{int(payload['episode_id']):04d}.{self.extension}"
        path = self.output_dir / filename
        manifest = _read_json(self.output_dir / "manifest.json")

        if path.exists() or any(
            item["episode_id"] == int(payload["episode_id"])
            for item in manifest["episodes"]
        ):
            raise FileExistsError(
                f"Episode {int(payload['episode_id'])} already exists"
            )

        self._write_episode(path, payload)
        manifest["episodes"].append(
            {
                "episode_id": int(payload["episode_id"]),
                "scenario_id": payload["scenario_id"],
                "termination_reason": payload["termination_reason"],
                "success": bool(payload["success"]),
                "num_transitions": int(payload["observations"].shape[0]),
                "file": filename,
                "integrity_sha256": digest,
            }
        )
        manifest["episodes"].sort(key=lambda item: item["episode_id"])

        try:
            _write_json(self.output_dir / "manifest.json", manifest)
        except BaseException:
            self._reconcile_manifest()
            raise

        return path

    def close(self):
        """Mark the session closed and release the recorder lock."""

        if self._closed:
            return

        self._closed = True
        _OPEN_RECORDERS.discard(id(self))

        try:
            _write_json(
                self.output_dir / "session.json",
                {"schema_version": SCHEMA_VERSION, "state": "closed"},
            )
        finally:
            self._release_lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _open_dataset(self):
        schema_path = self.output_dir / "schema.json"
        manifest_path = self.output_dir / "manifest.json"
        session_path = self.output_dir / "session.json"

        if not schema_path.exists():
            if any(self.output_dir.glob("episode_*")) or manifest_path.exists():
                raise RuntimeError(
                    "Output directory has episode data without a schema"
                )

            _write_json_exclusive(schema_path, self.schema)
            _write_json(
                manifest_path,
                {"schema_version": SCHEMA_VERSION, "episodes": []},
            )
        else:
            existing = _read_json(schema_path)
            if existing != self.schema:
                raise RuntimeError(
                    "Existing dataset schema does not match this recorder"
                )

            session = (
                _read_json(session_path) if session_path.exists() else {"state": "open"}
            )
            if session.get("state") != "closed":
                self.recovered_interrupt = True

            self._remove_partials()
            self._reconcile_manifest()

        _write_json(
            session_path,
            {"schema_version": SCHEMA_VERSION, "state": "open"},
        )

    def _reconcile_manifest(self):
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
        else:
            manifest = {"schema_version": SCHEMA_VERSION, "episodes": []}
            self.recovered_interrupt = True

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Manifest schema version does not match")

        known = {item["file"]: item for item in manifest["episodes"]}
        files = {
            path.name: path
            for path in self.output_dir.glob(f"episode_*.{self.extension}")
        }

        missing = [item["file"] for item in manifest["episodes"] if item["file"] not in files]
        if missing:
            raise RuntimeError(
                "Manifest references missing episode files: " + ", ".join(missing)
            )

        for name, path in sorted(files.items()):
            if name in known:
                continue

            payload = load_episode(path)
            manifest["episodes"].append(
                {
                    "episode_id": int(payload["episode_id"]),
                    "scenario_id": payload["scenario_id"],
                    "termination_reason": payload["termination_reason"],
                    "success": bool(payload["success"]),
                    "num_transitions": int(payload["observations"].shape[0]),
                    "file": name,
                    "integrity_sha256": payload["integrity_sha256"],
                }
            )
            self.recovered_interrupt = True

        manifest["episodes"].sort(key=lambda item: item["episode_id"])
        _write_json(manifest_path, manifest)

    def _remove_partials(self):
        for path in self.output_dir.glob("*.partial"):
            path.unlink()

    def _write_episode(self, path, payload):
        partial = path.with_name(path.name + ".partial")

        try:
            if self.storage == "npz":
                _write_npz(partial, payload)
            else:
                _write_hdf5(partial, payload)

            os.link(partial, path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        else:
            partial.unlink(missing_ok=True)

    def _acquire_lock(self):
        lock_path = self.output_dir / "recorder.lock"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

        try:
            fd = os.open(lock_path, flags, 0o644)
        except FileExistsError:
            if _lock_is_active(lock_path):
                raise RuntimeError("Another demonstration recorder is running")

            self.recovered_interrupt = True
            lock_path.unlink()
            fd = os.open(lock_path, flags, 0o644)

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
            handle.flush()
            os.fsync(handle.fileno())

    def _release_lock(self):
        lock_path = self.output_dir / "recorder.lock"

        try:
            payload = _read_json(lock_path)
        except FileNotFoundError:
            return

        if int(payload.get("pid", -1)) == os.getpid():
            lock_path.unlink(missing_ok=True)


def load_episode(path):
    """Load one episode and reject a failed integrity check."""

    path = Path(path)

    if path.suffix == ".npz":
        payload = _read_npz(path)
    elif path.suffix == ".h5":
        payload = _read_hdf5(path)
    else:
        raise ValueError("Episode file must be .npz or .h5")

    digest = payload.pop("integrity_sha256")
    expected = content_digest(payload)

    if digest != expected:
        raise ValueError(f"Integrity check failed for {path.name}")

    payload["integrity_sha256"] = digest
    return payload


def load_dataset(output_dir):
    """Load every completed episode listed in the manifest."""

    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "manifest.json")
    episodes = []

    for item in manifest["episodes"]:
        payload = load_episode(output_dir / item["file"])

        if payload["integrity_sha256"] != item["integrity_sha256"]:
            raise ValueError("Manifest digest does not match episode file")

        if int(payload["episode_id"]) != int(item["episode_id"]):
            raise ValueError("Manifest episode id does not match file")

        if payload["scenario_id"] != item["scenario_id"]:
            raise ValueError("Manifest scenario id does not match file")

        if payload["termination_reason"] != item["termination_reason"]:
            raise ValueError("Manifest termination reason does not match file")

        episodes.append(payload)

    return episodes


def _write_npz(path, payload):
    arrays = _storage_arrays(payload)

    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    return _restore_payload(payload)


def _write_hdf5(path, payload):
    h5py = _import_h5py()
    arrays = _storage_arrays(payload)

    with h5py.File(path, "w") as handle:
        for key, value in arrays.items():
            if value.dtype.kind in ("U", "O"):
                encoded = np.array(
                    value.item(),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                handle.create_dataset(key, data=encoded)
            elif value.ndim > 0 and value.size > 0:
                handle.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                handle.create_dataset(key, data=value)


def _read_hdf5(path):
    h5py = _import_h5py()

    with h5py.File(path, "r") as handle:
        payload = {}
        for key in handle.keys():
            value = handle[key][()]
            if getattr(value, "dtype", None) is not None and h5py.check_string_dtype(value.dtype):
                payload[key] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            else:
                payload[key] = value

    return _restore_payload(payload)


def _storage_arrays(payload):
    stored = {
        "episode_id": np.int64(payload["episode_id"]),
        "scenario_id": np.array(payload["scenario_id"]),
        "termination_reason": np.array(payload["termination_reason"]),
        "success": np.bool_(payload["success"]),
        "goal": np.asarray(payload["goal"], dtype=np.float64),
        "observations": np.asarray(payload["observations"], dtype=np.float32),
        "actions": np.asarray(payload["actions"], dtype=np.float32),
        "requested_actions": np.asarray(payload["requested_actions"], dtype=np.float32),
        "applied_velocities": np.asarray(payload["applied_velocities"], dtype=np.float32),
        "poses": np.asarray(payload["poses"], dtype=np.float64),
        "simulation_timestamps_ns": np.asarray(
            payload["simulation_timestamps_ns"],
            dtype=np.int64,
        ),
        "integrity_sha256": np.array(payload["integrity_sha256"]),
    }
    return stored


def _restore_payload(payload):
    restored = dict(payload)
    restored["episode_id"] = int(np.asarray(payload["episode_id"]))
    restored["scenario_id"] = _text(payload["scenario_id"])
    restored["termination_reason"] = _text(payload["termination_reason"])
    restored["success"] = bool(np.asarray(payload["success"]))
    restored["integrity_sha256"] = _text(payload["integrity_sha256"])
    restored["goal"] = np.asarray(payload["goal"], dtype=np.float64)
    restored["observations"] = np.asarray(payload["observations"], dtype=np.float32)
    restored["actions"] = np.asarray(payload["actions"], dtype=np.float32)
    restored["requested_actions"] = np.asarray(payload["requested_actions"], dtype=np.float32)
    restored["applied_velocities"] = np.asarray(
        payload["applied_velocities"],
        dtype=np.float32,
    )
    restored["poses"] = np.asarray(payload["poses"], dtype=np.float64)
    restored["simulation_timestamps_ns"] = np.asarray(
        payload["simulation_timestamps_ns"],
        dtype=np.int64,
    )
    return restored


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 storage requires h5py") from exc
    return h5py


def _lock_is_active(lock_path):
    try:
        payload = _read_json(lock_path)
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if pid == os.getpid() and not _OPEN_RECORDERS:
        return False

    if pid == os.getpid() and _OPEN_RECORDERS:
        return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")

    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(partial, path)


def _write_json_exclusive(path, payload):
    path = Path(path)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"{path.name} already exists") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
