"""Offline observation and action qualification harness.

Synthetic fixtures are qualified, passed through the existing pure observation
builder and action mapper, and recorded by an in-memory command sink.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from jackal_rl_navigation.action_mapper import ActionMapper, VelocityCommand
from jackal_rl_navigation.episode_outcome import evaluate_episode_outcome
from jackal_rl_navigation.goal_frames import world_goal_to_odom
from jackal_rl_navigation.observation_builder import ObservationBuilder


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_AGE_S = 0.5
SCAN_FRAME = "lidar_fixture"
ODOM_FRAME = "odom"
CHILD_FRAME = "base_link"
SCAN_SOURCE = "scan_fixture"
ODOM_SOURCE = "odom_fixture"
COMPARISON_ATOL = 1e-6

_REQUIRED_FIELDS = (
    "run_id",
    "episode_id",
    "snapshot_id",
    "transition_index",
    "scan_source_id",
    "odom_source_id",
    "scan_frame",
    "odom_frame",
    "child_frame",
    "ranges",
    "distance_m",
    "bearing_rad",
    "linear_mps",
    "angular_rps",
    "scan_stamp_ns",
    "odom_stamp_ns",
    "scan_received_mono_s",
    "odom_received_mono_s",
    "capture_mono_s",
    "angle_min_rad",
    "angle_increment_rad",
)

_EXACT_TOKENS = {
    "NaN": float("nan"),
    "+Inf": float("inf"),
    "-Inf": float("-inf"),
    "pi": math.pi,
    "-pi": -math.pi,
    "pi/2": math.pi / 2.0,
    "-pi/2": -(math.pi / 2.0),
    "3*pi/2": 3.0 * math.pi / 2.0,
}


class QualificationError(Exception):
    """Closed-form rejection raised before a command is accepted."""

    def __init__(self, code, message, prepared=None, snapshot=None):
        super().__init__(message)
        self.code = code
        self.prepared = prepared
        self.snapshot = snapshot


class _UnreadCache:
    """Sentinel whose fields must not be read by qualification."""

    def __getattribute__(self, name):
        raise RuntimeError("unrelated cache consulted")


def _reject_json_constant(name):
    raise ValueError(f"nonstandard JSON numeric: {name}")


def decode_fixture_token(value):
    """Decode the explicit fixture token allowlist. Other strings stay strings."""

    if isinstance(value, str):
        if value in _EXACT_TOKENS:
            return _EXACT_TOKENS[value]
        prefix = "2*pi/"
        if value.startswith(prefix) and value[len(prefix) :].isdigit():
            count = int(value[len(prefix) :])
            if count > 0:
                return 2.0 * math.pi / count
        return value
    if isinstance(value, list):
        return [decode_fixture_token(item) for item in value]
    if isinstance(value, dict):
        return {key: decode_fixture_token(item) for key, item in value.items()}
    return value


def load_manifest(path):
    """Load fixture JSON without nonstandard numeric literals, then decode tokens."""

    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text, parse_constant=_reject_json_constant)
    decoded = decode_fixture_token(raw)
    if not isinstance(decoded, dict):
        raise QualificationError("MALFORMED_INPUT", "manifest must be an object")
    return decoded


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_pins(pins):
    """Reject the run when a pinned pure-source hash does not match the tree."""

    if not isinstance(pins, dict) or not pins:
        raise QualificationError("SOURCE_PIN_MISMATCH", "source pins missing")
    mismatched = []
    for relative, digest in pins.items():
        path = REPO_ROOT / relative
        if not path.is_file() or _sha256(path) != digest:
            mismatched.append(str(relative))
    if mismatched:
        raise QualificationError("SOURCE_PIN_MISMATCH", " ".join(mismatched))


def _text_id(value):
    if not isinstance(value, str) or value == "":
        raise QualificationError("MALFORMED_INPUT", "identifier must be a nonempty string")
    return value


def _nonnegative_index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualificationError("MALFORMED_INPUT", "transition_index must be a nonnegative integer")
    return value


def _positive_stamp(value):
    if value is None or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationError("INVALID_STAMP", "stamp must be a positive integer")
    return value


def _real_scalar(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError("MALFORMED_INPUT", "scalar must be a real number")
    return float(value)


def _real_sample(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError("MALFORMED_INPUT", "range sample must be a real number")
    return float(value)


def _freeze_ranges(ranges):
    if ranges is None:
        raise QualificationError("MISSING_INPUT", "ranges are missing")
    if isinstance(ranges, (str, bytes)):
        raise QualificationError("MALFORMED_INPUT", "ranges must not be text")
    if isinstance(ranges, np.ndarray):
        if ranges.dtype == np.bool_ or ranges.dtype.kind not in "fiu":
            raise QualificationError("MALFORMED_INPUT", "ranges dtype")
        if ranges.ndim == 1:
            return tuple(_real_sample(item) for item in ranges.tolist())
        if ranges.ndim == 2:
            return tuple(tuple(_real_sample(item) for item in row) for row in ranges.tolist())
        raise QualificationError("MALFORMED_INPUT", "ranges rank")
    if isinstance(ranges, (list, tuple)):
        if len(ranges) > 0 and isinstance(ranges[0], (list, tuple, np.ndarray)):
            frozen_rows = []
            for row in ranges:
                if not isinstance(row, (list, tuple, np.ndarray)):
                    raise QualificationError("MALFORMED_INPUT", "ragged ranges")
                frozen_rows.append(tuple(_real_sample(item) for item in row))
            return tuple(frozen_rows)
        return tuple(_real_sample(item) for item in ranges)
    raise QualificationError("MALFORMED_INPUT", "ranges type")


def _range_rank(ranges):
    if isinstance(ranges, tuple) and ranges and isinstance(ranges[0], tuple):
        return 2
    return 1


@dataclass(frozen=True)
class SensorSnapshot:
    """Detached immutable copy of one synthetic sensor fixture."""

    run_id: str
    episode_id: str
    snapshot_id: str
    transition_index: int
    scan_source_id: str
    odom_source_id: str
    scan_frame: str
    odom_frame: str
    child_frame: str
    ranges: tuple
    distance_m: float
    bearing_rad: float
    linear_mps: float
    angular_rps: float
    scan_stamp_ns: int
    odom_stamp_ns: int
    scan_received_mono_s: float
    odom_received_mono_s: float
    capture_mono_s: float
    angle_min_rad: float
    angle_increment_rad: float

    def __post_init__(self):
        # Frozen fields alone do not detach caller-owned range containers.
        # Metadata is validated at entry so malformed typed inputs can reject
        # through the same structured dispatch path as dictionary inputs.
        object.__setattr__(self, "ranges", _freeze_ranges(self.ranges))

    @classmethod
    def from_fixture(cls, raw):
        if isinstance(raw, cls):
            raw = {name: getattr(raw, name) for name in _REQUIRED_FIELDS}
        if not isinstance(raw, dict):
            raise QualificationError("MALFORMED_INPUT", "snapshot fixture must be a dict")
        missing = [key for key in _REQUIRED_FIELDS if key not in raw]
        if missing:
            raise QualificationError("MISSING_INPUT", "missing " + ",".join(missing))
        return cls(
            run_id=_text_id(raw["run_id"]),
            episode_id=_text_id(raw["episode_id"]),
            snapshot_id=_text_id(raw["snapshot_id"]),
            transition_index=_nonnegative_index(raw["transition_index"]),
            scan_source_id=_text_id(raw["scan_source_id"]),
            odom_source_id=_text_id(raw["odom_source_id"]),
            scan_frame=_text_id(raw["scan_frame"]),
            odom_frame=_text_id(raw["odom_frame"]),
            child_frame=_text_id(raw["child_frame"]),
            ranges=_freeze_ranges(raw["ranges"]),
            distance_m=_real_scalar(raw["distance_m"]),
            bearing_rad=_real_scalar(raw["bearing_rad"]),
            linear_mps=_real_scalar(raw["linear_mps"]),
            angular_rps=_real_scalar(raw["angular_rps"]),
            scan_stamp_ns=_positive_stamp(raw["scan_stamp_ns"]),
            odom_stamp_ns=_positive_stamp(raw["odom_stamp_ns"]),
            scan_received_mono_s=_real_scalar(raw["scan_received_mono_s"]),
            odom_received_mono_s=_real_scalar(raw["odom_received_mono_s"]),
            capture_mono_s=_real_scalar(raw["capture_mono_s"]),
            angle_min_rad=_real_scalar(raw["angle_min_rad"]),
            angle_increment_rad=_real_scalar(raw["angle_increment_rad"]),
        )


@dataclass(frozen=True)
class PreparedObservation:
    """Little-endian float32 observation bytes and their identity."""

    snapshot: SensorSnapshot
    vector_bytes: bytes
    shape: tuple
    dtype_str: str
    sha256: str
    valid: bool
    reason: str
    dispatch_mono_s: float

    def readonly_vector(self):
        array = np.frombuffer(self.vector_bytes, dtype="<f4")
        return array.reshape(self.shape)


@dataclass(frozen=True)
class ActionRecord:
    """Immutable dummy-predictor output associated with one prepared observation."""

    snapshot_id: str
    observation_hash: str
    dummy_id: str
    action: object
    dt: object
    prepare_mono_s: float
    dispatch_mono_s: float


@dataclass(frozen=True)
class CapturedCommand:
    """One in-memory command record. This is not an actuator acknowledgement."""

    sequence: int
    run_id: str
    episode_id: str
    snapshot_id: str
    observation_hash: str
    dummy_id: str
    action: tuple
    linear: float
    angular: float
    monotonic_s: float
    decision: str
    reason: str


@dataclass(frozen=True)
class DispatchResult:
    """Result of one dispatch or stop attempt."""

    accepted: bool
    code: str
    latched: bool
    predictor_calls: int
    captured: bool
    command: tuple | None
    observation_valid: bool | None
    observation_reason: str | None
    observation_values: tuple | None
    observation_sha256: str | None
    observation_bytes: bytes | None
    observation_dtype: str | None
    observation_shape: tuple | None
    mapper_previous: tuple
    detail: str = ""

    def observation_view(self):
        if self.observation_bytes is None or self.observation_shape is None:
            raise RuntimeError("no observation bytes")
        array = np.frombuffer(self.observation_bytes, dtype="<f4")
        return array.reshape(self.observation_shape)


@dataclass(frozen=True)
class ResetResult:
    accepted: bool
    code: str


class ConstantDummyPredictor:
    """Fixed action source with no model, checkpoint, or random state."""

    def __init__(
        self,
        action=(0.5, -0.5),
        dt=0.1,
        dummy_id="constant-dummy-v1",
        snapshot_id=None,
        observation_hash=None,
    ):
        self.action = action
        self.dt = dt
        self.dummy_id = dummy_id
        self.snapshot_id = snapshot_id
        self.observation_hash = observation_hash
        self.calls = 0

    def predict(self, prepared):
        self.calls += 1
        view = prepared.readonly_vector()
        if view.size:
            _ = float(view[0])
        return ActionRecord(
            snapshot_id=prepared.snapshot.snapshot_id if self.snapshot_id is None else self.snapshot_id,
            observation_hash=prepared.sha256 if self.observation_hash is None else self.observation_hash,
            dummy_id=self.dummy_id,
            action=_freeze_tree(self.action),
            dt=self.dt,
            prepare_mono_s=prepared.snapshot.capture_mono_s,
            dispatch_mono_s=prepared.dispatch_mono_s,
        )


class CapturedCommandSink:
    """Append-only memory sink. External sink objects are rejected by the harness."""

    def __init__(self, fault=None):
        if fault is not None and not isinstance(fault, BaseException):
            raise TypeError("fault must be an exception instance")
        self._fault = fault
        self._records = []

    @property
    def records(self):
        return tuple(self._records)

    def capture(self, record):
        if type(record) is not CapturedCommand:
            raise TypeError("sink record type")
        if self._fault is not None:
            raise self._fault
        self._records.append(record)


def _freeze_tree(value):
    if isinstance(value, list):
        return tuple(_freeze_tree(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _finite_clock(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError("MALFORMED_INPUT", "clock must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise QualificationError("MALFORMED_INPUT", "clock must be finite")
    return number


def _check_identity(snapshot):
    if (
        snapshot.scan_frame != SCAN_FRAME
        or snapshot.odom_frame != ODOM_FRAME
        or snapshot.child_frame != CHILD_FRAME
    ):
        raise QualificationError("FRAME_MISMATCH", "frame identity", snapshot=snapshot)
    if snapshot.scan_source_id != SCAN_SOURCE or snapshot.odom_source_id != ODOM_SOURCE:
        raise QualificationError("SOURCE_MISMATCH", "source identity", snapshot=snapshot)


def _check_geometry(snapshot):
    angle_min = snapshot.angle_min_rad
    increment = snapshot.angle_increment_rad
    if not math.isfinite(angle_min) or not math.isfinite(increment) or increment <= 0:
        raise QualificationError("GEOMETRY_MISMATCH", "angle metadata", snapshot=snapshot)
    if abs(angle_min - (-math.pi)) > COMPARISON_ATOL:
        raise QualificationError("GEOMETRY_MISMATCH", "angle_min", snapshot=snapshot)
    if _range_rank(snapshot.ranges) != 1:
        return
    count = len(snapshot.ranges)
    if count < 1:
        raise QualificationError("GEOMETRY_MISMATCH", "ray count", snapshot=snapshot)
    expected = 2.0 * math.pi / count
    if abs(increment - expected) > COMPARISON_ATOL:
        raise QualificationError("GEOMETRY_MISMATCH", "angle_increment", snapshot=snapshot)


def _check_clock(snapshot, dispatch, phase):
    if (
        snapshot.scan_received_mono_s > snapshot.capture_mono_s
        or snapshot.odom_received_mono_s > snapshot.capture_mono_s
        or dispatch < snapshot.capture_mono_s
    ):
        raise QualificationError("CLOCK_ORDER", "receive, capture, dispatch order", snapshot=snapshot)
    reference = snapshot.capture_mono_s if phase == "prepare" else dispatch
    for received in (snapshot.scan_received_mono_s, snapshot.odom_received_mono_s):
        age = reference - received
        if age < 0:
            raise QualificationError("CLOCK_ORDER", "negative age", snapshot=snapshot)
        if age > MAX_AGE_S:
            raise QualificationError("STALE_INPUT", "receive age exceeds 0.5s", snapshot=snapshot)


def _check_progress(snapshot, dispatch, last_scan_stamp, last_odom_stamp, last_dispatch_mono_s):
    if last_dispatch_mono_s is not None and dispatch <= last_dispatch_mono_s:
        raise QualificationError("CLOCK_ORDER", "dispatch clock did not advance", snapshot=snapshot)
    if last_scan_stamp is not None and snapshot.scan_stamp_ns <= last_scan_stamp:
        raise QualificationError("NONADVANCING_STAMP", "scan stamp", snapshot=snapshot)
    if last_odom_stamp is not None and snapshot.odom_stamp_ns <= last_odom_stamp:
        raise QualificationError("NONADVANCING_STAMP", "odom stamp", snapshot=snapshot)


def qualify_snapshot(
    snapshot,
    dispatch_mono_s,
    last_scan_stamp=None,
    last_odom_stamp=None,
    last_dispatch_mono_s=None,
    phase="prepare",
):
    """Validate identity, synthetic geometry, clocks, ages, and stamp progress."""

    if not isinstance(snapshot, SensorSnapshot):
        raise QualificationError("MALFORMED_INPUT", "snapshot type")
    snapshot = SensorSnapshot.from_fixture(snapshot)
    for value in (
        snapshot.scan_received_mono_s,
        snapshot.odom_received_mono_s,
        snapshot.capture_mono_s,
    ):
        if not math.isfinite(value):
            raise QualificationError("MALFORMED_INPUT", "clock must be finite", snapshot=snapshot)
    dispatch = _finite_clock(dispatch_mono_s)
    _check_identity(snapshot)
    _check_geometry(snapshot)
    _check_clock(snapshot, dispatch, phase)
    if phase == "prepare":
        _check_progress(
            snapshot,
            dispatch,
            last_scan_stamp,
            last_odom_stamp,
            last_dispatch_mono_s,
        )


def _is_exact_zero_action(action):
    if isinstance(action, (str, bytes, bool)):
        return False
    try:
        values = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        return False
    return bool(np.all(values == 0.0))


def _safe_action(action):
    frozen = _freeze_tree(action)

    def convert(value):
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "malformed"
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "+Inf" if number > 0 else "-Inf"
        return number

    if isinstance(frozen, tuple):
        return convert(frozen)
    return (convert(frozen),)


class OfflineHarness:
    """Connect one snapshot, the pure builder, a dummy predictor, and the mapper."""

    def __init__(self, sink=None):
        if sink is None:
            sink = CapturedCommandSink()
        if type(sink) is not CapturedCommandSink:
            raise TypeError("external command sink is not permitted")
        self.sink = sink
        self._builder = ObservationBuilder()
        self._mapper = ActionMapper()
        self._latch = None
        self._episode_id = None
        self._episode_identity_unknown = False
        self._failed_episode_ids = set()
        self._last_scan_stamp = None
        self._last_odom_stamp = None
        self._last_dispatch_mono_s = None
        self._last_snapshot = None

    def seed_accepted_stamps(self, scan_stamp, odom_stamp):
        """Record fixture bookkeeping for an already accepted input."""

        self._last_scan_stamp = _positive_stamp(scan_stamp)
        self._last_odom_stamp = _positive_stamp(odom_stamp)

    def _coerce_snapshot(self, raw):
        if raw is None:
            raise QualificationError("MISSING_INPUT", "snapshot is missing")
        if isinstance(raw, (SensorSnapshot, dict)):
            return SensorSnapshot.from_fixture(raw)
        raise QualificationError("MALFORMED_INPUT", "snapshot type")

    def _build_prepared(self, snapshot, dispatch_mono_s):
        try:
            result = self._builder.build(
                snapshot.ranges,
                snapshot.distance_m,
                snapshot.bearing_rad,
                snapshot.linear_mps,
                snapshot.angular_rps,
            )
        except (TypeError, ValueError) as exc:
            raise QualificationError("CONVERSION_ERROR", str(exc), snapshot=snapshot) from exc
        vector = np.ascontiguousarray(result.vector, dtype="<f4")
        expected = (self._builder.observation_dim,)
        if vector.shape != expected:
            raise QualificationError(
                "INVALID_OBSERVATION",
                "observation shape",
                snapshot=snapshot,
            )
        blob = vector.tobytes()
        return PreparedObservation(
            snapshot=snapshot,
            vector_bytes=blob,
            shape=expected,
            dtype_str="<f4",
            sha256=hashlib.sha256(blob).hexdigest(),
            valid=bool(result.valid),
            reason=result.reason or "",
            dispatch_mono_s=float(dispatch_mono_s),
        )

    def prepare(self, raw, dispatch_mono_s):
        """Qualify a snapshot and build its observation without predicting."""

        if self._latch is not None:
            raise QualificationError(self._latch, "harness is latched")
        snapshot = self._coerce_snapshot(raw)
        if self._episode_id is not None and snapshot.episode_id != self._episode_id:
            raise QualificationError(
                "EPISODE_MISMATCH",
                "episode id does not match the active episode",
                snapshot=snapshot,
            )
        qualify_snapshot(
            snapshot,
            dispatch_mono_s,
            last_scan_stamp=self._last_scan_stamp,
            last_odom_stamp=self._last_odom_stamp,
            last_dispatch_mono_s=self._last_dispatch_mono_s,
            phase="prepare",
        )
        return self._build_prepared(snapshot, dispatch_mono_s)

    def _result(
        self,
        *,
        accepted,
        code,
        latched,
        predictor_calls,
        captured,
        command,
        prepared,
        detail="",
    ):
        if prepared is None:
            values = None
            digest = None
            dtype_name = None
            shape = None
            valid = None
            reason = None
            blob = None
        else:
            view = prepared.readonly_vector()
            values = tuple(float(item) for item in view.tolist())
            digest = prepared.sha256
            dtype_name = "float32"
            shape = tuple(prepared.shape)
            valid = prepared.valid
            reason = prepared.reason
            blob = prepared.vector_bytes
        command_tuple = None if command is None else (float(command[0]), float(command[1]))
        return DispatchResult(
            accepted=accepted,
            code=code,
            latched=latched,
            predictor_calls=predictor_calls,
            captured=captured,
            command=command_tuple,
            observation_valid=valid,
            observation_reason=reason,
            observation_values=values,
            observation_sha256=digest,
            observation_bytes=blob,
            observation_dtype=dtype_name,
            observation_shape=shape,
            mapper_previous=(
                float(self._mapper.previous.linear),
                float(self._mapper.previous.angular),
            ),
            detail=detail,
        )

    def _make_record(self, snapshot, prepared, action_record, command, decision, reason, monotonic_s):
        return CapturedCommand(
            sequence=len(self.sink.records) + 1,
            run_id="" if snapshot is None else snapshot.run_id,
            episode_id=self._episode_id or ("" if snapshot is None else snapshot.episode_id),
            snapshot_id="" if snapshot is None else snapshot.snapshot_id,
            observation_hash="" if prepared is None else prepared.sha256,
            dummy_id="" if action_record is None else action_record.dummy_id,
            action=() if action_record is None else _safe_action(action_record.action),
            linear=float(command.linear),
            angular=float(command.angular),
            monotonic_s=float(monotonic_s),
            decision=decision,
            reason=reason,
        )

    def _capture(self, record):
        try:
            self.sink.capture(record)
        except Exception:
            self._latch = "SINK_FAILED"
            if self._episode_id is None:
                self._episode_identity_unknown = True
            else:
                self._failed_episode_ids.add(self._episode_id)
            return False
        return True

    def _reject(self, exc, predictor_calls, monotonic_s):
        self._latch = exc.code
        stopped = self._mapper.emergency_stop()
        prepared = exc.prepared
        snapshot = exc.snapshot if exc.snapshot is not None else (None if prepared is None else prepared.snapshot)
        if self._episode_id is None:
            if snapshot is not None and isinstance(snapshot.episode_id, str) and snapshot.episode_id:
                self._episode_id = snapshot.episode_id
            else:
                self._episode_identity_unknown = True
        if self._episode_id is not None:
            self._failed_episode_ids.add(self._episode_id)
        record = self._make_record(
            snapshot,
            prepared,
            None,
            stopped,
            "REJECT",
            exc.code,
            monotonic_s,
        )
        captured = self._capture(record)
        if not captured:
            return self._result(
                accepted=False,
                code="SINK_FAILED",
                latched=True,
                predictor_calls=predictor_calls,
                captured=False,
                command=None,
                prepared=prepared,
                detail=exc.code,
            )
        return self._result(
            accepted=False,
            code=exc.code,
            latched=True,
            predictor_calls=predictor_calls,
            captured=True,
            command=(stopped.linear, stopped.angular),
            prepared=prepared,
            detail=str(exc),
        )

    def _check_association(self, action_record, prepared):
        if action_record.snapshot_id != prepared.snapshot.snapshot_id:
            raise QualificationError(
                "ASSOCIATION_MISMATCH",
                "snapshot association",
                prepared=prepared,
                snapshot=prepared.snapshot,
            )
        if action_record.observation_hash != prepared.sha256:
            raise QualificationError(
                "ASSOCIATION_MISMATCH",
                "observation association",
                prepared=prepared,
                snapshot=prepared.snapshot,
            )
        if not isinstance(action_record.dummy_id, str) or action_record.dummy_id == "":
            raise QualificationError(
                "ASSOCIATION_MISMATCH",
                "dummy association",
                prepared=prepared,
                snapshot=prepared.snapshot,
            )

    def _apply_action(self, action_record):
        if _is_exact_zero_action(action_record.action):
            probe = ActionMapper()
            try:
                probe.map_action(action_record.action, action_record.dt)
            except ValueError as exc:
                raise QualificationError("INVALID_ACTION", str(exc)) from exc
            return "STOP", self._mapper.emergency_stop()
        try:
            command = self._mapper.map_action(action_record.action, action_record.dt)
        except ValueError as exc:
            raise QualificationError("INVALID_ACTION", str(exc)) from exc
        return "ACCEPT", command

    def _commit(self, snapshot, dispatch_mono_s):
        self._last_scan_stamp = snapshot.scan_stamp_ns
        self._last_odom_stamp = snapshot.odom_stamp_ns
        self._last_dispatch_mono_s = float(dispatch_mono_s)
        self._episode_id = snapshot.episode_id
        self._last_snapshot = snapshot

    def dispatch(self, raw, dispatch_mono_s, predictor=None, unrelated_cache=None):
        """Predict and map one snapshot, or capture zero and latch on rejection."""

        if predictor is None:
            predictor = ConstantDummyPredictor()
        calls_before = predictor.calls
        if unrelated_cache is not None:
            _ = True
        try:
            monotonic_s = float(dispatch_mono_s)
        except (TypeError, ValueError):
            monotonic_s = 0.0
        if self._latch is not None:
            return self._reject(
                QualificationError(self._latch, "harness is latched", snapshot=self._last_snapshot),
                0,
                monotonic_s,
            )
        # Bind only the syntactically trustworthy episode label, before other
        # fields can fail validation. A latched dispatch cannot replace it.
        if self._episode_id is None:
            episode_id = raw.get("episode_id") if isinstance(raw, dict) else (
                raw.episode_id if isinstance(raw, SensorSnapshot) else None
            )
            if isinstance(episode_id, str) and episode_id:
                self._episode_id = episode_id
        try:
            prepared = self.prepare(raw, dispatch_mono_s)
        except QualificationError as exc:
            return self._reject(exc, 0, monotonic_s)
        if not prepared.valid:
            return self._reject(
                QualificationError(
                    "INVALID_OBSERVATION",
                    prepared.reason or "invalid observation",
                    prepared=prepared,
                    snapshot=prepared.snapshot,
                ),
                0,
                monotonic_s,
            )
        try:
            action_record = predictor.predict(prepared)
            if not isinstance(action_record, ActionRecord):
                raise QualificationError(
                    "INVALID_ACTION",
                    "predictor result",
                    prepared=prepared,
                    snapshot=prepared.snapshot,
                )
            qualify_snapshot(prepared.snapshot, dispatch_mono_s, phase="dispatch")
            self._check_association(action_record, prepared)
            decision, command = self._apply_action(action_record)
        except QualificationError as exc:
            if exc.prepared is None:
                exc.prepared = prepared
            if exc.snapshot is None:
                exc.snapshot = prepared.snapshot
            return self._reject(exc, predictor.calls - calls_before, monotonic_s)
        except (TypeError, ValueError) as exc:
            return self._reject(
                QualificationError(
                    "INVALID_ACTION",
                    str(exc),
                    prepared=prepared,
                    snapshot=prepared.snapshot,
                ),
                predictor.calls - calls_before,
                monotonic_s,
            )
        record = self._make_record(
            prepared.snapshot,
            prepared,
            action_record,
            command,
            decision,
            "STOP" if decision == "STOP" else "OK",
            monotonic_s,
        )
        if not self._capture(record):
            self._mapper.emergency_stop()
            return self._result(
                accepted=False,
                code="SINK_FAILED",
                latched=True,
                predictor_calls=predictor.calls - calls_before,
                captured=False,
                command=None,
                prepared=prepared,
                detail="SINK_FAILED",
            )
        self._commit(prepared.snapshot, monotonic_s)
        return self._result(
            accepted=True,
            code="STOP" if decision == "STOP" else "OK",
            latched=False,
            predictor_calls=predictor.calls - calls_before,
            captured=True,
            command=(command.linear, command.angular),
            prepared=prepared,
        )

    def stop(self):
        """Capture an immediate zero. Repeated calls keep the stopped mapper state."""

        command = self._mapper.emergency_stop()
        if self._latch is None:
            self._latch = "STOPPED"
        monotonic_s = 0.0 if self._last_dispatch_mono_s is None else self._last_dispatch_mono_s
        record = self._make_record(
            self._last_snapshot,
            None,
            None,
            command,
            "STOP",
            self._latch,
            monotonic_s,
        )
        captured = self._capture(record)
        if not captured:
            return self._result(
                accepted=False,
                code="SINK_FAILED",
                latched=True,
                predictor_calls=0,
                captured=False,
                command=None,
                prepared=None,
                detail="SINK_FAILED",
            )
        return self._result(
            accepted=False,
            code=self._latch,
            latched=True,
            predictor_calls=0,
            captured=True,
            command=(command.linear, command.angular),
            prepared=None,
            detail=self._latch,
        )

    def reset_episode(self, new_episode_id):
        """Clear software latch and mapper state for a different fixture episode."""

        if not isinstance(new_episode_id, str) or new_episode_id == "":
            return ResetResult(False, "MALFORMED_INPUT")
        if self._episode_identity_unknown:
            # No label can be proved different from the unidentified failure.
            # Retire this fixture harness; never clear its latch by guessing.
            return ResetResult(False, "EPISODE_IDENTITY_UNKNOWN")
        if new_episode_id in self._failed_episode_ids or new_episode_id == self._episode_id:
            return ResetResult(False, "EPISODE_REARM_REJECTED")
        self._episode_id = new_episode_id
        self._mapper.reset()
        self._latch = None
        self._last_scan_stamp = None
        self._last_odom_stamp = None
        self._last_dispatch_mono_s = None
        self._last_snapshot = None
        return ResetResult(True, "OK")

    def apply_outcome(
        self,
        goal_distance,
        collision_detected,
        episode_steps,
        max_episode_steps=300,
        goal_tolerance=0.30,
    ):
        """Apply the pure outcome function and latch terminal or timeout results."""

        outcome = evaluate_episode_outcome(
            goal_distance,
            goal_tolerance,
            collision_detected,
            episode_steps,
            max_episode_steps,
        )
        terminated, truncated, _reward, info = outcome
        if not terminated and not truncated:
            return outcome, "CONTINUE", False, None
        code = "TIME_LIMIT" if truncated else "TERMINAL"
        if self._latch is None:
            self._latch = code
        stopped = self._mapper.emergency_stop()
        record = self._make_record(
            self._last_snapshot,
            None,
            None,
            stopped,
            "REJECT",
            info["termination_reason"] or code,
            0.0 if self._last_dispatch_mono_s is None else self._last_dispatch_mono_s,
        )
        if not self._capture(record):
            return outcome, "SINK_FAILED", False, None
        return outcome, self._latch, True, (stopped.linear, stopped.angular)

    def convert_goal(self, world_goal, world_pose, odom_pose):
        """Convert one synthetic world goal with the existing pure function."""

        return world_goal_to_odom(world_goal, world_pose, odom_pose)


def _materialize_ranges(spec):
    if spec is None:
        return None, None
    if isinstance(spec, dict) and "shape" in spec:
        rows, columns = spec["shape"]
        fill = spec["fill"]
        return [[fill for _ in range(columns)] for _ in range(rows)], None
    if isinstance(spec, dict):
        count = int(spec["count"])
        values = [spec["fill"] for _ in range(count)]
        for key, value in spec.get("set", {}).items():
            values[int(key)] = value
        return values, count
    if isinstance(spec, list):
        nested = bool(spec) and isinstance(spec[0], list)
        return list(spec), None if nested else len(spec)
    raise QualificationError("MALFORMED_INPUT", "ranges spec")


def build_snapshot_input(base, step, case_id, index):
    """Merge the base fixture with one step's explicit overrides."""

    if "snapshot" in step and step["snapshot"] is None:
        return None

    def pick(name):
        if name in step:
            return step[name]
        return base[name]

    ranges_spec = step["ranges"] if "ranges" in step else base["ranges"]
    ranges, count = _materialize_ranges(ranges_spec)
    raw = {
        "run_id": pick("run_id"),
        "episode_id": pick("episode_id"),
        "snapshot_id": step.get("snapshot_id", f"{case_id}-s{index}"),
        "transition_index": step.get("transition_index", index),
        "scan_source_id": pick("scan_source_id"),
        "odom_source_id": pick("odom_source_id"),
        "scan_frame": pick("scan_frame"),
        "odom_frame": pick("odom_frame"),
        "child_frame": pick("child_frame"),
        "ranges": ranges,
        "distance_m": pick("distance_m"),
        "bearing_rad": pick("bearing_rad"),
        "linear_mps": pick("linear_mps"),
        "angular_rps": pick("angular_rps"),
        "scan_stamp_ns": pick("scan_stamp_ns"),
        "odom_stamp_ns": pick("odom_stamp_ns"),
        "scan_received_mono_s": pick("scan_received_mono_s"),
        "odom_received_mono_s": pick("odom_received_mono_s"),
        "capture_mono_s": pick("capture_mono_s"),
        "angle_min_rad": pick("angle_min_rad"),
    }
    if "angle_increment_rad" in step:
        raw["angle_increment_rad"] = step["angle_increment_rad"]
    elif count:
        raw["angle_increment_rad"] = 2.0 * math.pi / count
    else:
        raw["angle_increment_rad"] = pick("angle_increment_rad")
    for key in step.get("omit", ()):
        raw.pop(key, None)
    return raw


def _new_harness(case, base, step=None):
    harness = OfflineHarness()
    if step is not None and "prior_stamps" in step:
        seeds = step["prior_stamps"]
    elif "prior_stamps" in case:
        seeds = case["prior_stamps"]
    else:
        seeds = base.get("prior_stamps")
    if seeds:
        harness.seed_accepted_stamps(seeds[0], seeds[1])
    return harness


def _result_dict(result, op):
    command = None if result.command is None else [float(result.command[0]), float(result.command[1])]
    observation = None if result.observation_values is None else [float(item) for item in result.observation_values]
    return {
        "op": op,
        "accepted": result.accepted,
        "code": result.code,
        "latched": result.latched,
        "predictor_calls": result.predictor_calls,
        "captured": result.captured,
        "command": command,
        "valid": result.observation_valid,
        "reason": result.observation_reason,
        "observation": observation,
        "observation_sha256": result.observation_sha256,
        "observation_dtype": result.observation_dtype,
        "observation_shape": None if result.observation_shape is None else list(result.observation_shape),
        "mapper_previous": [float(result.mapper_previous[0]), float(result.mapper_previous[1])],
        "detail": result.detail,
    }


def _execute_direct(direct, step):
    if "set_previous" in step:
        previous = step["set_previous"]
        direct.previous = VelocityCommand(float(previous[0]), float(previous[1]))
    before = [direct.previous.linear, direct.previous.angular]
    try:
        command = direct.map_action(step["action"], step["dt"])
        raised = None
        message = ""
        values = [command.linear, command.angular]
    except ValueError as exc:
        raised = "ValueError"
        message = str(exc)
        values = None
    return {
        "op": "direct_mapper",
        "raised": raised,
        "message": message,
        "command": values,
        "previous_before": before,
        "previous_after": [direct.previous.linear, direct.previous.angular],
        "captured": False,
        "accepted": raised is None,
        "predictor_calls": 0,
        "latched": False,
    }


def _execute_dispatch(harness, base, case, step, index):
    raw = build_snapshot_input(base, step, case["id"], index)
    action = step["action"] if "action" in step else base["action"]
    dt = step["dt"] if "dt" in step else base["dt"]
    dispatch_s = step["dispatch_mono_s"] if "dispatch_mono_s" in step else base["dispatch_mono_s"]
    predictor = ConstantDummyPredictor(
        action=action,
        dt=dt,
        snapshot_id=step.get("association_snapshot_id"),
        observation_hash=step.get("association_hash"),
    )
    cache = _UnreadCache() if step.get("unrelated_cache") else None
    result = harness.dispatch(raw, dispatch_s, predictor=predictor, unrelated_cache=cache)
    return _result_dict(result, "dispatch")


def _execute_outcome(harness, step):
    outcome, code, captured, command = harness.apply_outcome(
        step["goal_distance"],
        step["collision_detected"],
        step["episode_steps"],
        step.get("max_episode_steps", 300),
        step.get("goal_tolerance", 0.30),
    )
    terminated, truncated, reward, info = outcome
    return {
        "op": "outcome",
        "terminated": terminated,
        "truncated": truncated,
        "reward_adjustment": reward,
        "termination_reason": info["termination_reason"],
        "collision_detected": info["collision_detected"],
        "goal_reached": info["goal_reached"],
        "time_limit_reached": info["time_limit_reached"],
        "code": code,
        "accepted": code == "CONTINUE",
        "latched": code != "CONTINUE",
        "captured": captured,
        "command": None if command is None else [float(command[0]), float(command[1])],
        "predictor_calls": 0,
    }


def _execute_goal(harness, step):
    goal_x, goal_y = harness.convert_goal(step["world_goal"], step["world_pose"], step["odom_pose"])
    return {
        "op": "goal_frame",
        "odom_goal": [goal_x, goal_y],
        "accepted": True,
        "code": "OK",
        "captured": False,
        "command": None,
        "predictor_calls": 0,
        "latched": False,
    }


def _execute_immutability(harness, base, case):
    step = {"ranges": {"count": 72, "fill": 6.0}}
    raw = build_snapshot_input(base, step, case["id"], 0)
    original = raw["ranges"]
    snapshot = SensorSnapshot.from_fixture(raw)
    original[0] = -5.0
    input_isolated = snapshot.ranges[0] == 6.0
    frozen_field_rejected = False
    try:
        snapshot.snapshot_id = "mutated"
    except AttributeError:
        frozen_field_rejected = True
    result = harness.dispatch(snapshot, base["dispatch_mono_s"])
    observation_write_rejected = False
    try:
        result.observation_view()[0] = 0.0
    except ValueError:
        observation_write_rejected = True
    sink_record_frozen = False
    try:
        harness.sink.records[-1].linear = 9.0
    except AttributeError:
        sink_record_frozen = True
    other = OfflineHarness()
    other.seed_accepted_stamps(100, 100)
    again = other.dispatch(snapshot, base["dispatch_mono_s"])
    return {
        "op": "immutability",
        "input_isolated": input_isolated,
        "frozen_field_rejected": frozen_field_rejected,
        "observation_write_rejected": observation_write_rejected,
        "sink_record_frozen": sink_record_frozen,
        "hash_unchanged": result.observation_sha256 == again.observation_sha256,
        "observation_sha256": result.observation_sha256,
        "accepted": result.accepted,
        "code": result.code,
        "captured": result.captured,
        "command": None if result.command is None else [float(result.command[0]), float(result.command[1])],
        "predictor_calls": result.predictor_calls,
        "latched": result.latched,
    }


def _fixture_capture_context(harness, base, case, step, index):
    """Expected record identity from fixture input and pre-operation state."""

    snapshot = harness._last_snapshot
    episode_id = harness._episode_id or ""
    monotonic_s = harness._last_dispatch_mono_s or 0.0
    if step["op"] in ("dispatch", "immutability"):
        overrides = {} if step["op"] == "immutability" else step
        raw = build_snapshot_input(base, overrides, case["id"], index)
        monotonic_s = overrides.get("dispatch_mono_s", base["dispatch_mono_s"])
        if harness._latch is None:
            if not episode_id and raw is not None:
                candidate = raw.get("episode_id")
                if isinstance(candidate, str) and candidate:
                    episode_id = candidate
            try:
                snapshot = SensorSnapshot.from_fixture(raw)
            except QualificationError:
                snapshot = None
    return {
        "run_id": "" if snapshot is None else snapshot.run_id,
        "episode_id": episode_id or ("" if snapshot is None else snapshot.episode_id),
        "snapshot_id": "" if snapshot is None else snapshot.snapshot_id,
        "monotonic_s": float(monotonic_s),
    }


def execute_case(case, base):
    """Execute one manifest case and return the observed trace."""

    harness = _new_harness(case, base)
    direct = ActionMapper()
    steps_out = []
    sink_commands = []
    for index, step in enumerate(case["steps"]):
        if step.get("new_harness"):
            harness = _new_harness(case, base, step)
            direct = ActionMapper()
        operation = step["op"]
        before = harness.sink.records
        context = _fixture_capture_context(harness, base, case, step, index)
        if operation == "dispatch":
            actual = _execute_dispatch(harness, base, case, step, index)
        elif operation == "direct_mapper":
            actual = _execute_direct(direct, step)
        elif operation == "outcome":
            actual = _execute_outcome(harness, step)
        elif operation == "stop":
            actual = _result_dict(harness.stop(), "stop")
        elif operation == "reset_episode":
            reset = harness.reset_episode(step["episode_id"])
            actual = {
                "op": "reset_episode",
                "accepted": reset.accepted,
                "code": reset.code,
                "captured": False,
                "command": None,
                "predictor_calls": 0,
                "latched": harness._latch is not None,
            }
        elif operation == "goal_frame":
            actual = _execute_goal(harness, step)
        elif operation == "immutability":
            actual = _execute_immutability(harness, base, case)
        else:
            raise QualificationError("MALFORMED_INPUT", "unknown case operation")
        after = harness.sink.records
        actual["sink_before"] = [record.__dict__.copy() for record in before]
        actual["sink_after"] = [record.__dict__.copy() for record in after]
        actual["sink_context"] = context
        steps_out.append(actual)
        sink_commands.extend([[record.linear, record.angular] for record in after[len(before):]])
    return {
        "id": case["id"], "steps": steps_out, "sink_commands": sink_commands,
        "default_action": _safe_action(base["action"]),
    }


def _close(expected, actual, atol=COMPARISON_ATOL):
    if actual is None or len(expected) != len(actual):
        return False
    return all(abs(float(left) - float(right)) <= atol for left, right in zip(expected, actual))


def _fail(case_id, index, key, expected, actual):
    raise AssertionError(f"{case_id} step {index} {key}: expected {expected!r} actual {actual!r}")


def _assert_step(case_id, index, expect, actual):
    scalar_keys = (
        "code",
        "accepted",
        "latched",
        "predictor_calls",
        "captured",
        "valid",
        "reason",
        "raised",
        "terminated",
        "truncated",
        "termination_reason",
        "collision_detected",
        "goal_reached",
        "time_limit_reached",
        "input_isolated",
        "frozen_field_rejected",
        "observation_write_rejected",
        "sink_record_frozen",
        "hash_unchanged",
    )
    for key in scalar_keys:
        if key in expect and actual.get(key) != expect[key]:
            _fail(case_id, index, key, expect[key], actual.get(key))
    if "reward_adjustment" in expect:
        if abs(float(actual["reward_adjustment"]) - float(expect["reward_adjustment"])) > COMPARISON_ATOL:
            _fail(case_id, index, "reward_adjustment", expect["reward_adjustment"], actual["reward_adjustment"])
    if "message_contains" in expect and expect["message_contains"] not in actual.get("message", ""):
        _fail(case_id, index, "message_contains", expect["message_contains"], actual.get("message"))
    if expect.get("previous_unchanged") and not _close(actual["previous_before"], actual["previous_after"]):
        _fail(case_id, index, "previous_unchanged", actual["previous_before"], actual["previous_after"])
    if "command" in expect:
        if expect["command"] is None:
            if actual.get("command") is not None:
                _fail(case_id, index, "command", None, actual.get("command"))
        elif not _close(expect["command"], actual.get("command")):
            _fail(case_id, index, "command", expect["command"], actual.get("command"))
    if "mapper_previous" in expect and not _close(expect["mapper_previous"], actual.get("mapper_previous")):
        _fail(case_id, index, "mapper_previous", expect["mapper_previous"], actual.get("mapper_previous"))
    if "previous_after" in expect and not _close(expect["previous_after"], actual.get("previous_after")):
        _fail(case_id, index, "previous_after", expect["previous_after"], actual.get("previous_after"))
    if "odom_goal" in expect and not _close(expect["odom_goal"], actual.get("odom_goal")):
        _fail(case_id, index, "odom_goal", expect["odom_goal"], actual.get("odom_goal"))
    if "observation" in expect:
        if expect["observation"] is None:
            if actual.get("observation") is not None:
                _fail(case_id, index, "observation", None, actual.get("observation"))
        else:
            if actual.get("observation_dtype") != "float32" or actual.get("observation_shape") != [40]:
                _fail(
                    case_id,
                    index,
                    "observation_layout",
                    ["float32", [40]],
                    [actual.get("observation_dtype"), actual.get("observation_shape")],
                )
            if not _close(expect["observation"], actual.get("observation")):
                _fail(case_id, index, "observation", expect["observation"], actual.get("observation"))
    if actual.get("accepted") is False and actual.get("captured") and actual.get("command") is not None:
        if not _close([0.0, 0.0], actual["command"]):
            _fail(case_id, index, "rejected_command", [0.0, 0.0], actual["command"])


def assert_case(case, actual):
    """Compare one executed trace with the fixture's independent expectations."""

    if actual["id"] != case["id"]:
        raise AssertionError(f"{case['id']} id mismatch")
    if len(actual["steps"]) != len(case["steps"]):
        raise AssertionError(f"{case['id']} step count")
    for index, (step, got) in enumerate(zip(case["steps"], actual["steps"])):
        _assert_step(case["id"], index, step["expect"], got)
        before, after = got["sink_before"], got["sink_after"]
        if index and not step.get("new_harness") and before != actual["steps"][index - 1]["sink_after"]:
            _fail(case["id"], index, "sink_continuity", actual["steps"][index - 1]["sink_after"], before)
        if after[:len(before)] != before:
            _fail(case["id"], index, "sink_history", before, after[:len(before)])
        records = after[len(before):]
        count = int(step["expect"].get("captured", False))
        if len(records) != count or bool(records) != got["captured"]:
            _fail(case["id"], index, "sink_count", count, len(records))
        for record in records:
            expected = dict(got["sink_context"])
            expected["sequence"] = len(before) + 1
            expected["observation_hash"] = got.get("observation_sha256") or ""
            predicted = got["accepted"] and step["op"] in ("dispatch", "immutability")
            expected["dummy_id"] = "constant-dummy-v1" if predicted else ""
            expected["action"] = _safe_action(step.get("action", actual["default_action"])) if predicted else ()
            expected["decision"] = (
                "STOP" if step["op"] == "stop" or (predicted and got["code"] == "STOP")
                else "ACCEPT" if predicted else "REJECT"
            )
            expected["reason"] = (
                got["termination_reason"] or got["code"] if step["op"] == "outcome"
                else "OK" if predicted and got["code"] == "OK" else got["code"]
            )
            for key, value in expected.items():
                if record[key] != value:
                    _fail(case["id"], index, "sink_" + key, value, record[key])
            command = [record["linear"], record["angular"]]
            if not _close(step["expect"]["command"], command):
                _fail(case["id"], index, "sink_command", step["expect"]["command"], command)
    if "sink_commands" in case and not _close(
        [item for row in case["sink_commands"] for item in row],
        [item for row in actual["sink_commands"] for item in row],
    ):
        raise AssertionError(f"{case['id']} sink_commands: expected {case['sink_commands']!r} actual {actual['sink_commands']!r}")
    if "sink_commands" in case and len(case["sink_commands"]) != len(actual["sink_commands"]):
        raise AssertionError(f"{case['id']} sink length")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite output")
        return value
    raise TypeError(f"unsupported output type {type(value).__name__}")


def _run_loaded(data):
    results = []
    for case in data["cases"]:
        item = {"id": case["id"], "group": case.get("group", "")}
        try:
            actual = execute_case(case, data["base"])
            assert_case(case, actual)
            item["passed"] = True
            item["actual"] = _jsonable(actual)
        except Exception as exc:
            item["passed"] = False
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    return results


def run_cases(manifest_path):
    """Execute every manifest case and return author-check traces."""

    data = load_manifest(manifest_path)
    verify_source_pins(data["source_hashes"])
    return _run_loaded(data)


def _parse_fixtures(argv):
    if not argv or argv[0] != "fixtures":
        raise ValueError("mode")
    manifest = None
    output = None
    index = 1
    while index < len(argv):
        flag = argv[index]
        if flag not in ("--manifest", "--output") or index + 1 >= len(argv):
            raise ValueError(flag)
        if flag == "--manifest":
            manifest = argv[index + 1]
        else:
            output = argv[index + 1]
        index += 2
    if manifest is None or output is None:
        raise ValueError("missing argument")
    return manifest, output


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    """Run fixtures mode only. Any other mode is refused before work starts."""

    if argv is None:
        argv = sys.argv[1:]
    argv = [str(arg) for arg in argv]
    if not argv or argv[0] != "fixtures":
        sys.stderr.write("unsupported mode\n")
        return 2
    try:
        manifest, output = _parse_fixtures(argv)
    except ValueError as exc:
        sys.stderr.write(f"unsupported mode: {exc}\n")
        return 2
    manifest_path = Path(manifest)
    output_path = Path(output)
    if not manifest_path.is_file():
        sys.stderr.write("manifest missing\n")
        return 2
    if output_path.exists():
        sys.stderr.write("output directory exists\n")
        return 2
    try:
        data = load_manifest(manifest_path)
        verify_source_pins(data["source_hashes"])
    except (OSError, ValueError, QualificationError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"manifest rejected: {exc}\n")
        return 2
    output_path.mkdir(parents=False, exist_ok=False)
    results = _run_loaded(data)
    failed = [item["id"] for item in results if not item["passed"]]
    summary = {
        "role": "author-check",
        "case_count": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "skipped": 0,
        "failures": failed,
        "manifest_sha256": _sha256(manifest_path),
    }
    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "case_results.json", results)
    print(f"passed={summary['passed']} failed={summary['failed']} skipped=0")
    if summary["passed"] + summary["failed"] != summary["case_count"] or summary["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
