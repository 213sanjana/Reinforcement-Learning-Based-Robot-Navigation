"""Stateless conversion of reduced, explicitly synthetic message-shaped data.

Declarations here authenticate no ROS source, clock, frame or transform. The
caller must separately qualify returned snapshots with the offline harness.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from navigation.jackal_interface_qualification import QualificationError, SensorSnapshot


@dataclass(frozen=True)
class ConversionProvenance:
    """Detached tuple tree of original fields, distinct from derived geometry."""

    original: tuple
    expanded_input_sha256: str
    scan_stamp_ns: int | None
    odom_stamp_ns: int | None


@dataclass(frozen=True)
class ConversionResult:
    converted: bool
    code: str
    snapshot: SensorSnapshot | None
    provenance: ConversionProvenance | None
    detail: str = ""
    inner_code: str | None = None


class _Invalid(Exception):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail


def _fail(code, detail):
    raise _Invalid(code, detail)


def _xyz():
    return dict.fromkeys(('x', 'y', 'z'))


def _schema():
    header = {'frame_id': None, 'stamp': {'sec': None, 'nanosec': None}}
    return {
        'identity': dict.fromkeys(('run_id', 'episode_id', 'snapshot_id', 'transition_index')),
        'scan_source': dict.fromkeys(('source_id', 'topic', 'message_type', 'origin', 'provenance')),
        'odom_source': dict.fromkeys(('source_id', 'topic', 'message_type', 'origin', 'provenance')),
        'clock': dict.fromkeys(('domain', 'epoch', 'scan_received_mono_s', 'odom_received_mono_s', 'capture_mono_s', 'dispatch_mono_s')),
        'scan': {'header': header, **dict.fromkeys(('ranges', 'angle_min', 'angle_max', 'angle_increment', 'range_min', 'range_max', 'time_increment', 'scan_time'))},
        'odom': {'header': header, 'child_frame_id': None,
                 'pose': {'pose': {'position': _xyz(), 'orientation': dict.fromkeys(('x', 'y', 'z', 'w'))}},
                 'twist': {'twist': {'linear': _xyz(), 'angular': _xyz()}}},
        'goal': dict.fromkeys(('frame_id', 'x', 'y', 'origin', 'alignment_reference')),
    }


def _check_shape(value, schema, path='', *, missing_only):
    if not isinstance(value, dict):
        if not missing_only:
            _fail('ADAPTER_SCHEMA', path + ': mapping required')
        return
    for key, child in schema.items():
        label = path + '.' + key
        if key not in value:
            _fail('ADAPTER_MISSING_FIELD', label)
        if child is not None:
            _check_shape(value[key], child, label, missing_only=missing_only)
    if not missing_only and value.keys() - schema.keys():
        _fail('ADAPTER_SCHEMA', path + ': unknown keys')


def _text(value, code, label):
    if not isinstance(value, str) or not value.strip():
        _fail(code, label)
    return value


def _real(value, code, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code, label)
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _fail(code, label)
    if not math.isfinite(number):
        _fail(code, label)
    return number


def _stamp(stamp):
    sec, nano = stamp['sec'], stamp['nanosec']
    if (type(sec) is not int or type(nano) is not int
            or not -2147483648 <= sec <= 2147483647
            or not 0 <= nano <= 999999999):
        _fail('ADAPTER_INVALID_STAMP', 'integer stamp components/ranges')
    total = sec * 1000000000 + nano
    if total <= 0:
        _fail('ADAPTER_INVALID_STAMP', 'nonpositive total stamp')
    return total


def _tokens(value):
    if isinstance(value, dict):
        return {key: _tokens(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tokens(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
    if value is None or type(value) in (str, int, float, bool):
        return value
    _fail('ADAPTER_SCHEMA', 'plain JSON-shaped values required')


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def convert_record(record, *, expected_clock_domain, expected_clock_epoch):
    """Convert one explicit pair; reject without dispatch, I/O or lifecycle work."""
    provenance = None
    try:
        schema = _schema()
        _check_shape(record, schema, missing_only=True)
        _check_shape(record, schema, missing_only=False)
        token_record = _tokens(record)
        digest = hashlib.sha256(json.dumps(token_record, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
        provenance = ConversionProvenance(_freeze(record), digest, None, None)
        identity = record['identity']
        for key in ('run_id', 'episode_id', 'snapshot_id'):
            _text(identity[key], 'ADAPTER_INVALID_IDENTITY', key)
        index = identity['transition_index']
        if type(index) is not int or index < 0:
            _fail('ADAPTER_INVALID_IDENTITY', 'transition_index')
        for name, expected_type in (('scan_source', 'sensor_msgs/msg/LaserScan'), ('odom_source', 'nav_msgs/msg/Odometry')):
            source = record[name]
            _text(source['source_id'], 'ADAPTER_INVALID_IDENTITY', name)
            _text(source['origin'], 'ADAPTER_SCHEMA', name + '.origin')
            topic = _text(source['topic'], 'ADAPTER_SCHEMA', name + '.topic')
            if not topic.startswith('/') or source['message_type'] != expected_type or source['provenance'] != 'synthetic':
                _fail('ADAPTER_SCHEMA', name + ': source declaration')
        scan, odom, clock, goal = (record[k] for k in ('scan', 'odom', 'clock', 'goal'))
        for value in (scan['header']['frame_id'], odom['header']['frame_id'], odom['child_frame_id'], goal['frame_id'], goal['origin'], goal['alignment_reference']):
            _text(value, 'ADAPTER_SCHEMA', 'frame/origin/alignment declaration')
        scan_stamp, odom_stamp = _stamp(scan['header']['stamp']), _stamp(odom['header']['stamp'])
        provenance = ConversionProvenance(provenance.original, digest, scan_stamp, odom_stamp)
        for key in ('domain', 'epoch'):
            _text(clock[key], 'ADAPTER_INVALID_CLOCK', key)
        if (clock['domain'] != expected_clock_domain or clock['epoch'] != expected_clock_epoch
                or not isinstance(expected_clock_domain, str) or not expected_clock_domain.strip()
                or not isinstance(expected_clock_epoch, str) or not expected_clock_epoch.strip()):
            _fail('ADAPTER_CLOCK_EPOCH', 'explicit clock declaration mismatch')
        for key in ('scan_received_mono_s', 'odom_received_mono_s', 'capture_mono_s', 'dispatch_mono_s'):
            if _real(clock[key], 'ADAPTER_INVALID_CLOCK', key) < 0:
                _fail('ADAPTER_INVALID_CLOCK', key)
        if (clock['scan_received_mono_s'] > clock['capture_mono_s']
                or clock['odom_received_mono_s'] > clock['capture_mono_s']
                or clock['capture_mono_s'] > clock['dispatch_mono_s']):
            _fail('ADAPTER_INVALID_CLOCK', 'receive <= capture <= dispatch required')
        if not isinstance(scan['ranges'], list):
            _fail('ADAPTER_SCHEMA', 'ranges must be a one-dimensional list')
        for value in scan['ranges']:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail('ADAPTER_SCHEMA', 'range must be a real scalar')
            try:
                float(value)
            except OverflowError:
                _fail('ADAPTER_SCHEMA', 'range not representable')
        for key in ('angle_min', 'angle_max', 'angle_increment', 'range_min', 'range_max', 'time_increment', 'scan_time'):
            _real(scan[key], 'ADAPTER_INVALID_SCALAR', 'scan.' + key)
        if (scan['angle_increment'] <= 0 or scan['angle_max'] < scan['angle_min']
                or not 0 <= scan['range_min'] < scan['range_max']
                or scan['time_increment'] < 0 or scan['scan_time'] < 0):
            _fail('ADAPTER_INVALID_SCALAR', 'scan metadata bounds')
        pose, twist = odom['pose']['pose'], odom['twist']['twist']
        q = pose['orientation']
        for key in ('x', 'y', 'z', 'w'):
            _real(q[key], 'ADAPTER_INVALID_QUATERNION', 'orientation.' + key)
        norm2 = sum(float(q[key]) * float(q[key]) for key in ('x', 'y', 'z', 'w'))
        if not math.isfinite(norm2) or norm2 == 0 or abs(norm2 - 1) > 1e-6:
            _fail('ADAPTER_INVALID_QUATERNION', 'unit quaternion required without normalization')
        for group in (pose['position'], twist['linear'], twist['angular']):
            for key in ('x', 'y', 'z'):
                _real(group[key], 'ADAPTER_INVALID_SCALAR', key)
        for key in ('x', 'y'):
            _real(goal[key], 'ADAPTER_INVALID_SCALAR', 'goal.' + key)
        if pose['position']['z'] != 0 or q['x'] != 0 or q['y'] != 0:
            _fail('ADAPTER_NONPLANAR', 'planar pose required')
        if goal['frame_id'] != odom['header']['frame_id']:
            _fail('ADAPTER_TRANSFORM_UNAVAILABLE', 'only same-frame goals supported')
        if goal['alignment_reference'] != 'none':
            _fail('ADAPTER_SCHEMA', 'same-frame alignment_reference must be none')
        yaw = math.atan2(2 * q['w'] * q['z'], 1 - 2 * q['z'] ** 2)
        dx, dy = goal['x'] - pose['position']['x'], goal['y'] - pose['position']['y']
        distance = math.hypot(dx, dy)
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
        _real(distance, 'ADAPTER_INVALID_SCALAR', 'derived distance')
        raw = {**identity, 'scan_source_id': record['scan_source']['source_id'], 'odom_source_id': record['odom_source']['source_id'],
               'scan_frame': scan['header']['frame_id'], 'odom_frame': odom['header']['frame_id'], 'child_frame': odom['child_frame_id'],
               'ranges': scan['ranges'], 'distance_m': distance, 'bearing_rad': bearing,
               'linear_mps': twist['linear']['x'], 'angular_rps': twist['angular']['z'],
               'scan_stamp_ns': scan_stamp, 'odom_stamp_ns': odom_stamp,
               'scan_received_mono_s': clock['scan_received_mono_s'], 'odom_received_mono_s': clock['odom_received_mono_s'],
               'capture_mono_s': clock['capture_mono_s'], 'angle_min_rad': scan['angle_min'], 'angle_increment_rad': scan['angle_increment']}
        try:
            snapshot = SensorSnapshot.from_fixture(raw)
        except QualificationError as exc:
            return ConversionResult(False, 'ADAPTER_SNAPSHOT_REJECTED', None, provenance, str(exc), exc.code)
        return ConversionResult(True, 'ADAPTER_OK', snapshot, provenance)
    except _Invalid as exc:
        return ConversionResult(False, exc.code, None, provenance, exc.detail)
