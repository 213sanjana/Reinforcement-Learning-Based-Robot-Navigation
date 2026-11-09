"""Synthetic, memory-only adapter contract; all serialization is test-local."""
import ast
import copy
from dataclasses import asdict, FrozenInstanceError, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from navigation.jackal_interface_qualification import (
    CapturedCommandSink, ConstantDummyPredictor, OfflineHarness,
    SensorSnapshot, decode_fixture_token,
)
from navigation.jackal_ros_message_contract import convert_record

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/jackal_ros_message_contract_v1.json'
CONVERTER = ROOT / 'navigation/jackal_ros_message_contract.py'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key: ' + key)
        result[key] = value
    return result


def _constant(name):
    raise ValueError('nonstandard numeric ' + name)


def _load(text):
    data = json.loads(text, object_pairs_hook=_unique, parse_constant=_constant)
    assert set(data) == {'schema_version', 'provenance', 'source_pins', 'base_record', 'cases'}
    assert type(data['schema_version']) is int and data['schema_version'] == 1
    assert data['provenance'] == 'synthetic'
    assert [c['id'] for c in data['cases']] == [f'C{i:03d}' for i in range(1, 25)]
    for case in data['cases']:
        assert case['steps']
        for step in case['steps']:
            assert step['operation'] in ('convert_dispatch', 'convert_only')
            assert type(step['fresh_harness']) is bool
            assert step['expected']['attempted'] and step['expected']['completed']
    return data


DATA = _load(FIXTURE.read_text(encoding='utf-8'))


def _encode(value):
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
    return value


def _canonical(value):
    return json.dumps(_encode(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _expanded(step):
    record = copy.deepcopy(DATA['base_record'])
    for path, value in step['overrides'].items():
        keys = path.split('.'); obj = record
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = copy.deepcopy(value)
    for path in step['delete_paths']:
        keys = path.split('.'); obj = record
        for key in keys[:-1]:
            obj = obj[key]
        del obj[keys[-1]]
    digest = hashlib.sha256(_canonical(record).encode('utf-8')).hexdigest()
    # Only exact range-position tokens are decoded; no generic coercion.
    if isinstance(record.get('scan', {}).get('ranges'), list):
        record['scan']['ranges'] = [decode_fixture_token(v) if isinstance(v, str) and v in ('NaN', '+Inf', '-Inf') else v for v in record['scan']['ranges']]
    return record, digest


def _convert(record):
    return convert_record(record, expected_clock_domain='synthetic_monotonic', expected_clock_epoch='epoch-A')


def _base():
    return copy.deepcopy(DATA['base_record'])


def _frozen(value):
    if isinstance(value, dict):
        return tuple((k, _frozen(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


def _state(h):
    return (h._episode_id, h._episode_identity_unknown, h._latch,
            frozenset(h._failed_episode_ids), h._last_scan_stamp,
            h._last_odom_stamp, h._last_dispatch_mono_s, h._last_snapshot,
            h._mapper.previous, h.sink.records)


def _accepted_history(h):
    if h._last_dispatch_mono_s is None:
        return None
    return dict(scan_stamp_ns=h._last_scan_stamp, odom_stamp_ns=h._last_odom_stamp,
                dispatch_mono_s=h._last_dispatch_mono_s)


def _records(records):
    return _encode([asdict(r) for r in records])


def _assert_records(actual, expected):
    # Exact fields, sequence, associations and retained prefix, not just count.
    assert actual == expected


def _assert_snapshot(snapshot, record, expected):
    assert type(snapshot) is SensorSnapshot
    for key in ('run_id', 'episode_id', 'snapshot_id', 'transition_index'):
        assert getattr(snapshot, key) == record['identity'][key]
    for field, value in {
        'scan_source_id': record['scan_source']['source_id'],
        'odom_source_id': record['odom_source']['source_id'],
        'scan_frame': record['scan']['header']['frame_id'],
        'odom_frame': record['odom']['header']['frame_id'],
        'child_frame': record['odom']['child_frame_id'],
        'angle_min_rad': record['scan']['angle_min'],
        'angle_increment_rad': record['scan']['angle_increment'],
        'linear_mps': record['odom']['twist']['twist']['linear']['x'],
        'angular_rps': record['odom']['twist']['twist']['angular']['z'],
        **{k: record['clock'][k] for k in ('scan_received_mono_s', 'odom_received_mono_s', 'capture_mono_s')},
    }.items():
        assert getattr(snapshot, field) == value
    assert _encode(snapshot.ranges) == _encode(record['scan']['ranges'])
    for stream in ('scan', 'odom'):
        stamp = record[stream]['header']['stamp']
        assert getattr(snapshot, stream + '_stamp_ns') == stamp['sec'] * 1000000000 + stamp['nanosec']
    obs = expected['observation']
    if isinstance(obs, dict) and obs['valid']:
        assert snapshot.distance_m == pytest.approx(obs['vector'][36] * 20, abs=1e-6, rel=0)
        assert snapshot.bearing_rad == pytest.approx(obs['vector'][37] * math.pi, abs=1e-6, rel=0)


_ANALYTIC_HASH = '7167efd3309991bbc6b4ea784ecee4044cebcce4d290010227e50884ff793839'


def _analytic_vector():
    # Fixed little-endian float32 oracle: 36 ones, then float32(0.1), 0, 0, 0.
    vector = np.array([1.0] * 36 + [float(np.float32(0.1)), 0.0, 0.0, 0.0], dtype=np.float32)
    assert hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest() == _ANALYTIC_HASH
    return vector


@pytest.mark.parametrize('case', DATA['cases'], ids=[c['id'] for c in DATA['cases']])
def test_conversion_case(case, tmp_path):
    observed = {'id': case['id'], 'completed': False, 'steps': []}
    h = predictor = None
    try:
        for index, step in enumerate(case['steps']):
            if step['fresh_harness']:
                h, predictor = OfflineHarness(), ConstantDummyPredictor()
            assert h is not None
            record, digest = _expanded(step)
            before, prefix, calls = _state(h), h.sink.records, predictor.calls
            actual = dict(index=index, attempted=True, completed=False, expanded_input_sha256=digest)
            observed['steps'].append(actual)
            conversion = _convert(record)
            actual.update(converter_status=conversion.converted, converter_code=conversion.code,
                          snapshot_present=conversion.snapshot is not None,
                          detail=conversion.detail, inner_code=conversion.inner_code)
            expected = step['expected']
            for key in ('converter_status', 'converter_code', 'snapshot_present'):
                assert actual[key] == expected[key]
            if conversion.provenance is not None:
                assert conversion.provenance.expanded_input_sha256 == digest
                assert _canonical(conversion.provenance.original) == _canonical(_frozen(record))
            if conversion.converted:
                _assert_snapshot(conversion.snapshot, record, expected)
            if conversion.converted and step['operation'] == 'convert_dispatch':
                result = h.dispatch(conversion.snapshot, record['clock']['dispatch_mono_s'], predictor=predictor)
                actual.update(qualifier_status=result.accepted, qualifier_code=result.code,
                              observation='not_produced' if result.observation_bytes is None else dict(
                                  valid=result.observation_valid, reason=result.observation_reason,
                                  shape=list(result.observation_shape), dtype=result.observation_dtype,
                                  vector=list(result.observation_values), sha256=result.observation_sha256),
                              captured=result.captured, command=result.command)
                assert result.latched == (expected['latch'] is not None)
                if result.observation_bytes is not None:
                    view = result.observation_view()
                    assert view.shape == (40,) and view.dtype == np.dtype('float32')
                    assert np.isfinite(view).all() and not view.flags.writeable
                    assert result.observation_sha256 == hashlib.sha256(result.observation_bytes).hexdigest()
                if not result.accepted:
                    assert result.command == (0.0, 0.0)
                    # Only the dispatch-stale C015 step predicts before rejection.
                    if (case['id'], index) == ('C015', 2):
                        clock = record['clock']
                        assert step['fresh_harness'] is True
                        assert clock['scan_received_mono_s'] == 9.5
                        assert clock['odom_received_mono_s'] == 9.5
                        assert clock['capture_mono_s'] == 10.0
                        assert clock['dispatch_mono_s'] == 10.000001
                        assert expected['qualifier_code'] == 'STALE_INPUT'
                        assert expected['predictor_call_delta'] == 1
                        assert result.code == 'STALE_INPUT'
                        assert result.observation_valid is True and result.observation_bytes is not None
                        view = result.observation_view()
                        assert view.shape == (40,) and view.dtype == np.dtype('float32')
                        assert np.isfinite(view).all() and not view.flags.writeable
                        np.testing.assert_allclose(view, _analytic_vector(), atol=1e-6, rtol=0)
                        assert result.observation_sha256 == _ANALYTIC_HASH
                    else:
                        assert expected['predictor_call_delta'] == 0
                        assert predictor.calls - calls == 0
                else:
                    assert len(result.command) == 2 and all(math.isfinite(v) for v in result.command)
                    assert -.15 <= result.command[0] <= .3 and abs(result.command[1]) <= .6
                assert result.predictor_calls == predictor.calls - calls == expected['predictor_call_delta']
            else:
                actual.update(qualifier_status='not_called', qualifier_code='not_called', observation='not_produced')
                assert _state(h) == before
            actual.update(predictor_call_delta=predictor.calls-calls,
                          appended_sink_count=len(h.sink.records)-len(prefix),
                          appended_sink_records=_records(h.sink.records[len(prefix):]),
                          retained_history=_records(h.sink.records), accepted_history=_accepted_history(h),
                          latch=h._latch, reset='not_called')
            assert h.sink.records[:len(prefix)] == prefix
            for key in ('qualifier_status', 'qualifier_code', 'predictor_call_delta', 'appended_sink_count', 'accepted_history', 'latch', 'reset'):
                assert actual[key] == expected[key], (case['id'], index, key, actual[key], expected[key])
            for key in ('appended_sink_records', 'retained_history'):
                _assert_records(actual[key], expected[key])
            if isinstance(expected['observation'], dict):
                for key in ('valid', 'reason', 'shape', 'dtype', 'sha256'):
                    assert actual['observation'][key] == expected['observation'][key]
                np.testing.assert_allclose(actual['observation']['vector'], expected['observation']['vector'], atol=1e-6, rtol=0)
            else:
                assert actual['observation'] == expected['observation']
            actual['completed'] = True
        observed['completed'] = True
    finally:
        with (tmp_path / (case['id'] + '_results.json')).open('x', encoding='utf-8') as output:
            json.dump(_encode(observed), output, indent=2, allow_nan=False)


def test_S01_fixture_loader_pins_and_rejections():
    for relative, digest in DATA['source_pins'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError):
        _load('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError):
        _load('{"bad":NaN}')
    changed = copy.deepcopy(DATA); changed['cases'][0]['steps'][0]['operation'] = 'runtime'
    with pytest.raises(AssertionError):
        _load(json.dumps(changed))


def test_S02_deep_detach_determinism_and_readonly():
    record = _base(); first = _convert(record)
    assert first.converted
    pinned = _canonical(asdict(first))
    second = _convert(copy.deepcopy(record))
    assert _canonical(asdict(second)) == pinned
    record['scan']['ranges'][0] = 0
    record['scan']['header']['frame_id'] = 'mutated'
    record['odom']['pose']['pose']['position']['x'] = 999
    record['odom']['pose']['pose']['orientation']['w'] = 0
    record['odom']['twist']['twist']['linear']['y'] = 999
    record['clock']['scan_received_mono_s'] = 0
    record['goal']['origin'] = 'mutated'
    record['scan_source']['origin'] = 'mutated'
    assert _canonical(asdict(first)) == pinned
    with pytest.raises(FrozenInstanceError):
        first.code = 'changed'
    with pytest.raises(FrozenInstanceError):
        first.provenance.expanded_input_sha256 = 'changed'
    with pytest.raises(FrozenInstanceError):
        first.snapshot.ranges = ()
    prepared = OfflineHarness().prepare(first.snapshot, 10)
    with pytest.raises(ValueError):
        prepared.readonly_vector()[0] = 0
    assert _encode(first.provenance.original) == _encode(_frozen(_base()))


def test_S03_original_receive_no_later_cache_rescue():
    record = _base(); record['clock']['scan_received_mono_s'] = 9.499999
    converted = _convert(record); h = OfflineHarness(); predictor = ConstantDummyPredictor()
    class Unread:
        def __getattribute__(self, name):
            raise AssertionError('later cache consulted')
    record['clock']['scan_received_mono_s'] = 10
    result = h.dispatch(converted.snapshot, 10, predictor=predictor, unrelated_cache=Unread())
    assert result.code == 'STALE_INPUT' and predictor.calls == 0
    assert len(h.sink.records) == 1 and result.command == (0, 0)


@pytest.mark.parametrize('kw', [{'snapshot_id': 'wrong'}, {'observation_hash': '0'*64}])
def test_S04_wrong_action_association(kw):
    h = OfflineHarness(); p = ConstantDummyPredictor(**kw)
    r = h.dispatch(_convert(_base()).snapshot, 10, predictor=p)
    assert r.code == 'ASSOCIATION_MISMATCH' and r.command == (0, 0)
    assert len(h.sink.records) == 1 and p.calls == 1
    calls = p.calls
    later = h.dispatch(_convert(_base()).snapshot, 10.1, predictor=p)
    assert later.latched and p.calls == calls and later.command == (0, 0)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'wrong', 'prefix', 'hash'])
def test_S05_comparator_detects_record_corruption(fault):
    expected = DATA['cases'][0]['steps'][0]['expected']['retained_history']
    actual = copy.deepcopy(expected)
    if fault == 'missing': actual.pop()
    elif fault == 'extra': actual.append(copy.deepcopy(actual[0]))
    elif fault == 'wrong': actual[0]['linear'] = .2
    elif fault == 'prefix': actual[0]['episode_id'] = 'rewrite'
    else: actual[0]['observation_hash'] = 'wrong'
    with pytest.raises(AssertionError):
        _assert_records(actual, expected)


def test_S06_failed_identity_reset_and_unknown_lockout():
    h = OfflineHarness(); r = _base(); r['scan']['header']['frame_id'] = 'wrong'
    s = _convert(r).snapshot
    assert h.dispatch(s, 10).code == 'FRAME_MISMATCH'
    before = _state(h)
    assert h.reset_episode('episode-A').code == 'EPISODE_REARM_REJECTED'
    assert _state(h) == before
    assert h.reset_episode('episode-B').accepted
    before = _state(h)
    assert h.reset_episode('episode-A').code == 'EPISODE_REARM_REJECTED'
    assert _state(h) == before
    # Existing raw seam only: converter deliberately rejects this input earlier.
    unknown = OfflineHarness(); raw = asdict(_convert(_base()).snapshot); raw['episode_id'] = None
    assert not unknown.dispatch(raw, 10).accepted
    before = _state(unknown)
    assert unknown.reset_episode('guessed-new').code == 'EPISODE_IDENTITY_UNKNOWN'
    assert _state(unknown) == before


def test_S07_sink_failure_retains_history_no_delivery_or_later_prediction():
    h = OfflineHarness(); p = ConstantDummyPredictor(); r = _base()
    assert h.dispatch(_convert(r).snapshot, 10, predictor=p).accepted
    prefix = h.sink.records
    h.sink._fault = RuntimeError('synthetic memory sink failure')
    r['scan']['header']['stamp']['nanosec'] = 2
    r['odom']['header']['stamp']['nanosec'] = 2
    r['clock']['capture_mono_s'] = r['clock']['dispatch_mono_s'] = 10.1
    result = h.dispatch(_convert(r).snapshot, 10.1, predictor=p)
    assert result.code == 'SINK_FAILED' and result.latched
    assert not result.captured and result.command is None and h.sink.records == prefix
    assert h._last_scan_stamp == 1000000001 and h._last_odom_stamp == 1000000001
    calls = p.calls
    later = h.dispatch(_convert(r).snapshot, 10.2, predictor=p)
    assert p.calls == calls and not later.captured and later.command is None
    assert h.sink.records == prefix


def test_S08_converter_rejection_has_no_harness_side_effect_or_rearm():
    h = OfflineHarness(); p = ConstantDummyPredictor()
    assert h.dispatch(_convert(_base()).snapshot, 10, predictor=p).accepted
    before = _state(h); calls = p.calls
    raw = _base(); raw['clock']['epoch'] = 'epoch-B'
    assert _convert(raw).code == 'ADAPTER_CLOCK_EPOCH'
    assert _state(h) == before and p.calls == calls
    assert h._latch is None  # Rejection does not falsely claim to latch a harness.


@pytest.mark.parametrize('path,code', [
    ('goal.x', 'ADAPTER_INVALID_SCALAR'),
    ('odom.twist.twist.linear.x', 'ADAPTER_INVALID_SCALAR'),
    ('scan.angle_min', 'ADAPTER_INVALID_SCALAR'),
    ('odom.pose.pose.orientation.z', 'ADAPTER_INVALID_QUATERNION'),
    ('clock.capture_mono_s', 'ADAPTER_INVALID_CLOCK'),
])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_S09_nonfinite_state_metadata_rejected(path, code, value):
    raw = _base(); keys = path.split('.'); obj = raw
    for key in keys[:-1]: obj = obj[key]
    obj[keys[-1]] = value
    result = _convert(raw)
    assert result.code == code and result.snapshot is None


def test_S10_schema_precedence_and_scalar_edges():
    raw = _base(); del raw['scan']['header']['stamp']; raw['goal']['x'] = True
    assert _convert(raw).code == 'ADAPTER_MISSING_FIELD'
    raw = _base(); raw['odom']['pose']['pose']['orientation']['z'] = 1e300
    assert _convert(raw).code == 'ADAPTER_INVALID_QUATERNION'
    raw = _base(); raw['goal']['x'] = 1e308; raw['odom']['pose']['pose']['position']['x'] = -1e308
    assert _convert(raw).code == 'ADAPTER_INVALID_SCALAR'
    raw = _base(); raw['goal']['x'] = raw['goal']['y'] = 0.0
    result = _convert(raw)
    assert result.converted and result.snapshot.distance_m == 0 and result.snapshot.bearing_rad == 0
    for field in ('identity', 'scan', 'clock', 'goal'):
        raw = _base(); raw[field]['unknown'] = 0
        assert _convert(raw).code == 'ADAPTER_SCHEMA'
    raw = _base(); raw['scan']['ranges'] = []
    converted = _convert(raw)
    assert converted.converted
    assert OfflineHarness().dispatch(converted.snapshot, 10).code == 'GEOMETRY_MISMATCH'


def test_S11_import_ast_exact_boundary():
    tree = ast.parse(CONVERTER.read_text(encoding='utf-8'))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom): imports.add(node.module)
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else '')
            assert name not in {'open', 'read_text', 'read_bytes', 'write_text', 'write_bytes', '__import__', 'eval', 'exec', 'sleep', 'Popen', 'system', 'socket', 'load'}
    assert imports == {'dataclasses', 'hashlib', 'json', 'math', 'navigation.jackal_interface_qualification'}
    assert not any(isinstance(n, (ast.With, ast.AsyncFunctionDef, ast.Await)) for n in ast.walk(tree))


def test_S12_bounded_import_only_guard():
    # One bounded child, no application entrypoint; operations guarded before import.
    script = r'''
import importlib.abc, sys, os, socket, subprocess, time, threading, json
import numpy
banned = ('rclpy','geometry_msgs','sensor_msgs','nav_msgs','std_msgs','torch','stable_baselines3','gym','gymnasium','navigation.jackal_bc','navigation.jackal_ppo','navigation.live','navigation.guarded','jackal_rl_navigation.gazebo','jackal_rl_navigation.robot_interface','jackal_rl_navigation.collision_monitor','jackal_rl_navigation.safety_monitor','jackal_rl_navigation.world_pose')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('ros_gz') or any(fullname == p or fullname.startswith(p + '.') or fullname.startswith(p + '_') for p in banned):
            raise ImportError('banned import ' + fullname)
sys.meta_path.insert(0, Guard())
def denied(*args, **kwargs):
    raise RuntimeError('forbidden runtime operation')
for module, names in [(subprocess, ('Popen','run','call','check_call','check_output')), (os, ('system','fork','forkpty','posix_spawn','posix_spawnp','spawnl','spawnv','execv','execve')), (time, ('sleep',)), (socket, ('socket','create_connection')), (threading.Thread, ('start',))]:
    for name in names:
        if hasattr(module, name): setattr(module, name, denied)
phase = ['import']
def audit(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.exec', 'os.spawn', 'os.fork', 'os.system')):
        denied()
    if event == 'open':
        path, mode, flags = args
        path = str(path)
        if phase[0] == 'convert' or '/models/' in path or path.endswith(('.pt','.pth','.zip','.pkl','.npz')):
            denied()
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            denied()
sys.addaudithook(audit)
from navigation.jackal_ros_message_contract import convert_record
phase[0] = 'convert'
record = json.loads(RECORD_JSON)
r = convert_record(record, expected_clock_domain='synthetic_monotonic', expected_clock_epoch='epoch-A')
assert r.converted
for name in sys.modules:
    assert not name.startswith('ros_gz')
    assert not any(name == p or name.startswith(p + '.') or name.startswith(p + '_') for p in banned)
# Verify the guard itself intercepts an attempted forbidden operation.
try: time.sleep(0)
except RuntimeError: pass
else: raise AssertionError('operation guard inactive')
print('guarded-import-and-conversion-ok')
'''.replace('RECORD_JSON', repr(json.dumps(DATA['base_record'], allow_nan=False)))
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert 'guarded-import-and-conversion-ok' in result.stdout


def _clock_from_provenance(provenance, key):
    return dict(dict(provenance.original)['clock'])[key]


@pytest.mark.parametrize('boundary', ['inclusive_accept', 'prepare_stale', 'dispatch_stale'])
def test_S13_freshness_boundary(boundary):
    # Fixed synthetic clocks. Outcomes are independent of the edited C015 expected object.
    spec = {
        'inclusive_accept': (10.0, 10.0, 'S13-accept'),
        'prepare_stale': (10.000001, 10.000001, 'S13-prepare'),
        'dispatch_stale': (10.0, 10.000001, 'S13-dispatch'),
    }[boundary]
    capture, dispatch, snapshot_id = spec
    record = _base()
    record['identity']['snapshot_id'] = snapshot_id
    record['clock']['scan_received_mono_s'] = 9.5
    record['clock']['odom_received_mono_s'] = 9.5
    record['clock']['capture_mono_s'] = capture
    record['clock']['dispatch_mono_s'] = dispatch
    converted = _convert(record)
    assert converted.converted and converted.code == 'ADAPTER_OK' and converted.snapshot is not None
    assert converted.snapshot.scan_received_mono_s == 9.5
    assert converted.snapshot.odom_received_mono_s == 9.5
    assert converted.snapshot.capture_mono_s == capture
    assert _clock_from_provenance(converted.provenance, 'scan_received_mono_s') == 9.5
    assert _clock_from_provenance(converted.provenance, 'odom_received_mono_s') == 9.5
    assert _clock_from_provenance(converted.provenance, 'capture_mono_s') == capture
    harness, predictor = OfflineHarness(), ConstantDummyPredictor()
    result = harness.dispatch(converted.snapshot, dispatch, predictor=predictor)
    analytic = _analytic_vector()
    if boundary == 'inclusive_accept':
        assert result.accepted and result.code == 'OK' and result.latched is False
        assert result.predictor_calls == predictor.calls == 1
        assert result.command[0] == pytest.approx(0.03, abs=1e-6, rel=0)
        assert result.command[1] == pytest.approx(-0.06, abs=1e-6, rel=0)
        view = result.observation_view()
        assert result.observation_valid is True and view.shape == (40,) and view.dtype == np.dtype('float32')
        assert np.isfinite(view).all() and not view.flags.writeable
        np.testing.assert_allclose(view, analytic, atol=1e-6, rtol=0)
        assert result.observation_sha256 == _ANALYTIC_HASH
        assert len(harness.sink.records) == 1
        saved = harness.sink.records[0]
        assert saved.sequence == 1 and saved.run_id == 'run-A' and saved.episode_id == 'episode-A'
        assert saved.snapshot_id == snapshot_id and saved.observation_hash == _ANALYTIC_HASH
        assert saved.dummy_id == 'constant-dummy-v1' and saved.action == (0.5, -0.5)
        assert saved.linear == pytest.approx(0.03, abs=1e-6, rel=0)
        assert saved.angular == pytest.approx(-0.06, abs=1e-6, rel=0)
        assert saved.monotonic_s == 10.0 and saved.decision == 'ACCEPT' and saved.reason == 'OK'
        assert _accepted_history(harness) == dict(
            scan_stamp_ns=1000000001, odom_stamp_ns=1000000001, dispatch_mono_s=10.0)
        assert harness._latch is None
        return
    assert result.accepted is False and result.code == 'STALE_INPUT' and result.latched is True
    assert result.command == (0.0, 0.0) and harness._latch == 'STALE_INPUT'
    assert len(harness.sink.records) == 1 and _accepted_history(harness) is None
    saved = harness.sink.records[0]
    assert saved.sequence == 1 and saved.run_id == 'run-A' and saved.episode_id == 'episode-A'
    assert saved.snapshot_id == snapshot_id and saved.dummy_id == '' and saved.action == ()
    assert saved.linear == 0.0 and saved.angular == 0.0 and saved.monotonic_s == dispatch
    assert saved.decision == 'REJECT' and saved.reason == 'STALE_INPUT'
    if boundary == 'prepare_stale':
        assert result.predictor_calls == predictor.calls == 0
        assert result.observation_bytes is None and saved.observation_hash == ''
        return
    assert result.predictor_calls == predictor.calls == 1
    view = result.observation_view()
    assert result.observation_valid is True and view.shape == (40,) and view.dtype == np.dtype('float32')
    assert np.isfinite(view).all() and not view.flags.writeable
    np.testing.assert_allclose(view, analytic, atol=1e-6, rtol=0)
    assert result.observation_sha256 == saved.observation_hash == _ANALYTIC_HASH
    prepared_harness, unused = OfflineHarness(), ConstantDummyPredictor()
    prepared = prepared_harness.prepare(converted.snapshot, dispatch)
    assert unused.calls == 0 and prepared_harness.sink.records == () and prepared_harness._latch is None
    assert prepared.valid is True and prepared.reason == '' and prepared.sha256 == _ANALYTIC_HASH
    prepared_view = prepared.readonly_vector()
    assert prepared_view.shape == (40,) and prepared_view.dtype == np.dtype('float32')
    assert np.isfinite(prepared_view).all() and not prepared_view.flags.writeable
    np.testing.assert_allclose(prepared_view, analytic, atol=1e-6, rtol=0)
    before_calls = predictor.calls
    first = asdict(saved)
    later = harness.dispatch(converted.snapshot, 10.1, predictor=predictor)
    assert later.latched and later.code == 'STALE_INPUT' and later.command == (0.0, 0.0)
    assert later.predictor_calls == 0 and predictor.calls == before_calls
    assert len(harness.sink.records) == 2 and asdict(harness.sink.records[0]) == first
    added = harness.sink.records[1]
    assert added.sequence == 2 and added.linear == 0.0 and added.angular == 0.0
    assert added.action == () and added.dummy_id == '' and added.observation_hash == ''
    assert added.decision == 'REJECT' and added.reason == 'STALE_INPUT' and added.monotonic_s == 10.1
    assert _accepted_history(harness) is None
