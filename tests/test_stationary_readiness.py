"""Offline fake-clock/state tests. No ROS imports, processes or simulation."""
import ast
from pathlib import Path
import pytest
from scripts.stationary_readiness import ReadinessGate

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts/external_warehouse_stationary_diagnostic.py'


def active():
    return {'controller':[{'name':n,'state':'active'} for n in
            ('joint_state_broadcaster','platform_velocity_controller')]}


def fresh(gate, now):
    # Two advancing samples per stream; latest acquisition stamps match /clock.
    for name in ('clock','odometry','lidar'):
        gate.ingest(name,int((now-.1)*1e9),now-.1)
        gate.ingest(name,int(now*1e9),now)


def ready(now=1, **kwargs):
    g=ReadinessGate(0,**kwargs);g.controller_response(active());fresh(g,now)
    return g


@pytest.mark.parametrize('chunks',[
    ['Configured and activated platform_velocity_controller'],
    ['Configured and activated \x1b[1mplatform_velocity_controller\x1b[0m'],
    ['Configured and act','ivated \x1b[1mplatform_','velocity_controller\x1b[0m'],
    [], ['irrelevant partial output'],
])
def test_terminal_output_cannot_establish_or_prevent_structured_readiness(chunks):
    # Output is intentionally not passed into the state API at all.
    g=ReadinessGate(0)
    assert g.evaluate(1,True)['status']=='waiting'
    g.controller_response(active());fresh(g,1)
    assert g.evaluate(1,True)['status']=='ready'
    tree=ast.parse(RUNNER.read_text())
    assert 'Configured and activated' not in RUNNER.read_text()
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('ingest','controller_response','evaluate')
        and any(isinstance(x,ast.Name) and x.id=='text' for x in ast.walk(n))
        for n in ast.walk(tree))


def test_manager_process_presence_does_not_imply_service_readiness():
    g=ReadinessGate(0,startup_timeout=2)
    state=g.evaluate(2,True)
    assert state['status']=='failed'
    assert 'controller_manager: no successful service response' in state['reason']


def test_service_response_with_inactive_or_missing_controllers_is_not_ready():
    g=ReadinessGate(0,startup_timeout=2)
    g.controller_response({'controller':[{'name':'joint_state_broadcaster','state':'inactive'}]})
    fresh(g,1)
    assert 'controllers: not active' in g.evaluate(1,True)['reason']
    assert g.evaluate(2,True)['status']=='failed'


@pytest.mark.parametrize('bad', [None, {}, {'controller':None}])
def test_malformed_service_response_fails_closed(bad):
    g=ReadinessGate(0);g.controller_response(bad)
    assert 'malformed' in g.evaluate(1,True)['reason']


def test_active_controllers_with_missing_sensor_data_wait_then_timeout():
    g=ReadinessGate(0,startup_timeout=2);g.controller_response(active())
    state=g.evaluate(2,True)
    assert state['status']=='failed'
    assert all(n+': missing' in state['reason'] for n in ('clock','odometry','lidar'))


def test_recent_arrivals_with_repeated_acquisition_stamps_are_not_fresh():
    g=ReadinessGate(0);g.controller_response(active())
    for name in ('clock','odometry','lidar'):
        for t in (.1,.2,.3):g.ingest(name,100,t)
    assert 'has not advanced' in g.evaluate(.3,True)['reason']


def test_stale_stream_and_clock_disagreement_are_reported():
    g=ready();assert 'stale' in g.evaluate(1.6,True)['reason']
    g=ready();g.ingest('lidar',10_000_000_000,1)
    assert 'disagrees with /clock' in g.evaluate(1,True)['reason']


def test_success_window_begins_at_full_readiness_not_launch_or_log_time():
    g=ready(observation_duration=1)
    assert g.evaluate(1,True)['ready_at']==1
    fresh(g,1.5);assert g.evaluate(1.5,True)['status']=='ready'
    fresh(g,2);assert g.evaluate(2,True)['status']=='complete'
    assert g.ready_at==1


def test_readiness_lost_during_observation_fails_without_extending_window():
    g=ready();g.evaluate(1,True)
    result=g.evaluate(1.6,True)
    assert result['status']=='failed' and 'readiness lost' in result['reason']
    fresh(g,2);assert g.evaluate(2,True)['status']=='failed'
    assert g.ready_at==1


def test_shutdown_while_pending_suppresses_late_service_or_sensor_success():
    g=ReadinessGate(0);g.begin_shutdown()
    g.controller_response(active());fresh(g,1)
    assert g.evaluate(1,True)['status']=='stopping'
    assert not g.manager_responded and not g.sensors and g.ready_at is None


def test_overall_deadline_and_explicit_service_failure_request_failure():
    g=ready(live_timeout=2);g.evaluate(1,True);fresh(g,2)
    assert g.evaluate(2,True)['reason']=='overall live deadline'
    g=ReadinessGate(0);g.fail('controller_manager: response timeout')
    g.controller_response(active());fresh(g,1)
    assert g.evaluate(1,True)['reason']=='controller_manager: response timeout'


def test_invalid_sensor_frame_and_time_regression_cannot_be_ready():
    g=ready();g.ingest('lidar',1_100_000_000,1.1,valid=False)
    assert 'invalid data/frame' in g.evaluate(1.1,True)['reason']
    g=ready();g.ingest('clock',0,1.1)
    assert 'backwards' in g.evaluate(1.1,True)['reason']


def test_runner_gates_execution_and_preserves_single_launch_shutdown_and_limits():
    tree=ast.parse(RUNNER.read_text());text=RUNNER.read_text()
    guard=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    assert 'rclpy.init()' in ast.unparse(guard)
    assert 'exist_ok=False' in text and '--execute-approved-stationary' in text
    assert text.count("cmd=['ros2','launch'")==1
    assert 'os.kill(launch.pid,signal.SIGINT)' in text
    assert 'now-stop_at>=15' in text and 'expected=' in text
    assert 'max(0.0,time.monotonic()-stop_at)' in text
    assert 'create_publisher' not in text and 'RobotInterface(' not in text
    assert "state['status'] in ('failed','complete')" in text
    assert "if not stopping and age>=20" in text  # no service checks during teardown
    assert "q['sent']>=5" in text and "q['start']>=3" in text
