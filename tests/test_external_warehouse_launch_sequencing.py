"""Offline launch callbacks/configuration only; no launch service or process runs."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from launch.actions import EmitEvent, LogInfo
from launch.events import Shutdown

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_external_warehouse.launch.py'
spec = importlib.util.spec_from_file_location('external_launch_offline', PATH)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)  # _launch_setup is deliberately never executed.


@pytest.mark.parametrize('stage', ['robot spawn', 'joint-state broadcaster', 'velocity controller'])
@pytest.mark.parametrize('code', [1, -2, -9])
def test_failure_requests_shutdown_instead_of_starting_next_stage(stage, code):
    next_action = object()
    actions = launch._after_success(stage, next_action)(
        SimpleNamespace(returncode=code), SimpleNamespace(is_shutdown=False))
    assert len(actions) == 2
    assert isinstance(actions[0], LogInfo)
    assert isinstance(actions[1], EmitEvent)
    assert isinstance(actions[1].event, Shutdown)
    assert stage in actions[1].event.reason
    assert str(code) in actions[1].event.reason
    assert next_action not in actions


def test_success_alone_permits_next_stage_and_final_success_has_no_action():
    event = SimpleNamespace(returncode=0); context = SimpleNamespace(is_shutdown=False)
    next_action = object()
    assert launch._after_success('spawn', next_action)(event, context) == [next_action]
    assert launch._after_success('velocity')(event, context) == []


@pytest.mark.parametrize('code', [0, 1, -2, -9])
def test_shutdown_never_starts_another_stage_or_reemits_shutdown(code):
    event = SimpleNamespace(returncode=code); context = SimpleNamespace(is_shutdown=True)
    assert launch._after_success('spawn', object())(event, context) == []
    assert launch._on_server_exit(event, context) == []


@pytest.mark.parametrize('code', [0, 1, -9])
def test_unexpected_server_exit_requests_launch_shutdown(code):
    actions = launch._on_server_exit(SimpleNamespace(returncode=code), SimpleNamespace(is_shutdown=False))
    assert isinstance(actions[1].event, Shutdown)


@pytest.mark.parametrize('controller', ['joint_state_broadcaster', 'platform_velocity_controller'])
def test_readiness_namespace_and_response_timeouts(controller):
    args = launch._controller_spawner_arguments(controller, 'j100_0000')
    assert args == [controller, '--controller-manager', '/j100_0000/controller_manager',
                    '--controller-manager-timeout', '90', '--service-call-timeout', '10',
                    '--switch-timeout', '5']


def test_handlers_are_wired_to_success_callbacks_for_all_three_stages():
    tree = ast.parse(PATH.read_text())
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'OnProcessExit']
    targets = {}
    for handler in handlers:
        kwargs = {k.arg:k.value for k in handler.keywords}
        targets[kwargs['target_action'].id] = kwargs['on_exit']
    for target in ('spawn_robot', 'joint_broadcaster', 'velocity_controller'):
        assert isinstance(targets[target], ast.Call)
        assert targets[target].func.id == '_after_success'
    assert targets['spawn_robot'].args[1].id == 'joint_broadcaster'
    assert targets['joint_broadcaster'].args[1].id == 'velocity_controller'
    assert targets['gazebo'].id == '_on_server_exit'


def test_bridges_and_stationary_branch_have_no_new_command_publisher():
    text = PATH.read_text(); tree = ast.parse(text)
    for topic, msg in [('/clock', 'Clock'), ('/jackal_rl/obstacle_contacts', 'Contacts')]:
        assert topic in text and f'[ignition.msgs.{msg}' in text
    assert 'sensors/lidar3d_0/scan' in text and '[ignition.msgs.LaserScan' in text
    assert 'create_publisher' not in text and 'initialized_ppo.zip' not in text
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "mode == 'simulation_only'")
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('Node', 'IncludeLaunchDescription', 'ExecuteProcess') for stmt in branch.body for n in ast.walk(stmt))
    assert 'default_value="simulation_only"' in text
    assert 'default_value="true"' in text
