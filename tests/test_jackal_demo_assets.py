"""Static checks for demonstration worlds, launch files, and the status page."""

from navigation.jackal_demo_preflight import (
    run_static_preflight,
    validate_navigation_parameters,
)
from navigation.jackal_status_dashboard import (
    DashboardError,
    render_status_html,
    write_status_html,
)
import pytest


def test_navigation_parameters_keep_the_36_bin_contract():
    topics = validate_navigation_parameters({
        "namespace": "j100_0000",
        "lidar_bins": 36,
        "goal_x": 4.0,
        "goal_y": 0.0,
        "goal_frame": "odom",
        "sensor_timeout": 0.5,
    })
    assert topics["scan_topic"] == "/j100_0000/sensors/lidar3d_0/scan"
    assert topics["observation_dim"] == 40
    with pytest.raises(Exception):
        validate_navigation_parameters({
            "namespace": "j100_0000",
            "lidar_bins": 18,
            "goal_x": 4.0,
            "goal_y": 0.0,
            "goal_frame": "odom",
            "sensor_timeout": 0.5,
        })


def test_demo_assets_pass_the_static_preflight():
    report = run_static_preflight()
    assert report.ok
    assert len(report.checks) >= 6
    assert report.live_authorization_files
    assert any(item.endswith("jackal_demo_sim.launch.py") for item in report.live_authorization_files)


def test_dashboard_is_read_only(tmp_path):
    status = {
        "state": "NAVIGATING",
        "scenario_id": "world1_open",
        "policy_id": "ClassicalGoalBaseline",
        "algorithm_family": "classical",
        "goal_x": 4.0,
        "goal_y": 0.0,
        "goal_distance_m": 3.5,
        "elapsed_s": 1.0,
        "collision_count": 0,
        "path_length_m": 0.4,
        "result": "in progress",
    }
    page = render_status_html(status)
    assert "NAVIGATING" in page
    assert "does not publish velocity" in page
    assert "cmd_vel" not in page
    with pytest.raises(DashboardError):
        render_status_html({**status, "publish_velocity": True})
    output = write_status_html(status, tmp_path / "status.html")
    assert output.is_file()
