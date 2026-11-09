"""Read-only HTML status page for a Jackal navigation episode.

The page displays state and metrics. It has no control that publishes
velocity and it does not import ROS.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


class DashboardError(ValueError):
    """The status payload asked the dashboard to command the robot."""


DISPLAY_FIELDS = (
    ("state", "State"),
    ("scenario_id", "Scenario"),
    ("policy_id", "Policy"),
    ("algorithm_family", "Algorithm family"),
    ("goal_x", "Goal x"),
    ("goal_y", "Goal y"),
    ("goal_distance_m", "Goal distance (m)"),
    ("elapsed_s", "Elapsed (s)"),
    ("collision_count", "Collisions"),
    ("path_length_m", "Path length (m)"),
    ("result", "Result"),
)


def assert_read_only(status):
    if not isinstance(status, dict):
        raise DashboardError("Status must be a mapping")
    if status.get("publish_velocity") or status.get("cmd_vel") is not None:
        raise DashboardError("Dashboard cannot publish velocity")
    return status


def render_status_html(status):
    """Return a single HTML document for one status mapping."""

    assert_read_only(status)
    rows = []
    for key, label in DISPLAY_FIELDS:
        value = status.get(key, "—")
        rows.append(
            "<tr><th>"
            + html.escape(label)
            + "</th><td>"
            + html.escape(str(value))
            + "</td></tr>"
        )
    body = "\n".join(rows)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head><meta charset=\"utf-8\"><title>Jackal navigation status</title></head>\n"
        "<body>\n"
        "<h1>Jackal navigation status</h1>\n"
        "<p>Read only. This page does not publish velocity.</p>\n"
        "<table>\n"
        + body
        + "\n</table>\n"
        "</body>\n"
        "</html>\n"
    )


def write_status_html(status, output_path):
    path = Path(output_path)
    path.write_text(render_status_html(status), encoding="utf-8")
    return path


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: python -m navigation.jackal_status_dashboard STATUS.json OUTPUT.html")
    status = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    write_status_html(status, argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
