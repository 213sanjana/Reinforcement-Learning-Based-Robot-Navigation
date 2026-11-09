"""Manual emergency stop through the existing command owner.

This module does not create a velocity publisher. Without both the CLI flag
and the authorization environment variable it exits before importing ROS.
"""

from __future__ import annotations

import argparse
import os
import sys


AUTHORIZATION_ENV = "WAREHOUSE_RL_LIVE_AUTHORIZATION"
AUTHORIZATION_VALUE = "I_AUTHORIZE_JACKAL_MOTION"


def live_authorized(cli_flag: bool) -> bool:
    return bool(cli_flag) and os.environ.get(AUTHORIZATION_ENV) == AUTHORIZATION_VALUE


def stop_through_owner(owner) -> None:
    """Latch emergency stop on the object that already owns the command topic."""

    method = getattr(owner, "emergency_stop", None)
    if not callable(method):
        raise RuntimeError("Command owner has no emergency_stop method")
    method()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Latch the existing Jackal emergency stop. Does nothing without authorization.",
    )
    parser.add_argument(
        "--authorize-live",
        action="store_true",
        help="Required together with WAREHOUSE_RL_LIVE_AUTHORIZATION before any ROS node is created.",
    )
    args = parser.parse_args(argv)
    if not live_authorized(args.authorize_live):
        print(
            "Emergency stop was not sent. Refusing to create a ROS node without "
            f"--authorize-live and {AUTHORIZATION_ENV}={AUTHORIZATION_VALUE}.",
            file=sys.stderr,
        )
        return 2
    from jackal_rl_navigation.robot_interface import JackalRLInterface

    interface = JackalRLInterface()
    try:
        stop_through_owner(interface)
    finally:
        interface.destroy_node()
    print("Emergency stop latched through the existing robot interface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
