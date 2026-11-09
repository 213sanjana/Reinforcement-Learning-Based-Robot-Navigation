"""Goal navigation states for the offline Jackal manager.

Terminal states do not produce another motion command until ``reset``.
"""

from __future__ import annotations

from enum import Enum


class NavState(Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    NAVIGATING = "NAVIGATING"
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION = "COLLISION"
    TIMEOUT = "TIMEOUT"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SENSOR_ERROR = "SENSOR_ERROR"
    POLICY_ERROR = "POLICY_ERROR"


TERMINAL_STATES = frozenset({
    NavState.GOAL_REACHED,
    NavState.COLLISION,
    NavState.TIMEOUT,
    NavState.EMERGENCY_STOP,
    NavState.SENSOR_ERROR,
    NavState.POLICY_ERROR,
})


class NavigationStateError(ValueError):
    """A goal or command was requested in a state that does not allow it."""


def commanding(state):
    """True only while the manager may issue a non-terminal motion command."""

    return state is NavState.NAVIGATING
