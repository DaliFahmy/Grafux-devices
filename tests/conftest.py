"""
conftest.py
Shared pytest fixtures for the MELFA ROS2 integration test suite.

All tests run without a real robot, ROS2 environment, or network connection.
Every ROS2 / WebSocket dependency is replaced by a MagicMock / AsyncMock.

Path setup
----------
This conftest inserts `server/devices/` at the front of sys.path so that
both the flat imports used by `devices_server.py` (e.g. `import commands`) and the
package imports used by the Mitsubishi_MELFA package work correctly.
"""

import asyncio
import math
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

# ---------------------------------------------------------------------------
# sys.path bootstrap — must come before any project imports
# ---------------------------------------------------------------------------

_DEVICES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _DEVICES_DIR not in sys.path:
    sys.path.insert(0, _DEVICES_DIR)

# ---------------------------------------------------------------------------
# Project imports (available after path setup)
# ---------------------------------------------------------------------------

from Mitsubishi_MELFA.handlers import AgentConfig  # noqa: E402
from Mitsubishi_MELFA.ros_bridge import RosBridgeClient  # noqa: E402
from Mitsubishi_MELFA import config as melfa_config  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_cfg() -> AgentConfig:
    """Default AgentConfig for unit tests."""
    return AgentConfig(
        rosbridge_url="ws://localhost:9090",
        move_group=melfa_config.DEFAULT_MOVE_GROUP,
        base_link=melfa_config.DEFAULT_BASE_LINK,
        ee_link=melfa_config.DEFAULT_EE_LINK,
        joint_names=list(melfa_config.JOINT_NAMES),
    )


def _make_mock_rb(connected: bool) -> MagicMock:
    """Return a MagicMock of RosBridgeClient with all async stubs wired."""
    rb = MagicMock(spec=RosBridgeClient)

    # is_connected is a @property — must mock via the class type
    type(rb).is_connected = PropertyMock(return_value=connected)

    # active_goals is a @property
    type(rb).active_goals = PropertyMock(return_value={})

    # Sync methods
    rb.get_cached.return_value = None
    rb.has_topic_data.return_value = False

    # Async methods
    rb.call_service = AsyncMock(return_value={})
    rb.send_action_goal = AsyncMock(
        return_value={"values": {"result": {"error_code": {"val": 1}, "planning_time": 0.3}}}
    )
    rb.cancel_all_goals = AsyncMock(return_value=0)
    rb.subscribe = AsyncMock()
    rb.unsubscribe = AsyncMock()
    rb.publish = AsyncMock()

    return rb


@pytest.fixture
def mock_rb() -> MagicMock:
    """Mock RosBridgeClient with is_connected=True and default success stubs."""
    return _make_mock_rb(connected=True)


@pytest.fixture
def mock_rb_disconnected() -> MagicMock:
    """Mock RosBridgeClient with is_connected=False."""
    return _make_mock_rb(connected=False)


@pytest.fixture
def fake_joint_states() -> dict:
    """
    Realistic /joint_states message with six joints at known radian values.

    Angles (degrees → radians):
        j1 =  0.0 deg  →  0.0 rad
        j2 = -30.0 deg → -pi/6  rad
        j3 =  90.0 deg →  pi/2  rad
        j4 =  0.0 deg  →  0.0 rad
        j5 =  45.0 deg →  pi/4  rad
        j6 =  0.0 deg  →  0.0 rad
    """
    names = list(melfa_config.JOINT_NAMES)
    degrees = [0.0, -30.0, 90.0, 0.0, 45.0, 0.0]
    positions_rad = [math.radians(d) for d in degrees]
    return {
        "header": {"stamp": {"sec": 100, "nanosec": 0}, "frame_id": ""},
        "name": names,
        "position": positions_rad,
        "velocity": [0.0] * 6,
        "effort": [0.0] * 6,
    }


@pytest.fixture
def fake_fk_response() -> dict:
    """
    Fake /compute_fk service response for a known end-effector pose:
      position  (0.3, 0.1, 0.5) metres
      orientation  identity quaternion (0, 0, 0, 1)  → roll=0, pitch=0, yaw=0
    """
    return {
        "values": {
            "error_code": {"val": 1},
            "pose_stamped": [
                {
                    "header": {"frame_id": "base_link"},
                    "pose": {
                        "position": {"x": 0.3, "y": 0.1, "z": 0.5},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                }
            ],
        }
    }


@pytest.fixture
def success_move_result() -> dict:
    """Fake MoveGroup action result with error_code SUCCESS (val=1)."""
    return {
        "values": {
            "result": {
                "error_code": {"val": 1},
                "planning_time": 0.35,
            }
        }
    }


@pytest.fixture
def failure_move_result() -> dict:
    """Fake MoveGroup action result with PLANNING_FAILED (val=-2)."""
    return {
        "values": {
            "result": {
                "error_code": {"val": -2},
                "planning_time": 0.0,
            }
        }
    }
