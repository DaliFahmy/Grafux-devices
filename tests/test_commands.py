"""
test_commands.py
Unit tests for server/devices/commands.py — the robot command builder functions.

Each builder must return a dict with:
  - "id"        : a valid UUID4 string
  - "type"      : the expected command type string
  - "timestamp" : a float close to the current time
  - "payload"   : a dict containing the expected fields
"""

import time
import uuid

import pytest

import commands as cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_base(result: dict, expected_type: str) -> None:
    """Assert all common fields are present and well-formed."""
    assert isinstance(result, dict), "builder must return a dict"
    assert result["type"] == expected_type
    # id must be a valid UUID
    parsed_uuid = uuid.UUID(result["id"])
    assert str(parsed_uuid) == result["id"]
    # timestamp must be a recent float
    assert isinstance(result["timestamp"], float)
    assert abs(result["timestamp"] - time.time()) < 5.0
    assert isinstance(result["payload"], dict)


# ---------------------------------------------------------------------------
# Generic command builders (existing — smoke-tested for completeness)
# ---------------------------------------------------------------------------

def test_ping_type():
    r = cmd.ping()
    _assert_base(r, "ping")
    assert r["payload"] == {}


def test_get_status_type():
    r = cmd.get_status()
    _assert_base(r, "get_status")


def test_run_code_payload():
    r = cmd.run_code("print(1+1)", timeout=15)
    _assert_base(r, "run_code")
    assert r["payload"]["code"] == "print(1+1)"
    assert r["payload"]["timeout"] == 15


def test_shell_payload():
    r = cmd.shell("ls -la", timeout=10)
    _assert_base(r, "shell")
    assert r["payload"]["command"] == "ls -la"
    assert r["payload"]["timeout"] == 10


# ---------------------------------------------------------------------------
# Robot command builders — type and empty payload
# ---------------------------------------------------------------------------

def test_robot_get_joint_states_type():
    r = cmd.robot_get_joint_states()
    _assert_base(r, "robot_get_joint_states")
    assert r["payload"] == {}


def test_robot_get_pose_type():
    r = cmd.robot_get_pose()
    _assert_base(r, "robot_get_pose")
    assert r["payload"] == {}


def test_robot_get_status_type():
    r = cmd.robot_get_status()
    _assert_base(r, "robot_get_status")
    assert r["payload"] == {}


def test_robot_stop_type():
    r = cmd.robot_stop()
    _assert_base(r, "robot_stop")
    assert r["payload"] == {}


# ---------------------------------------------------------------------------
# robot_move_joints
# ---------------------------------------------------------------------------

def test_robot_move_joints_defaults():
    r = cmd.robot_move_joints()
    _assert_base(r, "robot_move_joints")
    p = r["payload"]
    assert p["j1"] == 0.0
    assert p["j2"] == 0.0
    assert p["j3"] == 0.0
    assert p["j4"] == 0.0
    assert p["j5"] == 0.0
    assert p["j6"] == 0.0
    assert p["speed"] == 0.1


def test_robot_move_joints_custom():
    r = cmd.robot_move_joints(j1=10.0, j2=-30.0, j3=90.0, j4=5.0, j5=45.0, j6=-10.0, speed=0.5)
    _assert_base(r, "robot_move_joints")
    p = r["payload"]
    assert p["j1"] == 10.0
    assert p["j2"] == -30.0
    assert p["j3"] == 90.0
    assert p["j4"] == 5.0
    assert p["j5"] == 45.0
    assert p["j6"] == -10.0
    assert p["speed"] == 0.5


def test_robot_move_joints_all_keys_present():
    r = cmd.robot_move_joints()
    for key in ("j1", "j2", "j3", "j4", "j5", "j6", "speed"):
        assert key in r["payload"], f"missing key: {key}"


# ---------------------------------------------------------------------------
# robot_move_pose
# ---------------------------------------------------------------------------

def test_robot_move_pose_defaults():
    r = cmd.robot_move_pose()
    _assert_base(r, "robot_move_pose")
    p = r["payload"]
    assert p["x"] == 0.0
    assert p["y"] == 0.0
    assert p["z"] == 0.5
    assert p["roll"] == 180.0
    assert p["pitch"] == 0.0
    assert p["yaw"] == 0.0
    assert p["speed"] == 0.1


def test_robot_move_pose_custom():
    r = cmd.robot_move_pose(x=0.3, y=-0.1, z=0.6, roll=90.0, pitch=15.0, yaw=-45.0, speed=0.3)
    _assert_base(r, "robot_move_pose")
    p = r["payload"]
    assert p["x"] == 0.3
    assert p["y"] == -0.1
    assert p["z"] == 0.6
    assert p["roll"] == 90.0
    assert p["pitch"] == 15.0
    assert p["yaw"] == -45.0
    assert p["speed"] == 0.3


def test_robot_move_pose_all_keys_present():
    r = cmd.robot_move_pose()
    for key in ("x", "y", "z", "roll", "pitch", "yaw", "speed"):
        assert key in r["payload"], f"missing key: {key}"


# ---------------------------------------------------------------------------
# robot_home
# ---------------------------------------------------------------------------

def test_robot_home_default_speed():
    r = cmd.robot_home()
    _assert_base(r, "robot_home")
    assert r["payload"]["speed"] == 0.05


def test_robot_home_custom_speed():
    r = cmd.robot_home(speed=0.2)
    _assert_base(r, "robot_home")
    assert r["payload"]["speed"] == 0.2


# ---------------------------------------------------------------------------
# Uniqueness — each call must produce a distinct id
# ---------------------------------------------------------------------------

def test_unique_ids_same_builder():
    ids = [cmd.robot_get_joint_states()["id"] for _ in range(5)]
    assert len(set(ids)) == 5, "every command must have a unique id"


def test_unique_ids_across_builders():
    builders = [
        cmd.ping,
        cmd.robot_get_joint_states,
        cmd.robot_get_pose,
        cmd.robot_stop,
        cmd.robot_home,
    ]
    ids = [b()["id"] for b in builders]
    assert len(set(ids)) == len(builders)


def test_timestamp_increases_monotonically():
    before = time.time()
    r = cmd.robot_move_joints()
    after = time.time()
    assert before <= r["timestamp"] <= after
