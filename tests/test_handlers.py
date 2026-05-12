"""
test_handlers.py
Unit tests for server/devices/Mitsubishi_MELFA/handlers.py.

Coverage:
  - Pure math helpers (_deg_to_rad, _euler_to_quaternion, _quaternion_to_euler_deg, _parse_text_json)
  - Generic handlers: ping, get_status, run_code, shell
  - Robot state handlers: robot_get_joint_states, robot_get_pose
  - Motion handlers: robot_move_joints, robot_move_pose, robot_home, robot_stop
  - Guard: all robot handlers return an error when rosbridge is disconnected
  - Guard: state handlers return an error when the topic cache is empty

No real robot or ROS2 environment is needed — RosBridgeClient is mocked via conftest.
"""

import asyncio
import json
import math

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from Mitsubishi_MELFA.handlers import (
    AgentConfig,
    HANDLERS,
    handle_ping,
    handle_get_status,
    handle_run_code,
    handle_shell,
    handle_robot_get_joint_states,
    handle_robot_get_pose,
    handle_robot_move_joints,
    handle_robot_move_pose,
    handle_robot_home,
    handle_robot_stop,
    _deg_to_rad,
    _euler_to_quaternion,
    _quaternion_to_euler_deg,
    _parse_text_json,
)
from Mitsubishi_MELFA import config as melfa_config


# ===========================================================================
# Pure math helper tests (synchronous — no fixtures needed)
# ===========================================================================

class TestDegToRad:
    def test_zero(self):
        assert _deg_to_rad(0.0) == 0.0

    def test_90(self):
        assert abs(_deg_to_rad(90.0) - math.pi / 2) < 1e-12

    def test_180(self):
        assert abs(_deg_to_rad(180.0) - math.pi) < 1e-12

    def test_negative(self):
        assert abs(_deg_to_rad(-30.0) - (-math.pi / 6)) < 1e-12

    def test_360(self):
        assert abs(_deg_to_rad(360.0) - 2 * math.pi) < 1e-12


class TestEulerToQuaternion:
    def test_identity(self):
        q = _euler_to_quaternion(0.0, 0.0, 0.0)
        assert abs(q["x"]) < 1e-12
        assert abs(q["y"]) < 1e-12
        assert abs(q["z"]) < 1e-12
        assert abs(q["w"] - 1.0) < 1e-12

    def test_unit_quaternion_magnitude(self):
        for roll, pitch, yaw in [(30.0, -15.0, 45.0), (90.0, 0.0, 0.0), (0.0, 0.0, 180.0)]:
            q = _euler_to_quaternion(roll, pitch, yaw)
            mag = math.sqrt(q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2 + q["w"] ** 2)
            assert abs(mag - 1.0) < 1e-10, f"non-unit quaternion for rpy=({roll},{pitch},{yaw})"

    def test_roll_only_180(self):
        # 180° roll → x = 1, y = z = 0, w = 0  (approx)
        q = _euler_to_quaternion(180.0, 0.0, 0.0)
        assert abs(abs(q["x"]) - 1.0) < 1e-10
        assert abs(q["y"]) < 1e-10
        assert abs(q["z"]) < 1e-10

    def test_returns_all_keys(self):
        q = _euler_to_quaternion(10.0, 20.0, 30.0)
        for key in ("x", "y", "z", "w"):
            assert key in q


class TestQuaternionToEuler:
    def test_identity_roundtrip(self):
        roll, pitch, yaw = _quaternion_to_euler_deg(0.0, 0.0, 0.0, 1.0)
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10

    def test_roundtrip_arbitrary(self):
        for r, p, y in [(30.0, -15.0, 45.0), (10.0, 5.0, -20.0), (0.0, 0.0, 90.0)]:
            q = _euler_to_quaternion(r, p, y)
            r2, p2, y2 = _quaternion_to_euler_deg(q["x"], q["y"], q["z"], q["w"])
            assert abs(r2 - r) < 0.001, f"roll mismatch for ({r},{p},{y})"
            assert abs(p2 - p) < 0.001, f"pitch mismatch for ({r},{p},{y})"
            assert abs(y2 - y) < 0.001, f"yaw mismatch for ({r},{p},{y})"


class TestParseTextJson:
    def test_valid_json(self):
        parsed, err = _parse_text_json({"text": '{"j1": 30}'})
        assert err is None
        assert parsed["j1"] == 30

    def test_empty_text_field(self):
        parsed, err = _parse_text_json({})
        assert err is None
        assert parsed == {}

    def test_empty_string(self):
        parsed, err = _parse_text_json({"text": ""})
        assert err is None
        assert parsed == {}

    def test_whitespace_only(self):
        parsed, err = _parse_text_json({"text": "   "})
        assert err is None
        assert parsed == {}

    def test_invalid_json(self):
        parsed, err = _parse_text_json({"text": "not json"})
        assert err is not None
        assert err["type"] == "error"
        assert parsed == {}

    def test_partial_json(self):
        parsed, err = _parse_text_json({"text": '{"j1": 30'})
        assert err is not None
        assert err["type"] == "error"

    def test_nested_json(self):
        parsed, err = _parse_text_json({"text": '{"a": {"b": 1}}'})
        assert err is None
        assert parsed["a"]["b"] == 1


# ===========================================================================
# Generic handlers (work even when rosbridge is disconnected)
# ===========================================================================

@pytest.mark.asyncio
async def test_handle_ping(mock_rb, agent_cfg):
    result = await handle_ping({}, mock_rb, agent_cfg)
    assert result["type"] == "pong"
    assert result["message"] == "pong"


@pytest.mark.asyncio
async def test_handle_get_status_connected(mock_rb, agent_cfg):
    result = await handle_get_status({}, mock_rb, agent_cfg)
    assert result["type"] == "robot_status"
    assert result["ros_connected"] is True
    assert result["rosbridge_url"] == agent_cfg.rosbridge_url
    assert "uptime_s" in result
    assert "platform" in result
    assert "python" in result
    assert result["move_group"] == agent_cfg.move_group
    assert result["joint_names"] == agent_cfg.joint_names


@pytest.mark.asyncio
async def test_handle_get_status_disconnected(mock_rb_disconnected, agent_cfg):
    result = await handle_get_status({}, mock_rb_disconnected, agent_cfg)
    assert result["type"] == "robot_status"
    assert result["ros_connected"] is False


@pytest.mark.asyncio
async def test_handle_run_code_print(mock_rb, agent_cfg):
    result = await handle_run_code({"code": "print(1+1)"}, mock_rb, agent_cfg)
    assert result["type"] == "run_code_result"
    assert result["status"] == "ok"
    assert result["stdout"] == ["2"]
    assert result["stderr"] == []


@pytest.mark.asyncio
async def test_handle_run_code_multiple_lines(mock_rb, agent_cfg):
    result = await handle_run_code({"code": "print('a')\nprint('b')"}, mock_rb, agent_cfg)
    assert result["stdout"] == ["a", "b"]


@pytest.mark.asyncio
async def test_handle_run_code_syntax_error(mock_rb, agent_cfg):
    result = await handle_run_code({"code": "def broken(:"}, mock_rb, agent_cfg)
    assert result["status"] == "error"
    assert len(result["stderr"]) > 0


@pytest.mark.asyncio
async def test_handle_run_code_runtime_exception(mock_rb, agent_cfg):
    result = await handle_run_code({"code": "1/0"}, mock_rb, agent_cfg)
    assert result["status"] == "error"
    # handler writes str(exc) — "division by zero" — not the class name
    assert len(result["stderr"]) > 0
    assert any("zero" in line.lower() for line in result["stderr"])


@pytest.mark.asyncio
async def test_handle_shell_echo(mock_rb, agent_cfg):
    result = await handle_shell({"command": "echo hello", "timeout": 5}, mock_rb, agent_cfg)
    assert result["type"] == "shell_result"
    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert any("hello" in line for line in result["stdout"])


@pytest.mark.asyncio
async def test_handle_shell_returncode_nonzero(mock_rb, agent_cfg):
    result = await handle_shell({"command": "exit 2", "timeout": 5}, mock_rb, agent_cfg)
    assert result["status"] == "ok"
    assert result["returncode"] == 2


@pytest.mark.asyncio
async def test_handle_shell_fields_present(mock_rb, agent_cfg):
    result = await handle_shell({"command": "echo test"}, mock_rb, agent_cfg)
    for field in ("type", "status", "returncode", "stdout", "stderr"):
        assert field in result


# ===========================================================================
# Robot handlers — rosbridge disconnected guard
# ===========================================================================

ROBOT_HANDLERS_WITH_EMPTY_PAYLOAD = [
    (handle_robot_get_joint_states, {}),
    (handle_robot_get_pose, {}),
    (handle_robot_move_joints, {"text": '{"j1":0,"j2":0,"j3":0,"j4":0,"j5":0,"j6":0}'}),
    (handle_robot_move_pose, {"text": '{"x":0,"y":0,"z":0.5,"roll":180,"pitch":0,"yaw":0}'}),
    (handle_robot_home, {}),
    (handle_robot_stop, {}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler,payload", ROBOT_HANDLERS_WITH_EMPTY_PAYLOAD)
async def test_robot_handler_no_rosbridge(handler, payload, mock_rb_disconnected, agent_cfg):
    result = await handler(payload, mock_rb_disconnected, agent_cfg)
    assert result["type"] == "error"
    assert "rosbridge" in result["message"].lower()


@pytest.mark.asyncio
async def test_stop_does_not_cancel_when_disconnected(mock_rb_disconnected, agent_cfg):
    await handle_robot_stop({}, mock_rb_disconnected, agent_cfg)
    mock_rb_disconnected.cancel_all_goals.assert_not_called()


# ===========================================================================
# Robot handlers — no cached joint state data
# ===========================================================================

@pytest.mark.asyncio
async def test_get_joint_states_no_cache(mock_rb, agent_cfg):
    mock_rb.get_cached.return_value = None
    result = await handle_robot_get_joint_states({}, mock_rb, agent_cfg)
    assert result["type"] == "error"


@pytest.mark.asyncio
async def test_get_pose_no_cache(mock_rb, agent_cfg):
    mock_rb.get_cached.return_value = None
    result = await handle_robot_get_pose({}, mock_rb, agent_cfg)
    assert result["type"] == "error"


# ===========================================================================
# Robot state handlers — success paths
# ===========================================================================

@pytest.mark.asyncio
async def test_get_joint_states_success(mock_rb, agent_cfg, fake_joint_states):
    mock_rb.get_cached.return_value = fake_joint_states

    result = await handle_robot_get_joint_states({}, mock_rb, agent_cfg)

    assert result["type"] == "robot_joint_states"
    assert result["units"] == "deg"
    joints = result["joints"]
    assert set(joints.keys()) == set(melfa_config.JOINT_NAMES)

    # j1 = 0.0 deg
    assert abs(joints["j1"]) < 0.001
    # j2 = -30.0 deg
    assert abs(joints["j2"] - (-30.0)) < 0.001
    # j3 = 90.0 deg
    assert abs(joints["j3"] - 90.0) < 0.001
    # j5 = 45.0 deg
    assert abs(joints["j5"] - 45.0) < 0.001


@pytest.mark.asyncio
async def test_get_joint_states_empty_message(mock_rb, agent_cfg):
    mock_rb.get_cached.return_value = {"name": [], "position": []}
    result = await handle_robot_get_joint_states({}, mock_rb, agent_cfg)
    assert result["type"] == "error"


@pytest.mark.asyncio
async def test_get_pose_success(mock_rb, agent_cfg, fake_joint_states, fake_fk_response):
    mock_rb.get_cached.return_value = fake_joint_states
    mock_rb.call_service = AsyncMock(return_value=fake_fk_response)

    result = await handle_robot_get_pose({}, mock_rb, agent_cfg)

    assert result["type"] == "robot_pose"
    assert result["units"] == "m/deg"
    assert result["frame"] == agent_cfg.base_link
    assert result["ee_link"] == agent_cfg.ee_link
    assert abs(result["x"] - 0.3) < 0.001
    assert abs(result["y"] - 0.1) < 0.001
    assert abs(result["z"] - 0.5) < 0.001
    # identity orientation → roll=pitch=yaw=0
    assert abs(result["roll"]) < 0.001
    assert abs(result["pitch"]) < 0.001
    assert abs(result["yaw"]) < 0.001
    assert "quaternion" in result


@pytest.mark.asyncio
async def test_get_pose_fk_service_error(mock_rb, agent_cfg, fake_joint_states):
    mock_rb.get_cached.return_value = fake_joint_states
    mock_rb.call_service = AsyncMock(return_value={
        "values": {"error_code": {"val": -1}, "pose_stamped": []}
    })
    result = await handle_robot_get_pose({}, mock_rb, agent_cfg)
    assert result["type"] == "error"


@pytest.mark.asyncio
async def test_get_pose_fk_service_timeout(mock_rb, agent_cfg, fake_joint_states):
    mock_rb.get_cached.return_value = fake_joint_states
    mock_rb.call_service = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))
    result = await handle_robot_get_pose({}, mock_rb, agent_cfg)
    assert result["type"] == "error"


# ===========================================================================
# Motion handlers — robot_move_joints
# ===========================================================================

@pytest.mark.asyncio
async def test_move_joints_success(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)
    payload = {"text": '{"j1":0,"j2":-30,"j3":90,"j4":0,"j5":45,"j6":0,"speed":0.1}'}

    result = await handle_robot_move_joints(payload, mock_rb, agent_cfg)

    assert result["type"] == "robot_move_result"
    assert result["success"] is True
    assert result["error_code"] == 1
    assert result["error_string"] == "SUCCESS"
    assert "planning_time_s" in result


@pytest.mark.asyncio
async def test_move_joints_planning_failed(mock_rb, agent_cfg, failure_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=failure_move_result)
    payload = {"text": '{"j1":0,"j2":0,"j3":0,"j4":0,"j5":0,"j6":0}'}

    result = await handle_robot_move_joints(payload, mock_rb, agent_cfg)

    assert result["success"] is False
    assert result["error_code"] == -2
    assert result["error_string"] == "PLANNING_FAILED"


@pytest.mark.asyncio
async def test_move_joints_timeout(mock_rb, agent_cfg):
    mock_rb.send_action_goal = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))
    payload = {"text": '{"j1":0,"j2":0,"j3":0,"j4":0,"j5":0,"j6":0}'}

    result = await handle_robot_move_joints(payload, mock_rb, agent_cfg)

    assert result["type"] == "error"


@pytest.mark.asyncio
async def test_move_joints_invalid_json(mock_rb, agent_cfg):
    result = await handle_robot_move_joints({"text": "bad json"}, mock_rb, agent_cfg)
    assert result["type"] == "error"


@pytest.mark.asyncio
async def test_move_joints_goal_structure(mock_rb, agent_cfg):
    """The MoveGroup goal must contain correctly converted joint positions."""
    mock_rb.send_action_goal = AsyncMock(return_value={
        "values": {"result": {"error_code": {"val": 1}, "planning_time": 0.1}}
    })
    # j2 = -30 deg, j3 = 90 deg
    payload = {"text": '{"j1":0,"j2":-30,"j3":90,"j4":0,"j5":0,"j6":0,"speed":0.2}'}

    await handle_robot_move_joints(payload, mock_rb, agent_cfg)

    mock_rb.send_action_goal.assert_awaited_once()
    _action, _atype, goal, *_ = mock_rb.send_action_goal.call_args[0]

    assert _action == melfa_config.ACTION_MOVE_GROUP
    assert _atype == melfa_config.ACTION_MOVE_GROUP_TYPE

    mpr = goal["request"]["motion_plan_request"]
    assert mpr["planning_pipeline"] == melfa_config.PLANNER_PIPELINE_JOINT
    assert mpr["group_name"] == agent_cfg.move_group

    # speed clamped to [0.01, 1.0]
    assert mpr["max_velocity_scaling_factor"] == pytest.approx(0.2, abs=1e-6)

    jc = mpr["goal_constraints"][0]["joint_constraints"]
    jc_map = {c["joint_name"]: c["position"] for c in jc}

    assert abs(jc_map["j2"] - math.radians(-30.0)) < 1e-9
    assert abs(jc_map["j3"] - math.radians(90.0)) < 1e-9
    assert abs(jc_map["j1"]) < 1e-9


# ===========================================================================
# Motion handlers — robot_move_pose
# ===========================================================================

@pytest.mark.asyncio
async def test_move_pose_success(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)
    payload = {"text": '{"x":0.3,"y":0.0,"z":0.5,"roll":180,"pitch":0,"yaw":0,"speed":0.1}'}

    result = await handle_robot_move_pose(payload, mock_rb, agent_cfg)

    assert result["type"] == "robot_move_result"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_move_pose_uses_pilz_lin_planner(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)
    payload = {"text": '{"x":0.3,"y":0.0,"z":0.5,"roll":180,"pitch":0,"yaw":0}'}

    await handle_robot_move_pose(payload, mock_rb, agent_cfg)

    _, _, goal, *_ = mock_rb.send_action_goal.call_args[0]
    mpr = goal["request"]["motion_plan_request"]
    assert mpr["planner_id"] == melfa_config.PLANNER_ID_LIN
    assert mpr["planning_pipeline"] == melfa_config.PLANNER_PIPELINE_CARTESIAN


@pytest.mark.asyncio
async def test_move_pose_contains_position_constraints(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)
    payload = {"text": '{"x":0.3,"y":0.1,"z":0.5,"roll":0,"pitch":0,"yaw":0}'}

    await handle_robot_move_pose(payload, mock_rb, agent_cfg)

    _, _, goal, *_ = mock_rb.send_action_goal.call_args[0]
    constraints = goal["request"]["motion_plan_request"]["goal_constraints"][0]
    pos_c = constraints["position_constraints"][0]

    primitive_pose = pos_c["constraint_region"]["primitive_poses"][0]
    assert abs(primitive_pose["position"]["x"] - 0.3) < 1e-9
    assert abs(primitive_pose["position"]["y"] - 0.1) < 1e-9
    assert abs(primitive_pose["position"]["z"] - 0.5) < 1e-9

    assert pos_c["link_name"] == agent_cfg.ee_link
    assert pos_c["header"]["frame_id"] == agent_cfg.base_link


@pytest.mark.asyncio
async def test_move_pose_invalid_json(mock_rb, agent_cfg):
    result = await handle_robot_move_pose({"text": "invalid"}, mock_rb, agent_cfg)
    assert result["type"] == "error"


# ===========================================================================
# Motion handlers — robot_home
# ===========================================================================

@pytest.mark.asyncio
async def test_home_goes_to_all_zeros(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)

    result = await handle_robot_home({}, mock_rb, agent_cfg)

    assert result["type"] == "robot_move_result"
    assert result["success"] is True

    _, _, goal, *_ = mock_rb.send_action_goal.call_args[0]
    jc = goal["request"]["motion_plan_request"]["goal_constraints"][0]["joint_constraints"]
    for constraint in jc:
        assert abs(constraint["position"]) < 1e-12, (
            f"joint {constraint['joint_name']} should be at 0 rad for home"
        )


@pytest.mark.asyncio
async def test_home_uses_joint_planner(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)
    await handle_robot_home({}, mock_rb, agent_cfg)
    _, _, goal, *_ = mock_rb.send_action_goal.call_args[0]
    assert goal["request"]["motion_plan_request"]["planning_pipeline"] == melfa_config.PLANNER_PIPELINE_JOINT


@pytest.mark.asyncio
async def test_home_default_speed(mock_rb, agent_cfg, success_move_result):
    mock_rb.send_action_goal = AsyncMock(return_value=success_move_result)
    await handle_robot_home({}, mock_rb, agent_cfg)
    _, _, goal, *_ = mock_rb.send_action_goal.call_args[0]
    speed = goal["request"]["motion_plan_request"]["max_velocity_scaling_factor"]
    assert 0.0 < speed <= 0.1  # home is slow by default


# ===========================================================================
# Motion handlers — robot_stop
# ===========================================================================

@pytest.mark.asyncio
async def test_stop_calls_cancel_all_goals(mock_rb, agent_cfg):
    mock_rb.cancel_all_goals = AsyncMock(return_value=2)

    result = await handle_robot_stop({}, mock_rb, agent_cfg)

    mock_rb.cancel_all_goals.assert_awaited_once()
    assert result["type"] == "robot_stop_result"
    assert result["cancelled"] is True
    assert result["goals_cancelled"] == 2


@pytest.mark.asyncio
async def test_stop_zero_goals(mock_rb, agent_cfg):
    mock_rb.cancel_all_goals = AsyncMock(return_value=0)
    result = await handle_robot_stop({}, mock_rb, agent_cfg)
    assert result["cancelled"] is True
    assert result["goals_cancelled"] == 0


# ===========================================================================
# HANDLERS dispatch table
# ===========================================================================

def test_handlers_dict_contains_all_expected_keys():
    expected = {
        "ping", "get_status", "robot_get_status",
        "run_code", "shell",
        "robot_get_joint_states", "robot_get_pose",
        "robot_move_joints", "robot_move_pose",
        "robot_home", "robot_stop",
    }
    assert expected.issubset(set(HANDLERS.keys()))


def test_handlers_values_are_callable():
    for name, handler in HANDLERS.items():
        assert callable(handler), f"HANDLERS['{name}'] is not callable"
