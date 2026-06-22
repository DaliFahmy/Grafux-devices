"""
handlers.py
All MELFA command handlers.

Each handler is an async function with the signature:
    async def handle_*(payload: dict, rb: RosBridgeClient, cfg: AgentConfig) -> dict

Handlers return a plain dict that is JSON-serialised and sent back to the
Grafux device server as the command result.

Generic handlers (ping, get_status, run_code, shell) work even when rosbridge
is not connected.  Robot handlers require an active rosbridge connection and
will return an error dict if one is unavailable.
"""

import asyncio
import contextlib
import io
import json
import logging
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

from .ros_bridge import RosBridgeClient
from . import config

logger = logging.getLogger("melfa.handlers")

_agent_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Agent runtime config (passed into every handler)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    rosbridge_url: str
    move_group: str = config.DEFAULT_MOVE_GROUP
    base_link: str = config.DEFAULT_BASE_LINK
    ee_link: str = config.DEFAULT_EE_LINK
    joint_names: list = None  # defaults to config.JOINT_NAMES

    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = list(config.JOINT_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(message: str, details: str = "") -> dict:
    return {"type": "error", "status": "error", "message": message, "details": details}


def _require_rosbridge(rb: RosBridgeClient) -> Optional[dict]:
    """Return an error dict if rosbridge is not connected, else None."""
    if not rb.is_connected:
        return _error(
            "rosbridge not connected",
            "Ensure 'ros2 run rosbridge_server rosbridge_websocket' is running on the robot machine.",
        )
    return None


def _parse_text_json(payload: dict) -> tuple[dict, Optional[dict]]:
    """
    Parse the 'text' field of a Grafux command payload as JSON.
    Returns (parsed_dict, error_dict_or_None).
    """
    raw = payload.get("text", "").strip()
    if not raw:
        return {}, None
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return {}, _error(f"Invalid JSON in text field: {exc}", f"Received: {raw!r}")


def _deg_to_rad(deg: float) -> float:
    return math.radians(deg)


def _euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> dict:
    """
    Convert ZYX Euler angles (degrees) to a quaternion {x, y, z, w}.
    Convention: extrinsic Z-Y-X (yaw first, then pitch, then roll).
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)

    return {
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy,
    }


def _quaternion_to_euler_deg(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Convert quaternion to ZYX Euler angles (degrees): roll, pitch, yaw."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Generic handlers (work without rosbridge)
# ---------------------------------------------------------------------------

async def handle_ping(payload: dict, rb: RosBridgeClient, cfg: AgentConfig) -> dict:
    return {"type": "pong", "message": "pong"}


async def handle_get_status(payload: dict, rb: RosBridgeClient, cfg: AgentConfig) -> dict:
    return {
        "type": "robot_status",
        "ros_connected": rb.is_connected,
        "rosbridge_url": cfg.rosbridge_url,
        "joint_states_cached": rb.has_topic_data(config.TOPIC_JOINT_STATES),
        "active_goals": len(rb.active_goals),
        "uptime_s": round(time.time() - _agent_start_time, 1),
        "platform": platform.platform(),
        "python": sys.version,
        "move_group": cfg.move_group,
        "base_link": cfg.base_link,
        "ee_link": cfg.ee_link,
        "joint_names": cfg.joint_names,
    }


async def handle_run_code(payload: dict, rb: RosBridgeClient, cfg: AgentConfig) -> dict:
    """Execute a Python snippet on the robot machine and return stdout/stderr."""
    code = payload.get("code", "")
    timeout = int(payload.get("timeout", 30))
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    status = "ok"
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, "<melfa_agent>", "exec"), {})  # noqa: S102
    except Exception as exc:
        stderr_buf.write(str(exc))
        status = "error"
    return {
        "type": "run_code_result",
        "status": status,
        "stdout": stdout_buf.getvalue().splitlines(),
        "stderr": stderr_buf.getvalue().splitlines(),
    }


async def handle_shell(payload: dict, rb: RosBridgeClient, cfg: AgentConfig) -> dict:
    """Run a shell command on the robot machine and return stdout/stderr."""
    command = payload.get("command", "")
    timeout = int(payload.get("timeout", 30))
    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "type": "shell_result",
            "status": "ok",
            "returncode": result.returncode,
            "stdout": result.stdout.splitlines(),
            "stderr": result.stderr.splitlines(),
        }
    except subprocess.TimeoutExpired:
        return {"type": "shell_result", "status": "timeout", "command": command}
    except Exception as exc:  # noqa: BLE001
        return {"type": "shell_result", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Robot state handlers
# ---------------------------------------------------------------------------

async def handle_robot_get_joint_states(
    payload: dict, rb: RosBridgeClient, cfg: AgentConfig
) -> dict:
    """Return the latest cached joint positions as a {j1..j6} dict in degrees."""
    if err := _require_rosbridge(rb):
        return err

    cached = rb.get_cached(config.TOPIC_JOINT_STATES)
    if cached is None:
        return _error(
            "No joint state data yet",
            f"Topic {config.TOPIC_JOINT_STATES!r} has not published any messages. "
            "Ensure the MELFA driver is running.",
        )

    names: list[str] = cached.get("name", [])
    positions_rad: list[float] = cached.get("position", [])

    if not names or not positions_rad:
        return _error("Joint state message is empty", str(cached))

    # Build {j1: deg, j2: deg, ...} using either the configured joint_names order
    # or whatever names the driver reports.
    joints_deg: dict[str, float] = {}
    for name, pos_rad in zip(names, positions_rad):
        joints_deg[name] = round(math.degrees(pos_rad), 4)

    return {
        "type": "robot_joint_states",
        "joints": joints_deg,
        "units": "deg",
        "timestamp": cached.get("header", {}).get("stamp", {}),
    }


async def handle_robot_get_pose(
    payload: dict, rb: RosBridgeClient, cfg: AgentConfig
) -> dict:
    """Return the current end-effector pose via the /compute_fk service."""
    if err := _require_rosbridge(rb):
        return err

    cached_js = rb.get_cached(config.TOPIC_JOINT_STATES)
    if cached_js is None:
        return _error("No joint state data", "Cannot compute FK without joint states.")

    request = {
        "header": {"frame_id": cfg.base_link},
        "fk_link_names": [cfg.ee_link],
        "robot_state": {
            "joint_state": {
                "header": cached_js.get("header", {}),
                "name": cached_js.get("name", []),
                "position": cached_js.get("position", []),
                "velocity": cached_js.get("velocity", []),
                "effort": cached_js.get("effort", []),
            }
        },
    }

    try:
        response = await rb.call_service(
            config.SERVICE_COMPUTE_FK,
            "moveit_msgs/srv/GetPositionFK",
            args=request,
            timeout=config.SERVICE_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(f"FK service call failed: {exc}")

    values = response.get("values", {})
    error_code = values.get("error_code", {}).get("val", -1)
    if error_code != config.MOVEIT_SUCCESS:
        return _error(
            f"FK service error: {config.moveit_error_string(error_code)}",
            f"error_code={error_code}",
        )

    pose_stamped_list = values.get("pose_stamped", [])
    if not pose_stamped_list:
        return _error("FK service returned empty pose_stamped list")

    pose = pose_stamped_list[0].get("pose", {})
    pos = pose.get("position", {})
    ori = pose.get("orientation", {})

    roll, pitch, yaw = _quaternion_to_euler_deg(
        ori.get("x", 0.0),
        ori.get("y", 0.0),
        ori.get("z", 0.0),
        ori.get("w", 1.0),
    )

    return {
        "type": "robot_pose",
        "x": round(pos.get("x", 0.0), 6),
        "y": round(pos.get("y", 0.0), 6),
        "z": round(pos.get("z", 0.0), 6),
        "roll": round(roll, 4),
        "pitch": round(pitch, 4),
        "yaw": round(yaw, 4),
        "units": "m/deg",
        "frame": cfg.base_link,
        "ee_link": cfg.ee_link,
        "quaternion": {
            "x": ori.get("x", 0.0),
            "y": ori.get("y", 0.0),
            "z": ori.get("z", 0.0),
            "w": ori.get("w", 1.0),
        },
    }


# ---------------------------------------------------------------------------
# Motion handlers
# ---------------------------------------------------------------------------

def _build_joint_move_goal(
    joint_names: list[str],
    positions_rad: list[float],
    move_group: str,
    speed: float,
) -> dict:
    """Build a MoveGroup action goal for a joint-space move."""
    joint_constraints = [
        {
            "joint_name": name,
            "position": pos,
            "tolerance_above": 0.001,
            "tolerance_below": 0.001,
            "weight": 1.0,
        }
        for name, pos in zip(joint_names, positions_rad)
    ]
    return {
        "request": {
            "group_name": move_group,
            "motion_plan_request": {
                "group_name": move_group,
                "planner_id": config.PLANNER_ID_JOINT,
                "planning_pipeline": config.PLANNER_PIPELINE_JOINT,
                "num_planning_attempts": config.PLANNING_ATTEMPTS,
                "allowed_planning_time": config.PLANNING_TIME_S,
                "max_velocity_scaling_factor": max(0.01, min(1.0, speed)),
                "max_acceleration_scaling_factor": max(0.01, min(1.0, speed * 0.5)),
                "goal_constraints": [{"joint_constraints": joint_constraints}],
            },
            "planning_options": {
                "plan_only": False,
                "look_around": False,
                "replan": True,
                "replan_attempts": 4,
                "replan_delay": 10.0,
            },
        }
    }


def _build_pose_move_goal(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    ee_link: str,
    base_link: str,
    move_group: str,
    speed: float,
) -> dict:
    """Build a MoveGroup action goal for a Cartesian end-effector move (Pilz LIN)."""
    return {
        "request": {
            "group_name": move_group,
            "motion_plan_request": {
                "group_name": move_group,
                "planner_id": config.PLANNER_ID_LIN,
                "planning_pipeline": config.PLANNER_PIPELINE_CARTESIAN,
                "num_planning_attempts": config.PLANNING_ATTEMPTS,
                "allowed_planning_time": config.PLANNING_TIME_S,
                "max_velocity_scaling_factor": max(0.01, min(1.0, speed)),
                "max_acceleration_scaling_factor": max(0.01, min(1.0, speed * 0.5)),
                "goal_constraints": [
                    {
                        "position_constraints": [
                            {
                                "header": {"frame_id": base_link},
                                "link_name": ee_link,
                                "constraint_region": {
                                    "primitives": [
                                        {"type": 2, "dimensions": [0.001]}  # SPHERE r=1mm
                                    ],
                                    "primitive_poses": [
                                        {
                                            "position": {"x": x, "y": y, "z": z},
                                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                        }
                                    ],
                                },
                                "weight": 1.0,
                            }
                        ],
                        "orientation_constraints": [
                            {
                                "header": {"frame_id": base_link},
                                "link_name": ee_link,
                                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                                "absolute_x_axis_tolerance": 0.01,
                                "absolute_y_axis_tolerance": 0.01,
                                "absolute_z_axis_tolerance": 0.01,
                                "weight": 1.0,
                            }
                        ],
                    }
                ],
            },
            "planning_options": {
                "plan_only": False,
                "look_around": False,
                "replan": False,
                "replan_attempts": 0,
                "replan_delay": 0.0,
            },
        }
    }


async def _execute_move_goal(
    rb: RosBridgeClient,
    goal: dict,
    timeout: float = config.MOVE_TIMEOUT_S,
) -> dict:
    """Send a MoveGroup action goal and return a normalised result dict."""
    try:
        result = await rb.send_action_goal(
            config.ACTION_MOVE_GROUP,
            config.ACTION_MOVE_GROUP_TYPE,
            goal,
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(f"MoveGroup action failed: {exc}")

    values = result.get("values", {})
    move_result = values.get("result", {})
    error_code = move_result.get("error_code", {}).get("val", -1)
    planning_time = move_result.get("planning_time", 0.0)
    success = error_code == config.MOVEIT_SUCCESS

    return {
        "type": "robot_move_result",
        "success": success,
        "error_code": error_code,
        "error_string": config.moveit_error_string(error_code),
        "planning_time_s": round(planning_time, 3),
    }


async def handle_robot_move_joints(
    payload: dict, rb: RosBridgeClient, cfg: AgentConfig
) -> dict:
    """
    Plan and execute a joint-space move via MoveIt2 OMPL.

    Expects payload["text"] = JSON string with j1..j6 (degrees) and optional speed.
    Example: {"j1": 0, "j2": -30, "j3": 90, "j4": 0, "j5": 45, "j6": 0, "speed": 0.1}
    """
    if err := _require_rosbridge(rb):
        return err

    params, parse_err = _parse_text_json(payload)
    if parse_err:
        return parse_err

    speed = float(params.get("speed", 0.1))
    positions_deg = [float(params.get(name, 0.0)) for name in cfg.joint_names]
    positions_rad = [_deg_to_rad(d) for d in positions_deg]

    logger.info(
        "move_joints: %s deg  speed=%.2f",
        dict(zip(cfg.joint_names, [round(d, 2) for d in positions_deg])),
        speed,
    )

    goal = _build_joint_move_goal(cfg.joint_names, positions_rad, cfg.move_group, speed)
    return await _execute_move_goal(rb, goal)


async def handle_robot_move_pose(
    payload: dict, rb: RosBridgeClient, cfg: AgentConfig
) -> dict:
    """
    Plan and execute a Cartesian end-effector move via MoveIt2 Pilz LIN planner.

    Expects payload["text"] = JSON with x,y,z (metres), roll,pitch,yaw (degrees), speed.
    Example: {"x": 0.3, "y": 0.0, "z": 0.5, "roll": 180, "pitch": 0, "yaw": 0, "speed": 0.1}
    """
    if err := _require_rosbridge(rb):
        return err

    params, parse_err = _parse_text_json(payload)
    if parse_err:
        return parse_err

    x = float(params.get("x", 0.0))
    y = float(params.get("y", 0.0))
    z = float(params.get("z", 0.5))
    roll = float(params.get("roll", 180.0))
    pitch = float(params.get("pitch", 0.0))
    yaw = float(params.get("yaw", 0.0))
    speed = float(params.get("speed", 0.1))

    q = _euler_to_quaternion(roll, pitch, yaw)
    logger.info(
        "move_pose: (%.3f, %.3f, %.3f) rpy=(%.1f, %.1f, %.1f) speed=%.2f",
        x, y, z, roll, pitch, yaw, speed,
    )

    goal = _build_pose_move_goal(
        x, y, z,
        q["x"], q["y"], q["z"], q["w"],
        cfg.ee_link, cfg.base_link, cfg.move_group, speed,
    )
    return await _execute_move_goal(rb, goal)


async def handle_robot_home(
    payload: dict, rb: RosBridgeClient, cfg: AgentConfig
) -> dict:
    """Move all joints to the home (all-zeros) position."""
    if err := _require_rosbridge(rb):
        return err

    params, parse_err = _parse_text_json(payload)
    if parse_err:
        return parse_err

    speed = float(params.get("speed", config.HOME_JOINTS_DEG[0] if False else 0.05))
    positions_rad = [_deg_to_rad(d) for d in config.HOME_JOINTS_DEG]
    logger.info("robot_home: moving to zero configuration at speed=%.2f", speed)

    goal = _build_joint_move_goal(cfg.joint_names, positions_rad, cfg.move_group, speed)
    return await _execute_move_goal(rb, goal)


async def handle_robot_stop(
    payload: dict, rb: RosBridgeClient, cfg: AgentConfig
) -> dict:
    """Cancel all active MoveGroup goals."""
    if err := _require_rosbridge(rb):
        return err

    count = await rb.cancel_all_goals()
    logger.info("robot_stop: cancelled %d active goal(s)", count)
    return {"type": "robot_stop_result", "cancelled": True, "goals_cancelled": count}


# ---------------------------------------------------------------------------
# Handler dispatch table (imported by agent.py)
# ---------------------------------------------------------------------------

HANDLERS: dict[str, any] = {
    "ping":                   handle_ping,
    "get_status":             handle_get_status,
    "robot_get_status":       handle_get_status,
    "run_code":               handle_run_code,
    "shell":                  handle_shell,
    "robot_get_joint_states": handle_robot_get_joint_states,
    "robot_get_pose":         handle_robot_get_pose,
    "robot_move_joints":      handle_robot_move_joints,
    "robot_move_pose":        handle_robot_move_pose,
    "robot_home":             handle_robot_home,
    "robot_stop":             handle_robot_stop,
}
