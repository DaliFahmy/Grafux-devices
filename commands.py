"""
commands.py
Command block builders.

Each function returns a plain dict that is JSON-serialised and sent to the
target device over WebSocket.  The device client reads the "type" field and
acts accordingly.

Supported command types
-----------------------
ping        — request a liveness check; device replies {"type":"pong"}
get_status  — request system info (CPU, memory, uptime, etc.)
run_code    — execute a Python snippet; device replies with stdout / stderr
shell       — run an arbitrary shell command
set_config  — push a key/value config update to the device
restart     — ask the device agent to restart itself
"""

import time
import uuid
from typing import Any, Dict, Optional


def _base(command_type: str, payload: Optional[Dict[str, Any]] = None) -> dict:
    """Attach a unique ID and timestamp to every command block."""
    return {
        "id": str(uuid.uuid4()),
        "type": command_type,
        "timestamp": time.time(),
        "payload": payload or {},
    }


# ---------------------------------------------------------------------------
# Predefined command builders
# ---------------------------------------------------------------------------

def ping() -> dict:
    """Ask a device to respond with a pong."""
    return _base("ping")


def get_status() -> dict:
    """Request system metrics from a device."""
    return _base("get_status")


def run_code(code: str, timeout: int = 30) -> dict:
    """
    Ask the device to execute a Python snippet.

    Parameters
    ----------
    code:    Python source code to run.
    timeout: Maximum seconds the device should wait before killing execution.
    """
    return _base("run_code", {"code": code, "timeout": timeout})


def shell(command: str, timeout: int = 30) -> dict:
    """
    Ask the device to run a shell command.

    Parameters
    ----------
    command: Shell command string (e.g. "ls -la /home/pi").
    timeout: Maximum seconds before the device kills the process.
    """
    return _base("shell", {"command": command, "timeout": timeout})


def set_config(key: str, value: Any) -> dict:
    """Push a configuration key/value to the device."""
    return _base("set_config", {"key": key, "value": value})


def restart() -> dict:
    """Ask the device agent to restart itself."""
    return _base("restart")


# ---------------------------------------------------------------------------
# Robot (MELFA) command builders
# ---------------------------------------------------------------------------

def robot_get_joint_states() -> dict:
    """Read the current J1–J6 joint positions from the robot (returned in degrees)."""
    return _base("robot_get_joint_states")


def robot_get_pose() -> dict:
    """Read the current end-effector pose via forward-kinematics (x,y,z,roll,pitch,yaw)."""
    return _base("robot_get_pose")


def robot_get_status() -> dict:
    """Read robot/rosbridge connection health and active goal count."""
    return _base("robot_get_status")


def robot_move_joints(
    j1: float = 0.0,
    j2: float = 0.0,
    j3: float = 0.0,
    j4: float = 0.0,
    j5: float = 0.0,
    j6: float = 0.0,
    speed: float = 0.1,
) -> dict:
    """
    Plan and execute a joint-space move via MoveIt2 (OMPL).

    Parameters
    ----------
    j1..j6: Target joint angles in degrees.
    speed:  Velocity scaling factor [0.0, 1.0].
    """
    return _base(
        "robot_move_joints",
        {"j1": j1, "j2": j2, "j3": j3, "j4": j4, "j5": j5, "j6": j6, "speed": speed},
    )


def robot_move_pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.5,
    roll: float = 180.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    speed: float = 0.1,
) -> dict:
    """
    Plan and execute a Cartesian end-effector move via MoveIt2 Pilz LIN planner.

    Parameters
    ----------
    x,y,z:        Target position in metres (base_link frame).
    roll,pitch,yaw: Target orientation in degrees (ZYX Euler).
    speed:         Velocity scaling factor [0.0, 1.0].
    """
    return _base(
        "robot_move_pose",
        {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw, "speed": speed},
    )


def robot_home(speed: float = 0.05) -> dict:
    """Move all joints to the home (all-zeros) position."""
    return _base("robot_home", {"speed": speed})


def robot_stop() -> dict:
    """Cancel all active MoveIt2 goals immediately."""
    return _base("robot_stop")


# ---------------------------------------------------------------------------
# Raspberry Pi command builders
# ---------------------------------------------------------------------------


def download_from_s3(
    s3_key: str = "",
    filename: Optional[str] = None,
    bucket: Optional[str] = None,
    file_url: Optional[str] = None,
) -> dict:
    """
    Ask the Pi agent to download a file from S3 into its local workspace.

    Parameters
    ----------
    s3_key:   Key inside the S3 bucket (e.g. "users/42/alice/myproject/logic.py").
    filename: Local filename to save as.  Defaults to the basename of s3_key.
    bucket:   S3 bucket name.  Falls back to the agent's AWS_S3_BUCKET env var.
    file_url: Pre-signed or direct URL — used instead of s3_key when provided.
    """
    payload: Dict[str, Any] = {}
    if s3_key:
        payload["s3_key"] = s3_key
    if file_url:
        payload["file_url"] = file_url
    if filename:
        payload["filename"] = filename
    if bucket:
        payload["bucket"] = bucket
    return _base("download_from_s3", payload)


def compile_and_run(
    file_path: str,
    args: Optional[str] = None,
    timeout: int = 120,
) -> dict:
    """
    Ask the Pi agent to compile (if needed) and run a file already in its workspace.

    Parameters
    ----------
    file_path: Absolute or workspace-relative path to the file on the Pi.
    args:      CLI argument string passed to the program.
    timeout:   Maximum execution seconds. Default 120.
    """
    payload: Dict[str, Any] = {"file_path": file_path, "timeout": timeout}
    if args:
        payload["args"] = args
    return _base("compile_and_run", payload)


def download_and_run(
    s3_key: str = "",
    filename: Optional[str] = None,
    args: Optional[str] = None,
    timeout: int = 120,
    bucket: Optional[str] = None,
    file_url: Optional[str] = None,
) -> dict:
    """
    Ask the Pi agent to download a code file from S3 and immediately compile / run it.

    Parameters
    ----------
    s3_key:   S3 key of the code file to fetch.
    filename: Local filename to save as.  Defaults to the basename of s3_key.
    args:     CLI argument string passed to the compiled / interpreted program.
    timeout:  Execution timeout in seconds. Default 120.
    bucket:   S3 bucket name.  Falls back to the agent's AWS_S3_BUCKET env var.
    file_url: Pre-signed or direct URL — used instead of s3_key when provided.
    """
    payload: Dict[str, Any] = {"timeout": timeout}
    if s3_key:
        payload["s3_key"] = s3_key
    if file_url:
        payload["file_url"] = file_url
    if filename:
        payload["filename"] = filename
    if bucket:
        payload["bucket"] = bucket
    if args:
        payload["args"] = args
    return _base("download_and_run", payload)
