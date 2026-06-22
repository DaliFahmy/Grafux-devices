"""
device/models.py
Typed request bodies for the device-block REST endpoints.

These replace the previous untyped ``body: dict`` handlers.  Every field is
optional with a default that matches the old ``body.get(field, default)`` calls,
so the accepted wire format is unchanged — the models only add validation and
make the contract self-documenting.

Port → field mapping (the Grafux device block):
    code      -> RunCodeRequest.code
    command   -> ShellRequest.command
    file/s3   -> DownloadRequest.s3_key / file_url
    file_path -> CompileAndRunRequest.file_path
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RunCodeRequest(BaseModel):
    """POST /devices/{id}/run_code — execute a Python snippet on the device."""

    code: str = ""
    timeout: int = 30


class ShellRequest(BaseModel):
    """POST /devices/{id}/shell — run a shell command on the device."""

    command: str = ""
    timeout: int = 30


class DownloadRequest(BaseModel):
    """POST /devices/{id}/download_from_s3 — fetch a file into the device workspace."""

    s3_key: str = ""
    file_url: Optional[str] = None
    filename: Optional[str] = None
    bucket: Optional[str] = None


class CompileAndRunRequest(BaseModel):
    """POST /devices/{id}/compile_and_run — compile (if needed) and run a workspace file."""

    file_path: str = ""
    args: Optional[str] = None
    timeout: int = 120


class DownloadAndRunRequest(BaseModel):
    """POST /devices/{id}/download_and_run — fetch from S3 then compile/run immediately."""

    s3_key: str = ""
    file_url: Optional[str] = None
    filename: Optional[str] = None
    bucket: Optional[str] = None
    args: Optional[str] = None
    timeout: int = 120


class RobotMoveJointsRequest(BaseModel):
    """POST /devices/{id}/robot/move_joints — joint-space move (degrees)."""

    j1: float = 0.0
    j2: float = 0.0
    j3: float = 0.0
    j4: float = 0.0
    j5: float = 0.0
    j6: float = 0.0
    speed: float = 0.1


class RobotMovePoseRequest(BaseModel):
    """POST /devices/{id}/robot/move_pose — Cartesian move (metres + degrees)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.5
    roll: float = 180.0
    pitch: float = 0.0
    yaw: float = 0.0
    speed: float = 0.1


class RobotHomeRequest(BaseModel):
    """POST /devices/{id}/robot/home — move to the all-zeros home position."""

    speed: float = 0.05
