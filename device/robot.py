"""
device/robot.py
MELFA robot shortcut endpoints for the device block.

A thin convenience layer over ``device.commands``: each route builds a robot
command block and relays it to the connected MELFA agent.  Mounted by
``device.app`` alongside the main hub router, sharing its ``ConnectionManager``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter

from . import commands as cmd_builder
from .models import RobotHomeRequest, RobotMoveJointsRequest, RobotMovePoseRequest
from .router import manager

logger = logging.getLogger("devices.robot")

router = APIRouter(prefix="/devices/{device_id}/robot", tags=["robot"])


@router.post("/joint_states")
async def robot_joint_states(device_id: str):
    """Ask the robot agent for the current J1–J6 joint positions (degrees)."""
    command = cmd_builder.robot_get_joint_states()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/pose")
async def robot_pose(device_id: str):
    """Ask the robot agent for the current end-effector pose (x,y,z,roll,pitch,yaw)."""
    command = cmd_builder.robot_get_pose()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/status")
async def robot_status(device_id: str):
    """Ask the robot agent for ROS bridge health and active goal count."""
    command = cmd_builder.robot_get_status()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/move_joints")
async def robot_move_joints(device_id: str, body: RobotMoveJointsRequest):
    """Plan and execute a joint-space move via MoveIt2."""
    command = cmd_builder.robot_move_joints(
        j1=body.j1, j2=body.j2, j3=body.j3,
        j4=body.j4, j5=body.j5, j6=body.j6,
        speed=body.speed,
    )
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/move_pose")
async def robot_move_pose(device_id: str, body: RobotMovePoseRequest):
    """Plan and execute a Cartesian move via MoveIt2 Pilz LIN planner."""
    command = cmd_builder.robot_move_pose(
        x=body.x, y=body.y, z=body.z,
        roll=body.roll, pitch=body.pitch, yaw=body.yaw,
        speed=body.speed,
    )
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/home")
async def robot_home(device_id: str, body: Optional[RobotHomeRequest] = None):
    """Move the robot to the home (all-zeros) position."""
    speed = (body or RobotHomeRequest()).speed
    command = cmd_builder.robot_home(speed=speed)
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/stop")
async def robot_stop(device_id: str):
    """Cancel all active MoveIt2 goals on the robot immediately."""
    command = cmd_builder.robot_stop()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}
