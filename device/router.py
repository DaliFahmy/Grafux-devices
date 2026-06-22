"""
device/router.py
The device-block hub, exposed over REST + WebSocket and mounted by ``device.app``.

Topology
--------
    agent (Pi, Windows, MELFA, …)  ──ws://<host>/ws?device_id=…&token=…──►  this hub
    Grafux-app                     ──REST /devices/{id}/…──►                this hub

Agents connect over ``/ws``; Grafux-app sends them command blocks through the
REST endpoints and reads results back as Grafux output ports.  The companion
robot endpoints live in ``device.robot``.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from . import commands as cmd_builder
from . import runtime
from .constants import AGENT_TOKEN, DOWNLOAD_WAIT_BUFFER_S, RUN_WAIT_BUFFER_S
from .models import (
    CompileAndRunRequest,
    DownloadAndRunRequest,
    DownloadRequest,
    RunCodeRequest,
    ShellRequest,
)
from .registry import ConnectionManager
from .results import ResultStore

logger = logging.getLogger("devices.server")

# ---------------------------------------------------------------------------
# Shared singletons (re-exported by device.app)
# ---------------------------------------------------------------------------

manager = ConnectionManager()
results = ResultStore()

router = APIRouter(tags=["devices"])


# ---------------------------------------------------------------------------
# WebSocket endpoint — agents connect here
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    device_id: str = Query(..., description="Unique identifier for this device"),
    token: str = Query(..., description="Shared secret token"),
):
    """Accept a device agent connection and relay its result messages.

    Devices connect with ``ws://<host>/ws?device_id=<id>&token=<secret>``.
    Unauthorized connections are closed with code 1008 (Policy Violation).
    """
    if token != AGENT_TOKEN:
        logger.warning("Rejected connection from device_id='%s' — invalid token", device_id)
        await websocket.close(code=1008)
        return

    await manager.connect(device_id, websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                _ingest_device_message(device_id, raw)
            except Exception:  # noqa: BLE001 — one bad message must not drop the socket
                logger.exception("[%s] error handling device message (ignored)", device_id)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 — log unexpected socket faults instead of silently dropping
        logger.exception("[%s] websocket loop error", device_id)
    finally:
        # Only evict if we are still the registered socket (a reconnect may have
        # already replaced us).
        manager.disconnect(device_id, websocket)


def _ingest_device_message(device_id: str, raw: str) -> None:
    """Parse one inbound device frame and route its result (store + resolve)."""
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[%s] Received non-JSON message: %s", device_id, raw[:200])
        return

    msg_type = message.get("type", "unknown")
    logger.info("[%s] <- received type=%s", device_id, msg_type)
    logger.debug("[%s] payload: %s", device_id, message)

    # Make the reply retrievable by REST pollers (the wait=false path).
    # Devices use one of two id conventions, so index by both:
    #   - message["id"]         (e.g. device.ws_server echoes the request id)
    #   - message["command_id"] (e.g. the Pi agent echoes the original id)
    # plus a re-readable "latest:<device_id>" fallback.
    cmd_id = message.get("id")
    if cmd_id and cmd_id != "None":
        results.put(str(cmd_id), message)
    results.put(f"latest:{device_id}", message)

    # Deliver to any REST caller awaiting this result (the wait=true path).
    echo_id = message.get("command_id")
    if echo_id:
        if manager.resolve_result(str(echo_id), message):
            # A Future already delivered it — no need to also cache it.
            logger.debug("[%s] resolved pending waiter for command_id=%s", device_id, echo_id)
        else:
            results.put(str(echo_id), message)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Health check for Render and uptime monitors."""
    return {"status": "ok", "connected_devices": len(manager.list_devices())}


@router.get("/devices")
async def list_devices():
    """Return a list of currently connected device IDs."""
    return {"devices": manager.list_devices()}


@router.get("/devices/{device_id}/result/{command_id}")
async def get_command_result(device_id: str, command_id: str):
    """Poll for the result of a previously sent command.

    Returns ``{"status": "ready", "result": …}`` once the device has responded,
    or ``{"status": "pending"}`` otherwise.  A specific command result is
    consumed on first read; the special id ``latest`` returns (re-readably) the
    most recent result from this device.
    """
    if command_id == "latest":
        entry = results.peek(f"latest:{device_id}")
    else:
        entry = results.pop(command_id)
    if entry is None:
        return {"status": "pending"}
    return {"status": "ready", "result": entry}


@router.post("/devices/{device_id}/command")
async def send_command(device_id: str, body: dict):
    """Send a pre-built command block (a dict produced by ``commands.py``)."""
    await manager.send(device_id, body)
    return {"status": "sent", "device_id": device_id, "command": body}


@router.post("/devices/{device_id}/ping")
async def send_ping(device_id: str):
    """Send a ping to a device and expect a pong result back."""
    command = cmd_builder.ping()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/devices/{device_id}/status")
async def send_get_status(device_id: str):
    """Ask a device to report its current system status."""
    command = cmd_builder.get_status()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/devices/{device_id}/run_code")
async def send_run_code(device_id: str, body: RunCodeRequest, wait: bool = True):
    """Ask a device to execute Python code.

    ``wait=true`` (default) blocks until the device returns a result and maps it
    onto Grafux output ports; ``wait=false`` returns a command_id to poll.
    """
    command = cmd_builder.run_code(body.code, body.timeout)
    return await runtime.send_and_maybe_wait(
        manager, device_id, command, wait=wait, wait_timeout=body.timeout + RUN_WAIT_BUFFER_S
    )


@router.post("/devices/{device_id}/shell")
async def send_shell(device_id: str, body: ShellRequest):
    """Ask a device to run a shell command."""
    command = cmd_builder.shell(body.command, body.timeout)
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/broadcast")
async def broadcast_command(body: dict):
    """Send a command block to ALL connected devices at once."""
    await manager.broadcast(body)
    return {"status": "broadcast_sent", "devices": manager.list_devices(), "command": body}


@router.post("/devices/{device_id}/download_from_s3")
async def send_download_from_s3(device_id: str, body: DownloadRequest):
    """Ask a Pi agent to download a file from S3 into its local workspace.

    One of ``s3_key`` or ``file_url`` is required.
    """
    command = cmd_builder.download_from_s3(
        s3_key=body.s3_key,
        filename=body.filename,
        bucket=body.bucket,
        file_url=body.file_url,
    )
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@router.post("/devices/{device_id}/compile_and_run")
async def send_compile_and_run(device_id: str, body: CompileAndRunRequest, wait: bool = True):
    """Ask a Pi agent to compile (if needed) and run a file in its workspace."""
    command = cmd_builder.compile_and_run(
        file_path=body.file_path,
        args=body.args,
        timeout=body.timeout,
    )
    return await runtime.send_and_maybe_wait(
        manager, device_id, command, wait=wait, wait_timeout=body.timeout + RUN_WAIT_BUFFER_S
    )


@router.post("/devices/{device_id}/download_and_run")
async def send_download_and_run(device_id: str, body: DownloadAndRunRequest, wait: bool = True):
    """Download a code file from S3 and immediately compile / run it on a Pi agent.

    Primary endpoint for the Grafux device block when command='download_and_run'
    and the 'file' port holds an S3 key.  One of ``s3_key`` or ``file_url`` is
    required.  The wait timeout adds extra headroom for download + compile + run.
    """
    command = cmd_builder.download_and_run(
        s3_key=body.s3_key,
        filename=body.filename,
        args=body.args,
        timeout=body.timeout,
        bucket=body.bucket,
        file_url=body.file_url,
    )
    return await runtime.send_and_maybe_wait(
        manager, device_id, command, wait=wait, wait_timeout=body.timeout + DOWNLOAD_WAIT_BUFFER_S
    )
