"""
devices_server.py
Device Agent WebSocket Server.

Each remote agent (e.g. Raspberry Pi) connects via WebSocket:

    ws://<host>/ws?device_id=<id>&token=<secret>

The server can then send command blocks to any connected device through
the REST API, and receives results back over the same socket.

Environment variables
---------------------
PORT         Port to bind (set automatically by Render).
AGENT_TOKEN  Shared secret that every agent must supply in the URL.
             Default: "changeme"  — always override in production!
"""

import json
import logging
import os
import sys
import time

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import commands as cmd_builder
from connection_manager import ConnectionManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("devices.server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_TOKEN: str = os.environ.get("AGENT_TOKEN", "changeme")
PORT: int = int(os.environ.get("PORT", "8000"))

if AGENT_TOKEN == "changeme":
    logger.warning(
        "AGENT_TOKEN is set to the default value 'changeme'. "
        "Set the AGENT_TOKEN environment variable in production!"
    )

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Device Agent Server",
    description="WebSocket hub for managing remote agents (Raspberry Pi, etc.)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------
# Maps command_id → {"result": <device message>, "ts": <unix time>}
# Entries are consumed (deleted) when fetched and cleaned up after RESULT_TTL_S.
_result_store: dict[str, dict] = {}
RESULT_TTL_S: int = 120  # seconds before an uncollected result is discarded


def _store_result(command_id: str, message: dict) -> None:
    """Persist a device result and purge stale entries."""
    _result_store[command_id] = {"result": message, "ts": time.time()}
    # Purge entries older than RESULT_TTL_S to avoid unbounded growth
    stale = [k for k, v in _result_store.items() if time.time() - v["ts"] > RESULT_TTL_S]
    for k in stale:
        del _result_store[k]

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    device_id: str = Query(..., description="Unique identifier for this device"),
    token: str = Query(..., description="Shared secret token"),
):
    """
    WebSocket connection endpoint for device agents.

    Devices connect with:
        ws://<host>/ws?device_id=<id>&token=<secret>

    Unauthorized connections are closed with code 1008 (Policy Violation).
    """
    if token != AGENT_TOKEN:
        logger.warning(
            "Rejected connection from device_id='%s' — invalid token", device_id
        )
        await websocket.close(code=1008)
        return

    await manager.connect(device_id, websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[%s] Received non-JSON message: %s", device_id, raw[:200])
                continue

            msg_type = message.get("type", "unknown")
            logger.info("[%s] ← received type=%s", device_id, msg_type)

            # Log the full payload at DEBUG level (avoid flooding INFO in prod)
            logger.debug("[%s] payload: %s", device_id, message)

            # Store result so REST clients can retrieve it by command_id.
            # We index by:
            #   - the command id embedded in the message (preferred, exact match)
            #   - "latest:<device_id>" (fallback for commands without ids)
            cmd_id = message.get("id")
            if cmd_id and cmd_id != "None":
                _store_result(str(cmd_id), message)
            _store_result(f"latest:{device_id}", message)

            # Resolve any REST caller that is awaiting this result.
            # The Pi echoes the original command_id back in "command_id".
            # We store under that key so the polling endpoint
            # GET /devices/{id}/result/{cmdId} can find it.
            echo_id = message.get("command_id")
            if echo_id:
                _store_result(str(echo_id), message)
                resolved = manager.resolve_result(str(echo_id), message)
                if resolved:
                    logger.debug("[%s] resolved pending waiter for command_id=%s", device_id, echo_id)

    except WebSocketDisconnect:
        manager.disconnect(device_id)


# ---------------------------------------------------------------------------
# Port extractor
# ---------------------------------------------------------------------------


def _extract_ports(result: dict) -> dict:
    """Map a device result dict to Grafux output port values.

    Falls back gracefully when the agent has not yet added the enriched
    port fields (e.g. older Pi firmware or non-execution result types).

    The 'files' port carries a list of base64-encoded output files (images,
    generated data, etc.) produced by the device command — for example a JPEG
    captured by the Windows agent's capture_image command.  Each entry is a
    dict with keys: name, content (base64), mime_type.
    """
    return {
        "output":   result.get("output",   "\n".join(result.get("stdout", []))),
        "errors":   result.get("errors",   "\n".join(result.get("stderr", []))),
        "warnings": result.get("warnings", ""),
        "status":   result.get("status",   "unknown"),
        "response": result.get("response", ""),
        "files":    result.get("files",    []),
        "file":     result.get("file",     ""),
    }


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check for Render and uptime monitors."""
    return {"status": "ok", "connected_devices": len(manager.list_devices())}


@app.get("/devices")
async def list_devices():
    """Return a list of currently connected device IDs."""
    return {"devices": manager.list_devices()}


@app.get("/devices/{device_id}/result/{command_id}")
async def get_command_result(device_id: str, command_id: str):
    """
    Poll for the result of a previously sent command.

    Returns {"status": "ready", "result": <device message>} when the device
    has responded, or {"status": "pending"} if not yet received.

    The result is consumed (deleted) on first successful retrieval.
    The special command_id "latest" returns the most recent result from
    this device regardless of which command triggered it.
    """
    key = f"latest:{device_id}" if command_id == "latest" else command_id
    entry = _result_store.get(key)
    if entry is None:
        return {"status": "pending"}
    # Consume the entry (but keep "latest:<device>" so it can be re-read)
    if command_id != "latest":
        del _result_store[key]
    return {"status": "ready", "result": entry["result"]}


@app.post("/devices/{device_id}/command")
async def send_command(device_id: str, body: dict):
    """
    Send a pre-built command block to a specific device.

    The body should be a valid command dict produced by commands.py, e.g.:

        POST /devices/pi-001/command
        {"type": "ping"}

        POST /devices/pi-001/command
        {"type": "run_code", "payload": {"code": "print(1+1)", "timeout": 10}}

    You can also use the helper endpoints below for common commands.
    """
    await manager.send(device_id, body)
    return {"status": "sent", "device_id": device_id, "command": body}


@app.post("/devices/{device_id}/ping")
async def send_ping(device_id: str):
    """Send a ping to a device and expect a pong result back."""
    command = cmd_builder.ping()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/status")
async def send_get_status(device_id: str):
    """Ask a device to report its current system status."""
    command = cmd_builder.get_status()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/run_code")
async def send_run_code(device_id: str, body: dict, wait: bool = True):
    """
    Ask a device to execute Python code.

    Body fields:
    - code    (str, required): Python source to run.
    - timeout (int, optional): Max execution seconds. Default 30.

    Query params:
    - wait (bool, default true): If true, block until the Pi returns a result
      and return it with a "ports" key ready for Grafux output ports.
      Set wait=false to get the fire-and-forget behaviour (returns command_id
      so you can poll GET /devices/{id}/result/{command_id} yourself).
    """
    code = body.get("code", "")
    timeout = int(body.get("timeout", 30))
    command = cmd_builder.run_code(code, timeout)
    if wait:
        manager.register_waiter(command["id"])   # register BEFORE send to avoid race
    await manager.send(device_id, command)
    if not wait:
        return {"status": "sent", "command_id": command["id"], "command": command}
    result = await manager.wait_for_result(command["id"], timeout=timeout + 10)
    if result is None:
        return {"status": "timeout", "command_id": command["id"]}
    return {
        "status":     "ready",
        "command_id": command["id"],
        "ports":      _extract_ports(result),
        "result":     result,
    }


@app.post("/devices/{device_id}/shell")
async def send_shell(device_id: str, body: dict):
    """
    Ask a device to run a shell command.

    Body fields:
    - command (str, required): Shell command string.
    - timeout (int, optional): Max execution seconds. Default 30.
    """
    command = cmd_builder.shell(body.get("command", ""), int(body.get("timeout", 30)))
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/broadcast")
async def broadcast_command(body: dict):
    """Send a command block to ALL connected devices at once."""
    await manager.broadcast(body)
    return {"status": "broadcast_sent", "devices": manager.list_devices(), "command": body}


# ---------------------------------------------------------------------------
# Robot (MELFA) shortcut endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Raspberry Pi shortcut endpoints
# ---------------------------------------------------------------------------


@app.post("/devices/{device_id}/download_from_s3")
async def send_download_from_s3(device_id: str, body: dict):
    """
    Ask a Raspberry Pi agent to download a file from S3 into its local workspace.

    Body fields:
    - s3_key  (str, required*): S3 key of the file (e.g. "users/42/alice/logic.py").
    - file_url (str, required*): Pre-signed / direct URL — alternative to s3_key.
    - filename (str, optional): Local filename to save as. Defaults to s3_key basename.
    - bucket  (str, optional): S3 bucket name. Falls back to agent's AWS_S3_BUCKET env var.

    *One of s3_key or file_url is required.
    """
    command = cmd_builder.download_from_s3(
        s3_key=body.get("s3_key", ""),
        filename=body.get("filename"),
        bucket=body.get("bucket"),
        file_url=body.get("file_url"),
    )
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/compile_and_run")
async def send_compile_and_run(device_id: str, body: dict, wait: bool = True):
    """
    Ask a Raspberry Pi agent to compile (if needed) and run a file in its workspace.

    Body fields:
    - file_path (str, required): Absolute or workspace-relative path on the Pi.
    - args      (str, optional): CLI argument string.
    - timeout   (int, optional): Execution timeout seconds. Default 120.

    Query params:
    - wait (bool, default true): Block until the Pi returns port-mapped results.
    """
    timeout = int(body.get("timeout", 120))
    command = cmd_builder.compile_and_run(
        file_path=body.get("file_path", ""),
        args=body.get("args"),
        timeout=timeout,
    )
    if wait:
        manager.register_waiter(command["id"])   # register BEFORE send to avoid race
    await manager.send(device_id, command)
    if not wait:
        return {"status": "sent", "command_id": command["id"], "command": command}
    result = await manager.wait_for_result(command["id"], timeout=timeout + 10)
    if result is None:
        return {"status": "timeout", "command_id": command["id"]}
    return {
        "status":     "ready",
        "command_id": command["id"],
        "ports":      _extract_ports(result),
        "result":     result,
    }


@app.post("/devices/{device_id}/download_and_run")
async def send_download_and_run(device_id: str, body: dict, wait: bool = True):
    """
    Ask a Raspberry Pi agent to download a code file from S3 and immediately run it.

    This is the primary endpoint used by the Grafux diagram Device block when the
    user sets command='download_and_run' and populates the 'file' port with an S3 key.

    Body fields:
    - s3_key  (str, required*): S3 key of the code file.
    - file_url (str, required*): Pre-signed / direct URL — alternative to s3_key.
    - filename (str, optional): Local filename.  Defaults to s3_key basename.
    - bucket  (str, optional): S3 bucket name.
    - args    (str, optional): CLI arguments passed to the program.
    - timeout (int, optional): Execution timeout seconds. Default 120.

    Query params:
    - wait (bool, default true): Block until the Pi returns port-mapped results.
      The response will contain a "ports" dict with keys: output, errors, warnings,
      status, response — one per Grafux output port of the Device block.

    *One of s3_key or file_url is required.
    """
    timeout = int(body.get("timeout", 120))
    command = cmd_builder.download_and_run(
        s3_key=body.get("s3_key", ""),
        filename=body.get("filename"),
        args=body.get("args"),
        timeout=timeout,
        bucket=body.get("bucket"),
        file_url=body.get("file_url"),
    )
    if wait:
        manager.register_waiter(command["id"])   # register BEFORE send to avoid race
    await manager.send(device_id, command)
    if not wait:
        return {"status": "sent", "command_id": command["id"], "command": command}
    # Add extra headroom: download time + compile time + run timeout
    result = await manager.wait_for_result(command["id"], timeout=timeout + 30)
    if result is None:
        return {"status": "timeout", "command_id": command["id"]}
    return {
        "status":     "ready",
        "command_id": command["id"],
        "ports":      _extract_ports(result),
        "result":     result,
    }


@app.post("/devices/{device_id}/robot/joint_states")
async def robot_joint_states(device_id: str):
    """Ask the robot agent for the current J1–J6 joint positions (degrees)."""
    command = cmd_builder.robot_get_joint_states()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/robot/pose")
async def robot_pose(device_id: str):
    """Ask the robot agent for the current end-effector pose (x,y,z,roll,pitch,yaw)."""
    command = cmd_builder.robot_get_pose()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/robot/status")
async def robot_status(device_id: str):
    """Ask the robot agent for ROS bridge health and active goal count."""
    command = cmd_builder.robot_get_status()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/robot/move_joints")
async def robot_move_joints(device_id: str, body: dict):
    """
    Ask the robot to execute a joint-space move via MoveIt2.

    Body fields:
    - j1..j6 (float, degrees): target joint angles. Default 0.
    - speed  (float, 0–1):     velocity scaling factor. Default 0.1.
    """
    command = cmd_builder.robot_move_joints(
        j1=float(body.get("j1", 0.0)),
        j2=float(body.get("j2", 0.0)),
        j3=float(body.get("j3", 0.0)),
        j4=float(body.get("j4", 0.0)),
        j5=float(body.get("j5", 0.0)),
        j6=float(body.get("j6", 0.0)),
        speed=float(body.get("speed", 0.1)),
    )
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/robot/move_pose")
async def robot_move_pose(device_id: str, body: dict):
    """
    Ask the robot to execute a Cartesian move via MoveIt2 Pilz LIN planner.

    Body fields:
    - x,y,z       (float, metres):  target position in base_link frame.
    - roll,pitch,yaw (float, degrees): target ZYX Euler orientation.
    - speed        (float, 0–1):    velocity scaling factor. Default 0.1.
    """
    command = cmd_builder.robot_move_pose(
        x=float(body.get("x", 0.0)),
        y=float(body.get("y", 0.0)),
        z=float(body.get("z", 0.5)),
        roll=float(body.get("roll", 180.0)),
        pitch=float(body.get("pitch", 0.0)),
        yaw=float(body.get("yaw", 0.0)),
        speed=float(body.get("speed", 0.1)),
    )
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/robot/home")
async def robot_home(device_id: str, body: dict = None):
    """Move the robot to the home (all-zeros) position."""
    speed = float((body or {}).get("speed", 0.05))
    command = cmd_builder.robot_home(speed=speed)
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


@app.post("/devices/{device_id}/robot/stop")
async def robot_stop(device_id: str):
    """Cancel all active MoveIt2 goals on the robot immediately."""
    command = cmd_builder.robot_stop()
    await manager.send(device_id, command)
    return {"status": "sent", "command": command}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Device Agent Server on 0.0.0.0:%d", PORT)
    uvicorn.run("devices_server:app", host="0.0.0.0", port=PORT, reload=False)
