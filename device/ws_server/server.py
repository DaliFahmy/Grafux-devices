"""
device/ws_server/server.py
Device-side WebSocket server.

Runs *on the device* (Raspberry Pi, laptop, workstation, …) and lets Grafux
connect to it directly — no central hub.  The Grafux "devices" block opens a
WebSocket to this server, sends the source code the user wrote, and this server
compiles + runs it locally and sends the results back for the output ports.

Topology
--------
    Grafux app  ──ws://device-host:8765/?token=…──►  this server (the device)

Protocol (clean direct API)
---------------------------
Client → device (one JSON text frame per request):

    { "id": "<uuid>", "action": "compile_and_run",
      "language": "cpp", "code": "<source>", "args": "", "timeout": 120 }
    { "id": "<uuid>", "action": "run_code", "code": "print(42)", "timeout": 30 }
    { "id": "<uuid>", "action": "shell",  "command": "uname -a", "timeout": 30 }
    { "id": "<uuid>", "action": "status" }
    { "id": "<uuid>", "action": "ping" }

device → client:

    { "id": "<uuid>", "action": "compile_and_run", "status": "ok",
      "output": "42", "errors": "", "warnings": "", "response": "...",
      "stdout": [...], "stderr": [...], "files": [...],
      "device_id": "mydev", "timestamp": 169... }

Usage
-----
    pip install -r requirements.txt
    python server.py --port 8765 --token <secret> --device-id mydev

Environment variables (all optional when flags are used):
    DEVICE_WS_PORT   Port to bind (default 8765)
    AGENT_TOKEN      Shared secret clients must supply as ?token=…
    DEVICE_ID        Identifier echoed back in every result
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from urllib.parse import parse_qs, urlparse

try:
    import websockets
except ImportError:
    print("ERROR: websockets is not installed.  Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    # When run as a script from this directory: ``python server.py``
    from handlers import dispatch
    from discovery import start_advertiser, stop_advertiser
except ImportError:
    # When imported as a package member: ``from device.ws_server import server``
    from device.ws_server.handlers import dispatch
    from device.ws_server.discovery import start_advertiser, stop_advertiser

#: Advertised protocol version (TXT record); bump on protocol changes.
VERSION = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("device.ws.server")

DEFAULT_PORT      = int(os.environ.get("DEVICE_WS_PORT", "8765"))
DEFAULT_TOKEN     = os.environ.get("AGENT_TOKEN", "changeme")
DEFAULT_DEVICE_ID = os.environ.get("DEVICE_ID", "device-001")


def _token_from_path(path: str) -> str:
    """Extract the ?token=… query parameter from the WebSocket request path."""
    query = urlparse(path).query
    values = parse_qs(query).get("token", [])
    return values[0] if values else ""


async def _handle_connection(websocket, token: str, device_id: str) -> None:
    """Authenticate, then serve compile/run requests over this socket."""
    # Request path location differs across websockets versions:
    #   v13+  → websocket.request.path
    #   <v13  → websocket.path
    request = getattr(websocket, "request", None)
    path = (getattr(request, "path", None)
            or getattr(websocket, "path", "")
            or "")
    if _token_from_path(path) != token:
        logger.warning("Rejected connection — invalid token")
        await websocket.close(code=1008, reason="invalid token")
        return

    peer = getattr(websocket, "remote_address", None)
    logger.info("Client connected from %s (device_id=%s)", peer, device_id)

    loop = asyncio.get_running_loop()

    try:
        async for raw in websocket:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "status": "error", "errors": "invalid JSON", "response": "error — invalid JSON",
                }))
                continue

            req_id = message.get("id")
            action = message.get("action") or message.get("type") or "ping"
            logger.info("recv action=%s id=%s", action, req_id)

            async def _send_progress(frame: dict) -> None:
                frame["id"]        = req_id
                frame["action"]    = action
                frame["type"]      = "progress"
                frame["device_id"] = device_id
                frame["timestamp"] = time.time()
                try:
                    await websocket.send(json.dumps(frame))
                except Exception:  # noqa: BLE001  — socket closed mid-run, ignore
                    pass

            # Bridge the worker thread's sync progress calls onto the event loop.
            def on_progress(frame: dict) -> None:
                asyncio.run_coroutine_threadsafe(_send_progress(frame), loop)

            # The clean protocol carries fields at the top level; pass the whole
            # message as the payload (handlers ignore id/action/type). Streaming
            # actions emit progress frames before this final result frame.
            result = await asyncio.to_thread(dispatch, action, message, on_progress)

            result["id"]        = req_id
            result["action"]    = action
            result["device_id"] = device_id
            result["timestamp"] = time.time()

            await websocket.send(json.dumps(result))
            logger.info("sent id=%s status=%s", req_id, result.get("status", "-"))
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected (%s)", peer)
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection error: %s", exc)


async def run_server(host: str, port: int, token: str, device_id: str,
                     advertise: bool = True) -> None:
    async def handler(websocket):
        await _handle_connection(websocket, token, device_id)

    logger.info("Device WebSocket server listening on ws://%s:%d  (device_id=%s)",
                host, port, device_id)
    mdns = start_advertiser(device_id, port, VERSION) if advertise else None
    try:
        async with websockets.serve(handler, host, port):
            await asyncio.Future()  # run forever
    finally:
        stop_advertiser(mdns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafux device-side WebSocket server")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default 8765)")
    parser.add_argument("--token", default=DEFAULT_TOKEN,
                        help="Shared secret clients must supply as ?token=…")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID,
                        help="Identifier echoed back in every result")
    parser.add_argument("--no-mdns", action="store_true",
                        help="Disable LAN mDNS discovery advertisement")
    args = parser.parse_args()

    if args.token == "changeme":
        logger.warning("Token is the default 'changeme' — set --token / AGENT_TOKEN in production!")

    try:
        asyncio.run(run_server(args.host, args.port, args.token, args.device_id,
                               advertise=not args.no_mdns))
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
