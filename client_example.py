"""
client_example.py
Minimal device agent client — runs on a Raspberry Pi (or any machine).

Usage
-----
Install the dependency:
    pip install websockets

Run:
    python client_example.py --host wss://your-app.onrender.com --device-id pi-001 --token YOUR_AGENT_TOKEN

You can also use environment variables instead of flags:

```bash
export AGENT_HOST=wss://grafux.onrender.com
export DEVICE_ID=pi-001
export AGENT_TOKEN=YOUR_AGENT_TOKEN
python client_example.py
```

The client will:
  1. Connect to the server WebSocket.
  2. Wait for command blocks (JSON dicts).
  3. Handle each command and send a result back.
  4. Reconnect automatically on disconnect.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time

try:
    import websockets
except ImportError:
    print("ERROR: websockets is not installed.")
    print("Install it with:  pip install websockets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("device.client")

# ---------------------------------------------------------------------------
# Default config — override with CLI args or env variables
# ---------------------------------------------------------------------------

DEFAULT_HOST = "ws://localhost:8000"
DEFAULT_DEVICE_ID = "pi-001"
DEFAULT_TOKEN = "changeme"
RECONNECT_DELAY = 5  # seconds between reconnect attempts


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def handle_ping(payload: dict) -> dict:
    return {"type": "pong", "message": "pong"}


def handle_get_status(payload: dict) -> dict:
    import os
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
        "uptime_seconds": None,
    }
    try:
        # Works on Linux / Raspberry Pi OS
        with open("/proc/uptime") as f:
            info["uptime_seconds"] = float(f.read().split()[0])
    except Exception:
        pass
    return {"type": "status_report", "data": info}


def _collect_files(directory: str) -> list:
    """Scan a directory and return base64-encoded file objects."""
    import base64
    import mimetypes
    files = []
    try:
        for fname in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "rb") as f:
                    content = base64.b64encode(f.read()).decode("utf-8")
                mime_type, _ = mimetypes.guess_type(fname)
                files.append({
                    "name":      fname,
                    "content":   content,
                    "mime_type": mime_type or "application/octet-stream",
                })
            except Exception:
                pass
    except Exception:
        pass
    return files


def handle_run_code(payload: dict) -> dict:
    import io
    import contextlib
    import tempfile

    code = payload.get("code", "")
    timeout = int(payload.get("timeout", 30))
    stdout_lines = []
    stderr_lines = []

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(compile(code, "<device>", "exec"), {})  # noqa: S102
                stdout_lines = stdout_buf.getvalue().splitlines()
                stderr_lines = stderr_buf.getvalue().splitlines()
                status = "ok"
            except Exception as exc:
                stdout_lines = stdout_buf.getvalue().splitlines()
                stderr_lines = stderr_buf.getvalue().splitlines() or [str(exc)]
                status = "error"
        finally:
            os.chdir(old_cwd)

        files = _collect_files(tmpdir)

    return {
        "type":   "run_code_result",
        "status": status,
        "stdout": stdout_lines,
        "stderr": stderr_lines,
        "files":  files,
    }


def handle_shell(payload: dict) -> dict:
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
    except Exception as exc:
        return {"type": "shell_result", "status": "error", "error": str(exc)}


def handle_set_config(payload: dict) -> dict:
    key = payload.get("key")
    value = payload.get("value")
    logger.info("Config update: %s = %s", key, value)
    return {"type": "config_ack", "key": key, "value": value}


def handle_restart(payload: dict) -> dict:
    logger.info("Restart requested — restarting in 2 seconds…")
    # Schedule actual restart after sending the ack
    asyncio.get_event_loop().call_later(2, lambda: os.execv(sys.executable, [sys.executable] + sys.argv))
    return {"type": "restart_ack", "message": "Restarting agent..."}


HANDLERS = {
    "ping": handle_ping,
    "get_status": handle_get_status,
    "run_code": handle_run_code,
    "shell": handle_shell,
    "set_config": handle_set_config,
    "restart": handle_restart,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_agent(host: str, device_id: str, token: str) -> None:
    url = f"{host}/ws?device_id={device_id}&token={token}"

    while True:
        try:
            logger.info("Connecting to %s …", url)
            async with websockets.connect(url) as ws:
                logger.info("Connected as device_id='%s'", device_id)

                async for raw_message in ws:
                    try:
                        message = json.loads(raw_message)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message received, ignoring.")
                        continue

                    command_type = message.get("type", "unknown")
                    command_id = message.get("id")
                    payload = message.get("payload", {})

                    logger.info("← received command type=%s id=%s", command_type, command_id)

                    handler = HANDLERS.get(command_type)
                    if handler:
                        try:
                            result = handler(payload)
                        except Exception as exc:
                            result = {"type": "error", "error": str(exc)}
                    else:
                        result = {"type": "unknown_command", "received_type": command_type}

                    # Attach the original command ID so the server can correlate
                    result["command_id"] = command_id
                    result["device_id"] = device_id
                    result["timestamp"] = time.time()

                    await ws.send(json.dumps(result))
                    logger.info("→ sent result type=%s", result.get("type"))

        except websockets.exceptions.ConnectionClosedError as exc:
            logger.warning("Connection closed: %s", exc)
        except OSError as exc:
            logger.error("Connection failed: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)

        logger.info("Reconnecting in %d seconds…", RECONNECT_DELAY)
        await asyncio.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Device agent client")
    parser.add_argument(
        "--host",
        default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
        help="Server WebSocket URL, e.g. wss://your-app.onrender.com",
    )
    parser.add_argument(
        "--device-id",
        default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
        help="Unique ID for this device",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
        help="Shared secret token (must match server AGENT_TOKEN)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_agent(args.host, args.device_id, args.token))
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
