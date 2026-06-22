"""
raspberry_pi/agent.py
Raspberry Pi device agent — runs on the Pi itself.

Connects to the Grafux device hub, listens for command blocks, executes them
locally, and sends results back.  Supports downloading user code from S3/URL and
compiling / running it on the Pi.

The connection loop, reconnect/keepalive, dispatch and result envelope live in
``device.agents.base.BaseAgent``; the cross-platform handlers and helpers live in
``device.agents.common``.  This module only contributes the Pi-specific bits:
``get_status`` (reads ``/proc/uptime``) and the compile/run command resolution
for .py / .c / .cpp / .sh.

Usage
-----
    pip install -r requirements.txt
    python agent.py --host wss://grafux.onrender.com --device-id pi-001 --token topsecret

Environment variables (all optional when flags are used):
    AGENT_HOST, DEVICE_ID, AGENT_TOKEN, WORKSPACE_DIR,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_REGION
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys

# Make the shared ``device`` package importable when this file is run directly
# as ``python agent.py`` (deploy the device/agents/ tree alongside this script).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from device.agents import common  # noqa: E402
from device.agents.base import BaseAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pi.agent")

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "ws://localhost:8000"
DEFAULT_DEVICE_ID = "pi-001"
DEFAULT_TOKEN = "changeme"

WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR", os.path.join(os.path.expanduser("~"), "pi_workspace")
)

# Extensions beyond common's base map (.py/.c/.cpp/.cc/.cxx).
_LANGUAGE_EXTRA = {".sh": "shell", ".bash": "shell"}


def _ensure_workspace() -> str:
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR


# ---------------------------------------------------------------------------
# Pi-specific handlers
# ---------------------------------------------------------------------------

def handle_get_status(payload: dict) -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
        "uptime_seconds": None,
        "workspace_dir": WORKSPACE_DIR,
    }
    try:
        with open("/proc/uptime") as f:
            info["uptime_seconds"] = float(f.read().split()[0])
    except Exception:  # noqa: BLE001 — /proc/uptime is Linux-only
        pass
    return {"type": "status_report", "status": "ok", "data": info}


def _resolve_run_cmd(language, file_path, args_list, workspace):
    """Map a detected language to its run command, compiling C/C++ as needed."""
    if language == "python":
        return [sys.executable, file_path] + args_list, None, None

    if language == "c":
        binary = file_path.replace(".c", "")
        cp = common.compile_source(["gcc", "-o", binary, file_path, "-lm"])
        diag = (cp.stdout.splitlines(), cp.stderr.splitlines(), cp.returncode)
        if cp.returncode != 0:
            return None, common.compile_error_result(*diag), None
        return [binary] + args_list, None, diag

    if language == "cpp":
        binary = os.path.splitext(file_path)[0]
        cp = common.compile_source(["g++", "-o", binary, file_path, "-lm", "-lstdc++"])
        diag = (cp.stdout.splitlines(), cp.stderr.splitlines(), cp.returncode)
        if cp.returncode != 0:
            return None, common.compile_error_result(*diag), None
        return [binary] + args_list, None, diag

    if language == "shell":
        os.chmod(file_path, 0o755)
        return ["/bin/bash", file_path] + args_list, None, None

    return None, {
        "type": "compile_run_result",
        "status": "error",
        "error": f"Unsupported file type: {os.path.basename(file_path)}. "
                 "Supported: .py, .c, .cpp, .cc, .cxx, .sh",
    }, None


def handle_compile_and_run(payload: dict) -> dict:
    return common.compile_and_run(
        payload,
        workspace=_ensure_workspace(),
        resolve=_resolve_run_cmd,
        language_extra=_LANGUAGE_EXTRA,
        tool_hint="Install gcc/g++ on the Pi.",
    )


def handle_download_from_s3(payload: dict) -> dict:
    return common.download_from_s3(payload, workspace=_ensure_workspace())


def handle_download_and_run(payload: dict) -> dict:
    return common.download_and_run(
        payload, workspace=_ensure_workspace(), compile_and_run_fn=handle_compile_and_run
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

def handle_shell(payload: dict) -> dict:
    """Run an arbitrary shell command (POSIX shell)."""
    command = payload.get("command", "")
    timeout = common.safe_int(payload.get("timeout"), 30)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,  # noqa: S602
        )
        return {
            "type": "shell_result",
            "status": "ok" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout.splitlines(),
            "stderr": result.stderr.splitlines(),
        }
    except subprocess.TimeoutExpired:
        return {"type": "shell_result", "status": "timeout", "command": command}
    except Exception as exc:  # noqa: BLE001
        return {"type": "shell_result", "status": "error", "error": str(exc)}


HANDLERS = {
    "ping": common.handle_ping,
    "get_status": handle_get_status,
    "run_code": common.run_code,
    "shell": handle_shell,
    "set_config": common.handle_set_config,
    "restart": common.handle_restart,
    "download_from_s3": handle_download_from_s3,
    "compile_and_run": handle_compile_and_run,
    "download_and_run": handle_download_and_run,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi device agent")
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
                        help="Server WebSocket URL (e.g. wss://your-app.onrender.com)")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
                        help="Unique ID for this Pi")
    parser.add_argument("--token", default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
                        help="Shared secret token (must match server AGENT_TOKEN)")
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE_DIR", WORKSPACE_DIR),
                        help="Local directory for downloaded / compiled files")
    args = parser.parse_args()

    WORKSPACE_DIR = args.workspace
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    logger.info("workspace = %s", WORKSPACE_DIR)
    BaseAgent(args.host, args.device_id, args.token, handlers=HANDLERS, logger=logger).run_forever()
