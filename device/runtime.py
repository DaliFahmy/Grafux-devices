"""
device/runtime.py
Orchestration for the device-block hub: mapping device replies onto Grafux
output ports, and the shared "send a command, optionally wait for the result"
flow used by the run_code / compile_and_run / download_and_run endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from .registry import ConnectionManager


def extract_ports(result: dict) -> dict:
    """Map a device result dict onto the Grafux device block's output ports.

    Falls back gracefully when the agent has not added the enriched port fields
    (e.g. older firmware, or non-execution result types).

    The 'files' port carries a list of base64-encoded output files (images,
    generated data, …) produced by the command — each a dict with keys
    ``name``, ``content`` (base64), ``mime_type``.
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


async def send_and_maybe_wait(
    manager: "ConnectionManager",
    device_id: str,
    command: dict,
    *,
    wait: bool,
    wait_timeout: float,
) -> dict:
    """Send *command* to a device and, when *wait* is true, block for its result.

    This collapses the three near-identical endpoint bodies (run_code,
    compile_and_run, download_and_run) into one place.

    Returns one of:
      * ``{"status": "sent", "command_id", "command"}``         (wait=false)
      * ``{"status": "timeout", "command_id"}``                  (wait=true, no reply)
      * ``{"status": "ready", "command_id", "ports", "result"}`` (wait=true, reply)

    The waiter is registered *before* the send to avoid a race where the device
    replies before we are listening.
    """
    command_id = command["id"]
    if wait:
        manager.register_waiter(command_id)  # register BEFORE send to avoid a race

    try:
        await manager.send(device_id, command)
    except Exception:
        # send failed (device gone / dead socket).  Drop the pre-registered
        # waiter so it doesn't leak in _pending, then let the error propagate.
        if wait:
            manager.cancel_waiter(command_id)
        raise

    if not wait:
        return {"status": "sent", "command_id": command_id, "command": command}

    result = await manager.wait_for_result(command_id, timeout=wait_timeout)
    if result is None:
        return {"status": "timeout", "command_id": command_id}
    return {
        "status":     "ready",
        "command_id": command_id,
        "ports":      extract_ports(result),
        "result":     result,
    }
