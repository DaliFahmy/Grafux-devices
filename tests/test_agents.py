"""
test_agents.py
Unit tests for the shared agent runtime (device.agents.base) and the common
handler helpers (device.agents.common), including the reliability improvements:
  * BaseAgent.dispatch isolates handler errors and supports sync + async handlers
  * run_code runs in a subprocess and the timeout is actually enforced
  * compile_and_run confines file paths to the workspace
"""

import os
import tempfile

import pytest

from device.agents import common
from device.agents.base import BaseAgent


# ---------------------------------------------------------------------------
# BaseAgent.dispatch
# ---------------------------------------------------------------------------

async def test_dispatch_sync_handler():
    a = BaseAgent("ws://h", "d", "t", handlers={"ping": common.handle_ping})
    r = await a.dispatch("ping", {})
    assert r["type"] == "pong" and r["status"] == "ok"


async def test_dispatch_async_handler():
    async def ah(payload):
        return {"type": "async_ok", "status": "ok", "echo": payload.get("v")}

    a = BaseAgent("ws://h", "d", "t", handlers={"a": ah})
    r = await a.dispatch("a", {"v": 5})
    assert r == {"type": "async_ok", "status": "ok", "echo": 5}


async def test_dispatch_unknown_command():
    a = BaseAgent("ws://h", "d", "t", handlers={"ping": common.handle_ping})
    r = await a.dispatch("nope", {})
    assert r["type"] == "unknown_command"
    assert "ping" in r["available_commands"]


async def test_dispatch_handler_error_becomes_result():
    def boom(payload):
        raise ValueError("kaboom")

    a = BaseAgent("ws://h", "d", "t", handlers={"boom": boom})
    r = await a.dispatch("boom", {})
    assert r["type"] == "error" and r["status"] == "error"
    assert "kaboom" in r["error"]


def test_envelope_adds_correlation_fields():
    a = BaseAgent("ws://h", "dev42", "t")
    r = a._envelope({"type": "pong"}, "cmd-1")
    assert r["command_id"] == "cmd-1"
    assert r["device_id"] == "dev42"
    assert "timestamp" in r


# ---------------------------------------------------------------------------
# common helpers
# ---------------------------------------------------------------------------

def test_safe_int():
    assert common.safe_int("3", 0) == 3
    assert common.safe_int("abc", 9) == 9
    assert common.safe_int(None, 7) == 7


def test_split_diagnostics():
    w, e = common.split_diagnostics(["foo warning: x", "bar error: y", "fatal: z", "note"])
    assert w == ["foo warning: x"]
    assert e == ["bar error: y", "fatal: z"]


def test_detect_language_with_extra():
    assert common.detect_language("a.py") == "python"
    assert common.detect_language("a.cpp") == "cpp"
    assert common.detect_language("a.sh", {".sh": "shell"}) == "shell"
    assert common.detect_language("a.xyz") == "unknown"


def test_confine_path_allows_inside_blocks_outside():
    ws = tempfile.mkdtemp()
    inside = common.confine_path("sub/x.py", ws)
    assert inside and inside.startswith(os.path.abspath(ws))
    outside = common.confine_path(os.path.abspath(os.path.join(ws, "..", "evil.py")), ws)
    assert outside is None


# ---------------------------------------------------------------------------
# run_code — subprocess execution with an enforced timeout
# ---------------------------------------------------------------------------

def test_run_code_captures_output():
    r = common.run_code({"code": "print(6 * 7)"})
    assert r["status"] == "ok"
    assert r["output"] == "42"


def test_run_code_reports_error_status():
    r = common.run_code({"code": "raise SystemExit(3)"})
    assert r["status"] == "error"


def test_run_code_enforces_timeout():
    r = common.run_code({"code": "import time; time.sleep(5)", "timeout": 1})
    assert r["status"] == "timeout"


# ---------------------------------------------------------------------------
# compile_and_run — workspace confinement (no compiler needed for this path)
# ---------------------------------------------------------------------------

def test_compile_and_run_rejects_path_outside_workspace():
    ws = tempfile.mkdtemp()

    def resolve(language, file_path, args_list, workspace):  # pragma: no cover - not reached
        return [], None, None

    r = common.compile_and_run(
        {"file_path": os.path.abspath(os.path.join(ws, "..", "etc", "passwd"))},
        workspace=ws,
        resolve=resolve,
    )
    assert r["status"] == "error"
    assert "inside the workspace" in r["error"]


def test_compile_and_run_runs_python(tmp_path):
    ws = str(tmp_path)
    src = os.path.join(ws, "hello.py")
    with open(src, "w") as f:
        f.write("print('hi from file')")

    import sys

    def resolve(language, file_path, args_list, workspace):
        assert language == "python"
        return [sys.executable, file_path] + args_list, None, None

    r = common.compile_and_run({"file_path": "hello.py"}, workspace=ws, resolve=resolve)
    assert r["status"] == "ok"
    assert r["output"] == "hi from file"
