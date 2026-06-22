"""
test_ws_handlers.py
Unit tests for the device-side ws_server handlers, focused on the run_code
timeout hardening (previously in-process exec with no enforceable timeout).
"""

from device.ws_server import handlers as h


def test_run_code_captures_output():
    r = h.handle_run_code({"code": "print(1 + 2)"})
    assert r["status"] == "ok"
    assert r["output"] == "3"


def test_run_code_enforces_timeout():
    r = h.handle_run_code({"code": "while True: pass", "timeout": 1})
    assert r["status"] == "timeout"


def test_run_code_bad_timeout_does_not_crash():
    r = h.handle_run_code({"code": "print(9)", "timeout": "not-an-int"})
    assert r["status"] == "ok"
    assert r["output"] == "9"


def test_run_code_requires_code():
    r = h.handle_run_code({})
    assert r["status"] == "error"


def test_ping_and_unknown_dispatch():
    assert h.dispatch("ping", {})["type"] == "pong"
    assert h.dispatch("nope", {})["type"] == "unknown_command"


def test_compile_and_run_python_inline():
    r = h.handle_compile_and_run({"code": "print('inline')", "language": "python"})
    assert r["status"] == "ok"
    assert r["output"] == "inline"
