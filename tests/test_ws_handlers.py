"""
test_ws_handlers.py
Unit tests for the device-side ws_server handlers, focused on the run_code
timeout hardening (previously in-process exec with no enforceable timeout).
"""

from device.ws_server import handlers as h
from device.ws_server import discovery as d


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


# ── Progress streaming (on_progress) ─────────────────────────────────────────


def test_run_code_streams_progress():
    frames = []
    r = h.handle_run_code(
        {"code": "print('a'); print('b')"},
        on_progress=lambda f: frames.append(f),
    )
    # Final dict is unchanged …
    assert r["status"] == "ok"
    assert r["output"] == "a\nb"
    # … and progress frames streamed the running phase + each stdout line.
    assert any(f.get("phase") == "running" for f in frames)
    streamed = [f.get("line") for f in frames if "line" in f]
    assert "a" in streamed and "b" in streamed


def test_compile_and_run_streams_progress():
    frames = []
    r = h.handle_compile_and_run(
        {"code": "print('x')", "language": "python"},
        on_progress=lambda f: frames.append(f),
    )
    assert r["status"] == "ok"
    assert r["output"] == "x"
    assert any(f.get("phase") == "running" for f in frames)


def test_streaming_final_matches_nonstreaming():
    """The final result dict must be identical with and without a callback."""
    payload = {"code": "print('same')", "language": "python"}
    plain = h.handle_compile_and_run(dict(payload))
    streamed = h.handle_compile_and_run(dict(payload), on_progress=lambda f: None)
    assert plain["status"] == streamed["status"]
    assert plain["output"] == streamed["output"]
    assert plain["errors"] == streamed["errors"]


def test_dispatch_two_and_three_arg_compatible():
    # Legacy two-arg call still works …
    assert h.dispatch("run_code", {"code": "print(1)"})["status"] == "ok"
    # … and the three-arg form with a callback streams.
    frames = []
    r = h.dispatch("run_code", {"code": "print(2)"}, lambda f: frames.append(f))
    assert r["status"] == "ok"
    assert frames  # at least one progress frame


def test_run_code_streaming_enforces_timeout():
    frames = []
    r = h.handle_run_code(
        {"code": "while True: pass", "timeout": 1},
        on_progress=lambda f: frames.append(f),
    )
    assert r["status"] == "timeout"


# ── mDNS advertiser guard ────────────────────────────────────────────────────


def test_mdns_disabled_via_env(monkeypatch):
    monkeypatch.setenv("DEVICE_MDNS_DISABLE", "1")
    assert d.start_advertiser("dev1", 8765) is None


def test_stop_advertiser_handles_none():
    # Must be a safe no-op when advertisement never started.
    d.stop_advertiser(None)
