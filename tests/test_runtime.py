"""
test_runtime.py
Unit tests for device.runtime — the output-port mapping and the shared
send-and-maybe-wait flow, including the pending-waiter leak fix.
"""

import pytest
from fastapi.exceptions import HTTPException

from device.registry import ConnectionManager
from device.runtime import extract_ports, send_and_maybe_wait


# ---------------------------------------------------------------------------
# extract_ports
# ---------------------------------------------------------------------------

def test_extract_ports_uses_enriched_fields():
    p = extract_ports({"output": "hi", "status": "ok", "files": [{"name": "x"}]})
    assert p["output"] == "hi"
    assert p["status"] == "ok"
    assert p["files"] == [{"name": "x"}]


def test_extract_ports_falls_back_to_stdout_stderr():
    p = extract_ports({"stdout": ["a", "b"], "stderr": ["e"]})
    assert p["output"] == "a\nb"
    assert p["errors"] == "e"
    assert p["status"] == "unknown"     # default when absent


# ---------------------------------------------------------------------------
# send_and_maybe_wait — a stub manager exercises the three success branches
# ---------------------------------------------------------------------------

class StubMgr:
    def __init__(self, reply=None):
        self.reply = reply
        self.waiters = set()
        self.cancelled = []
        self.sent = []

    def register_waiter(self, cid):
        self.waiters.add(cid)

    def cancel_waiter(self, cid):
        self.cancelled.append(cid)
        self.waiters.discard(cid)

    async def send(self, device_id, command):
        self.sent.append((device_id, command))

    async def wait_for_result(self, cid, timeout):
        return self.reply


async def test_wait_false_returns_sent():
    m = StubMgr()
    r = await send_and_maybe_wait(m, "d", {"id": "C"}, wait=False, wait_timeout=5)
    assert r["status"] == "sent" and r["command_id"] == "C"
    assert not m.waiters                 # no waiter registered for fire-and-forget


async def test_wait_true_returns_ready_with_ports():
    m = StubMgr(reply={"stdout": ["4"], "status": "ok"})
    r = await send_and_maybe_wait(m, "d", {"id": "C"}, wait=True, wait_timeout=5)
    assert r["status"] == "ready"
    assert r["ports"]["output"] == "4"
    assert "C" in m.waiters


async def test_wait_true_returns_timeout_when_no_reply():
    m = StubMgr(reply=None)
    r = await send_and_maybe_wait(m, "d", {"id": "C"}, wait=True, wait_timeout=5)
    assert r["status"] == "timeout"


# ---------------------------------------------------------------------------
# The leak fix: a failed send must not leave an orphaned waiter behind
# ---------------------------------------------------------------------------

class FailWS:
    async def accept(self):
        pass

    async def close(self, code=1000):
        pass

    async def send_text(self, data):
        raise RuntimeError("dead socket")


async def test_failed_send_cleans_up_pending_waiter():
    m = ConnectionManager()
    await m.connect("d", FailWS())
    with pytest.raises(HTTPException) as exc:
        await send_and_maybe_wait(m, "d", {"id": "C1"}, wait=True, wait_timeout=1)
    assert exc.value.status_code == 503
    assert "C1" not in m._pending         # waiter was cancelled, not leaked
