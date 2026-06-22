"""
test_registry.py
Unit tests for device.registry.ConnectionManager — connection lifecycle and the
request/reply waiter machinery, including the reliability fixes:
  * a duplicate device_id connection closes the stale socket
  * disconnect is identity-aware (a closing old socket can't evict a new one)
  * a failed send drops the dead connection and raises 503
"""

import asyncio

import pytest
from fastapi.exceptions import HTTPException

from device.registry import ConnectionManager


class FakeWS:
    """Minimal stand-in for a Starlette WebSocket."""

    def __init__(self, fail_send: bool = False) -> None:
        self.accepted = False
        self.closed_code = None
        self.sent: list[str] = []
        self.fail_send = fail_send

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int = 1000) -> None:
        self.closed_code = code

    async def send_text(self, data: str) -> None:
        if self.fail_send:
            raise RuntimeError("dead socket")
        self.sent.append(data)


async def test_duplicate_connection_closes_stale_socket():
    m = ConnectionManager()
    ws1, ws2 = FakeWS(), FakeWS()
    await m.connect("d", ws1)
    await m.connect("d", ws2)
    assert ws1.closed_code == 1012          # stale socket was closed
    assert m._connections["d"] is ws2       # newest wins
    assert m.list_devices() == ["d"]


async def test_disconnect_is_identity_aware():
    m = ConnectionManager()
    ws1, ws2 = FakeWS(), FakeWS()
    await m.connect("d", ws1)
    await m.connect("d", ws2)               # ws2 now current
    m.disconnect("d", ws1)                  # stale socket trying to evict — ignored
    assert m.is_connected("d")
    m.disconnect("d", ws2)                  # current socket — really removes
    assert not m.is_connected("d")


async def test_send_to_unknown_device_raises_404():
    m = ConnectionManager()
    with pytest.raises(HTTPException) as exc:
        await m.send("ghost", {"type": "ping"})
    assert exc.value.status_code == 404


async def test_send_failure_drops_connection_and_raises_503():
    m = ConnectionManager()
    ws = FakeWS(fail_send=True)
    await m.connect("d", ws)
    with pytest.raises(HTTPException) as exc:
        await m.send("d", {"type": "ping"})
    assert exc.value.status_code == 503
    assert not m.is_connected("d")          # dead socket no longer reported connected


async def test_register_wait_resolve_roundtrip():
    m = ConnectionManager()
    m.register_waiter("c1")

    async def reply():
        await asyncio.sleep(0.01)
        assert m.resolve_result("c1", {"ok": True}) is True

    asyncio.create_task(reply())
    result = await m.wait_for_result("c1", timeout=1.0)
    assert result == {"ok": True}
    assert "c1" not in m._pending            # cleaned up after delivery


async def test_wait_times_out_when_no_reply():
    m = ConnectionManager()
    m.register_waiter("c2")
    result = await m.wait_for_result("c2", timeout=0.05)
    assert result is None
    assert "c2" not in m._pending


async def test_cancel_waiter_removes_pending():
    m = ConnectionManager()
    m.register_waiter("c3")
    m.cancel_waiter("c3")
    assert "c3" not in m._pending
    # resolving a cancelled waiter is a harmless no-op
    assert m.resolve_result("c3", {"x": 1}) is False


async def test_broadcast_drops_failed_socket():
    m = ConnectionManager()
    good, bad = FakeWS(), FakeWS(fail_send=True)
    await m.connect("good", good)
    await m.connect("bad", bad)
    await m.broadcast({"type": "ping"})
    assert good.sent and m.is_connected("good")
    assert not m.is_connected("bad")         # failed send evicted the bad one
