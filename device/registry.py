"""
device/registry.py
Tracks active WebSocket connections, keyed by device_id, and the per-command
Futures that REST callers await on (the request/reply path).
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

from fastapi import WebSocket
from fastapi.exceptions import HTTPException

logger = logging.getLogger("devices.manager")


class ConnectionManager:
    """Manages all active device WebSocket connections."""

    def __init__(self) -> None:
        # device_id -> WebSocket
        self._connections: Dict[str, WebSocket] = {}
        # command_id -> Future waiting for the device's result
        self._pending: Dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, device_id: str, websocket: WebSocket) -> None:
        """Accept the handshake and register the device.

        If a connection already exists for *device_id* (e.g. the device
        reconnected before the server noticed the old socket died), the stale
        socket is closed first so it cannot linger half-open or deliver a late
        result onto the new connection's waiters.
        """
        await websocket.accept()
        old = self._connections.get(device_id)
        if old is not None:
            logger.warning("[%s] Replacing existing connection — closing the stale socket", device_id)
            try:
                await old.close(code=1012)  # 1012 = Service Restart / superseded
            except Exception as exc:  # noqa: BLE001 — best effort; never block the new connection
                logger.debug("[%s] error closing stale socket: %s", device_id, exc)
        self._connections[device_id] = websocket
        logger.info("[%s] Connected  (total: %d)", device_id, len(self._connections))

    def disconnect(self, device_id: str, websocket: Optional[WebSocket] = None) -> None:
        """Remove a device from the registry.

        If *websocket* is given, the device is only removed when it is still the
        currently-registered socket.  This prevents a closing old connection from
        evicting a newer one that already replaced it (see ``connect``).
        """
        if websocket is not None and self._connections.get(device_id) is not websocket:
            return
        self._connections.pop(device_id, None)
        logger.info("[%s] Disconnected  (total: %d)", device_id, len(self._connections))

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, device_id: str, payload: dict) -> None:
        """Send a JSON payload to one specific device.

        Raises HTTPException(404) if the device is not connected, or HTTPException(503)
        if the socket is dead — in which case the device is dropped from the
        registry so it is not reported as connected until it reconnects.
        """
        websocket = self._connections.get(device_id)
        if websocket is None:
            raise HTTPException(
                status_code=404,
                detail=f"Device '{device_id}' is not connected.",
            )
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001 — a dead socket must not look "sent"
            logger.warning("[%s] send failed (%s) — dropping stale connection", device_id, exc)
            self.disconnect(device_id)
            raise HTTPException(
                status_code=503,
                detail=f"Device '{device_id}' connection is no longer usable.",
            ) from exc
        logger.info("[%s] -> sent command type=%s", device_id, payload.get("type"))

    async def broadcast(self, payload: dict) -> None:
        """Send a JSON payload to every connected device concurrently.

        Sockets are written in parallel via ``asyncio.gather`` so one slow
        device does not hold up delivery to the rest.  A failed send drops that
        device from the registry.
        """
        if not self._connections:
            logger.warning("broadcast called but no devices are connected")
            return
        message = json.dumps(payload)
        targets = list(self._connections.items())

        async def _send_one(device_id: str, websocket: WebSocket) -> None:
            try:
                await websocket.send_text(message)
                logger.info("[%s] -> broadcast type=%s", device_id, payload.get("type"))
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] broadcast failed: %s", device_id, exc)
                self.disconnect(device_id)

        await asyncio.gather(*(_send_one(d, ws) for d, ws in targets))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_devices(self) -> List[str]:
        """Return a list of currently connected device IDs."""
        return list(self._connections.keys())

    def is_connected(self, device_id: str) -> bool:
        return device_id in self._connections

    # ------------------------------------------------------------------
    # Request / reply — wait for a device result by command_id
    # ------------------------------------------------------------------

    def register_waiter(self, command_id: str) -> None:
        """Pre-create a Future for *command_id* BEFORE sending the command.

        This must be called before ``send()`` to avoid a race condition where
        the device replies before ``wait_for_result`` has registered the Future.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[command_id] = fut

    async def wait_for_result(
        self, command_id: str, timeout: float = 120.0
    ) -> Optional[dict]:
        """Suspend until the device sends back a result for *command_id*.

        Call ``register_waiter(command_id)`` BEFORE sending the command to
        avoid a race condition.  This method awaits the pre-registered Future
        (or creates one if not already registered).

        Returns the result dict, or ``None`` if *timeout* seconds elapse
        without a matching result arriving.
        """
        fut = self._pending.get(command_id)
        if fut is None:
            # Fallback: register now (only safe when the round-trip is slow)
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._pending[command_id] = fut
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending.pop(command_id, None)

    def cancel_waiter(self, command_id: str) -> None:
        """Drop a pre-registered waiter that will never be awaited.

        Call this if ``send()`` fails after ``register_waiter()`` so the orphaned
        Future does not accumulate in ``_pending`` (a slow memory leak otherwise).
        """
        self._pending.pop(command_id, None)

    def resolve_result(self, command_id: str, result: dict) -> bool:
        """Deliver *result* to any caller waiting on *command_id*.

        Returns True if a waiter was found and resolved, False otherwise.
        """
        fut = self._pending.get(command_id)
        if fut is not None and not fut.done():
            fut.set_result(result)
            return True
        return False
