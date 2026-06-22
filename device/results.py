"""
device/results.py
A small TTL cache for device command results.

When a REST caller uses the fire-and-forget path (``wait=false``) it later polls
``GET /devices/{id}/result/{command_id}`` to collect the device's reply.  This
store holds those replies until they are collected, or until they age out.

Two indexing schemes are used by the WebSocket handler:

  * the command id echoed back by the device  -> consumed on first read
  * ``latest:<device_id>``                     -> the most recent reply, re-readable

Performance
-----------
The previous implementation purged stale entries with a full O(n) scan on *every*
inbound message (several times per message).  Here purging is decoupled from
writes: a single background task sweeps the store once per TTL window, so a write
is O(1) and memory stays bounded (the never-consumed ``latest:`` keys now expire
too).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger("devices.results")

DEFAULT_TTL_S: int = 120  # seconds before an uncollected result is discarded


class ResultStore:
    """In-memory ``command_id -> result`` cache with a background TTL sweeper."""

    def __init__(self, ttl_s: int = DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s
        # key -> (result message, stored-at unix time)
        self._store: dict[str, tuple[dict, float]] = {}
        self._sweeper: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Reads / writes (O(1))
    # ------------------------------------------------------------------

    def put(self, key: str, message: dict) -> None:
        """Store *message* under *key* (overwriting any previous value)."""
        self._store[key] = (message, time.time())

    def pop(self, key: str) -> Optional[dict]:
        """Return and remove the result for *key*, or ``None`` if absent/expired."""
        entry = self._store.pop(key, None)
        if entry is None:
            return None
        message, ts = entry
        if time.time() - ts > self._ttl_s:
            return None
        return message

    def peek(self, key: str) -> Optional[dict]:
        """Return the result for *key* without removing it (``None`` if expired)."""
        entry = self._store.get(key)
        if entry is None:
            return None
        message, ts = entry
        if time.time() - ts > self._ttl_s:
            self._store.pop(key, None)
            return None
        return message

    # ------------------------------------------------------------------
    # Background sweeper — one O(n) pass per TTL window, not per write
    # ------------------------------------------------------------------

    async def _sweep_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._ttl_s)
                cutoff = time.time() - self._ttl_s
                stale = [k for k, (_, ts) in self._store.items() if ts < cutoff]
                for k in stale:
                    self._store.pop(k, None)
                if stale:
                    logger.debug("Swept %d stale result(s)", len(stale))
        except asyncio.CancelledError:
            pass

    def start_sweeper(self) -> None:
        """Start the periodic purge task (idempotent).  Call from app startup."""
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop_sweeper(self) -> None:
        """Cancel the purge task.  Call from app shutdown."""
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None
