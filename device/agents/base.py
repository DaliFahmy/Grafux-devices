"""
device/agents/base.py
Shared runtime for every device agent.

All agents do the same thing around the edges: dial the Grafux hub over a
WebSocket, authenticate, receive command blocks, run the matching handler, and
send a correlated result back — reconnecting forever when the link drops.  That
machinery lived (copy-pasted) in raspberry_pi, windows and melfa; it now lives
here once, in :class:`BaseAgent`.

Reliability features built in for every agent:
  * WebSocket keepalive (``ping_interval`` / ``ping_timeout``) so a half-open
    connection is detected promptly instead of hanging until the next message.
  * Exponential reconnect backoff (capped) so a flapping hub is not hammered.
  * Blocking (synchronous) handlers run in a worker thread via
    ``asyncio.to_thread`` so a long compile/subprocess never stalls the event
    loop — keepalive pings keep flowing during the work.
  * One bad command can never kill the agent: handler exceptions become an
    error result; loop/transport faults are logged and retried.

Subclasses provide a ``handlers`` mapping (``command_type -> callable``).  A
handler may be synchronous (``def h(payload) -> dict``) or asynchronous
(``async def h(payload) -> dict``); both are supported transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Awaitable, Callable, Dict, List, Optional, Union

try:
    import websockets
except ImportError:  # pragma: no cover - environment guard
    print("ERROR: websockets is not installed.  Run: pip install websockets")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Keepalive + reconnect tunables (overridable via environment)
# ---------------------------------------------------------------------------

PING_INTERVAL_S: float = float(os.environ.get("AGENT_PING_INTERVAL_S", "20"))
PING_TIMEOUT_S: float = float(os.environ.get("AGENT_PING_TIMEOUT_S", "20"))
RECONNECT_MIN_S: float = float(os.environ.get("AGENT_RECONNECT_MIN_S", "1"))
RECONNECT_MAX_S: float = float(os.environ.get("AGENT_RECONNECT_MAX_S", "30"))

Handler = Callable[[dict], Union[dict, Awaitable[dict]]]


class BaseAgent:
    """Connection loop + dispatch shared by all device agents."""

    def __init__(
        self,
        host: str,
        device_id: str,
        token: str,
        *,
        handlers: Optional[Dict[str, Handler]] = None,
        logger: Optional[logging.Logger] = None,
        ping_interval: float = PING_INTERVAL_S,
        ping_timeout: float = PING_TIMEOUT_S,
        reconnect_min: float = RECONNECT_MIN_S,
        reconnect_max: float = RECONNECT_MAX_S,
    ) -> None:
        self.host = host.rstrip("/")
        self.device_id = device_id
        self.token = token
        self.handlers: Dict[str, Handler] = dict(handlers or {})
        self.log = logger or logging.getLogger("device.agent")
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.reconnect_min = reconnect_min
        self.reconnect_max = reconnect_max
        self._ws = None

    @property
    def url(self) -> str:
        return f"{self.host}/ws?device_id={self.device_id}&token={self.token}"

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        """Run once before the connection loop starts (override as needed)."""

    async def on_shutdown(self) -> None:
        """Run once when the agent stops (override to release resources)."""

    async def extra_tasks(self) -> List[Awaitable[None]]:
        """Return background coroutines to run alongside the main loop.

        E.g. the MELFA agent returns its rosbridge keepalive task here.
        """
        return []

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, command_type: str, payload: dict) -> dict:
        """Run the handler for *command_type* and return its result dict.

        Sync handlers run in a worker thread so blocking work (subprocess,
        file IO) does not stall the event loop.  Any handler exception is
        converted into an error result rather than killing the loop.
        """
        handler = self.handlers.get(command_type)
        if handler is None:
            return {
                "type": "unknown_command",
                "status": "error",
                "received_type": command_type,
                "available_commands": sorted(self.handlers),
            }
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(payload)
            result = await asyncio.to_thread(handler, payload)
            if asyncio.iscoroutine(result):  # sync fn that returned a coroutine
                result = await result
            return result
        except Exception as exc:  # noqa: BLE001 — never let one command kill the agent
            self.log.exception("handler '%s' raised", command_type)
            return {"type": "error", "status": "error", "error": str(exc)}

    def _envelope(self, result: dict, command_id) -> dict:
        """Attach the correlation fields the hub uses to match the reply."""
        result["command_id"] = command_id
        result["device_id"] = self.device_id
        result["timestamp"] = time.time()
        return result

    async def _serve(self, ws) -> None:
        """Process inbound command frames on a live connection until it closes."""
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                self.log.warning("non-JSON message received, ignoring")
                continue

            command_type = message.get("type", "unknown")
            command_id = message.get("id")
            payload = message.get("payload", {}) or {}
            self.log.info("<- command type=%s id=%s", command_type, command_id)

            result = await self.dispatch(command_type, payload)
            await ws.send(json.dumps(self._envelope(result, command_id)))
            self.log.info(
                "-> result type=%s status=%s",
                result.get("type"),
                result.get("status", result.get("success", "-")),
            )

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _grafux_loop(self) -> None:
        delay = self.reconnect_min
        while True:
            try:
                self.log.info("connecting to %s …", self.url)
                async with websockets.connect(
                    self.url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                ) as ws:
                    self._ws = ws
                    self.log.info("connected as device_id='%s'", self.device_id)
                    delay = self.reconnect_min  # reset backoff after a good connect
                    await self._serve(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — survive every transport fault
                self.log.warning("connection error: %s", exc)
            finally:
                self._ws = None

            self.log.info("reconnecting in %.0fs …", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.reconnect_max)  # exponential backoff

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.on_startup()
        try:
            await asyncio.gather(self._grafux_loop(), *(await self.extra_tasks()))
        finally:
            await self.on_shutdown()

    def run_forever(self) -> None:
        """Blocking entry point for ``__main__`` — runs until Ctrl-C."""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.log.info("agent stopped.")
