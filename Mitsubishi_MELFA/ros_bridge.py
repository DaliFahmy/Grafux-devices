"""
ros_bridge.py
Async rosbridge v2 WebSocket client.

Implements the rosbridge protocol operations needed by the MELFA agent:
  - subscribe   → receive ROS topic messages via async callbacks
  - unsubscribe → stop receiving a topic
  - publish     → fire-and-forget publish to a topic
  - call_service → request/response with asyncio.Future
  - send_action_goal → send a ROS2 action goal, await result
  - cancel_action_goal / cancel_all_goals → halt active motion

All public methods are async-safe and can be awaited from the agent's
main event loop.  The internal receive loop runs as a background task
and routes incoming rosbridge messages by their "op" field.

rosbridge protocol reference:
  https://github.com/RobotWebTools/rosbridge_suite/blob/ros2/ROSBRIDGE_PROTOCOL.md
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from . import config as cfg

logger = logging.getLogger("melfa.rosbridge")


class RosBridgeClient:
    """
    Persistent async WebSocket client for the rosbridge v2 protocol.

    Usage
    -----
    client = RosBridgeClient("ws://localhost:9090")
    await client.connect()
    await client.subscribe("/joint_states", "sensor_msgs/msg/JointState", my_callback)
    result = await client.send_action_goal("/move_group", "moveit_msgs/action/MoveGroup", goal)
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()

        # topic → async callback
        self._subscriptions: Dict[str, Callable[[dict], Coroutine]] = {}
        # topic → latest received message dict (for instant polling)
        self._topic_cache: Dict[str, dict] = {}

        # id → asyncio.Future  (service call results)
        self._pending_services: Dict[str, asyncio.Future] = {}

        # id → {"future": asyncio.Future, "feedback_cb": Optional[Callable]}
        self._pending_actions: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection to rosbridge and start the receive loop."""
        self._ws = await websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop(), name="rosbridge-recv")
        logger.info("Connected to rosbridge at %s", self.url)

    async def disconnect(self) -> None:
        """Close the WebSocket gracefully."""
        self._connected.clear()
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from rosbridge")

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and self._ws is not None

    # ------------------------------------------------------------------
    # Internal receive loop
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """Dispatch incoming rosbridge messages to the right handlers."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("rosbridge: non-JSON message received")
                    continue

                op = msg.get("op", "")

                if op == "publish":
                    await self._handle_publish(msg)

                elif op == "service_response":
                    self._handle_service_response(msg)

                elif op == "action_result":
                    self._handle_action_result(msg)

                elif op == "action_feedback":
                    await self._handle_action_feedback(msg)

                elif op in ("set_level", "status"):
                    logger.debug("rosbridge %s: %s", op, msg.get("msg", ""))

                else:
                    logger.debug("rosbridge unhandled op=%s", op)

        except (ConnectionClosedError, ConnectionClosedOK) as exc:
            logger.warning("rosbridge connection closed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("rosbridge recv loop error: %s", exc)
        finally:
            self._connected.clear()

    async def _handle_publish(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        data = msg.get("msg", {})
        self._topic_cache[topic] = data
        cb = self._subscriptions.get(topic)
        if cb:
            try:
                await cb(data)
            except Exception as exc:  # noqa: BLE001
                logger.error("rosbridge subscription callback error [%s]: %s", topic, exc)

    def _handle_service_response(self, msg: dict) -> None:
        call_id = msg.get("id")
        future = self._pending_services.pop(call_id, None)
        if future and not future.done():
            future.set_result(msg)

    def _handle_action_result(self, msg: dict) -> None:
        goal_id = msg.get("id")
        entry = self._pending_actions.pop(goal_id, None)
        if entry:
            future: asyncio.Future = entry["future"]
            if not future.done():
                future.set_result(msg)

    async def _handle_action_feedback(self, msg: dict) -> None:
        goal_id = msg.get("id")
        entry = self._pending_actions.get(goal_id)
        if entry and entry.get("feedback_cb"):
            try:
                await entry["feedback_cb"](msg.get("feedback", {}))
            except Exception as exc:  # noqa: BLE001
                logger.error("rosbridge action feedback callback error: %s", exc)

    # ------------------------------------------------------------------
    # Public API — topic operations
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        topic: str,
        msg_type: str,
        callback: Optional[Callable[[dict], Coroutine]] = None,
        throttle_rate: int = 0,
    ) -> None:
        """
        Subscribe to a ROS topic.

        Parameters
        ----------
        topic:         Full ROS topic name, e.g. "/joint_states".
        msg_type:      ROS message type, e.g. "sensor_msgs/msg/JointState".
        callback:      Async function called with each new message dict.
        throttle_rate: Minimum ms between messages (0 = no throttle).
        """
        if callback:
            self._subscriptions[topic] = callback
        payload: dict[str, Any] = {
            "op": "subscribe",
            "topic": topic,
            "type": msg_type,
        }
        if throttle_rate > 0:
            payload["throttle_rate"] = throttle_rate
        await self._send(payload)
        logger.info("rosbridge: subscribed to %s (%s)", topic, msg_type)

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a ROS topic."""
        self._subscriptions.pop(topic, None)
        await self._send({"op": "unsubscribe", "topic": topic})
        logger.info("rosbridge: unsubscribed from %s", topic)

    async def publish(self, topic: str, msg_type: str, message: dict) -> None:
        """Publish a single message to a ROS topic (fire-and-forget)."""
        await self._send({
            "op": "publish",
            "topic": topic,
            "type": msg_type,
            "msg": message,
        })

    # ------------------------------------------------------------------
    # Public API — service calls
    # ------------------------------------------------------------------

    async def call_service(
        self,
        service: str,
        service_type: str,
        args: Optional[dict] = None,
        timeout: float = cfg.SERVICE_TIMEOUT_S,
    ) -> dict:
        """
        Call a ROS service and wait for the response.

        Returns the full rosbridge service_response dict on success.
        Raises asyncio.TimeoutError if no response within `timeout` seconds.
        """
        call_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_services[call_id] = future

        await self._send({
            "op": "call_service",
            "id": call_id,
            "service": service,
            "type": service_type,
            "args": args or {},
        })
        logger.debug("rosbridge: called service %s id=%s", service, call_id)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_services.pop(call_id, None)
            raise asyncio.TimeoutError(
                f"rosbridge service {service!r} did not respond within {timeout}s"
            )

    # ------------------------------------------------------------------
    # Public API — action goals
    # ------------------------------------------------------------------

    async def send_action_goal(
        self,
        action: str,
        action_type: str,
        goal: dict,
        feedback_cb: Optional[Callable[[dict], Coroutine]] = None,
        timeout: float = cfg.MOVE_TIMEOUT_S,
    ) -> dict:
        """
        Send an action goal to a ROS2 action server and wait for the result.

        Returns the full rosbridge action_result dict.
        Raises asyncio.TimeoutError if no result within `timeout` seconds.
        """
        goal_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_actions[goal_id] = {"future": future, "feedback_cb": feedback_cb}

        await self._send({
            "op": "send_action_goal",
            "id": goal_id,
            "action": action,
            "action_type": action_type,
            "goal": goal,
        })
        logger.info("rosbridge: sent action goal to %s id=%s", action, goal_id)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            # Try to cancel the goal before re-raising
            try:
                await self.cancel_action_goal(action, goal_id)
            except Exception:  # noqa: BLE001
                pass
            self._pending_actions.pop(goal_id, None)
            raise asyncio.TimeoutError(
                f"Action {action!r} goal {goal_id!r} timed out after {timeout}s"
            )

    async def cancel_action_goal(self, action: str, goal_id: str) -> None:
        """Send a cancel request for a specific action goal."""
        await self._send({
            "op": "cancel_action_goal",
            "id": goal_id,
            "action": action,
        })
        logger.info("rosbridge: cancelled goal %s on %s", goal_id, action)

    async def cancel_all_goals(self) -> int:
        """
        Cancel every pending action goal tracked by this client.
        Returns the number of goals cancelled.
        """
        cancelled = 0
        for goal_id, entry in list(self._pending_actions.items()):
            try:
                await self.cancel_action_goal(cfg.ACTION_MOVE_GROUP, goal_id)
            except Exception:  # noqa: BLE001
                pass
            future: asyncio.Future = entry["future"]
            if not future.done():
                future.cancel()
            cancelled += 1
        self._pending_actions.clear()
        return cancelled

    # ------------------------------------------------------------------
    # Cached topic data helpers
    # ------------------------------------------------------------------

    def get_cached(self, topic: str) -> Optional[dict]:
        """Return the latest cached message for a subscribed topic, or None."""
        return self._topic_cache.get(topic)

    def has_topic_data(self, topic: str) -> bool:
        """True if at least one message has been received from this topic."""
        return topic in self._topic_cache

    @property
    def active_goals(self) -> dict:
        """Return the dict of pending action goal futures."""
        return self._pending_actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send(self, payload: dict) -> None:
        if self._ws is None or not self.is_connected:
            raise ConnectionError("rosbridge WebSocket is not connected")
        await self._ws.send(json.dumps(payload))
