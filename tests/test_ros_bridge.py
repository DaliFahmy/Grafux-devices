"""
test_ros_bridge.py
Unit tests for server/devices/Mitsubishi_MELFA/ros_bridge.py — RosBridgeClient.

Strategy
--------
Rather than starting a real WebSocket server, each test works with a
RosBridgeClient whose internal `_ws` is replaced with an `AsyncMock`.
This lets us:
  - Verify outgoing rosbridge JSON messages via `mock_ws.send.assert_called_with(...)`
  - Inject incoming messages by calling the client's internal dispatch methods
    (`_handle_publish`, `_handle_service_response`, `_handle_action_result`) directly
  - Test Future-based call_service / send_action_goal by creating a task, letting
    it register its Future, then manually resolving it

No real WebSocket server or ROS2 environment is required.
"""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from device.agents.melfa.ros_bridge import RosBridgeClient


# ---------------------------------------------------------------------------
# Fixture: a RosBridgeClient whose WebSocket is an AsyncMock
# ---------------------------------------------------------------------------

@pytest.fixture
def rb() -> RosBridgeClient:
    """
    Returns a RosBridgeClient in the 'connected' state without running connect().
    The internal _ws is an AsyncMock so all send calls can be inspected.
    """
    client = RosBridgeClient("ws://localhost:9090")
    mock_ws = AsyncMock()
    client._ws = mock_ws
    client._connected.set()
    return client


def _last_sent(rb: RosBridgeClient) -> dict:
    """Return the last JSON message sent over the mocked WebSocket."""
    raw = rb._ws.send.call_args[0][0]
    return json.loads(raw)


# ===========================================================================
# Connectivity helpers
# ===========================================================================

def test_is_connected_false_on_init():
    client = RosBridgeClient("ws://localhost:9090")
    # No _ws set, no _connected event set
    assert client.is_connected is False


def test_is_connected_true_after_setup(rb):
    assert rb.is_connected is True


@pytest.mark.asyncio
async def test_disconnect_clears_connected(rb):
    rb._ws.close = AsyncMock()
    # No recv_task to cancel
    await rb.disconnect()
    assert rb.is_connected is False
    assert rb._ws is None


# ===========================================================================
# Topic cache helpers (synchronous — no async needed)
# ===========================================================================

def test_get_cached_returns_none_before_any_message(rb):
    assert rb.get_cached("/joint_states") is None


def test_has_topic_data_false_before_message(rb):
    assert rb.has_topic_data("/joint_states") is False


def test_get_cached_after_publish(rb):
    data = {"name": ["j1"], "position": [0.5]}
    # Simulate an incoming publish — call the internal handler directly
    asyncio.get_event_loop().run_until_complete(
        rb._handle_publish({"op": "publish", "topic": "/joint_states", "msg": data})
    )
    assert rb.get_cached("/joint_states") == data


def test_has_topic_data_true_after_publish(rb):
    asyncio.get_event_loop().run_until_complete(
        rb._handle_publish({"op": "publish", "topic": "/joint_states", "msg": {"position": []}})
    )
    assert rb.has_topic_data("/joint_states") is True


# ===========================================================================
# subscribe / unsubscribe
# ===========================================================================

@pytest.mark.asyncio
async def test_subscribe_sends_correct_op(rb):
    await rb.subscribe("/joint_states", "sensor_msgs/msg/JointState")
    msg = _last_sent(rb)
    assert msg["op"] == "subscribe"
    assert msg["topic"] == "/joint_states"
    assert msg["type"] == "sensor_msgs/msg/JointState"


@pytest.mark.asyncio
async def test_subscribe_registers_callback(rb):
    async def my_cb(data):
        pass

    await rb.subscribe("/joint_states", "sensor_msgs/msg/JointState", callback=my_cb)
    assert rb._subscriptions["/joint_states"] is my_cb


@pytest.mark.asyncio
async def test_subscribe_with_throttle_rate(rb):
    await rb.subscribe("/joint_states", "sensor_msgs/msg/JointState", throttle_rate=100)
    msg = _last_sent(rb)
    assert msg["throttle_rate"] == 100


@pytest.mark.asyncio
async def test_unsubscribe_sends_op(rb):
    async def cb(d):
        pass

    rb._subscriptions["/joint_states"] = cb
    await rb.unsubscribe("/joint_states")
    msg = _last_sent(rb)
    assert msg["op"] == "unsubscribe"
    assert msg["topic"] == "/joint_states"
    assert "/joint_states" not in rb._subscriptions


# ===========================================================================
# publish
# ===========================================================================

@pytest.mark.asyncio
async def test_publish_sends_correct_op(rb):
    await rb.publish("/cmd_vel", "geometry_msgs/msg/Twist", {"linear": {"x": 0.1}})
    msg = _last_sent(rb)
    assert msg["op"] == "publish"
    assert msg["topic"] == "/cmd_vel"
    assert msg["type"] == "geometry_msgs/msg/Twist"
    assert msg["msg"]["linear"]["x"] == 0.1


# ===========================================================================
# _send while disconnected
# ===========================================================================

@pytest.mark.asyncio
async def test_send_while_disconnected_raises():
    client = RosBridgeClient("ws://localhost:9090")
    # _ws is None, _connected not set
    with pytest.raises(ConnectionError):
        await client._send({"op": "ping"})


# ===========================================================================
# call_service — Future resolution
# ===========================================================================

@pytest.mark.asyncio
async def test_call_service_sends_correct_op(rb):
    # Fire call_service as a task; resolve immediately
    task = asyncio.create_task(
        rb.call_service("/compute_fk", "moveit_msgs/srv/GetPositionFK", {}, timeout=5.0)
    )
    await asyncio.sleep(0)  # let the task register its Future and send the message

    msg = _last_sent(rb)
    assert msg["op"] == "call_service"
    assert msg["service"] == "/compute_fk"
    assert msg["type"] == "moveit_msgs/srv/GetPositionFK"
    call_id = msg["id"]

    # Resolve the Future so the task finishes
    rb._handle_service_response({"id": call_id, "values": {"result": "ok"}})
    result = await task
    assert result["values"]["result"] == "ok"


@pytest.mark.asyncio
async def test_call_service_resolves_with_correct_data(rb):
    response_data = {"values": {"pose_stamped": [{"pose": {}}]}, "id": None}

    async def _resolve():
        await asyncio.sleep(0)
        # Capture the call_id from the outgoing message
        call_id = json.loads(rb._ws.send.call_args[0][0])["id"]
        response_data["id"] = call_id
        rb._handle_service_response(response_data)

    asyncio.create_task(_resolve())
    result = await rb.call_service("/compute_fk", "moveit_msgs/srv/GetPositionFK", {}, timeout=2.0)
    assert "pose_stamped" in result["values"]


@pytest.mark.asyncio
async def test_call_service_timeout_raises(rb):
    with pytest.raises(asyncio.TimeoutError):
        await rb.call_service("/slow_service", "some/Type", {}, timeout=0.05)


@pytest.mark.asyncio
async def test_call_service_consumed_from_pending(rb):
    task = asyncio.create_task(
        rb.call_service("/compute_fk", "t", {}, timeout=2.0)
    )
    await asyncio.sleep(0)
    call_id = json.loads(rb._ws.send.call_args[0][0])["id"]
    rb._handle_service_response({"id": call_id})
    await task
    # Future should be removed from pending after resolution
    assert call_id not in rb._pending_services


# ===========================================================================
# send_action_goal — Future resolution
# ===========================================================================

@pytest.mark.asyncio
async def test_send_action_goal_sends_correct_op(rb):
    task = asyncio.create_task(
        rb.send_action_goal("/move_group", "moveit_msgs/action/MoveGroup", {}, timeout=5.0)
    )
    await asyncio.sleep(0)

    msg = _last_sent(rb)
    assert msg["op"] == "send_action_goal"
    assert msg["action"] == "/move_group"
    assert msg["action_type"] == "moveit_msgs/action/MoveGroup"
    goal_id = msg["id"]

    # Resolve
    rb._handle_action_result({
        "id": goal_id,
        "values": {"result": {"error_code": {"val": 1}}}
    })
    result = await task
    assert result["values"]["result"]["error_code"]["val"] == 1


@pytest.mark.asyncio
async def test_send_action_goal_timeout_raises(rb):
    with pytest.raises(asyncio.TimeoutError):
        await rb.send_action_goal("/move_group", "t", {}, timeout=0.05)


@pytest.mark.asyncio
async def test_send_action_goal_timeout_sends_cancel(rb):
    try:
        await rb.send_action_goal("/move_group", "t", {}, timeout=0.05)
    except asyncio.TimeoutError:
        pass

    # At minimum two messages sent: the goal itself + the cancel
    assert rb._ws.send.await_count >= 2
    messages = [json.loads(c[0][0]) for c in rb._ws.send.call_args_list]
    ops = [m["op"] for m in messages]
    assert "cancel_action_goal" in ops


@pytest.mark.asyncio
async def test_action_goal_consumed_from_pending(rb):
    task = asyncio.create_task(
        rb.send_action_goal("/move_group", "t", {}, timeout=2.0)
    )
    await asyncio.sleep(0)
    goal_id = json.loads(rb._ws.send.call_args[0][0])["id"]
    rb._handle_action_result({"id": goal_id, "values": {}})
    await task
    assert goal_id not in rb._pending_actions


# ===========================================================================
# cancel_all_goals
# ===========================================================================

@pytest.mark.asyncio
async def test_cancel_all_goals_empties_pending(rb):
    loop = asyncio.get_event_loop()
    f1 = loop.create_future()
    f2 = loop.create_future()
    rb._pending_actions["goal-1"] = {"future": f1, "feedback_cb": None}
    rb._pending_actions["goal-2"] = {"future": f2, "feedback_cb": None}

    count = await rb.cancel_all_goals()

    assert count == 2
    assert len(rb._pending_actions) == 0
    assert f1.cancelled()
    assert f2.cancelled()


@pytest.mark.asyncio
async def test_cancel_all_goals_returns_zero_when_empty(rb):
    count = await rb.cancel_all_goals()
    assert count == 0


# ===========================================================================
# _handle_publish — subscription callback dispatch
# ===========================================================================

@pytest.mark.asyncio
async def test_handle_publish_invokes_callback(rb):
    received = []

    async def cb(data):
        received.append(data)

    rb._subscriptions["/joint_states"] = cb
    await rb._handle_publish({
        "op": "publish",
        "topic": "/joint_states",
        "msg": {"name": ["j1"], "position": [0.0]},
    })

    assert len(received) == 1
    assert received[0]["name"] == ["j1"]


@pytest.mark.asyncio
async def test_handle_publish_updates_cache(rb):
    data = {"position": [1.0, 2.0]}
    await rb._handle_publish({"op": "publish", "topic": "/joint_states", "msg": data})
    assert rb.get_cached("/joint_states") == data


@pytest.mark.asyncio
async def test_handle_publish_no_callback_still_caches(rb):
    # No subscription registered — should still cache
    await rb._handle_publish({"op": "publish", "topic": "/tf", "msg": {"transforms": []}})
    assert rb.has_topic_data("/tf")


# ===========================================================================
# action_feedback dispatch
# ===========================================================================

@pytest.mark.asyncio
async def test_handle_action_feedback_calls_callback(rb):
    feedback_received = []

    async def fb_cb(fb):
        feedback_received.append(fb)

    goal_id = "test-goal-id"
    loop = asyncio.get_event_loop()
    f = loop.create_future()
    rb._pending_actions[goal_id] = {"future": f, "feedback_cb": fb_cb}

    await rb._handle_action_feedback({
        "id": goal_id,
        "feedback": {"state": "EXECUTING"},
    })

    assert len(feedback_received) == 1
    assert feedback_received[0]["state"] == "EXECUTING"


# ===========================================================================
# active_goals property
# ===========================================================================

def test_active_goals_reflects_pending_actions(rb):
    loop = asyncio.get_event_loop()
    f = loop.create_future()
    rb._pending_actions["g1"] = {"future": f, "feedback_cb": None}
    assert "g1" in rb.active_goals
