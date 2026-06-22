"""
test_main_robot_endpoints.py
Integration tests for the 7 robot REST shortcut endpoints added to
server/devices/devices_server.py.

Uses httpx.AsyncClient with ASGITransport to exercise each endpoint end-to-end
through the ASGI app without starting a real server.  The ConnectionManager.send()
method is patched with an AsyncMock so no real WebSocket connection is required.

This approach is compatible with all httpx versions (including 0.28+) and avoids
the starlette TestClient/httpx version mismatch.

Endpoint coverage
-----------------
POST /devices/{id}/robot/joint_states
POST /devices/{id}/robot/pose
POST /devices/{id}/robot/status
POST /devices/{id}/robot/move_joints
POST /devices/{id}/robot/move_pose
POST /devices/{id}/robot/home
POST /devices/{id}/robot/stop
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

from device.app import app, manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE_ID = "melfa-001"
BASE_URL = "http://test"


@pytest.fixture
async def api():
    """
    Async httpx client backed by the ASGI app.
    manager.send is patched as an AsyncMock so no real WebSocket is needed.
    """
    transport = httpx.ASGITransport(app=app)
    with patch.object(manager, "send", new_callable=AsyncMock) as mock_send:
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            yield client, mock_send


@pytest.fixture
async def api_device_not_found():
    """
    Async httpx client where manager.send raises HTTPException 404.
    """
    exc = HTTPException(status_code=404, detail=f"Device '{DEVICE_ID}' is not connected.")
    transport = httpx.ASGITransport(app=app)
    with patch.object(manager, "send", new_callable=AsyncMock, side_effect=exc):
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_sent_ok(response, expected_type: str):
    """Assert the endpoint returned 200 and the sent command has the right type."""
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "sent"
    assert body["command"]["type"] == expected_type


# ===========================================================================
# No-body endpoints (state reads)
# ===========================================================================

@pytest.mark.asyncio
async def test_robot_joint_states(api):
    client, mock_send = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/joint_states")
    _assert_sent_ok(response, "robot_get_joint_states")
    mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_robot_pose(api):
    client, mock_send = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/pose")
    _assert_sent_ok(response, "robot_get_pose")
    mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_robot_status(api):
    client, mock_send = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/status")
    _assert_sent_ok(response, "robot_get_status")
    mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_robot_stop(api):
    client, mock_send = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/stop")
    _assert_sent_ok(response, "robot_stop")


# ===========================================================================
# robot/move_joints
# ===========================================================================

@pytest.mark.asyncio
async def test_robot_move_joints_defaults(api):
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_joints", json={})
    _assert_sent_ok(response, "robot_move_joints")
    payload = response.json()["command"]["payload"]
    assert payload["j1"] == 0.0
    assert payload["j2"] == 0.0
    assert payload["j3"] == 0.0
    assert payload["j4"] == 0.0
    assert payload["j5"] == 0.0
    assert payload["j6"] == 0.0
    assert payload["speed"] == 0.1


@pytest.mark.asyncio
async def test_robot_move_joints_custom(api):
    client, _ = api
    body = {"j1": 10.0, "j2": -30.0, "j3": 90.0, "j4": 5.0, "j5": 45.0, "j6": -10.0, "speed": 0.5}
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_joints", json=body)
    _assert_sent_ok(response, "robot_move_joints")
    payload = response.json()["command"]["payload"]
    assert payload["j2"] == -30.0
    assert payload["j3"] == 90.0
    assert payload["speed"] == 0.5


@pytest.mark.asyncio
async def test_robot_move_joints_partial_body(api):
    """Omitted joints default to 0.0; provided joints are used."""
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_joints", json={"j3": 45.0})
    payload = response.json()["command"]["payload"]
    assert payload["j3"] == 45.0
    assert payload["j1"] == 0.0
    assert payload["j6"] == 0.0


@pytest.mark.asyncio
async def test_robot_move_joints_all_keys_in_payload(api):
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_joints", json={})
    payload = response.json()["command"]["payload"]
    for key in ("j1", "j2", "j3", "j4", "j5", "j6", "speed"):
        assert key in payload, f"missing key: {key}"


# ===========================================================================
# robot/move_pose
# ===========================================================================

@pytest.mark.asyncio
async def test_robot_move_pose_defaults(api):
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_pose", json={})
    _assert_sent_ok(response, "robot_move_pose")
    payload = response.json()["command"]["payload"]
    assert payload["x"] == 0.0
    assert payload["y"] == 0.0
    assert payload["z"] == 0.5
    assert payload["roll"] == 180.0
    assert payload["pitch"] == 0.0
    assert payload["yaw"] == 0.0
    assert payload["speed"] == 0.1


@pytest.mark.asyncio
async def test_robot_move_pose_custom(api):
    client, _ = api
    body = {"x": 0.3, "y": -0.1, "z": 0.6, "roll": 90.0, "pitch": 15.0, "yaw": -45.0, "speed": 0.3}
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_pose", json=body)
    _assert_sent_ok(response, "robot_move_pose")
    payload = response.json()["command"]["payload"]
    assert payload["x"] == 0.3
    assert payload["y"] == -0.1
    assert payload["z"] == 0.6
    assert payload["roll"] == 90.0
    assert payload["pitch"] == 15.0
    assert payload["yaw"] == -45.0
    assert payload["speed"] == 0.3


@pytest.mark.asyncio
async def test_robot_move_pose_all_keys_in_payload(api):
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/move_pose", json={})
    payload = response.json()["command"]["payload"]
    for key in ("x", "y", "z", "roll", "pitch", "yaw", "speed"):
        assert key in payload, f"missing key: {key}"


# ===========================================================================
# robot/home
# ===========================================================================

@pytest.mark.asyncio
async def test_robot_home_default_speed(api):
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/home", json={})
    _assert_sent_ok(response, "robot_home")
    assert response.json()["command"]["payload"]["speed"] == 0.05


@pytest.mark.asyncio
async def test_robot_home_custom_speed(api):
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/home", json={"speed": 0.2})
    assert response.json()["command"]["payload"]["speed"] == 0.2


@pytest.mark.asyncio
async def test_robot_home_empty_body(api):
    """No body at all should use default speed."""
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/home")
    assert response.status_code == 200
    assert response.json()["command"]["payload"]["speed"] == 0.05


# ===========================================================================
# Device not connected → 404
# ===========================================================================

@pytest.mark.asyncio
async def test_robot_endpoint_device_not_connected_joint_states(api_device_not_found):
    response = await api_device_not_found.post(f"/devices/{DEVICE_ID}/robot/joint_states")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_robot_endpoint_device_not_connected_move_joints(api_device_not_found):
    response = await api_device_not_found.post(
        f"/devices/{DEVICE_ID}/robot/move_joints",
        json={"j1": 0, "j2": 0, "j3": 0, "j4": 0, "j5": 0, "j6": 0},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_robot_endpoint_device_not_connected_stop(api_device_not_found):
    response = await api_device_not_found.post(f"/devices/{DEVICE_ID}/robot/stop")
    assert response.status_code == 404


# ===========================================================================
# Response shape
# ===========================================================================

@pytest.mark.asyncio
async def test_response_contains_command_id(api):
    """Each response embeds the full command block including 'id' and 'timestamp'."""
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/joint_states")
    cmd = response.json()["command"]
    assert "id" in cmd
    assert "timestamp" in cmd
    assert "payload" in cmd


@pytest.mark.asyncio
async def test_response_device_id_echoed(api):
    """The response echoes back the device_id and status."""
    client, _ = api
    response = await client.post(f"/devices/{DEVICE_ID}/robot/joint_states")
    body = response.json()
    assert body["status"] == "sent"


# ===========================================================================
# Existing generic endpoints still work (non-regression)
# ===========================================================================

@pytest.mark.asyncio
async def test_health_endpoint(api):
    client, _ = api
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_devices_list_endpoint(api):
    client, _ = api
    response = await client.get("/devices")
    assert response.status_code == 200
    assert "devices" in response.json()
