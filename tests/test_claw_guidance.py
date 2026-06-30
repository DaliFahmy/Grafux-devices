"""
test_claw_guidance.py
Unit tests for the claw guidance engine, QR rendering, and the friendlier
``connections`` shapes (app-name list + one-click Tool-Router).

All pure/offline except the no-key run_claw fallback, which uses a tiny stub
``anthropic`` module so the lazy import inside claw_runtime resolves.
"""

import sys
import types

import pytest

from openclaw import claw_runtime, connections, guidance, qr
from openclaw.connections import TOOL_ROUTER_URL
from openclaw.models import ClawSpec


def _spec(**kw) -> ClawSpec:
    return ClawSpec(**kw)


# ---------------------------------------------------------------------------
# guidance.analyze
# ---------------------------------------------------------------------------

def test_guidance_missing_api_key_not_ready():
    report = guidance.analyze(_spec(soul="You are a test claw."))
    assert report["ready"] is False
    assert report["setup_status"] == "Needs Anthropic API key"
    assert "api_keys" in report["guidance"]
    assert "sk-ant" in report["guidance"]


def test_guidance_missing_soul_warns_but_ready():
    report = guidance.analyze(_spec(api_keys="sk-ant-test"))
    assert report["ready"] is True                       # only the key blocks a run
    assert "persona" in report["guidance"].lower() or "soul" in report["guidance"].lower()


def test_guidance_fully_configured_is_ready():
    report = guidance.analyze(_spec(api_keys="sk-ant-test", soul="You are a helpful claw."))
    assert report["ready"] is True
    assert report["setup_status"] == "Ready"
    assert "ready to run" in report["guidance"].lower()


def test_guidance_app_without_composio_key_needs_auth():
    report = guidance.analyze(
        _spec(api_keys="sk-ant-test", soul="hi", connections='["whatsapp"]')
    )
    statuses = report["connections_status"]
    assert len(statuses) == 1
    assert statuses[0]["app"] == "whatsapp"
    assert statuses[0]["needs_auth"] is True
    assert statuses[0]["configured"] is False
    # setup_status flags the pending connection
    assert "whatsapp" in report["setup_status"].lower()
    assert "Manage Connections" in report["guidance"]


def test_guidance_no_connections_offers_apps():
    report = guidance.analyze(_spec(api_keys="sk-ant-test", soul="hi"))
    # The "connect apps (optional)" section lists catalog apps.
    assert "whatsapp" in report["guidance"]
    assert "telegram" in report["guidance"]


# ---------------------------------------------------------------------------
# connections — friendly shapes + Tool-Router
# ---------------------------------------------------------------------------

def test_parse_bare_app_name_list():
    conns = connections.parse_connections(_spec(connections='["whatsapp", "telegram"]'))
    assert [c.app for c in conns] == ["whatsapp", "telegram"]
    # No url/headers ⇒ not configured for tools yet (needs auth / a url).
    assert all(not connections._mcp_url_for(c) for c in conns)


def test_parse_app_only_dict():
    conns = connections.parse_connections(_spec(connections='[{"app": "slack", "enabled": true}]'))
    assert len(conns) == 1 and conns[0].app == "slack" and conns[0].enabled is True


def test_tool_router_resolves_with_composio_key():
    spec = _spec(
        api_keys='{"composio": "ck_test"}',
        connections='[{"app": "telegram", "tool_router": true}]',
    )
    conns = connections.parse_connections(spec)
    assert conns[0].mcp_url == TOOL_ROUTER_URL
    assert conns[0].header_auth is True
    assert conns[0].api_key == "ck_test"
    # Header-auth ⇒ driven by the local loop, excluded from the bearer connector.
    assert connections.build_mcp_servers(spec) == []
    assert connections.local_loop_connections(spec)  # non-empty


def test_tool_router_noop_without_composio_key():
    spec = _spec(connections='[{"app": "telegram", "tool_router": true}]')
    conns = connections.parse_connections(spec)
    # No key to authenticate with ⇒ left unresolved (guidance tells the user to add one).
    assert conns[0].mcp_url == ""
    assert conns[0].header_auth is False


def test_legacy_mcpservers_shape_still_parses():
    spec = _spec(
        connections='{"mcpServers": {"composio": {"url": "https://x/mcp", '
                    '"headers": {"x-consumer-api-key": "ck_1"}}}}'
    )
    conns = connections.parse_connections(spec)
    assert len(conns) == 1
    assert conns[0].mcp_url == "https://x/mcp"
    assert conns[0].header_auth is True  # custom headers ⇒ local loop


def test_legacy_list_shape_still_parses():
    spec = _spec(connections='[{"app": "slack", "mcp_url": "https://y/mcp"}]')
    conns = connections.parse_connections(spec)
    assert conns[0].mcp_url == "https://y/mcp"


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------

def test_qr_data_uri_empty_input():
    assert qr.qr_data_uri("") == ""


def test_qr_data_uri_renders_png():
    pytest.importorskip("segno")
    uri = qr.qr_data_uri("https://example.com/connect")
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 64


# ---------------------------------------------------------------------------
# tool-schema cache
# ---------------------------------------------------------------------------

def test_clear_tool_cache():
    connections._TOOL_SCHEMA_CACHE[("u", "h")] = [{"name": "x"}]
    connections.clear_tool_cache()
    assert connections._TOOL_SCHEMA_CACHE == {}


def test_tool_cache_key_changes_with_headers():
    k1 = connections._tool_cache_key("https://x/mcp", {"x-consumer-api-key": "a"})
    k2 = connections._tool_cache_key("https://x/mcp", {"x-consumer-api-key": "b"})
    assert k1 != k2


# ---------------------------------------------------------------------------
# run_claw — teach-don't-fail when the key is missing
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_anthropic(monkeypatch):
    """Minimal 'anthropic' module so the lazy import resolves (calls are never made here)."""
    mod = types.ModuleType("anthropic")

    class _AsyncAnthropic:
        def __init__(self, **kw):
            pass

    mod.AsyncAnthropic = _AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    claw_runtime._ANTHROPIC_CLIENTS.clear()
    return mod


async def test_run_without_key_returns_guidance_as_response(stub_anthropic):
    result = await claw_runtime.run_claw(_spec(soul="hi"), task="do a thing")
    assert result["status"] == "error"
    # The response teaches the user what to add instead of being blank.
    assert "api_keys" in result["response"]
    assert result["setup_status"] == "Needs Anthropic API key"
    assert "No Anthropic API key" in result["errors"]


async def test_run_result_always_carries_guidance_fields(stub_anthropic):
    # Every run result (even the error/no-key path) carries the guidance surfaces so the
    # block's guidance/setup_status/connections_status ports are always populated.
    result = await claw_runtime.run_claw(_spec(), task="hi")
    assert "guidance" in result and "setup_status" in result and "connections_status" in result
