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
    assert "Composio" in report["guidance"]


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


def test_app_name_connection_is_not_synthesised_to_mcp_url():
    # App-name connections now get tools via Composio REST, NOT a synthesised Tool-Router MCP URL.
    spec = _spec(
        api_keys='{"composio": "ck_test"}',
        connections='[{"app": "telegram", "tool_router": true}]',
    )
    conns = connections.parse_connections(spec)
    assert conns[0].mcp_url == ""          # no MCP-URL synthesis anymore
    assert conns[0].header_auth is False
    assert connections.build_mcp_servers(spec) == []
    assert connections.local_loop_connections(spec) == []  # not an MCP-path connection


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


# ---------------------------------------------------------------------------
# Composio REST tools — a connected app gives the claw real tools (no MCP URL)
# ---------------------------------------------------------------------------

from openclaw import composio_tools  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_composio_cache():
    composio_tools.clear_cache()
    yield
    composio_tools.clear_cache()


async def test_composio_build_empty_without_key():
    # No Composio key ⇒ no REST tools (byte-identical to before).
    assert await composio_tools.build(ClawSpec(connections='["telegram"]')) == ([], {})


async def test_composio_build_telegram_curated(monkeypatch):
    # Auto-discovery returns nothing → the curated TELEGRAM_SEND_MESSAGE is still present.
    async def no_discovery(key, app):
        return []

    async def fake_accounts(key):
        return [{"app": "telegram", "connection_id": "ca_1", "status": "ACTIVE", "account_label": ""}]

    monkeypatch.setattr(composio_tools, "_list_actions_for_app", no_discovery)
    monkeypatch.setattr(connections, "list_connections", fake_accounts)

    spec = ClawSpec(api_keys='{"composio": "ck_test"}', connections='["telegram"]')
    tool_defs, dispatch = await composio_tools.build(spec)
    assert any(d["name"] == "TELEGRAM_SEND_MESSAGE" for d in tool_defs)
    assert "TELEGRAM_SEND_MESSAGE" in dispatch
    assert composio_tools.has_rest_tools(spec) is True


async def test_composio_build_merges_discovered(monkeypatch):
    async def discovery(key, app):
        return [{"name": "TELEGRAM_GET_ME", "description": "who am I", "input_schema": {"type": "object", "properties": {}}}]

    async def fake_accounts(key):
        return []

    monkeypatch.setattr(composio_tools, "_list_actions_for_app", discovery)
    monkeypatch.setattr(connections, "list_connections", fake_accounts)
    tool_defs, _ = await composio_tools.build(
        ClawSpec(api_keys='{"composio": "ck"}', connections='["telegram"]')
    )
    names = {d["name"] for d in tool_defs}
    assert "TELEGRAM_SEND_MESSAGE" in names   # curated
    assert "TELEGRAM_GET_ME" in names         # discovered


# --- REST execute shape (fake httpx) ---------------------------------------

class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload, self.content = status, payload or {}, b"x"
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    captured: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.captured = {"url": url, "headers": headers, "json": json}
        return _FakeResp(200, {"response": "ok done"})


@pytest.fixture
def fake_httpx(monkeypatch):
    mod = types.ModuleType("httpx")
    mod.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", mod)
    return _FakeAsyncClient


async def test_composio_execute_posts_rest(fake_httpx):
    out = await composio_tools._execute(
        "ck_1", "TELEGRAM_SEND_MESSAGE", {"chat_id": "5", "text": "hi"}, "ca_9"
    )
    cap = fake_httpx.captured
    assert cap["url"].endswith("/api/v2/actions/TELEGRAM_SEND_MESSAGE/execute")
    assert cap["headers"]["x-api-key"] == "ck_1"
    assert cap["json"]["input"]["connectedAccountId"] == "ca_9"
    assert cap["json"]["input"]["chat_id"] == "5"
    assert "ok done" in out


# --- run_claw end-to-end via a fake tool-use client ------------------------

class _TU:
    def __init__(self, id, name, inp):
        self.type, self.id, self.name, self.input = "tool_use", id, name, inp


class _TX:
    def __init__(self, t):
        self.type, self.text = "text", t


class _M:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason, self.usage = content, stop_reason, None


@pytest.fixture
def tool_anthropic(monkeypatch):
    mod = types.ModuleType("anthropic")
    state = {"n": 0, "calls": []}

    class _Messages:
        async def create(self, **kwargs):
            state["calls"].append(kwargs)
            state["n"] += 1
            if state["n"] == 1:
                return _M([_TU("t1", "TELEGRAM_SEND_MESSAGE", {"chat_id": "5", "text": "hi"})], "tool_use")
            return _M([_TX("Sent it!")], "end_turn")

    class _AsyncAnthropic:
        def __init__(self, **kw):
            self.messages = _Messages()

    mod.AsyncAnthropic = _AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    claw_runtime._ANTHROPIC_CLIENTS.clear()
    return state


async def test_run_claw_uses_composio_tool(tool_anthropic, monkeypatch):
    executed = {}

    async def fake_execute(key, action, args, account_id):
        executed.update(key=key, action=action, args=args, account_id=account_id)
        return "Message sent."

    async def no_discovery(key, app):
        return []

    async def fake_accounts(key):
        return [{"app": "telegram", "connection_id": "ca_1", "status": "ACTIVE", "account_label": ""}]

    monkeypatch.setattr(composio_tools, "_execute", fake_execute)
    monkeypatch.setattr(composio_tools, "_list_actions_for_app", no_discovery)
    monkeypatch.setattr(connections, "list_connections", fake_accounts)

    spec = ClawSpec(
        api_keys='{"anthropic": "sk-ant", "composio": "ck_test"}',
        connections='["telegram"]',
        soul="You message Telegram.",
    )
    result = await claw_runtime.run_claw(spec, task="send hi to my telegram")
    assert result["status"] == "ok"
    assert result["response"] == "Sent it!"
    assert executed["action"] == "TELEGRAM_SEND_MESSAGE"
    assert executed["args"] == {"chat_id": "5", "text": "hi"}
    assert executed["account_id"] == "ca_1"          # resolved from list_connections
    assert any("tools" in c for c in tool_anthropic["calls"])


async def test_run_claw_surfaces_composio_tool_error(tool_anthropic, monkeypatch):
    # A failing Composio action must surface its real cause on the errors port (status error),
    # not just be paraphrased by the model.
    async def failing_execute(key, action, args, account_id):
        raise RuntimeError("Composio 'TELEGRAM_SEND_MESSAGE' failed (HTTP 401): invalid api key")

    async def no_discovery(key, app):
        return []

    async def fake_accounts(key):
        return [{"app": "telegram", "connection_id": "ca_1", "status": "ACTIVE"}]

    monkeypatch.setattr(composio_tools, "_execute", failing_execute)
    monkeypatch.setattr(composio_tools, "_list_actions_for_app", no_discovery)
    monkeypatch.setattr(connections, "list_connections", fake_accounts)

    spec = ClawSpec(api_keys='{"anthropic": "sk-ant", "composio": "ck_bad"}', connections='["telegram"]')
    result = await claw_runtime.run_claw(spec, task="send hi")
    assert result["status"] == "error"
    assert "HTTP 401" in result["errors"]
