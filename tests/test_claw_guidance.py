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

    async def fake_account(key, app, conn):
        return ("user1", "ca_1")

    monkeypatch.setattr(composio_tools, "_list_actions_for_app", no_discovery)
    monkeypatch.setattr(composio_tools, "_account_for_app", fake_account)

    spec = ClawSpec(api_keys='{"composio": "ck_test"}', connections='["telegram"]')
    tool_defs, dispatch = await composio_tools.build(spec)
    assert any(d["name"] == "TELEGRAM_SEND_MESSAGE" for d in tool_defs)
    assert "TELEGRAM_SEND_MESSAGE" in dispatch
    assert composio_tools.has_rest_tools(spec) is True


async def test_composio_build_merges_discovered(monkeypatch):
    async def discovery(key, app):
        return [{"name": "GOOGLESHEETS_GET_SPREADSHEET_INFO", "description": "info",
                 "input_schema": {"type": "object", "properties": {}}, "no_auth": False}]

    async def fake_account(key, app, conn):
        return ("", "")   # not connected yet — tools still listed

    monkeypatch.setattr(composio_tools, "_list_actions_for_app", discovery)
    monkeypatch.setattr(composio_tools, "_account_for_app", fake_account)
    tool_defs, _ = await composio_tools.build(
        ClawSpec(api_keys='{"composio": "ck"}', connections='["googlesheets"]')
    )
    names = {d["name"] for d in tool_defs}
    assert "GOOGLESHEETS_GET_SPREADSHEET_INFO" in names   # discovered via v3 listing


async def test_composio_list_actions_v3(fake_httpx_get):
    # _list_actions_for_app hits the v3 tools endpoint and parses items[].slug/input_parameters.
    fake_httpx_get.payload = {"items": [
        {"slug": "WEATHERMAP_WEATHER", "description": "get weather",
         "input_parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
         "no_auth": True},
        {"slug": "WEATHERMAP_OLD", "description": "x", "input_parameters": {}, "is_deprecated": True},
    ]}
    out = await composio_tools._list_actions_for_app("ck_1", "weathermap")
    assert fake_httpx_get.captured["url"].endswith("/api/v3/tools")
    assert fake_httpx_get.captured["params"]["toolkit_slug"] == "weathermap"
    names = {a["name"] for a in out}
    assert "WEATHERMAP_WEATHER" in names          # kept
    assert "WEATHERMAP_OLD" not in names          # deprecated dropped
    assert next(a for a in out if a["name"] == "WEATHERMAP_WEATHER")["no_auth"] is True


# --- REST shapes (fake httpx supporting get + post) ------------------------

class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload, self.content = status, payload or {}, b"x"
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    # Class-level knobs the fixtures set; captured records the last call, calls records all.
    payload: dict = {"data": "ok done", "successful": True}
    status: int = 200
    routes: list = []          # list of (url_substring, status, payload) matched in order
    captured: dict = {}
    calls: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def _resp(self, url):
        for sub, status, payload in type(self).routes:
            if sub in url:
                return _FakeResp(status, payload)
        return _FakeResp(type(self).status, type(self).payload)

    async def post(self, url, headers=None, json=None):
        rec = {"method": "POST", "url": url, "headers": headers, "json": json}
        type(self).captured = rec
        type(self).calls.append(rec)
        return self._resp(url)

    async def get(self, url, headers=None, params=None):
        rec = {"method": "GET", "url": url, "headers": headers, "params": params}
        type(self).captured = rec
        type(self).calls.append(rec)
        return self._resp(url)


def _install_fake_httpx(monkeypatch):
    _FakeAsyncClient.payload = {"data": "ok done", "successful": True}
    _FakeAsyncClient.status = 200
    _FakeAsyncClient.routes = []
    _FakeAsyncClient.captured = {}
    _FakeAsyncClient.calls = []
    mod = types.ModuleType("httpx")
    mod.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", mod)
    return _FakeAsyncClient


@pytest.fixture
def fake_httpx(monkeypatch):
    return _install_fake_httpx(monkeypatch)


@pytest.fixture
def fake_httpx_get(monkeypatch):
    return _install_fake_httpx(monkeypatch)


async def test_composio_execute_posts_v3(fake_httpx):
    out = await composio_tools._execute(
        "ck_1", "TELEGRAM_SEND_MESSAGE", {"chat_id": "5", "text": "hi"}, "user1", "ca_9"
    )
    cap = fake_httpx.captured
    assert cap["url"].endswith("/api/v3/tools/execute/TELEGRAM_SEND_MESSAGE")
    assert cap["headers"]["x-api-key"] == "ck_1"
    assert cap["json"]["user_id"] == "user1"
    assert cap["json"]["connected_account_id"] == "ca_9"
    assert cap["json"]["arguments"] == {"chat_id": "5", "text": "hi"}
    assert "ok done" in out


def test_coerce_schema_drops_anthropic_invalid_property_keys():
    # Composio schemas sometimes carry property keys Anthropic rejects (spaces, >64 chars, etc.).
    bad = {
        "type": "object",
        "properties": {
            "good_key": {"type": "string"},
            "bad key with spaces": {"type": "string"},
            "x" * 80: {"type": "string"},
            "nested": {"type": "object", "properties": {"also bad!": {"type": "string"}}},
        },
        "required": ["good_key", "bad key with spaces"],
    }
    out = composio_tools._coerce_schema(bad)
    assert set(out["properties"].keys()) == {"good_key", "nested"}
    assert out["required"] == ["good_key"]                       # pruned to surviving keys
    assert out["properties"]["nested"]["properties"] == {}       # nested bad key dropped


async def test_composio_execute_raises_on_error(fake_httpx):
    fake_httpx.status = 400
    fake_httpx.payload = {"error": "bad key"}
    with pytest.raises(RuntimeError):
        await composio_tools._execute("ck", "X_DO", {}, "u", "")


# --- connect flow (auth config + link) -------------------------------------

async def test_resolve_auth_config_reuses_existing(fake_httpx):
    fake_httpx.routes = [("/api/v3/auth_configs", 200, {"items": [{"id": "ac_existing"}]})]
    acid = await composio_tools.resolve_or_create_auth_config("ck", "googlecalendar")
    assert acid == "ac_existing"


async def test_resolve_auth_config_creates_when_missing(monkeypatch):
    # GET returns no items → POST /auth_configs creates one (managed auth) and returns its id.
    seen = {"post_body": None}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None, params=None):
            return _FakeResp(200, {"items": []})

        async def post(self, url, headers=None, json=None):
            seen["post_body"] = json
            return _FakeResp(200, {"auth_config": {"id": "ac_new"}})

    mod = types.ModuleType("httpx")
    mod.AsyncClient = _Client
    monkeypatch.setitem(sys.modules, "httpx", mod)

    acid = await composio_tools.resolve_or_create_auth_config("ck", "gmail")
    assert acid == "ac_new"
    assert seen["post_body"]["toolkit"]["slug"] == "gmail"
    assert seen["post_body"]["auth_config"]["type"] == "use_composio_managed_auth"


async def test_initiate_connect_returns_redirect(fake_httpx):
    fake_httpx.routes = [
        ("/connected_accounts/link", 200,
         {"redirect_url": "https://accounts.google.com/o/oauth2/auth?x=1",
          "connected_account_id": "ca_new", "status": "INITIATED"}),
    ]
    out = await composio_tools.initiate_connect("ck", "ac_1", "default", "https://cb")
    assert out["redirect_url"].startswith("https://accounts.google.com")
    assert out["connection_id"] == "ca_new"
    cap = fake_httpx.captured
    assert cap["url"].endswith("/api/v3/connected_accounts/link")
    assert cap["json"]["auth_config_id"] == "ac_1"
    assert cap["json"]["user_id"] == "default"


# --- Manage Connections HTML page ------------------------------------------

def _manage_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from openclaw import router as claw_router
    app = FastAPI()
    app.include_router(claw_router.router)
    return TestClient(app)


def test_composio_manage_page_renders(monkeypatch):
    from openclaw.registry import registry

    async def fake_tools(key, app):
        return [{"name": "GOOGLECALENDAR_X", "description": "", "input_schema": {}, "no_auth": False}]

    async def fake_account(key, app, conn):
        return ("", "")  # not connected yet

    monkeypatch.setattr(composio_tools, "_list_actions_for_app", fake_tools)
    monkeypatch.setattr(composio_tools, "_account_for_app", fake_account)

    cid = registry.create(ClawSpec(api_keys='{"composio":"ck"}', connections='["googlecalendar"]', name="t"))
    try:
        r = _manage_client().get(f"/claw/{cid}/composio")
        assert r.status_code == 200
        assert "googlecalendar" in r.text
        assert "Connect" in r.text
        assert f"/claw/{cid}/composio/connect/googlecalendar" in r.text
        assert "Not connected" in r.text
    finally:
        registry.delete(cid)


def test_composio_manage_page_no_key():
    from openclaw.registry import registry
    cid = registry.create(ClawSpec(connections='["googlecalendar"]', name="t"))
    try:
        r = _manage_client().get(f"/claw/{cid}/composio")
        assert r.status_code == 200
        assert "api_keys" in r.text  # tells the user to add a Composio key
    finally:
        registry.delete(cid)


async def test_initiate_connect_falls_back_to_initiate(fake_httpx):
    fake_httpx.routes = [
        ("/connected_accounts/link", 404, {"error": "gone"}),
        ("/connected_accounts", 200,
         {"id": "ca_2", "connection_data": {"val": {"redirect_url": "https://x/auth", "status": "INITIATED"}}}),
    ]
    out = await composio_tools.initiate_connect("ck", "ac_1", "default", "https://cb")
    assert out["redirect_url"] == "https://x/auth"
    assert out["connection_id"] == "ca_2"


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

    async def fake_execute(key, tool_slug, args, user_id, account_id, no_auth=False):
        executed.update(key=key, action=tool_slug, args=args, user_id=user_id, account_id=account_id)
        return "Message sent."

    async def no_discovery(key, app):
        return []

    async def fake_account(key, app, conn):
        return ("user1", "ca_1")

    monkeypatch.setattr(composio_tools, "_execute", fake_execute)
    monkeypatch.setattr(composio_tools, "_list_actions_for_app", no_discovery)
    monkeypatch.setattr(composio_tools, "_account_for_app", fake_account)

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
    assert executed["user_id"] == "user1"            # v3 user scope, resolved from the account
    assert executed["account_id"] == "ca_1"
    assert any("tools" in c for c in tool_anthropic["calls"])


async def test_run_claw_surfaces_composio_tool_error(tool_anthropic, monkeypatch):
    # A failing Composio action must surface its real cause on the errors port (status error),
    # not just be paraphrased by the model.
    async def failing_execute(key, tool_slug, args, user_id, account_id, no_auth=False):
        raise RuntimeError("Composio 'TELEGRAM_SEND_MESSAGE' failed (HTTP 401): invalid api key")

    async def no_discovery(key, app):
        return []

    async def fake_account(key, app, conn):
        return ("user1", "ca_1")

    monkeypatch.setattr(composio_tools, "_execute", failing_execute)
    monkeypatch.setattr(composio_tools, "_list_actions_for_app", no_discovery)
    monkeypatch.setattr(composio_tools, "_account_for_app", fake_account)

    spec = ClawSpec(api_keys='{"anthropic": "sk-ant", "composio": "ck_bad"}', connections='["telegram"]')
    result = await claw_runtime.run_claw(spec, task="send hi")
    assert result["status"] == "error"
    assert "HTTP 401" in result["errors"]
