"""
test_claw_toolkits.py
Tests for the claw connection grounding added for stream voice/text block creation:

  * ``composio_tools.list_toolkit_slugs`` — the authoritative app-slug list.
  * ``GET /claw/toolkits`` — the endpoint the orchestrator fetches for grounding.
  * ``claw_runtime.scaffold_claw`` — normalizes drafted/hinted ``connections`` to valid slugs.

Pure/offline: the Composio HTTP calls and (optional) Anthropic scaffold are monkeypatched.
"""

import json

from openclaw import claw_runtime, composio_tools, connections
from openclaw import router as claw_router


# ── list_toolkit_slugs ────────────────────────────────────────────────────────

async def test_list_toolkit_slugs_sorted(monkeypatch):
    async def fake_slugs(key):
        return {"slack", "gmail", "googlesheets"}

    monkeypatch.setattr(composio_tools, "_toolkit_slugs", fake_slugs)
    assert await composio_tools.list_toolkit_slugs("ck") == ["gmail", "googlesheets", "slack"]


async def test_list_toolkit_slugs_no_key():
    assert await composio_tools.list_toolkit_slugs("") == []


# ── GET /claw/toolkits ────────────────────────────────────────────────────────

async def test_toolkits_endpoint(monkeypatch):
    async def fake_list(key):
        assert key == "ck"
        return ["gmail", "googlesheets"]

    monkeypatch.setattr(claw_router.composio_tools, "list_toolkit_slugs", fake_list)
    monkeypatch.setattr(claw_router.connections, "resolve_composio_key", lambda spec: "ck")
    resp = await claw_router.list_toolkits()
    assert resp.toolkits == ["gmail", "googlesheets"]


async def test_toolkits_endpoint_no_key(monkeypatch):
    # No server key → resolve returns None → empty list, no crash.
    monkeypatch.setattr(claw_router.connections, "resolve_composio_key", lambda spec: None)

    async def fake_list(key):
        return []

    monkeypatch.setattr(claw_router.composio_tools, "list_toolkit_slugs", fake_list)
    resp = await claw_router.list_toolkits()
    assert resp.toolkits == []


# ── scaffold_claw connection normalization ────────────────────────────────────

def _patch_resolvers(monkeypatch):
    """Resolve any "…sheet…" app to the canonical googlesheets slug; pass others through."""
    async def fake_resolve(key, app):
        return "googlesheets" if "sheet" in app.lower() else app.strip().lower()

    monkeypatch.setattr(claw_runtime.composio_tools, "resolve_toolkit_slug", fake_resolve)
    monkeypatch.setattr(claw_runtime.connections, "resolve_composio_key", lambda spec: "ck")


async def test_scaffold_normalizes_explicit_hint(monkeypatch):
    # Force the AI-less fallback draft (empty design ports) so we isolate the hint path.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_resolvers(monkeypatch)

    out = await claw_runtime.scaffold_claw(
        "an agent that logs data", connections_hint=["google sheets", "googlesheet"]
    )
    # Both spellings resolve to the same canonical slug and de-dupe.
    assert json.loads(out["connections"]) == ["googlesheets"]


async def test_scaffold_normalizes_ai_drafted_connections(monkeypatch):
    _patch_resolvers(monkeypatch)

    async def fake_draft(description, name=""):
        return {
            "soul": "You append rows.",
            "skills": "",
            "agent": claw_runtime.DEFAULT_MODEL,
            "task": "log a row",
            "memory": "",
            "tools_config": "",
            # AI returns the dict-shaped connections that the model tends to emit.
            "connections": json.dumps([{"app": "googlesheet", "enabled": True}]),
            "credentials": "<x>",
            "api_keys": "<y>",
            "guidance": "",
        }

    monkeypatch.setattr(claw_runtime, "_scaffold_draft", fake_draft)
    out = await claw_runtime.scaffold_claw("send info to google sheets")
    assert json.loads(out["connections"]) == ["googlesheets"]


async def test_scaffold_no_connections_stays_empty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_resolvers(monkeypatch)

    out = await claw_runtime.scaffold_claw("a plain assistant")
    assert out["connections"] == ""  # nothing implied, nothing hinted
