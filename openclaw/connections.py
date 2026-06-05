"""
connections.py
External-app connections for a claw — powered by Composio.

A claw's ``connections`` port carries a JSON list of connected applications
(Telegram, WhatsApp, Slack, …).  This module turns that list into:

  * **Outbound tools** — Composio-hosted MCP servers passed straight to Anthropic's
    native remote-MCP connector (``build_mcp_servers``), so the claw can *act* on the
    apps (send a Telegram message, post to Slack) without us hand-rolling a tool loop.
  * **OAuth wiring** — ``initiate_connection`` / ``list_connections`` / ``delete_connection``
    drive Composio's "connected accounts" flow so a user can link their own account.
  * **Inbound replies** — ``send_channel_reply`` posts the claw's response back to the
    chat a message arrived on (used by the channel webhook).

The REST shapes mirror the existing integration at
``Grafux-mcp/app/integrations/composio/{client.py,auth.py}`` so behavior stays
consistent across services.  ``httpx`` is imported lazily (like ``anthropic`` in
``claw_runtime``) so the devices server still boots where it is absent.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .models import ClawConnection, ClawSpec

logger = logging.getLogger("openclaw.connections")

# Composio endpoints (kept identical to the Grafux-mcp integration).
_COMPOSIO_BACKEND = "https://backend.composio.dev"
# Base for Composio's hosted MCP servers; an explicit per-connection ``mcp_url``
# always wins over anything derived from this base.
_COMPOSIO_MCP_BASE = os.environ.get("COMPOSIO_MCP_BASE", "https://mcp.composio.dev")

# Anthropic beta flag that enables the remote-MCP connector on the Messages API.
MCP_CONNECTOR_BETA = "mcp-client-2025-04-04"

_PLACEHOLDER_VALUES = {"empty", "unconnected"}


# ---------------------------------------------------------------------------
# Small local copies of the port helpers (kept here so this module does not
# import claw_runtime — claw_runtime imports us).
# ---------------------------------------------------------------------------

def _clean_port(text: Optional[str]) -> str:
    text = (text or "").strip()
    if text.lower() in _PLACEHOLDER_VALUES:
        return ""
    return text


def _maybe_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Composio API key + connection parsing
# ---------------------------------------------------------------------------

def resolve_composio_key(spec: ClawSpec) -> Optional[str]:
    """
    Find the Composio API key.

    Order: the api_keys port (JSON {"composio": "..."}), then credentials (same
    shape), then the COMPOSIO_API_KEY env var.  Mirrors ``_resolve_api_key`` in
    claw_runtime but for the Composio key rather than the Anthropic one.
    """
    for raw in (spec.api_keys, spec.credentials):
        raw = _clean_port(raw)
        if not raw:
            continue
        parsed = _maybe_json(raw)
        if isinstance(parsed, dict):
            for key in ("composio", "composio_api_key", "COMPOSIO_API_KEY"):
                if parsed.get(key):
                    return str(parsed[key])
    return os.environ.get("COMPOSIO_API_KEY") or None


def parse_connections(spec: ClawSpec) -> List[ClawConnection]:
    """Parse the ``connections`` port into ClawConnection objects (best-effort)."""
    parsed = _maybe_json(_clean_port(spec.connections))
    if not isinstance(parsed, list):
        return []
    out: List[ClawConnection] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ClawConnection(**item))
        except Exception as exc:  # noqa: BLE001 — skip malformed entries, keep the rest
            logger.warning("connections: skipping malformed entry %r (%s)", item, exc)
    return out


# ---------------------------------------------------------------------------
# Outbound tools — Composio MCP servers for the Anthropic connector
# ---------------------------------------------------------------------------

def _mcp_url_for(conn: ClawConnection) -> str:
    """
    Resolve the Composio MCP server URL for a connection.

    An explicit ``mcp_url`` on the connection is authoritative (the UI stores the
    URL Composio returns when the MCP server is created).  Otherwise we derive a
    best-effort URL from the connection id against the configured MCP base.
    """
    if _clean_port(conn.mcp_url):
        return conn.mcp_url.strip()
    cid = _clean_port(conn.connection_id)
    if not cid:
        return ""
    return f"{_COMPOSIO_MCP_BASE}/v3/mcp/{cid}"


def build_mcp_servers(spec: ClawSpec) -> List[Dict[str, Any]]:
    """
    Build the ``mcp_servers`` array for Anthropic's remote-MCP connector from the
    claw's enabled connections.  Returns ``[]`` when there are no usable
    connections or no Composio key (the runtime then falls back to a plain call).
    """
    key = resolve_composio_key(spec)
    if not key:
        return []
    servers: List[Dict[str, Any]] = []
    for conn in parse_connections(spec):
        if not conn.enabled:
            continue
        url = _mcp_url_for(conn)
        if not url:
            continue
        servers.append(
            {
                "type": "url",
                "url": url,
                "name": (_clean_port(conn.app) or "app").replace(" ", "_"),
                "authorization_token": key,
            }
        )
    return servers


def describe_connections(spec: ClawSpec) -> str:
    """A short human-readable list of enabled apps for the system prompt (no secrets)."""
    apps = [
        _clean_port(c.app) for c in parse_connections(spec) if c.enabled and _clean_port(c.app)
    ]
    return ", ".join(dict.fromkeys(apps))  # de-dupe, preserve order


# ---------------------------------------------------------------------------
# OAuth / connected-accounts flow (mirrors composio/auth.py)
# ---------------------------------------------------------------------------

async def initiate_connection(
    app: str, user_id: str, redirect_uri: str, key: str
) -> Dict[str, str]:
    """Start a Composio OAuth connection for ``app`` and return the redirect URL."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_COMPOSIO_BACKEND}/api/v1/connectedAccounts",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={
                "appName": app,
                "userUuid": user_id or f"claw:{app}",
                "redirectUri": redirect_uri,
                "authMode": "OAUTH2",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return {
        "app": app,
        "connection_id": data.get("connectedAccountId", ""),
        "redirect_url": data.get("redirectUrl", ""),
        "status": data.get("connectionStatus", "pending"),
    }


async def list_connections(key: str) -> List[Dict[str, Any]]:
    """List Composio connected accounts (id + app + status; never tokens)."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_COMPOSIO_BACKEND}/api/v1/connectedAccounts",
            headers={"x-api-key": key},
            params={"pageSize": 100},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "connection_id": it.get("id") or it.get("connectedAccountId", ""),
                "app": (it.get("appName") or it.get("appUniqueId") or "").lower(),
                "account_label": it.get("label") or it.get("clientUniqueUserId", ""),
                "status": it.get("status", ""),
            }
        )
    return out


async def delete_connection(connection_id: str, key: str) -> bool:
    """Delete a Composio connected account.  Returns True on success."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.delete(
            f"{_COMPOSIO_BACKEND}/api/v1/connectedAccounts/{connection_id}",
            headers={"x-api-key": key},
        )
        return 200 <= resp.status_code < 300


# ---------------------------------------------------------------------------
# Inbound — parse a provider webhook + send the reply back
# ---------------------------------------------------------------------------

def parse_inbound(provider: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract ``(chat_id, text)`` from a provider's inbound webhook payload.

    Returns ``("", "")`` for payloads that carry no user message (e.g. delivery
    receipts) so the caller can ignore them.
    """
    provider = (provider or "").lower()
    if provider == "telegram":
        msg = payload.get("message") or payload.get("edited_message") or {}
        chat = msg.get("chat", {})
        return str(chat.get("id", "")), str(msg.get("text", ""))
    if provider == "whatsapp":
        # Meta Cloud API shape: entry[].changes[].value.messages[]
        try:
            value = payload["entry"][0]["changes"][0]["value"]
            message = value["messages"][0]
            chat_id = message.get("from", "")
            text = (message.get("text") or {}).get("body", "")
            return str(chat_id), str(text)
        except (KeyError, IndexError, TypeError):
            return "", ""
    if provider == "slack":
        event = payload.get("event", {})
        return str(event.get("channel", "")), str(event.get("text", ""))
    # Generic fallback: look for common keys.
    return str(payload.get("chat_id", "")), str(payload.get("text", ""))


# Composio send-message action per provider.
_SEND_ACTIONS = {
    "telegram": "TELEGRAM_SEND_MESSAGE",
    "whatsapp": "WHATSAPP_SEND_MESSAGE",
    "slack": "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL",
}


def _send_arguments(provider: str, connection_id: str, chat_id: str, text: str) -> Dict[str, Any]:
    provider = (provider or "").lower()
    if provider == "telegram":
        return {"connectedAccountId": connection_id, "chat_id": chat_id, "text": text}
    if provider == "whatsapp":
        return {"connectedAccountId": connection_id, "to": chat_id, "message": text}
    if provider == "slack":
        return {"connectedAccountId": connection_id, "channel": chat_id, "text": text}
    return {"connectedAccountId": connection_id, "chat_id": chat_id, "text": text}


async def send_channel_reply(
    provider: str, connection_id: str, chat_id: str, text: str, key: str
) -> bool:
    """
    Post ``text`` back to ``chat_id`` on ``provider`` via the Composio send-message
    action (REST execute, as in composio/client.py).  Returns True on success.
    """
    import httpx

    provider = (provider or "").lower()
    action = _SEND_ACTIONS.get(provider)
    if not action or not chat_id or not text:
        return False
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_COMPOSIO_BACKEND}/api/v2/actions/{action}/execute",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"input": _send_arguments(provider, connection_id, chat_id, text)},
        )
        if resp.status_code >= 400:
            logger.warning("send_channel_reply %s failed: %s %s", provider, resp.status_code, resp.text[:200])
        return 200 <= resp.status_code < 300
