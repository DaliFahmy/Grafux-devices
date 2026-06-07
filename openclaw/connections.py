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

# Anthropic beta flag that enables the remote-MCP connector on the Messages API.
# (mcp-client-2025-04-04 is deprecated; 2025-11-20 requires a matching mcp_toolset
# entry in the tools array for every server — see claw_runtime.run_claw.)
MCP_CONNECTOR_BETA = "mcp-client-2025-11-20"

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
    """
    Parse the ``connections`` port into ClawConnection objects (best-effort).

    Two shapes are accepted:

      * **Standard MCP client config** — the JSON Composio (and other MCP hosts) hand you::

            {"mcpServers": {"composio": {"url": "https://…/mcp",
                                          "headers": {"x-consumer-api-key": "ck_…"}}}}

        Each server becomes a connection; any ``headers`` route it through the local MCP
        loop (Anthropic's connector cannot send custom headers).

      * **Legacy list** — ``[{"app", "mcp_url", "header_auth", "api_key", …}]``.
    """
    parsed = _maybe_json(_clean_port(spec.connections))
    if isinstance(parsed, dict) and isinstance(parsed.get("mcpServers"), dict):
        return _connections_from_mcp_servers(parsed["mcpServers"])
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


def _connections_from_mcp_servers(servers: Dict[str, Any]) -> List[ClawConnection]:
    """Turn a standard ``mcpServers`` config map into ClawConnection objects."""
    out: List[ClawConnection] = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        url = str(cfg.get("url", "")).strip()
        if not url:
            continue
        raw_headers = cfg.get("headers") if isinstance(cfg.get("headers"), dict) else {}
        headers = {str(k): str(v) for k, v in raw_headers.items()}
        out.append(
            ClawConnection(
                app=str(name),
                mcp_url=url,
                user_id=str(cfg.get("user_id", "")),
                headers=headers,
                header_auth=bool(headers),  # custom headers ⇒ local loop
                enabled=bool(cfg.get("enabled", True)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Outbound tools — Composio MCP servers for the Anthropic connector
# ---------------------------------------------------------------------------

def _mcp_url_for(conn: ClawConnection) -> str:
    """
    Resolve the Composio MCP server URL for a connection.

    The connection's ``mcp_url`` (the URL Composio gives when you create an MCP
    server, e.g. ``https://backend.composio.dev/v3/mcp/<server_id>``) is required —
    Anthropic's connector connects to it directly.  When a ``user_id`` is set it is
    appended as a query param so Composio selects that user's connected accounts.
    """
    url = _clean_port(conn.mcp_url)
    if not url:
        return ""
    user_id = _clean_port(conn.user_id)
    if user_id and "user_id=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}user_id={user_id}"
    return url


def _server_name_for(conn: ClawConnection, index: int) -> str:
    base = (_clean_port(conn.app) or f"app{index}").replace(" ", "_")
    return f"{base}_{index}"  # unique per server (Anthropic requires unique names)


def build_mcp_servers(spec: ClawSpec) -> List[Dict[str, Any]]:
    """
    Build the ``mcp_servers`` array for Anthropic's remote-MCP connector from the
    claw's enabled connections.  Returns ``[]`` when no connection carries an
    ``mcp_url`` (the runtime then falls back to a plain call).

    Note: the Composio MCP server must be self-authenticating (api-key requirement
    disabled) because Anthropic's connector can only send an OAuth bearer via
    ``authorization_token`` — it cannot send Composio's ``x-api-key`` header. We set
    ``authorization_token`` only when the connection provides an explicit bearer.
    """
    servers: List[Dict[str, Any]] = []
    for i, conn in enumerate(parse_connections(spec)):
        if not conn.enabled:
            continue
        if _is_local_loop(conn):
            # Header-authenticated servers (e.g. Composio's Connect/Tool-Router URL) cannot
            # go through Anthropic's connector — it only sends a bearer, never a custom
            # header. They are handled by the local MCP loop instead (run_local_agent_loop).
            continue
        url = _mcp_url_for(conn)
        if not url:
            continue
        entry: Dict[str, Any] = {
            "type": "url",
            "url": url,
            "name": _server_name_for(conn, i),
        }
        token = _clean_port(conn.auth_token)
        if token:
            entry["authorization_token"] = token
        servers.append(entry)
    return servers


def build_mcp_toolsets(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the matching ``tools`` array (one ``mcp_toolset`` per server).

    The 2025-11-20 connector requires every entry in ``mcp_servers`` to be referenced
    by exactly one toolset; with no per-tool config this enables all of the server's
    tools.
    """
    return [{"type": "mcp_toolset", "mcp_server_name": s["name"]} for s in servers]


def describe_connections(spec: ClawSpec) -> str:
    """A short human-readable list of enabled apps for the system prompt (no secrets)."""
    apps = [
        _clean_port(c.app) for c in parse_connections(spec) if c.enabled and _clean_port(c.app)
    ]
    return ", ".join(dict.fromkeys(apps))  # de-dupe, preserve order


# ---------------------------------------------------------------------------
# Local MCP loop — header-authenticated servers (Composio Connect / Tool-Router)
#
# Anthropic's remote-MCP connector can only send an OAuth bearer (authorization_token),
# never a custom header, so it cannot reach a Composio server that wants
# ``x-consumer-api-key``.  For those connections (``header_auth: true``) we act as the
# MCP client ourselves — connect over the MCP SDK's Streamable-HTTP (or SSE) transport
# with the api-key header, expose the server's tools to Claude as ordinary tools, and
# run a normal tool-use loop, executing each call against the MCP server.
# ---------------------------------------------------------------------------

# Composio's Connect/Tool-Router MCP servers authenticate with this header.
COMPOSIO_API_KEY_HEADER = "x-consumer-api-key"

# Guard so a tool that keeps asking for more tools cannot loop forever.
_MAX_TOOL_ITERATIONS = 8


def _is_local_loop(conn: ClawConnection) -> bool:
    """True if this connection must be driven by the local MCP loop (carries custom headers)."""
    return bool(conn.header_auth or conn.headers)


def local_loop_connections(spec: ClawSpec) -> List[ClawConnection]:
    """Enabled connections that must be driven by the local MCP loop (header auth)."""
    return [
        c
        for c in parse_connections(spec)
        if c.enabled and _is_local_loop(c) and _mcp_url_for(c)
    ]


def _header_key_for(conn: ClawConnection, spec: ClawSpec) -> str:
    """The ``x-consumer-api-key`` value: the connection's own key, else the claw's Composio key."""
    explicit = _clean_port(conn.api_key)
    return explicit or (resolve_composio_key(spec) or "")


def _resolve_headers(conn: ClawConnection, spec: ClawSpec) -> Dict[str, str]:
    """
    The full header set to send to this connection's MCP server.

    Explicit ``headers`` (from the mcpServers config) are used as-is. When ``header_auth``
    is set but no ``x-consumer-api-key`` was supplied, one is synthesized from the
    connection's ``api_key`` / the claw's resolved Composio key, so the legacy
    "tick header-auth + key in api_keys port" path keeps working.
    """
    headers = {str(k): str(v) for k, v in (conn.headers or {}).items()}
    has_composio = any(k.lower() == COMPOSIO_API_KEY_HEADER for k in headers)
    if conn.header_auth and not has_composio:
        key = _header_key_for(conn, spec)
        if key:
            headers[COMPOSIO_API_KEY_HEADER] = key
    return headers


def _sanitize_tool_name(name: str) -> str:
    """Coerce an MCP tool name into Anthropic's ``^[a-zA-Z0-9_-]{1,128}$`` constraint."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (name or ""))
    cleaned = cleaned.strip("_") or "tool"
    return cleaned[:128]


def _tool_result_text(result: Any) -> str:
    """Flatten an MCP ``call_tool`` result's content blocks into text for a tool_result."""
    parts: List[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "\n".join(parts)


def _extract_text(message: Any) -> str:
    """Join the text blocks of an Anthropic message (mirrors claw_runtime's extraction)."""
    return "".join(
        b.text for b in getattr(message, "content", []) if getattr(b, "type", None) == "text"
    )


async def run_local_agent_loop(
    client: Any,
    base_kwargs: Dict[str, Any],
    spec: ClawSpec,
    connector_servers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Run a tool-use loop against the claw's header-authenticated MCP servers and return
    the final assistant text.

    ``base_kwargs`` is the same dict claw_runtime builds for ``messages.create`` (model,
    max_tokens, temperature, system, messages).  ``connector_servers`` (the bearer/self-auth
    servers from ``build_mcp_servers``) are passed through on the same call so a claw that
    mixes both connection styles still works; those tools resolve server-side and never
    surface as local ``tool_use`` blocks.

    Raises RuntimeError with a friendly message when the ``mcp`` SDK is not installed.
    """
    try:
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:  # noqa: F841
        raise RuntimeError(
            "Header-authenticated connections need the 'mcp' package on the devices server. "
            "Add 'mcp>=1.0.0' to requirements.txt (pip install mcp)."
        ) from exc

    conns = local_loop_connections(spec)
    connector_servers = connector_servers or []

    # dispatch maps the (sanitized, unique) tool name we expose to Claude back to the
    # owning MCP session and the tool's real name on that server.
    dispatch: Dict[str, Any] = {}
    tool_defs: List[Dict[str, Any]] = []

    async with AsyncExitStack() as stack:
        for i, conn in enumerate(conns):
            url = _mcp_url_for(conn)
            headers = _resolve_headers(conn, spec)
            if not headers:
                # No auth header at all would draw an opaque 401 — fail fast instead.
                raise RuntimeError(
                    "No auth header for connection "
                    f"'{_clean_port(conn.app) or url}'. Provide it in the mcpServers config "
                    '(headers: {"x-consumer-api-key": "<key>"}), set COMPOSIO_API_KEY on the '
                    'devices server, or add {"composio": "<key>"} to the claw\'s api_keys port.'
                )

            # Streamable-HTTP is Composio's transport; fall back to SSE for ``…/sse`` URLs.
            if url.rstrip("/").endswith("/sse"):
                transport_ctx = sse_client(url, headers=headers)
            else:
                transport_ctx = streamablehttp_client(url, headers=headers)
            streams = await stack.enter_async_context(transport_ctx)
            # streamable_http yields (read, write, get_session_id); sse yields (read, write).
            read, write = streams[0], streams[1]
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            prefix = _server_name_for(conn, i)
            listed = await session.list_tools()
            logger.info(
                "local MCP '%s' (%s) exposed %d tool(s): %s",
                _clean_port(conn.app) or prefix,
                url,
                len(listed.tools),
                ", ".join(t.name for t in listed.tools) or "(none)",
            )
            for tool in listed.tools:
                # Prefix when there is more than one server so names stay unique.
                raw = f"{prefix}__{tool.name}" if len(conns) > 1 else tool.name
                exposed = _sanitize_tool_name(raw)
                dispatch[exposed] = (session, tool.name)
                schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else None
                tool_defs.append(
                    {
                        "name": exposed,
                        "description": tool.description or "",
                        "input_schema": schema or {"type": "object", "properties": {}},
                    }
                )

        if not tool_defs:
            # Connected and authenticated, but the server(s) advertised no tools — the
            # model would then claim it "has no integration". Surface the real cause.
            apps = ", ".join(_clean_port(c.app) or _mcp_url_for(c) for c in conns)
            raise RuntimeError(
                f"Connected to the MCP server(s) for [{apps}] but they exposed no tools. "
                "For Composio: enable the toolkit (e.g. Telegram) on the MCP server / "
                "Tool-Router config, and make sure the user_id has a connected, ACTIVE "
                "account for that toolkit. A generic Tool-Router URL can surface no toolkit "
                "tools — prefer a per-toolkit MCP server URL for the app."
            )

        messages = list(base_kwargs.get("messages", []))
        loop_kwargs = {k: v for k, v in base_kwargs.items() if k != "messages"}
        use_connector = bool(connector_servers)

        last_message: Any = None
        for _ in range(_MAX_TOOL_ITERATIONS):
            if use_connector:
                last_message = await client.beta.messages.create(
                    betas=[MCP_CONNECTOR_BETA],
                    mcp_servers=connector_servers,
                    tools=build_mcp_toolsets(connector_servers) + tool_defs,
                    messages=messages,
                    **loop_kwargs,
                )
            else:
                last_message = await client.messages.create(
                    tools=tool_defs, messages=messages, **loop_kwargs
                )

            tool_uses = [
                b for b in last_message.content if getattr(b, "type", None) == "tool_use"
            ]
            if last_message.stop_reason != "tool_use" or not tool_uses:
                return _extract_text(last_message)

            # Echo the assistant turn back verbatim (preserves connector blocks too).
            messages.append({"role": "assistant", "content": last_message.content})
            tool_results: List[Dict[str, Any]] = []
            for tu in tool_uses:
                session_real = dispatch.get(tu.name)
                if session_real is None:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "is_error": True,
                            "content": f"Unknown tool: {tu.name}",
                        }
                    )
                    continue
                session, real_name = session_real
                try:
                    result = await session.call_tool(real_name, tu.input or {})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "is_error": bool(getattr(result, "isError", False)),
                            "content": _tool_result_text(result) or "(no output)",
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — report the failure to the model
                    logger.warning("local MCP tool %s failed: %s", tu.name, exc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "is_error": True,
                            "content": f"Tool {tu.name} failed: {exc}",
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

        # Ran out of iterations — return whatever text the last turn produced.
        logger.warning("local MCP loop hit the %d-iteration cap", _MAX_TOOL_ITERATIONS)
        return _extract_text(last_message)


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
