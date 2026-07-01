"""
composio_tools.py
Give a claw real tools for its connected Composio apps — via Composio's REST API.

This is the reliable path (the one Grafux-mcp and the claw's own ``send_channel_reply`` already
use): actions are executed with ``POST /api/v2/actions/{ACTION}/execute`` + the ``x-api-key``
header, NOT via a Composio MCP-server URL (which returns no tools unless a toolkit is enabled
server-side).  It generalises to every Composio app.

For each enabled app-name connection (with a resolvable Composio key) the claw exposes that app's
actions as Anthropic tools:
  * a small **curated** set that is known to exist (e.g. TELEGRAM_SEND_MESSAGE), plus
  * a **best-effort auto-discovered** set from ``GET /api/v2/actions`` filtered by app.
Each tool call is executed over REST, injecting the app's connected-account id.

``httpx`` is imported lazily (like elsewhere in OpenClaw) so the devices server still boots where
it is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from . import connections
from .models import ClawConnection, ClawSpec

logger = logging.getLogger("openclaw.composio_tools")

# Guard so a tool-happy model cannot loop forever (mirrors the MCP loop cap).
_MAX_TOOL_ITERATIONS = 8

# Cap on auto-discovered actions per app (keeps the tool list + token cost reasonable).
_MAX_DISCOVERED = 25

# Curated per-action parameter schemas (the model fills these; connectedAccountId is injected
# by the claw, never exposed). Keyed by Composio action name.
_CURATED_PARAMS: Dict[str, Dict[str, Any]] = {
    "TELEGRAM_SEND_MESSAGE": {
        "chat_id": {"type": "string", "description": "Target chat id (a number as a string)."},
        "text": {"type": "string", "description": "The message text to send."},
    },
    "WHATSAPP_SEND_MESSAGE": {
        "to": {"type": "string", "description": "Recipient phone number in international format."},
        "message": {"type": "string", "description": "The message text to send."},
    },
    "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL": {
        "channel": {"type": "string", "description": "Channel id or name."},
        "text": {"type": "string", "description": "The message text to send."},
    },
}

# Caches so repeat runs of a connection-claw don't re-list actions / re-resolve accounts.
# Keyed by (composio_key, app).  Cleared on config patch via clear_cache().
_ACTIONS_CACHE: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
_ACCOUNT_CACHE: Dict[Tuple[str, str], str] = {}


def clear_cache() -> None:
    """Drop cached action lists + resolved account ids (call when a claw's config changes)."""
    _ACTIONS_CACHE.clear()
    _ACCOUNT_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_tool_name(name: str) -> str:
    """Coerce a Composio action name into Anthropic's ``^[a-zA-Z0-9_-]{1,128}$`` constraint."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (name or ""))
    return (cleaned.strip("_") or "action")[:128]


def _strip_account(schema: Any) -> Dict[str, Any]:
    """Drop connectedAccountId from a discovered action schema (the claw injects it)."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    props = schema.get("properties")
    if isinstance(props, dict):
        props = {k: v for k, v in props.items() if k.lower() != "connectedaccountid"}
        schema = dict(schema, properties=props)
        req = schema.get("required")
        if isinstance(req, list):
            schema["required"] = [r for r in req if r.lower() != "connectedaccountid"]
    return schema


def _curated_actions(app: str) -> List[Dict[str, Any]]:
    """The reliable, always-present action(s) for ``app`` (from connections._SEND_ACTIONS)."""
    action = connections._SEND_ACTIONS.get(app.lower())
    if not action:
        return []
    params = _CURATED_PARAMS.get(action, {})
    return [
        {
            "name": action,
            "description": f"Send a message via {app} (Composio).",
            "input_schema": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        }
    ]


async def _list_actions_for_app(key: str, app: str) -> List[Dict[str, Any]]:
    """
    Best-effort: list an app's Composio actions via ``GET /api/v2/actions`` filtered by app.

    Composio's filter param name has varied, so we try ``appNames`` then ``apps``; we also filter
    client-side by app so a param that is silently ignored can't flood the tool list with unrelated
    actions.  Returns ``[]`` on any error (the curated set still applies).
    """
    import httpx

    prefix = app.upper() + "_"
    for params in ({"appNames": app, "limit": 100}, {"apps": app, "limit": 100}):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{connections._COMPOSIO_BACKEND}/api/v2/actions",
                    headers={"x-api-key": key},
                    params=params,
                )
            if resp.status_code >= 400:
                continue
            items = (resp.json() or {}).get("items", [])
        except Exception as exc:  # noqa: BLE001 — discovery is best-effort
            logger.warning("composio: list actions for %s failed: %s", app, exc)
            continue
        out: List[Dict[str, Any]] = []
        for it in items:
            name = str(it.get("name") or it.get("enum") or "").strip()
            if not name:
                continue
            # Guard against an ignored filter param returning every app's actions.
            item_app = str(it.get("appName") or it.get("appKey") or "").lower()
            if item_app and item_app != app.lower() and not name.upper().startswith(prefix):
                continue
            out.append(
                {
                    "name": name,
                    "description": str(it.get("description", "")),
                    "input_schema": _strip_account(it.get("parameters") or it.get("input_schema")),
                }
            )
            if len(out) >= _MAX_DISCOVERED:
                break
        if out:
            return out
    return []


async def _actions_for_app(key: str, app: str) -> List[Dict[str, Any]]:
    """Curated + auto-discovered actions for ``app`` (curated first, de-duped by name; cached)."""
    ck = (key, app.lower())
    if ck in _ACTIONS_CACHE:
        return _ACTIONS_CACHE[ck]
    curated = _curated_actions(app)
    seen = {a["name"] for a in curated}
    discovered = [d for d in await _list_actions_for_app(key, app) if d["name"] not in seen]
    merged = curated + discovered
    _ACTIONS_CACHE[ck] = merged
    return merged


async def _account_for_app(key: str, app: str, conn: Optional[ClawConnection]) -> str:
    """
    The Composio connected-account id to scope actions to.

    Prefer the connection's own ``connection_id``; else look one up via list_connections matching
    the app (preferring an ACTIVE account).  Cached per (key, app).  Empty string when none — the
    action then executes without a connectedAccountId (Composio errors are surfaced to the model).
    """
    if conn is not None:
        cid = connections._clean_port(conn.connection_id)
        if cid:
            return cid
    ck = (key, app.lower())
    if ck in _ACCOUNT_CACHE:
        return _ACCOUNT_CACHE[ck]
    acct = ""
    try:
        for c in await connections.list_connections(key):
            if c.get("app", "").lower() != app.lower():
                continue
            acct = c.get("connection_id", "") or acct
            if str(c.get("status", "")).upper() == "ACTIVE":
                acct = c.get("connection_id", "")
                break
    except Exception as exc:  # noqa: BLE001 — account lookup is best-effort
        logger.warning("composio: account lookup for %s failed: %s", app, exc)
    _ACCOUNT_CACHE[ck] = acct
    return acct


async def _execute(key: str, action: str, args: Dict[str, Any], account_id: str) -> str:
    """
    Execute a Composio action over REST and return its output as text.

    Mirrors ``connections.send_channel_reply`` / Grafux-mcp's REST path:
    ``POST /api/v2/actions/{ACTION}/execute`` with ``x-api-key`` and
    ``{"input": {…args…, "connectedAccountId": <id>}}``.

    Raises RuntimeError with a clear message on any failure (no account, HTTP error, Composio
    error) so the run surfaces the real cause on the block's ``errors`` port — not just to the model.
    """
    import httpx

    payload = dict(args or {})
    if account_id and "connectedAccountId" not in payload:
        payload["connectedAccountId"] = account_id
    logger.info("composio: executing %s (account=%s) args=%s",
                action, account_id or "(none)", list(payload.keys()))
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{connections._COMPOSIO_BACKEND}/api/v2/actions/{action}/execute",
                headers={"x-api-key": key, "Content-Type": "application/json"},
                json={"input": payload},
            )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Composio '{action}' could not be reached: {exc}") from exc
    if resp.status_code >= 400:
        hint = "" if account_id else (
            f" — no connected '{action.split('_')[0].lower()}' account was found for this Composio "
            "key. Connect the app in your Composio account, then try again."
        )
        raise RuntimeError(
            f"Composio '{action}' failed (HTTP {resp.status_code}): {resp.text[:300]}{hint}"
        )
    data = resp.json() if resp.content else {}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Composio '{action}' returned an error: {data.get('error')}")
    if isinstance(data, dict):
        output = data.get("response") or data.get("output") or data.get("data") or data
    else:
        output = data
    logger.info("composio: %s succeeded", action)
    return str(output)


def _make_executor(key: str, action: str, account_id: str) -> Callable[[Dict[str, Any]], Awaitable[str]]:
    async def _run(inp: Dict[str, Any]) -> str:
        return await _execute(key, action, inp, account_id)
    return _run


# ---------------------------------------------------------------------------
# Public: which connections are handled here, tool building, system-prompt line
# ---------------------------------------------------------------------------

def _rest_connections(spec: ClawSpec) -> List[ClawConnection]:
    """
    Enabled app-name connections this REST provider handles.

    Excludes connections with an explicit ``mcp_url``/``headers`` (those stay on the MCP path) — so
    advanced users who paste a real MCP server config are unaffected.
    """
    out: List[ClawConnection] = []
    for c in connections.parse_connections(spec):
        if not c.enabled:
            continue
        if not connections._clean_port(c.app):
            continue
        if connections._is_local_loop(c) or connections._clean_port(c.mcp_url):
            continue
        out.append(c)
    return out


def has_rest_tools(spec: ClawSpec) -> bool:
    """Cheap (no-network) check: does this claw have Composio REST tools? (for the stream gate)."""
    if not connections.resolve_composio_key(spec):
        return False
    return bool(_rest_connections(spec))


def unloaded_reason(spec: ClawSpec, tool_defs: List[Dict[str, Any]]) -> str:
    """
    Explain why a claw with apps in its ``connections`` port ended up with NO tools.

    Returns "" when tools loaded, or there are no app connections to load.  Otherwise a clear,
    user-facing reason so the block's errors port flags the misconfiguration instead of the claw
    silently behaving like a plain chatbot ("I can't send messages").
    """
    if tool_defs:
        return ""
    apps = list(dict.fromkeys(connections._clean_port(c.app) for c in _rest_connections(spec)))
    if not apps:
        return ""
    app_list = ", ".join(apps)
    if not connections.resolve_composio_key(spec):
        return (
            f"Apps [{app_list}] are listed in the 'connections' port but NO Composio key was found. "
            "Add it to the 'api_keys' port as {\"composio\": \"ck_…\"} (a bare ck_… key also works)."
        )
    return (
        f"Apps [{app_list}] are listed and a Composio key is set, but no tools loaded — most likely "
        f"no connected account exists in Composio for [{app_list}]. Connect the app in your Composio "
        "account (for Telegram: add your bot token to the Telegram integration in the Composio "
        "dashboard), then run again."
    )


async def build(spec: ClawSpec) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]]]:
    """
    Build ``(tool_defs, dispatch)`` for the claw's connected Composio apps.

    Returns ``([], {})`` when there is no Composio key or no app-name connection — so a claw with
    no Composio setup is byte-identical to before.
    """
    key = connections.resolve_composio_key(spec)
    if not key:
        return [], {}
    tool_defs: List[Dict[str, Any]] = []
    dispatch: Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]] = {}
    seen: set = set()
    for conn in _rest_connections(spec):
        app = connections._clean_port(conn.app)
        actions = await _actions_for_app(key, app)
        if not actions:
            continue
        account_id = await _account_for_app(key, app, conn)
        for a in actions:
            name = _sanitize_tool_name(a["name"])
            if name in seen:
                continue
            seen.add(name)
            tool_defs.append(
                {
                    "name": name,
                    "description": a.get("description") or f"{app} action ({a['name']}).",
                    "input_schema": a.get("input_schema") or {"type": "object", "properties": {}},
                }
            )
            dispatch[name] = _make_executor(key, a["name"], account_id)
    return tool_defs, dispatch


def describe(spec: ClawSpec) -> str:
    """A system-prompt line naming the apps the claw has Composio tools for (cheap, no network)."""
    if not connections.resolve_composio_key(spec):
        return ""
    apps = list(dict.fromkeys(connections._clean_port(c.app) for c in _rest_connections(spec)))
    if not apps:
        return ""
    return (
        "You have Composio tools for these apps: " + ", ".join(apps) + ". Use them to take real "
        "actions (send messages, post, read) when asked — actually call the tools, do not say you "
        "cannot."
    )


# ---------------------------------------------------------------------------
# Generic Anthropic tool-use loop (python-executed tools)
# ---------------------------------------------------------------------------

def _extract_text(message: Any) -> str:
    return "".join(
        b.text for b in getattr(message, "content", []) if getattr(b, "type", None) == "text"
    )


async def run_tool_loop(
    client: Any,
    base_kwargs: Dict[str, Any],
    tool_defs: List[Dict[str, Any]],
    dispatch: Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]],
    connector_servers: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[str]]:
    """
    Run an Anthropic tool-use loop over python-executed tools.

    Returns ``(final_text, tool_errors)``.  ``tool_errors`` is a list of human-readable notes for
    any tool call that failed (e.g. a Composio 401/404 or "no connected account") so the caller can
    surface the real cause on the block's ``errors`` port — the model also sees each error as a
    tool_result and can explain it.  ``connector_servers`` ride along on the same calls.
    """
    connector_servers = connector_servers or []
    messages = list(base_kwargs.get("messages", []))
    loop_kwargs = {k: v for k, v in base_kwargs.items() if k != "messages"}
    use_connector = bool(connector_servers)
    tool_errors: List[str] = []

    last_message: Any = None
    for _ in range(_MAX_TOOL_ITERATIONS):
        if use_connector:
            last_message = await client.beta.messages.create(
                betas=[connections.MCP_CONNECTOR_BETA],
                mcp_servers=connector_servers,
                tools=connections.build_mcp_toolsets(connector_servers) + tool_defs,
                messages=messages,
                **loop_kwargs,
            )
        else:
            last_message = await client.messages.create(
                tools=tool_defs, messages=messages, **loop_kwargs
            )

        tool_uses = [b for b in last_message.content if getattr(b, "type", None) == "tool_use"]
        if last_message.stop_reason != "tool_use" or not tool_uses:
            return _extract_text(last_message), tool_errors

        messages.append({"role": "assistant", "content": last_message.content})
        results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            fn = dispatch.get(tu.name)
            if fn is None:
                tool_errors.append(f"Unknown tool: {tu.name}")
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "is_error": True, "content": f"Unknown tool: {tu.name}"})
                continue
            try:
                out = await fn(tu.input or {})
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out or "(no output)"})
            except Exception as exc:  # noqa: BLE001 — report the failure to the model + the port
                logger.warning("composio tool %s failed: %s", tu.name, exc)
                tool_errors.append(str(exc))
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "is_error": True, "content": f"{tu.name} failed: {exc}"})
        messages.append({"role": "user", "content": results})

    logger.warning("composio tool loop hit the %d-iteration cap", _MAX_TOOL_ITERATIONS)
    return _extract_text(last_message), tool_errors
