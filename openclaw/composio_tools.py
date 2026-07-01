"""
composio_tools.py
Give a claw real tools for its connected Composio apps — via Composio's REST API.

This uses Composio's **v3** REST API with the ``x-api-key`` header — tools are listed with
``GET /api/v3/tools?toolkit_slug=<slug>`` and executed with
``POST /api/v3/tools/execute/{tool_slug}`` — NOT a Composio MCP-server URL (which returns no tools
unless a toolkit is enabled server-side).  v3 tools are scoped to a ``user_id``, which the claw
auto-discovers from the toolkit's connected account.  It generalises to every Composio app.

For each enabled app-name connection (with a resolvable Composio key) the claw exposes that app's
actions as Anthropic tools:
  * a small **curated** set that is known to exist (e.g. TELEGRAM_SEND_MESSAGE), plus
  * a **best-effort auto-discovered** set from ``GET /api/v3/tools?toolkit_slug=<app>``.
Each tool call is executed over REST, injecting the app's connected-account id.

``httpx`` is imported lazily (like elsewhere in OpenClaw) so the devices server still boots where
it is absent.
"""

from __future__ import annotations

import difflib
import logging
import re
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
# (key, app) -> (user_id, connected_account_id).  v3 tools are scoped to a user_id.
_ACCOUNT_CACHE: Dict[Tuple[str, str], Tuple[str, str]] = {}
# The set of all Composio toolkit slugs (account-independent), fetched once per key.
_TOOLKIT_SLUGS_CACHE: Dict[str, set] = {}
# (key, typed_app) -> canonical toolkit slug (fuzzy-resolved).
_SLUG_RESOLVE_CACHE: Dict[Tuple[str, str], str] = {}


def clear_cache() -> None:
    """Drop cached action lists + resolved account ids (call when a claw's config changes)."""
    _ACTIONS_CACHE.clear()
    _ACCOUNT_CACHE.clear()
    _SLUG_RESOLVE_CACHE.clear()
    _TOOLKIT_SLUGS_CACHE.clear()


# ---------------------------------------------------------------------------
# Toolkit-slug resolution — forgive typos (googlesheet → googlesheets)
# ---------------------------------------------------------------------------

async def _toolkit_slugs(key: str) -> set:
    """All Composio toolkit slugs (lowercased), fetched once per key. ``set()`` on any error."""
    if key in _TOOLKIT_SLUGS_CACHE:
        return _TOOLKIT_SLUGS_CACHE[key]
    import httpx

    slugs: set = set()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{connections._COMPOSIO_BACKEND}/api/v3/toolkits",
                headers={"x-api-key": key},
                params={"limit": 500},
            )
        if resp.status_code < 400:
            for it in (resp.json() or {}).get("items", []) or []:
                s = str(it.get("slug") or "").strip().lower()
                if s:
                    slugs.add(s)
    except Exception as exc:  # noqa: BLE001 — resolution is best-effort
        logger.warning("composio: toolkits list failed: %s", exc)
    _TOOLKIT_SLUGS_CACHE[key] = slugs
    return slugs


async def list_toolkit_slugs(key: str) -> list:
    """Sorted list of every Composio toolkit slug (for grounding claw connections). ``[]`` on no key/error."""
    if not key:
        return []
    return sorted(await _toolkit_slugs(key))


async def resolve_toolkit_slug(key: str, app: str) -> str:
    """
    Resolve a user-typed app name to the canonical Composio toolkit slug.

    Exact (case-insensitive) match wins; otherwise a close match via difflib (cutoff 0.8) fixes
    typos like ``googlesheet`` → ``googlesheets`` / ``googlecalender`` → ``googlecalendar``.  Returns
    the input unchanged when toolkits can't be listed or nothing matches (so behavior degrades to
    today's strict slug).
    """
    app = (app or "").strip()
    ck = (key, app.lower())
    if ck in _SLUG_RESOLVE_CACHE:
        return _SLUG_RESOLVE_CACHE[ck]
    resolved = app
    slugs = await _toolkit_slugs(key)
    if slugs:
        low = app.lower()
        if low in slugs:
            resolved = low
        else:
            match = difflib.get_close_matches(low, sorted(slugs), n=1, cutoff=0.8)
            if match:
                resolved = match[0]
                logger.info("composio: resolved toolkit slug '%s' → '%s'", app, resolved)
    _SLUG_RESOLVE_CACHE[ck] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_tool_name(name: str) -> str:
    """Coerce a Composio action name into Anthropic's ``^[a-zA-Z0-9_-]{1,64}$`` constraint."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (name or ""))
    return (cleaned.strip("_") or "action")[:64]


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


# Anthropic requires every JSON-schema property KEY to match this pattern; Composio schemas
# sometimes carry keys that violate it, which 400s the whole Messages call. We drop such keys.
_PROP_KEY_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


def _sanitize_schema(node: Any) -> Any:
    """Recursively drop object-property keys Anthropic rejects (and prune them from ``required``)."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k == "properties" and isinstance(v, dict):
                out[k] = {
                    str(pk): _sanitize_schema(pv)
                    for pk, pv in v.items()
                    if _PROP_KEY_RE.match(str(pk))
                }
            else:
                out[k] = _sanitize_schema(v)
        req, props = out.get("required"), out.get("properties")
        if isinstance(req, list) and isinstance(props, dict):
            out["required"] = [r for r in req if r in props]
        return out
    if isinstance(node, list):
        return [_sanitize_schema(i) for i in node]
    return node


def _coerce_schema(schema: Any) -> Dict[str, Any]:
    """Ensure an action's input schema is a valid Anthropic object schema (keys sanitized)."""
    s = _sanitize_schema(_strip_account(schema))
    if not isinstance(s, dict):
        return {"type": "object", "properties": {}}
    if not s.get("type"):
        s = dict(s, type="object")
    if "properties" not in s:
        s = dict(s, properties={})
    return s


async def _list_actions_for_app(key: str, app: str) -> List[Dict[str, Any]]:
    """
    List an app's Composio tools via the v3 API: ``GET /api/v3/tools?toolkit_slug=<app>``.

    Listing does NOT require a connected account (it returns the toolkit's available tools), so this
    works as long as the app name is a valid Composio toolkit slug (e.g. ``googlesheets``,
    ``weathermap``).  Returns ``[]`` on any error (the curated set still applies).
    """
    import httpx

    app = await resolve_toolkit_slug(key, app)  # forgive typos (googlesheet → googlesheets)

    # Composio's filter param name varies (toolkit_slug vs toolkit_slugs) — try both.
    items: List[Dict[str, Any]] = []
    for pkey in ("toolkit_slug", "toolkit_slugs"):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{connections._COMPOSIO_BACKEND}/api/v3/tools",
                    headers={"x-api-key": key},
                    params={pkey: app, "limit": _MAX_DISCOVERED},
                )
            if resp.status_code >= 400:
                logger.warning("composio: v3 tools list for %s (%s) → HTTP %s: %s",
                               app, pkey, resp.status_code, resp.text[:200])
                continue
            items = (resp.json() or {}).get("items", []) or []
        except Exception as exc:  # noqa: BLE001 — discovery is best-effort
            logger.warning("composio: v3 tools list for %s (%s) failed: %s", app, pkey, exc)
            continue
        if items:
            break
    out: List[Dict[str, Any]] = []
    for it in items:
        if it.get("is_deprecated"):
            continue
        # Keep only tools that actually belong to the requested toolkit — a wrong/misspelled slug
        # can make Composio return unrelated tools (e.g. 1Password for "googlecalender").
        tk = str((it.get("toolkit") or {}).get("slug") or "").lower()
        if tk and tk != app.lower():
            continue
        slug = str(it.get("slug") or "").strip()
        if not slug:
            continue
        out.append(
            {
                "name": slug,
                "description": str(it.get("description", "")),
                "input_schema": _coerce_schema(it.get("input_parameters")),
                "no_auth": bool(it.get("no_auth", False)),
            }
        )
        if len(out) >= _MAX_DISCOVERED:
            break
    return out


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


async def _account_for_app(
    key: str, app: str, conn: Optional[ClawConnection], identity: str = "default"
) -> Tuple[str, str]:
    """
    Resolve ``(user_id, connected_account_id)`` for an app via v3 connected accounts, **scoped to
    the given ``identity``** (the claw's owner / Composio user_id) so one Grafux user cannot see
    another's accounts under the shared Composio key.

    Prefers an ACTIVE account.  Cached per ``(key, identity, app)``.  Returns ``(identity, "")`` when
    none for this identity — execution then reports a clear "connect the app" error.
    """
    app = await resolve_toolkit_slug(key, app)  # forgive typos (googlesheet → googlesheets)
    ck = (key, identity, app.lower())
    if ck in _ACCOUNT_CACHE:
        return _ACCOUNT_CACHE[ck]

    import httpx

    account_id = ""
    for params in ({"toolkit_slugs": app, "user_ids": identity, "statuses": "ACTIVE", "limit": 10},
                   {"toolkit_slugs": app, "user_ids": identity, "limit": 10}):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{connections._COMPOSIO_BACKEND}/api/v3/connected_accounts",
                    headers={"x-api-key": key},
                    params=params,
                )
            if resp.status_code >= 400:
                continue
            items = (resp.json() or {}).get("items", [])
        except Exception as exc:  # noqa: BLE001 — account lookup is best-effort
            logger.warning("composio: v3 connected accounts for %s failed: %s", app, exc)
            continue
        for it in items:
            tk = str((it.get("toolkit") or {}).get("slug") or "").lower()
            uid, aid = str(it.get("user_id") or ""), str(it.get("id") or "")
            # Belt-and-braces: only accept an account that matches BOTH this toolkit AND this identity
            # (in case the API ignores a filter param).
            if (tk and tk != app.lower()) or (uid and uid != identity):
                continue
            if str(it.get("status", "")).upper() == "ACTIVE":
                account_id = aid or account_id
                break
            if not account_id:
                account_id = aid
        if account_id:
            break

    result = (identity, account_id)
    _ACCOUNT_CACHE[ck] = result
    return result


def _owner(spec: ClawSpec) -> str:
    """The claw's Composio user_id scope (its Grafux owner), or ``"default"`` (single-tenant)."""
    return connections._clean_port(spec.owner) or "default"


async def list_connected_accounts(key: str, user_id: str = "") -> List[Dict[str, Any]]:
    """
    List Composio connected accounts for ``key`` → ``[{id, toolkit, user_id, status}]``.

    When ``user_id`` is given, only that user's (owner's) accounts are returned — so the Manage
    Connections page/probe show a Grafux user only their own connections.  Best-effort → ``[]``.
    """
    import httpx

    params: Dict[str, Any] = {"limit": 100}
    if user_id:
        params["user_ids"] = user_id
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{connections._COMPOSIO_BACKEND}/api/v3/connected_accounts",
                headers={"x-api-key": key},
                params=params,
            )
        if resp.status_code >= 400:
            return []
        items = (resp.json() or {}).get("items", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("composio: list connected accounts failed: %s", exc)
        return []
    out = [
        {
            "id": str(a.get("id") or ""),
            "toolkit": str((a.get("toolkit") or {}).get("slug") or "").lower(),
            "user_id": str(a.get("user_id") or ""),
            "status": str(a.get("status") or ""),
        }
        for a in items
    ]
    # Belt-and-braces client-side filter in case the API ignores user_ids.
    if user_id:
        out = [a for a in out if a["user_id"] == user_id]
    return out


async def delete_connected_account(key: str, account_id: str) -> bool:
    """Delete a Composio connected account (``DELETE /api/v3/connected_accounts/{id}``)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.delete(
                f"{connections._COMPOSIO_BACKEND}/api/v3/connected_accounts/{account_id}",
                headers={"x-api-key": key},
            )
        return 200 <= resp.status_code < 300
    except Exception as exc:  # noqa: BLE001
        logger.warning("composio: delete connected account %s failed: %s", account_id, exc)
        return False


async def _execute(
    key: str,
    tool_slug: str,
    args: Dict[str, Any],
    user_id: str,
    account_id: str,
    no_auth: bool = False,
) -> str:
    """
    Execute a Composio tool via the v3 API and return its output as text.

    ``POST /api/v3/tools/execute/{tool_slug}`` with ``x-api-key`` and body
    ``{"user_id", "connected_account_id"?, "arguments"}``.  v3 tools are scoped to a user_id;
    the connected account is passed at the top level (not inside arguments).

    Raises RuntimeError with a clear message on any failure so the run surfaces the real cause on
    the block's ``errors`` port — not just to the model.
    """
    import httpx

    body: Dict[str, Any] = {"user_id": user_id or "default", "arguments": dict(args or {})}
    if account_id:
        body["connected_account_id"] = account_id
    logger.info("composio: v3 execute %s (user=%s account=%s)",
                tool_slug, body["user_id"], account_id or "(none)")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{connections._COMPOSIO_BACKEND}/api/v3/tools/execute/{tool_slug}",
                headers={"x-api-key": key, "Content-Type": "application/json"},
                json=body,
            )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Composio '{tool_slug}' could not be reached: {exc}") from exc
    if resp.status_code >= 400:
        hint = "" if (account_id or no_auth) else (
            " — no connected account was found for this toolkit. Connect the app in your Composio "
            "account, then try again."
        )
        raise RuntimeError(
            f"Composio '{tool_slug}' failed (HTTP {resp.status_code}): {resp.text[:300]}{hint}"
        )
    data = resp.json() if resp.content else {}
    if isinstance(data, dict):
        # v3 wraps results as {"data": …, "successful": bool, "error": …}.
        if data.get("successful") is False or data.get("error"):
            raise RuntimeError(
                f"Composio '{tool_slug}' error: "
                f"{data.get('error') or data.get('message') or 'the action was unsuccessful'}"
            )
        output = data.get("data")
        if output is None:
            output = data.get("response") or data.get("output") or data
    else:
        output = data
    logger.info("composio: %s succeeded", tool_slug)
    return str(output)


def _make_executor(
    key: str, tool_slug: str, user_id: str, account_id: str, no_auth: bool
) -> Callable[[Dict[str, Any]], Awaitable[str]]:
    async def _run(inp: Dict[str, Any]) -> str:
        return await _execute(key, tool_slug, inp, user_id, account_id, no_auth)
    return _run


# ---------------------------------------------------------------------------
# Connect flow — create an Auth Config + a connect (auth-link) session (v3)
# ---------------------------------------------------------------------------

async def resolve_or_create_auth_config(key: str, app: str) -> str:
    """
    Return an auth-config id (``ac_…``) for ``app``, creating a Composio-managed one if none exists.

    ``GET /api/v3/auth_configs?toolkit_slug=<app>`` → reuse the first item; else
    ``POST /api/v3/auth_configs`` with ``{"toolkit":{"slug"},"auth_config":{"type":
    "use_composio_managed_auth"}}``.  Raises RuntimeError with the raw body on failure.
    """
    import httpx

    app = await resolve_toolkit_slug(key, app)  # create the auth config under the canonical slug

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{connections._COMPOSIO_BACKEND}/api/v3/auth_configs",
            headers={"x-api-key": key},
            params={"toolkit_slug": app, "limit": 5},
        )
        if resp.status_code < 400:
            items = (resp.json() or {}).get("items", []) or []
            if items and items[0].get("id"):
                return str(items[0]["id"])

        create = await client.post(
            f"{connections._COMPOSIO_BACKEND}/api/v3/auth_configs",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"toolkit": {"slug": app}, "auth_config": {"type": "use_composio_managed_auth"}},
        )
    if create.status_code >= 400:
        raise RuntimeError(f"Composio auth_configs create failed (HTTP {create.status_code}): "
                           f"{create.text[:400]}")
    data = create.json() if create.content else {}
    acid = (data.get("id")
            or (data.get("auth_config") or {}).get("id")
            or (data.get("data") or {}).get("id"))
    if not acid:
        raise RuntimeError(f"Composio auth_configs create returned no id: {str(data)[:400]}")
    return str(acid)


def _redirect_from(data: Any) -> str:
    """Pull a redirect URL out of the several shapes the connect endpoints return."""
    if not isinstance(data, dict):
        return ""
    if data.get("redirect_url"):
        return str(data["redirect_url"])
    if data.get("redirect_uri"):
        return str(data["redirect_uri"])
    cd = data.get("connection_data") or data.get("connectionData") or {}
    val = cd.get("val") if isinstance(cd, dict) else {}
    if isinstance(val, dict) and val.get("redirect_url"):
        return str(val["redirect_url"])
    return ""


async def initiate_connect(
    key: str, auth_config_id: str, user_id: str, callback_url: str, api_key: str = ""
) -> Dict[str, Any]:
    """
    Create a connect (auth-link) session and return ``{redirect_url, connection_id, status, raw}``.

    Tries the current ``POST /api/v3/connected_accounts/link`` first, then falls back to the older
    ``POST /api/v3/connected_accounts`` initiate shape.  For API-key toolkits, ``api_key`` is passed
    in the connection ``config`` so the account connects without an OAuth redirect.
    """
    import httpx

    link_body: Dict[str, Any] = {"auth_config_id": auth_config_id, "user_id": user_id}
    if callback_url:
        link_body["callback_url"] = callback_url
    if api_key:
        link_body["config"] = {"api_key": api_key}

    initiate_body: Dict[str, Any] = {
        "auth_config": {"id": auth_config_id},
        "connection": {"user_id": user_id, **({"callback_url": callback_url} if callback_url else {}),
                       **({"config": {"api_key": api_key}} if api_key else {})},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        attempts = (
            ("/api/v3/connected_accounts/link", link_body),
            ("/api/v3/connected_accounts", initiate_body),
        )
        last_text, last_code = "", 0
        for path, body in attempts:
            resp = await client.post(
                f"{connections._COMPOSIO_BACKEND}{path}",
                headers={"x-api-key": key, "Content-Type": "application/json"},
                json=body,
            )
            if resp.status_code < 400:
                data = resp.json() if resp.content else {}
                return {
                    "redirect_url": _redirect_from(data),
                    "connection_id": str(data.get("connected_account_id") or data.get("id") or ""),
                    "status": str(data.get("status")
                                  or ((data.get("connection_data") or {}).get("val") or {}).get("status")
                                  or ""),
                    "raw": data,
                }
            last_text, last_code = resp.text[:400], resp.status_code
    raise RuntimeError(f"Composio connect failed (HTTP {last_code}): {last_text}")


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
        f"Apps [{app_list}] are listed and a Composio key is set, but Composio returned no tools for "
        f"them. Check that each name is a valid Composio toolkit slug (e.g. 'googlesheets', "
        f"'weathermap', 'gmail') and that your Composio key is valid. Connecting the app in Composio "
        "is only needed to *use* a tool, not to list it — so if this persists the slug or key is wrong."
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
    identity = _owner(spec)  # scope every account lookup + execution to this claw's Grafux owner
    tool_defs: List[Dict[str, Any]] = []
    dispatch: Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]] = {}
    seen: set = set()
    for conn in _rest_connections(spec):
        app = connections._clean_port(conn.app)
        actions = await _actions_for_app(key, app)
        if not actions:
            continue
        user_id, account_id = await _account_for_app(key, app, conn, identity)
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
            dispatch[name] = _make_executor(
                key, a["name"], user_id, account_id, bool(a.get("no_auth", False))
            )
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
