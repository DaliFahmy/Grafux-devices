"""
native_tools.py
Built-in tools a claw can use directly from its own credentials — no Composio / MCP.

The first (and most requested) is **Telegram**: paste a BotFather bot token into the
claw's ``credentials`` (or ``api_keys``) port and the claw gains real tools to send
messages and discover chat ids, calling the Telegram Bot API directly. This is the
frictionless path for "I have a bot token, let the claw message Telegram" — Composio's
connector path (connections.py) stays available for everything else.

``httpx`` is imported lazily (like elsewhere in OpenClaw) so the devices server still
boots where it is absent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from . import connections
from .models import ClawSpec

logger = logging.getLogger("openclaw.native_tools")

# A BotFather token looks like "<bot_id>:<35-char secret>", e.g. 123456789:AAE…-xyz.
_TELEGRAM_TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")

# JSON keys under which a Telegram bot token may live in api_keys / credentials.
_TELEGRAM_KEYS = (
    "telegram",
    "telegram_bot_token",
    "telegram_token",
    "TELEGRAM_BOT_TOKEN",
    "bot_token",
)

# Guard so a tool-happy model cannot loop forever (mirrors the MCP loop cap).
_MAX_TOOL_ITERATIONS = 8


def _clean(text: Optional[str]) -> str:
    return connections._clean_port(text)


def _maybe_json(text: str) -> Optional[Any]:
    return connections._maybe_json(text)


def resolve_telegram_token(spec: ClawSpec) -> str:
    """
    Find a Telegram bot token in the claw's ports.

    Order: api_keys / credentials as JSON ({"telegram": "…"} etc.), then a *bare* token
    pasted straight into either port (the common case — the user pastes the BotFather
    token as-is), then the TELEGRAM_BOT_TOKEN env var. Returns "" when none is found.
    """
    import os

    for raw in (spec.api_keys, spec.credentials):
        raw = _clean(raw)
        if not raw:
            continue
        parsed = _maybe_json(raw)
        if isinstance(parsed, dict):
            for key in _TELEGRAM_KEYS:
                val = parsed.get(key)
                if val and _TELEGRAM_TOKEN_RE.match(str(val).strip()):
                    return str(val).strip()
        elif _TELEGRAM_TOKEN_RE.match(raw):
            return raw
    env = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    return env if _TELEGRAM_TOKEN_RE.match(env) else ""


# ---------------------------------------------------------------------------
# Telegram Bot API executors
# ---------------------------------------------------------------------------

_TELEGRAM_API = "https://api.telegram.org"


async def _telegram_send_message(token: str, chat_id: str, text: str) -> str:
    import httpx

    if not chat_id:
        return ("Missing chat_id. Call telegram_list_chats to find one — a bot can only "
                "message users/chats that have messaged it first.")
    if not text:
        return "Missing text to send."
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_TELEGRAM_API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        data = resp.json() if resp.content else {}
    if not data.get("ok"):
        return f"Telegram API error: {data.get('description', resp.text[:200] or 'unknown error')}"
    return f"Message sent to chat {chat_id}."


async def _telegram_list_chats(token: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{_TELEGRAM_API}/bot{token}/getUpdates")
        data = resp.json() if resp.content else {}
    if not data.get("ok"):
        return f"Telegram API error: {data.get('description', 'unknown error')}"
    chats: Dict[Any, Dict[str, Any]] = {}
    for update in data.get("result", []):
        msg = (update.get("message") or update.get("edited_message")
               or update.get("channel_post") or {})
        chat = msg.get("chat", {})
        cid = chat.get("id")
        if cid is None:
            continue
        name = (chat.get("title")
                or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")])).strip()
                or chat.get("username", ""))
        chats[cid] = {"chat_id": str(cid), "name": name, "last_message": msg.get("text", "")}
    if not chats:
        return ("No recent chats. Ask the user to open Telegram, find the bot, and send it any "
                "message first — a bot cannot start a conversation, it can only reply to chats "
                "that have messaged it.")
    return json.dumps(list(chats.values()))


# ---------------------------------------------------------------------------
# Tool definitions + dispatch
# ---------------------------------------------------------------------------

_TELEGRAM_TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "name": "telegram_send_message",
        "description": (
            "Send a text message through the user's own Telegram bot to a chat. Requires the "
            "numeric chat_id; if you don't have it, call telegram_list_chats first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "description": "Target chat id (a number as a string)."},
                "text": {"type": "string", "description": "The message text to send."},
            },
            "required": ["chat_id", "text"],
        },
    },
    {
        "name": "telegram_list_chats",
        "description": (
            "List recent chats that have messaged the bot, to discover a chat_id. A Telegram bot "
            "can only message users/chats that have first sent it a message."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

# A dispatch value takes the tool's input dict and returns the tool_result text.
Dispatch = Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]]


def build(spec: ClawSpec) -> Tuple[List[Dict[str, Any]], Dispatch]:
    """
    Return ``(tool_defs, dispatch)`` for the native tools this claw's credentials unlock.

    Currently: Telegram, when a bot token is resolvable.  Returns ``([], {})`` otherwise,
    so a claw with no native credentials is byte-identical to before.
    """
    token = resolve_telegram_token(spec)
    if not token:
        return [], {}
    dispatch: Dispatch = {
        "telegram_send_message":
            lambda inp: _telegram_send_message(token, str(inp.get("chat_id", "")), str(inp.get("text", ""))),
        "telegram_list_chats":
            lambda inp: _telegram_list_chats(token),
    }
    return list(_TELEGRAM_TOOL_DEFS), dispatch


def describe(spec: ClawSpec) -> str:
    """A system-prompt line telling the claw about its native tools (no secrets)."""
    if resolve_telegram_token(spec):
        return (
            "You have a Telegram bot connected. Use the telegram_send_message tool to actually "
            "send messages, and telegram_list_chats to find a chat_id when you don't have one "
            "(a bot can only message users who have messaged it first). Do send real messages "
            "when asked — do not just say you cannot."
        )
    return ""


def _extract_text(message: Any) -> str:
    return "".join(
        b.text for b in getattr(message, "content", []) if getattr(b, "type", None) == "text"
    )


async def run_native_tool_loop(
    client: Any,
    base_kwargs: Dict[str, Any],
    tool_defs: List[Dict[str, Any]],
    dispatch: Dispatch,
    connector_servers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Run an Anthropic tool-use loop over native (python-executed) tools and return the final text.

    ``base_kwargs`` is the dict claw_runtime builds for ``messages.create``.  ``connector_servers``
    (Composio bearer/self-auth MCP servers) ride along on the same calls so a claw mixing native
    tools with connector apps still works.
    """
    connector_servers = connector_servers or []
    messages = list(base_kwargs.get("messages", []))
    loop_kwargs = {k: v for k, v in base_kwargs.items() if k != "messages"}
    use_connector = bool(connector_servers)

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
            return _extract_text(last_message)

        messages.append({"role": "assistant", "content": last_message.content})
        results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            fn = dispatch.get(tu.name)
            if fn is None:
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "is_error": True, "content": f"Unknown tool: {tu.name}"})
                continue
            try:
                out = await fn(tu.input or {})
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out or "(no output)"})
            except Exception as exc:  # noqa: BLE001 — report the failure to the model
                logger.warning("native tool %s failed: %s", tu.name, exc)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "is_error": True, "content": f"{tu.name} failed: {exc}"})
        messages.append({"role": "user", "content": results})

    logger.warning("native tool loop hit the %d-iteration cap", _MAX_TOOL_ITERATIONS)
    return _extract_text(last_message)
