"""
claw_runtime.py
Turns a ClawSpec into a live Claude agent and runs it against a task.

The runtime maps the claw's block ports onto an Anthropic Messages API call:

    soul          -> system prompt (the claw's persona / standing instructions)
    skills        -> capabilities described in the system prompt
    tools_config  -> additional standing configuration in the system prompt
    agent         -> model id + generation params
    api_keys      -> Anthropic API key (+ task keys made available in the prompt)
    credentials   -> task secrets made available to the claw
    memory + task -> the user turn

Prompt caching is applied to the (large, stable) system block so repeated runs of
the same claw only pay to process the soul/skills once.

The Anthropic SDK is imported lazily so the devices server still boots (and the
hardware-device endpoints keep working) on a host where ``anthropic`` is absent.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from . import connections, native_tools
from .models import ClawSpec

logger = logging.getLogger("openclaw.runtime")

# The claw model is configurable per-deployment (OPENCLAW_DEFAULT_MODEL) so an
# operator can default every new claw to a cheaper/faster tier without a code
# change; the user can still override it per-claw via the ``agent`` port.
DEFAULT_MODEL = os.environ.get("OPENCLAW_DEFAULT_MODEL", "claude-opus-4-8")
DEFAULT_MAX_TOKENS = 4096

# Reuse one AsyncAnthropic client per API key instead of constructing one per run — each
# construction spins up an httpx connection pool, so reuse cuts per-run setup cost and lets
# connections stay warm across runs of the same claw.  Keyed by api_key (process-local).
_ANTHROPIC_CLIENTS: Dict[str, Any] = {}


def _client_for(api_key: str, ctor: Any) -> Any:
    """Return a cached AsyncAnthropic client for ``api_key`` (constructing one on first use)."""
    client = _ANTHROPIC_CLIENTS.get(api_key)
    if client is None:
        client = ctor(api_key=api_key)
        _ANTHROPIC_CLIENTS[api_key] = client
    return client

# Model catalog — pricing ($ per 1M tokens) and whether the model accepts the
# sampling params (temperature/top_p/top_k).  Sourced from the Claude API
# reference (2026-06).  CRITICAL: the Opus 4.7+/Fable family REJECTS sampling
# params with HTTP 400, so we must NOT send them for those models (see
# _model_accepts_sampling / run_claw).  Pricing drives the per-run cost badge.
#   "in"/"out" = $/1M input/output tokens; "sampling" = accepts temperature etc.
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "claude-fable-5":    {"label": "Claude Fable 5",    "in": 10.0, "out": 50.0, "sampling": False},
    "claude-opus-4-8":   {"label": "Claude Opus 4.8",   "in": 5.0,  "out": 25.0, "sampling": False},
    "claude-opus-4-7":   {"label": "Claude Opus 4.7",   "in": 5.0,  "out": 25.0, "sampling": False},
    "claude-opus-4-6":   {"label": "Claude Opus 4.6",   "in": 5.0,  "out": 25.0, "sampling": True},
    "claude-sonnet-4-6": {"label": "Claude Sonnet 4.6", "in": 3.0,  "out": 15.0, "sampling": True},
    "claude-haiku-4-5":  {"label": "Claude Haiku 4.5",  "in": 1.0,  "out": 5.0,  "sampling": True},
}


def _model_accepts_sampling(model: str) -> bool:
    """
    Whether ``model`` accepts temperature/top_p/top_k.

    The Opus 4.7/4.8 and Fable families return HTTP 400 when sent any sampling
    param, so the claw must omit them.  Unknown models default to *not* sending
    sampling params — the safe choice, since the current frontier default
    (claude-opus-4-8) rejects them and most new models follow suit.
    """
    entry = MODEL_CATALOG.get(model)
    if entry is not None:
        return bool(entry["sampling"])
    return False


def _estimate_cost_usd(model: str, usage: Any) -> float:
    """
    Estimate the USD cost of a run from ``message.usage`` and the model tier.

    Mirrors Anthropic's cache pricing: uncached input + cache *writes* at 1.25x,
    cache *reads* at 0.1x, output at the output rate.  Returns 0.0 for an unknown
    model or missing usage (the badge then simply shows token counts).
    """
    entry = MODEL_CATALOG.get(model)
    if entry is None or usage is None:
        return 0.0
    in_rate = entry["in"] / 1_000_000.0
    out_rate = entry["out"] / 1_000_000.0
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    return round(
        inp * in_rate + cache_write * in_rate * 1.25 + cache_read * in_rate * 0.1 + out * out_rate,
        6,
    )

# Sentinels the Grafux frontend writes into a port file when nothing is wired to
# it (see PortDataService::kEmptyPortValue and the "unconnected" literal in the
# block runner).  They are NOT real values — treat them as an empty port so an
# unconnected ``agent`` port never becomes the model id "unconnected" (which the
# Anthropic API rejects with 404 not_found).
_PLACEHOLDER_VALUES = {"empty", "unconnected"}


# ---------------------------------------------------------------------------
# Port parsing helpers — every port is free text that MAY contain JSON.
# ---------------------------------------------------------------------------

def _clean_port(text: Optional[str]) -> str:
    """Return the port's real value, mapping placeholder sentinels to ``""``."""
    text = (text or "").strip()
    if text.lower() in _PLACEHOLDER_VALUES:
        return ""
    return text


def _maybe_json(text: str) -> Optional[Any]:
    """Parse ``text`` as JSON, returning None when it is not valid JSON."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _resolve_api_key(spec: ClawSpec) -> Optional[str]:
    """
    Find the Anthropic API key.

    Order: the api_keys port (bare key or JSON {"anthropic": "..."}), then the
    credentials port (same shapes), then the ANTHROPIC_API_KEY env var.
    """
    for raw in (spec.api_keys, spec.credentials):
        raw = _clean_port(raw)
        if not raw:
            continue
        parsed = _maybe_json(raw)
        if isinstance(parsed, dict):
            for key in ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "api_key"):
                if parsed.get(key):
                    return str(parsed[key])
        elif raw.startswith("sk-"):
            return raw
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _resolve_model_params(spec: ClawSpec) -> Dict[str, Any]:
    """
    Resolve model id and generation params from the ``agent`` port.

    The port may be a bare model id ("claude-opus-4-8") or a JSON object such as
    {"model": "claude-opus-4-8", "max_tokens": 8192, "temperature": 0.7}.
    """
    params: Dict[str, Any] = {"model": DEFAULT_MODEL, "max_tokens": DEFAULT_MAX_TOKENS}
    agent = _clean_port(spec.agent)
    if not agent:
        return params
    parsed = _maybe_json(agent)
    if isinstance(parsed, dict):
        model = _clean_port(str(parsed.get("model", "")))
        if model:
            params["model"] = model
        if isinstance(parsed.get("max_tokens"), int):
            params["max_tokens"] = parsed["max_tokens"]
        if isinstance(parsed.get("temperature"), (int, float)):
            params["temperature"] = float(parsed["temperature"])
    else:
        params["model"] = agent
    return params


def _build_system_prompt(spec: ClawSpec, from_channel: bool = False) -> str:
    """Assemble the claw's standing instructions from soul + skills + tools_config."""
    parts = []
    soul = _clean_port(spec.soul)
    parts.append(soul if soul else "You are a helpful AI agent (a Grafux claw).")
    skills = _clean_port(spec.skills)
    if skills:
        parts.append("Your skills and capabilities:\n" + skills)
    tools_config = _clean_port(spec.tools_config)
    if tools_config:
        parts.append("Tool / environment configuration:\n" + tools_config)
    connected = connections.describe_connections(spec)
    if connected:
        # The claw has real tools for these apps via the MCP connector — tell it so.
        parts.append(
            "Connected apps you can act on with tools: " + connected + ".\n"
            "Use the available tools to take real actions (send messages, post, read) "
            "when the task requires it."
        )
    native_desc = native_tools.describe(spec)
    if native_desc:
        # Built-in tools from the claw's own credentials (e.g. a Telegram bot token).
        parts.append(native_desc)
    if from_channel:
        parts.append(
            "You are replying inside a live chat. Your text response is delivered to the "
            "user automatically — write it as the reply itself. Use tools only for "
            "additional side effects, not to send your own reply."
        )
    if _clean_port(spec.credentials):
        # Credentials are provided so the claw is *aware* it has access; we do not
        # dump raw secret blobs into the prompt beyond what the user supplied.
        parts.append("You have been provisioned with credentials needed for your tasks.")
    return "\n\n".join(parts)


def _build_user_turn(task: str, memory: str, text_message: str = "") -> str:
    task = _clean_port(task)
    memory = _clean_port(memory)
    text_message = _clean_port(text_message)
    parts = []
    if memory:
        parts.append(f"Relevant prior context (memory):\n{memory}")
    if task:
        parts.append(f"Task:\n{task}")
    if text_message:
        parts.append(f"Message:\n{text_message}")
    return "\n\n---\n\n".join(parts) or "Introduce yourself and describe what you can do."


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _describe_exception(exc: BaseException) -> str:
    """
    Render an exception for the ``errors`` port, unwrapping ExceptionGroups.

    The ``mcp`` SDK runs over anyio task groups, so a failed MCP handshake/tool call
    surfaces as an ExceptionGroup whose ``str()`` is the useless "unhandled errors in a
    TaskGroup (N sub-exceptions)". We recurse into ``.exceptions`` and join the real
    leaf errors so the actual cause (401, 404, connection refused, …) is visible.
    """
    leaves: List[str] = []

    def walk(e: BaseException) -> None:
        sub = getattr(e, "exceptions", None)
        if sub:
            for child in sub:
                walk(child)
        else:
            leaves.append(f"{type(e).__name__}: {e}".strip())

    walk(exc)
    # No sub-exceptions (plain error) → just its own message.
    if len(leaves) == 1 and not getattr(exc, "exceptions", None):
        return str(exc) or type(exc).__name__
    return " | ".join(dict.fromkeys(leaves)) or (str(exc) or type(exc).__name__)


async def run_claw(
    spec: ClawSpec,
    task: str,
    memory: str = "",
    text_message: str = "",
    from_channel: bool = False,
) -> Dict[str, str]:
    """
    Execute the claw described by ``spec`` against ``task`` and return a result dict
    with keys: status ("ok"|"error"), response, errors.

    When the claw has enabled app connections, their Composio-hosted MCP servers are
    passed to Anthropic's remote-MCP connector so the claw can take real actions.
    With no connections the call is identical to before (plain Messages request).
    ``from_channel`` tweaks the system prompt for inbound chat replies.
    """
    from . import guidance as guidance_mod  # lazy — avoids the guidance <-> claw_runtime cycle

    report = guidance_mod.analyze(spec)
    guide = {
        "guidance": report["guidance"],
        "setup_status": report["setup_status"],
        "connections_status": report["connections_status"],
    }

    try:
        from anthropic import AsyncAnthropic  # lazy import — see module docstring
    except ImportError:
        return {
            "status": "error",
            "response": "",
            "errors": "The 'anthropic' package is not installed on the devices server. "
                      "Add it to requirements.txt (pip install anthropic).",
            **guide,
        }

    api_key = _resolve_api_key(spec)
    if not api_key:
        # Teach instead of failing: surface the setup guidance as the response so the block
        # shows the user exactly what to add (API key, persona, connections) rather than a
        # bare error.  status stays "error" so the block still flags it as not-yet-runnable.
        return {
            "status": "error",
            "response": report["guidance"],
            "errors": "No Anthropic API key found. Set the claw's api_keys port "
                      "({\"anthropic\": \"sk-ant-…\"}) or the ANTHROPIC_API_KEY env var.",
            **guide,
        }

    params = _resolve_model_params(spec)
    system_prompt = _build_system_prompt(spec, from_channel=from_channel)
    user_turn = _build_user_turn(task, memory, text_message)
    mcp_servers = connections.build_mcp_servers(spec)
    local_conns = connections.local_loop_connections(spec)
    native_defs, native_dispatch = native_tools.build(spec)

    client = _client_for(api_key, AsyncAnthropic)
    request_kwargs: Dict[str, Any] = dict(
        model=params["model"],
        max_tokens=params["max_tokens"],
        system=[
            {
                "type": "text",
                "text": system_prompt,
                # Cache the stable persona so repeated runs are cheap/fast.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_turn}],
    )
    # Only send sampling params when the user explicitly set them AND the model
    # accepts them — the Opus 4.7+/Fable family returns HTTP 400 otherwise, which
    # is exactly the default model (claude-opus-4-8).  This is why the claw never
    # sends a default temperature.
    if "temperature" in params and _model_accepts_sampling(params["model"]):
        request_kwargs["temperature"] = params["temperature"]

    try:
        if local_conns:
            # Header-authenticated servers (e.g. Composio's Connect/Tool-Router URL) can't
            # use Anthropic's connector — we run them via a local MCP client loop that sends
            # the x-consumer-api-key header. Any bearer/self-auth connector servers ride
            # along on the same calls, and native tools (e.g. Telegram) are merged in, so a
            # mixed claw still works.
            text = await connections.run_local_agent_loop(
                client, request_kwargs, spec, connector_servers=mcp_servers,
                native_tool_defs=native_defs, native_dispatch=native_dispatch,
            )
            return {"status": "ok", "response": text, "errors": "", **guide}
        if native_defs:
            # Built-in tools from the claw's own credentials (Telegram bot token, …). Run a
            # local tool loop so the claw takes the real action; any connector servers ride along.
            text = await native_tools.run_native_tool_loop(
                client, request_kwargs, native_defs, native_dispatch, connector_servers=mcp_servers
            )
            return {"status": "ok", "response": text, "errors": "", **guide}
        if mcp_servers:
            # Remote-MCP connector: Claude calls the Composio app tools server-side,
            # so a single round-trip still yields the final text (no local loop).
            # The 2025-11-20 connector requires a matching mcp_toolset per server.
            message = await client.beta.messages.create(
                betas=[connections.MCP_CONNECTOR_BETA],
                mcp_servers=mcp_servers,
                tools=connections.build_mcp_toolsets(mcp_servers),
                **request_kwargs,
            )
        else:
            message = await client.messages.create(**request_kwargs)
    except Exception as exc:  # noqa: BLE001 — surface any SDK/API error to the block
        logger.exception("claw run failed")
        return {"status": "error", "response": "", "errors": _describe_exception(exc), **guide}

    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    usage = getattr(message, "usage", None)
    return {
        "status": "ok",
        "response": text,
        "errors": "",
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0)
        + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cost_usd": _estimate_cost_usd(params["model"], usage),
        **guide,
    }


async def stream_claw(
    spec: ClawSpec,
    task: str,
    memory: str = "",
    text_message: str = "",
    from_channel: bool = False,
):
    """
    Stream a claw run as an async generator of ``(kind, payload)`` tuples:

        ("delta", "<text chunk>")   — incremental assistant text
        ("done",  {usage + status}) — final frame (usage/cost, status, errors)
        ("error", "<message>")      — terminal error (no further frames)

    The plain (no-connections) path uses Anthropic's token stream so the block can
    render the answer as it arrives.  Connection paths (MCP connector / local tool
    loop) can't be token-streamed simply, so they fall back to a single ``delta``
    with the full text followed by ``done`` — the caller's UX is identical, just
    without intra-message streaming.  Never raises; errors arrive as an ``error``
    frame.
    """
    try:
        from anthropic import AsyncAnthropic  # lazy import — see module docstring
    except ImportError:
        yield ("error", "The 'anthropic' package is not installed on the devices server.")
        return

    api_key = _resolve_api_key(spec)
    if not api_key:
        yield ("error", "No Anthropic API key found. Set the claw's api_keys port "
                        "(or the ANTHROPIC_API_KEY env var on the server).")
        return

    params = _resolve_model_params(spec)
    mcp_servers = connections.build_mcp_servers(spec)
    local_conns = connections.local_loop_connections(spec)
    native_defs, _ = native_tools.build(spec)

    # Connection / native-tool paths aren't token-streamed — run them normally and emit once.
    if local_conns or mcp_servers or native_defs:
        result = await run_claw(spec, task, memory, text_message, from_channel)
        if result.get("status") == "ok":
            if result.get("response"):
                yield ("delta", result["response"])
            yield ("done", {k: result.get(k, 0) for k in (
                "input_tokens", "output_tokens", "cache_read_input_tokens", "cost_usd")}
                | {"status": "ok", "errors": ""})
        else:
            yield ("error", result.get("errors", "claw run failed"))
        return

    system_prompt = _build_system_prompt(spec, from_channel=from_channel)
    user_turn = _build_user_turn(task, memory, text_message)
    client = _client_for(api_key, AsyncAnthropic)
    request_kwargs: Dict[str, Any] = dict(
        model=params["model"],
        max_tokens=params["max_tokens"],
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_turn}],
    )
    if "temperature" in params and _model_accepts_sampling(params["model"]):
        request_kwargs["temperature"] = params["temperature"]

    try:
        async with client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                yield ("delta", text)
            message = await stream.get_final_message()
    except Exception as exc:  # noqa: BLE001 — surface to the socket as an error frame
        logger.exception("claw stream failed")
        yield ("error", _describe_exception(exc))
        return

    usage = getattr(message, "usage", None)
    yield ("done", {
        "status": "ok",
        "errors": "",
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0)
        + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cost_usd": _estimate_cost_usd(params["model"], usage),
    })


# ---------------------------------------------------------------------------
# Scaffold — draft a claw's design ports from a description
# ---------------------------------------------------------------------------

# Placeholder hints for the secret ports — never AI-fabricated.
_CREDENTIALS_HINT = "<your-service-credentials>"
_API_KEYS_HINT = "<your-anthropic-api-key>"

# Keys the scaffold returns to the dialog.  Secret keys are appended last and are
# always overwritten with the placeholder hints below.
_DESIGN_KEYS = ("soul", "skills", "agent", "task", "memory", "tools_config", "connections")

_SCAFFOLD_SYSTEM = (
    "You design 'claws' — small AI agents — for the Grafux platform. Given a short "
    "description, draft sensible values for a claw's configuration ports and respond "
    "with ONLY a single JSON object (no prose, no markdown fences) with EXACTLY these "
    "keys:\n"
    '  "soul"         — the agent\'s persona / system prompt (2-5 sentences).\n'
    '  "skills"       — a concise comma- or newline-separated list of capabilities.\n'
    '  "agent"        — a model id; default to "claude-opus-4-8" unless the '
    "description clearly implies otherwise.\n"
    '  "task"         — a concrete example task the claw would perform.\n'
    '  "memory"       — initial context worth remembering, or "" if none.\n'
    '  "tools_config" — tool/MCP configuration notes, or "" if none.\n'
    '  "connections"  — a JSON array of apps the claw should connect to (Telegram, '
    'WhatsApp, Slack, Gmail, …) inferred from the description, e.g. '
    '[{"app":"telegram","enabled":true}]. Use "[]" if none are implied. NEVER include '
    "tokens or connection ids — those are added later via OAuth.\n"
    "Do NOT include API keys, credentials, or secrets in any field."
)


def _scaffold_guidance(out: Dict[str, str]) -> str:
    """Compute the setup guidance for scaffolded port values (so the create dialog can show it)."""
    from . import guidance as guidance_mod  # lazy — avoid import cycle

    fields = {k: out.get(k, "") for k in ("soul", "skills", "agent", "tools_config", "connections")}
    try:
        return guidance_mod.analyze(ClawSpec(**fields))["guidance"]
    except Exception:  # noqa: BLE001 — guidance is best-effort, never break scaffolding
        return ""


def _scaffold_fallback() -> Dict[str, str]:
    """Empty design ports + placeholder secrets, used when the AI is unavailable."""
    out = {k: "" for k in _DESIGN_KEYS}
    out["agent"] = DEFAULT_MODEL
    out["credentials"] = _CREDENTIALS_HINT
    out["api_keys"] = _API_KEYS_HINT
    out["guidance"] = _scaffold_guidance(out)
    return out


async def scaffold_claw(description: str, name: str = "") -> Dict[str, str]:
    """
    Draft the claw's input-port values from ``description``.

    Returns a dict with the six design keys plus ``credentials``/``api_keys``
    (always placeholder hints).  Never raises — returns a best-effort fallback so
    the block can still be created when the AI is unavailable.
    """
    description = (description or "").strip()
    if not description:
        return _scaffold_fallback()

    try:
        from anthropic import AsyncAnthropic  # lazy import — see module docstring
    except ImportError:
        logger.warning("scaffold: anthropic not installed — returning fallback")
        return _scaffold_fallback()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # At scaffold time the user hasn't entered api_keys yet (that's what we're
        # generating), so we rely on the server's own key.
        logger.warning("scaffold: ANTHROPIC_API_KEY not set — returning fallback")
        return _scaffold_fallback()

    user_msg = f"Description: {description}"
    if name:
        user_msg = f"Claw name: {name}\n{user_msg}"

    client = AsyncAnthropic(api_key=api_key)
    try:
        message = await client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=_SCAFFOLD_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on any SDK/API error
        logger.warning("scaffold: API call failed (%s) — returning fallback", exc)
        return _scaffold_fallback()

    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    parsed = _maybe_json(text)
    if not isinstance(parsed, dict):
        logger.warning("scaffold: model did not return JSON — returning fallback")
        return _scaffold_fallback()

    # Keep only the design keys we asked for; default agent to a model id.
    out = {}
    for k in _DESIGN_KEYS:
        val = parsed.get(k, "") or ""
        # ``connections`` is a structured array — serialise it as JSON, not a Python
        # repr, so the port stores valid JSON the runtime can re-parse.
        if k == "connections" and isinstance(val, (list, dict)):
            val = json.dumps(val)
        out[k] = str(val)
    if not out["agent"].strip():
        out["agent"] = DEFAULT_MODEL
    # Secrets are ALWAYS placeholders, regardless of what the model returned.
    out["credentials"] = _CREDENTIALS_HINT
    out["api_keys"] = _API_KEYS_HINT
    out["guidance"] = _scaffold_guidance(out)
    return out
