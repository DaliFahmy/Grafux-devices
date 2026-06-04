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
from typing import Any, Dict, Optional

from .models import ClawSpec

logger = logging.getLogger("openclaw.runtime")

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Port parsing helpers — every port is free text that MAY contain JSON.
# ---------------------------------------------------------------------------

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
        raw = (raw or "").strip()
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
    agent = (spec.agent or "").strip()
    if not agent:
        return params
    parsed = _maybe_json(agent)
    if isinstance(parsed, dict):
        if parsed.get("model"):
            params["model"] = str(parsed["model"])
        if isinstance(parsed.get("max_tokens"), int):
            params["max_tokens"] = parsed["max_tokens"]
        if isinstance(parsed.get("temperature"), (int, float)):
            params["temperature"] = float(parsed["temperature"])
    else:
        params["model"] = agent
    return params


def _build_system_prompt(spec: ClawSpec) -> str:
    """Assemble the claw's standing instructions from soul + skills + tools_config."""
    parts = []
    soul = (spec.soul or "").strip()
    parts.append(soul if soul else "You are a helpful AI agent (a Grafux claw).")
    if spec.skills.strip():
        parts.append("Your skills and capabilities:\n" + spec.skills.strip())
    if spec.tools_config.strip():
        parts.append("Tool / environment configuration:\n" + spec.tools_config.strip())
    if spec.credentials.strip():
        # Credentials are provided so the claw is *aware* it has access; we do not
        # dump raw secret blobs into the prompt beyond what the user supplied.
        parts.append("You have been provisioned with credentials needed for your tasks.")
    return "\n\n".join(parts)


def _build_user_turn(task: str, memory: str) -> str:
    task = (task or "").strip()
    memory = (memory or "").strip()
    if memory:
        return f"Relevant prior context (memory):\n{memory}\n\n---\n\nTask:\n{task}"
    return task or "Introduce yourself and describe what you can do."


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def run_claw(spec: ClawSpec, task: str, memory: str = "") -> Dict[str, str]:
    """
    Execute the claw described by ``spec`` against ``task`` and return a result dict
    with keys: status ("ok"|"error"), response, errors.
    """
    try:
        from anthropic import AsyncAnthropic  # lazy import — see module docstring
    except ImportError:
        return {
            "status": "error",
            "response": "",
            "errors": "The 'anthropic' package is not installed on the devices server. "
                      "Add it to requirements.txt (pip install anthropic).",
        }

    api_key = _resolve_api_key(spec)
    if not api_key:
        return {
            "status": "error",
            "response": "",
            "errors": "No Anthropic API key found. Set the claw's api_keys port "
                      "(or the ANTHROPIC_API_KEY env var on the server).",
        }

    params = _resolve_model_params(spec)
    system_prompt = _build_system_prompt(spec)
    user_turn = _build_user_turn(task, memory)

    client = AsyncAnthropic(api_key=api_key)
    try:
        message = await client.messages.create(
            model=params["model"],
            max_tokens=params["max_tokens"],
            temperature=params.get("temperature", 1.0),
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
    except Exception as exc:  # noqa: BLE001 — surface any SDK/API error to the block
        logger.exception("claw run failed")
        return {"status": "error", "response": "", "errors": str(exc)}

    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    return {"status": "ok", "response": text, "errors": ""}


# ---------------------------------------------------------------------------
# Scaffold — draft a claw's design ports from a description
# ---------------------------------------------------------------------------

# Placeholder hints for the secret ports — never AI-fabricated.
_CREDENTIALS_HINT = "<your-service-credentials>"
_API_KEYS_HINT = "<your-anthropic-api-key>"

# Keys the scaffold returns to the dialog.  Secret keys are appended last and are
# always overwritten with the placeholder hints below.
_DESIGN_KEYS = ("soul", "skills", "agent", "task", "memory", "tools_config")

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
    "Do NOT include API keys, credentials, or secrets in any field."
)


def _scaffold_fallback() -> Dict[str, str]:
    """Empty design ports + placeholder secrets, used when the AI is unavailable."""
    out = {k: "" for k in _DESIGN_KEYS}
    out["agent"] = DEFAULT_MODEL
    out["credentials"] = _CREDENTIALS_HINT
    out["api_keys"] = _API_KEYS_HINT
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
    out = {k: str(parsed.get(k, "") or "") for k in _DESIGN_KEYS}
    if not out["agent"].strip():
        out["agent"] = DEFAULT_MODEL
    # Secrets are ALWAYS placeholders, regardless of what the model returned.
    out["credentials"] = _CREDENTIALS_HINT
    out["api_keys"] = _API_KEYS_HINT
    return out
