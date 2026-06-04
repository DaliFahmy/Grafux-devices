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
