"""
guidance.py
Readiness analysis for a claw — "what do I still need to add to which port?".

``analyze(spec)`` inspects a ClawSpec (no network, no API calls — pure & cheap, so it adds
~no latency to a run) and returns a small report the block surfaces on three output ports:

    setup_status         — a one-line readiness summary ("Ready" / "Needs Anthropic API key" …)
    guidance             — friendly markdown telling the user what to put in each input port
    connections_status   — per-app readiness (configured / needs_auth / channel)

It also gates the run: when ``ready`` is False (e.g. no Anthropic key) the runtime returns the
guidance text as the response instead of a bare error, so the block teaches instead of failing.

The module imports ``claw_runtime`` lazily inside ``analyze`` to avoid an import cycle
(``claw_runtime`` imports this module lazily in turn).
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import connections
from .models import ClawSpec


def _clean(text: str) -> str:
    return connections._clean_port(text)


def analyze(spec: ClawSpec) -> Dict[str, Any]:
    """Return ``{ready, setup_status, guidance, connections_status}`` for ``spec``."""
    from . import claw_runtime  # lazy — avoid the claw_runtime <-> guidance import cycle

    api_key = claw_runtime._resolve_api_key(spec)
    composio_key = connections.resolve_composio_key(spec)
    soul = _clean(spec.soul)
    model = claw_runtime._resolve_model_params(spec)["model"]
    model_known = model in claw_runtime.MODEL_CATALOG
    statuses = connections.connection_statuses(spec)

    pending = [s for s in statuses if s["needs_auth"]]
    ready = bool(api_key)  # only the Anthropic key blocks a run; the rest are warnings

    # ---- one-line status -------------------------------------------------
    if not api_key:
        setup_status = "Needs Anthropic API key"
    elif pending:
        first = pending[0]["app"]
        setup_status = f"{first}: not connected — open Manage Connections to scan a QR"
    elif not soul:
        setup_status = "Ready (tip: give it a persona in 'soul')"
    else:
        setup_status = "Ready"

    # ---- guidance markdown ----------------------------------------------
    parts: List[str] = []

    if not api_key:
        parts.append(
            "### 🔑 Anthropic API key — required\n"
            "Add your key to the **api_keys** port:\n"
            "```json\n{\"anthropic\": \"sk-ant-…\"}\n```\n"
            "Get one at https://console.anthropic.com/ . To also connect apps "
            "(WhatsApp / Telegram / …) add a Composio key in the same object:\n"
            "```json\n{\"anthropic\": \"sk-ant-…\", \"composio\": \"ck_…\"}\n```"
        )

    if not soul:
        parts.append(
            "### 🧠 Persona — recommended\n"
            "The **soul** port is the claw's system prompt / personality. Describe who it is "
            "and how it should behave, e.g. *\"You are a concise support agent for Acme. Be "
            "friendly, never invent order numbers.\"*"
        )

    if not model_known:
        parts.append(
            f"### 🤖 Model\n"
            f"The **agent** port resolves to `{model}`, which isn't in the known model list. "
            "Use a bare model id (e.g. `claude-opus-4-8`) or JSON "
            "`{\"model\": \"claude-opus-4-8\", \"max_tokens\": 4096}`."
        )

    # Connections section
    if statuses:
        lines = ["### 🔌 Connected apps"]
        for s in statuses:
            app = s["app"]
            tag = " (inbound channel)" if s["channel"] else ""
            if s["configured"]:
                lines.append(f"- ✅ **{app}**{tag} — tools ready.")
            else:
                hint = "" if composio_key else (
                    " First add a Composio key to **api_keys** "
                    "(`{\"composio\": \"ck_…\"}`)."
                )
                lines.append(
                    f"- ⚠️ **{app}**{tag} — not connected yet. Open **Manage Connections…** on "
                    f"the block, click **Connect**, and scan the QR with your phone to "
                    f"authorize it.{hint}"
                )
        parts.append("\n".join(lines))
    else:
        parts.append(
            "### 🔌 Connect apps (optional)\n"
            "Give the claw real tools (send a Telegram message, read Gmail, post to Slack…). "
            "Either type app names into the **connections** port — e.g. "
            "`[\"whatsapp\", \"telegram\"]` — or open **Manage Connections…** on the block to "
            "pick apps and scan a QR to authorize them. "
            f"Available: {', '.join(connections.APP_CATALOG.keys())}."
        )

    if ready and not pending:
        parts.insert(0, "✅ **This claw is ready to run.**")

    guidance = "\n\n".join(parts)

    return {
        "ready": ready,
        "setup_status": setup_status,
        "guidance": guidance,
        "connections_status": statuses,
    }
