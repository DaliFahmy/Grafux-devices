"""
models.py
Request/response schemas for the OpenClaw runtime.

A claw is assembled from the eight Grafux "claw" block input ports.  Every field
is optional so a partially-wired block still produces a usable (if limited) claw;
``soul``, ``agent`` and an API key are the only fields that meaningfully change
the produced agent.

Port → field mapping
--------------------
soul         -> ClawSpec.soul          (system prompt / persona)
skills       -> ClawSpec.skills        (capabilities; free text or JSON list)
agent        -> ClawSpec.agent         (model id + params; "claude-opus-4-8" or JSON)
credentials  -> ClawSpec.credentials   (free-form secrets injected at run time)
api_keys     -> ClawSpec.api_keys      (Anthropic key + task keys; text or JSON)
tools_config -> ClawSpec.tools_config  (tool/server configuration; text or JSON)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ClawSpec(BaseModel):
    """The persistent definition of a claw (everything except the live task)."""

    soul: str = Field("", description="Persona / system prompt for the claw.")
    skills: str = Field("", description="Capabilities — free text or a JSON list of skills/tools.")
    agent: str = Field("", description="Model id and params — a bare model id or a JSON object.")
    credentials: str = Field("", description="Free-form secrets the claw needs for its tasks.")
    api_keys: str = Field("", description="API keys — the Anthropic key and any task keys (text or JSON).")
    tools_config: str = Field("", description="Tool / MCP server configuration (text or JSON).")
    connections: str = Field(
        "",
        description=(
            "Connected applications (Telegram, WhatsApp, Slack, …) as a JSON list of "
            "connection objects: [{\"app\", \"connection_id\", \"account_label\", "
            "\"enabled\", \"channel\", \"mcp_url\"}]. Parsed at run time to give the "
            "claw real tools via Composio's hosted MCP servers."
        ),
    )
    channel: str = Field(
        "",
        description=(
            "JSON describing the inbound messaging channel this claw is registered on, "
            "e.g. {\"provider\": \"telegram\", \"connection_id\": \"...\"}. Empty when the "
            "claw is not driven by an external channel."
        ),
    )
    name: str = Field("", description="Optional human-friendly name for the claw.")


class ClawConnection(BaseModel):
    """A single external-app connection wired to a claw (one entry of ``connections``)."""

    app: str = Field("", description="App / toolkit name, e.g. 'telegram', 'whatsapp', 'slack'.")
    connection_id: str = Field("", description="Composio connected-account id for this app.")
    account_label: str = Field("", description="Human-friendly label for the connected account.")
    enabled: bool = Field(True, description="Whether the claw may use this connection's tools.")
    channel: bool = Field(False, description="True if this connection is the inbound messaging channel.")
    mcp_url: str = Field("", description="Explicit Composio MCP server URL override (optional).")


class CreateClawResponse(BaseModel):
    claw_id: str
    status: str = "created"


class RunRequest(BaseModel):
    """The live inputs supplied on every run of an existing claw."""

    task: str = Field("", description="The task / instruction to run the claw against.")
    memory: str = Field("", description="Prior context prepended to the task.")
    text_message: str = Field("", description="A free-text message the user sends to the claw on this run.")


class RunResponse(BaseModel):
    claw_id: str
    status: str  # "ok" | "error"
    response: str = ""
    errors: str = ""


class ClawSummary(BaseModel):
    """A claw entry as returned by the list endpoint (no secrets echoed back)."""

    claw_id: str
    name: str = ""
    agent: str = ""


class InitiateConnectionRequest(BaseModel):
    """Begin a Composio OAuth flow to connect an app to a claw."""

    app: str = Field(..., description="App / toolkit to connect, e.g. 'telegram'.")
    user_id: str = Field("", description="An opaque user/entity id used to scope the connection.")
    redirect_uri: str = Field("", description="Where Composio returns the user after authorizing.")


class InitiateConnectionResponse(BaseModel):
    app: str = ""
    connection_id: str = ""
    redirect_url: str = ""
    status: str = "pending"  # "pending" until the user authorizes, then "active"


class ConnectionSummary(BaseModel):
    """A connected app as echoed back to the UI (never carries secret tokens)."""

    app: str = ""
    connection_id: str = ""
    account_label: str = ""
    status: str = ""  # Composio account status, e.g. "ACTIVE" / "INITIATED"
    enabled: bool = True
    channel: bool = False


class RegisterChannelRequest(BaseModel):
    """Register (or update) the inbound messaging channel for a claw."""

    provider: str = Field(..., description="Channel provider, e.g. 'telegram', 'whatsapp'.")
    connection_id: str = Field("", description="Composio connected-account id backing the channel.")
    mode: str = Field("webhook", description="'webhook' (provider pushes to us) or 'polling'.")
    webhook_url: str = Field("", description="Public URL of this claw's webhook endpoint.")


class RegisterChannelResponse(BaseModel):
    status: str = "registered"
    provider: str = ""
    webhook_url: str = ""
    errors: str = ""


class ScaffoldRequest(BaseModel):
    """Ask the AI to draft a claw's design ports from a free-text description."""

    description: str = Field("", description="What the claw should be / do.")
    name: str = Field("", description="Optional claw name for extra context.")
    category: str = Field("", description="Optional category for extra context.")


class ScaffoldResponse(BaseModel):
    """
    AI-drafted values for the claw block's input ports.

    The design ports (soul/skills/agent/task/memory/tools_config) are filled from
    the description.  The secret ports (credentials/api_keys) are NEVER fabricated
    — they carry placeholder hints the user replaces with real secrets.
    """

    soul: str = ""
    skills: str = ""
    agent: str = ""
    task: str = ""
    memory: str = ""
    tools_config: str = ""
    connections: str = ""
    credentials: str = ""
    api_keys: str = ""
