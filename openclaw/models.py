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

from typing import Dict, List, Optional

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
            "\"enabled\", \"channel\", \"mcp_url\", \"header_auth\", \"api_key\"}]. Parsed at "
            "run time to give the claw real tools via Composio's hosted MCP servers — through "
            "Anthropic's connector (bearer auth) or, when header_auth is set, a local MCP loop "
            "that authenticates with the x-consumer-api-key header."
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
    mcp_url: str = Field(
        "",
        description=(
            "The Composio MCP server URL for this app, e.g. "
            "'https://backend.composio.dev/v3/mcp/<server_id>'. This is the primary field — the "
            "claw passes it to Anthropic's MCP connector to gain real tools. The Composio server "
            "should have its api-key requirement disabled so the URL is self-authenticating."
        ),
    )
    user_id: str = Field(
        "",
        description="Composio user id appended to the MCP URL (?user_id=…) to select connected accounts.",
    )
    auth_token: str = Field(
        "",
        description="Optional OAuth bearer token for the MCP server (sent as Authorization: Bearer).",
    )
    header_auth: bool = Field(
        False,
        description=(
            "When True, the claw does NOT hand this server's URL to Anthropic's MCP connector "
            "(which can only send an OAuth bearer). Instead it connects to the server itself via "
            "a local MCP client loop and authenticates with the 'x-consumer-api-key' header — "
            "required for Composio's Connect / Tool-Router URL "
            "(https://connect.composio.dev/mcp). Default False keeps the original connector path."
        ),
    )
    api_key: str = Field(
        "",
        description=(
            "Composio API key sent as 'x-consumer-api-key' when header_auth is True. When empty, "
            "the claw falls back to the Composio key resolved from its api_keys/credentials ports "
            "or the COMPOSIO_API_KEY env var, so the secret can live in api_keys rather than here."
        ),
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Arbitrary HTTP headers to send to the MCP server (e.g. {'x-consumer-api-key': '…'}). "
            "Populated when the connections port uses the standard MCP config shape "
            "({'mcpServers': {'<name>': {'url', 'headers'}}}). Any non-empty headers force the "
            "local MCP loop (Anthropic's connector cannot send custom headers)."
        ),
    )
    connection_id: str = Field("", description="Composio connected-account id (used for inbound replies).")
    account_label: str = Field("", description="Human-friendly label for the connected account.")
    enabled: bool = Field(True, description="Whether the claw may use this connection's tools.")
    channel: bool = Field(False, description="True if this connection is the inbound messaging channel.")


class CreateClawResponse(BaseModel):
    claw_id: str
    status: str = "created"


class RunRequest(BaseModel):
    """The live inputs supplied on every run of an existing claw."""

    task: str = Field("", description="The task / instruction to run the claw against.")
    memory: str = Field("", description="Prior context prepended to the task.")
    text_message: str = Field("", description="A free-text message the user sends to the claw on this run.")
    session_id: str = Field(
        "",
        description=(
            "Optional conversation id.  When set, the run is threaded into a rolling "
            "block-level transcript (provider 'block') so the claw remembers prior turns "
            "across Run clicks.  Empty ⇒ stateless run, identical to before."
        ),
    )
    remember: bool = Field(
        True,
        description="When session_id is set, whether to append this turn to the transcript.",
    )


class ConfigPatchRequest(BaseModel):
    """
    Patch the mutable, NON-secret config of an existing claw in place.

    Sent by the block on Run when its config ports changed, so connection / soul /
    skills / tools_config / agent edits take effect WITHOUT a full Regenerate.
    Every field is optional; only provided (non-None) fields are applied.  Secret
    ports (credentials, api_keys) are intentionally absent — those still require
    Regenerate so they are never silently patched.
    """

    soul: Optional[str] = None
    skills: Optional[str] = None
    agent: Optional[str] = None
    tools_config: Optional[str] = None
    connections: Optional[str] = None


class RunResponse(BaseModel):
    claw_id: str
    status: str  # "ok" | "error"
    response: str = ""
    errors: str = ""
    # Per-run usage for the block's cost badge.  All default 0 so older callers
    # (and the error / local-loop paths) stay valid.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0


class ClawModel(BaseModel):
    """One selectable Claude model for the creation-dialog dropdown."""

    id: str                 # model id passed back in ClawSpec.agent
    label: str = ""         # human-friendly display name
    input_per_mtok: float = 0.0   # $ per 1M input tokens
    output_per_mtok: float = 0.0  # $ per 1M output tokens


class ClawModelsResponse(BaseModel):
    models: list[ClawModel] = []


class ClawSummary(BaseModel):
    """A claw entry as returned by the list endpoint (no secrets echoed back)."""

    claw_id: str
    name: str = ""
    agent: str = ""
    # Enrichment for the block's badges/tooltips (non-secret, all optional).
    model: str = ""               # resolved model id the claw will run with
    apps: list[str] = []          # connected app names (e.g. ["telegram", "slack"])
    tool_count: int = 0           # number of tools the claw currently exposes


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
