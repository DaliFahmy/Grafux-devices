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

from typing import Optional

from pydantic import BaseModel, Field


class ClawSpec(BaseModel):
    """The persistent definition of a claw (everything except the live task)."""

    soul: str = Field("", description="Persona / system prompt for the claw.")
    skills: str = Field("", description="Capabilities — free text or a JSON list of skills/tools.")
    agent: str = Field("", description="Model id and params — a bare model id or a JSON object.")
    credentials: str = Field("", description="Free-form secrets the claw needs for its tasks.")
    api_keys: str = Field("", description="API keys — the Anthropic key and any task keys (text or JSON).")
    tools_config: str = Field("", description="Tool / MCP server configuration (text or JSON).")
    name: str = Field("", description="Optional human-friendly name for the claw.")


class CreateClawResponse(BaseModel):
    claw_id: str
    status: str = "created"


class RunRequest(BaseModel):
    """The live inputs supplied on every run of an existing claw."""

    task: str = Field("", description="The task / instruction to run the claw against.")
    memory: str = Field("", description="Prior context prepended to the task.")


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
