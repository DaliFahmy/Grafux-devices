"""
models.py
Request schemas for the Generator, key resolution, and the sealing of what an
agent job needs to know but a workflow input must not show.

There is no sandbox POD any more: agent turns run as GitHub Actions jobs
(``actions.py``).  RunPod pods are rented only by a block's Regenerate.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("generator.models")

# The CLI versions the agent job installs.  PINNED: a CLI that changes its JSONL
# schema under us turns every turn into a parse error.  Bump deliberately, after
# re-running the parser tests against captured output.
CLI_VERSIONS = {
    "claude_code": os.environ.get("GENERATOR_CLAUDE_CODE_VERSION", "2.1.283"),
    "codex": os.environ.get("GENERATOR_CODEX_VERSION", "0.157.1"),
}


class CreateSessionRequest(BaseModel):
    agent: str = Field("claude_code", description="claude_code | codex")
    model: str = Field("", description="Model id; empty = the agent's default.")
    mode: str = Field("plan", description="plan (propose, write nothing) | edit (build the block)")
    prompt: str = Field(..., description="The user's idea, in words.")
    repo_url: str = Field("", description="Upstream git repository (GitHub tree URLs accepted).")
    owner: str = Field("", description="The user's name or id; namespaces the image tag.")
    # Optional: the user's own model keys.  Empty = Grafux's (resolve_agent_key).
    # Held in memory for the session only, handed to the job SEALED, never logged.
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Kept for request compatibility; the Generator rents no pods.
    api_keys: str = ""


# Which provider each agent bills, and the env vars that hold GRAFUX's key for it.
# The dedicated GENERATOR_* key is tried first on purpose: the agent job also runs
# code from the user's repo, so a key placed there can be read by it.  Grafux's
# key for this job should be a separate, spend-limited one; the general key the
# claw block uses is only the fallback.
AGENT_PROVIDER = {"claude_code": "anthropic", "codex": "openai"}
_SERVER_KEY_ENV = {
    "anthropic": ("GENERATOR_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("GENERATOR_OPENAI_API_KEY", "OPENAI_API_KEY"),
}


def resolve_agent_key(provider: str, given: str = "") -> Tuple[str, str]:
    """
    ``(key, source)`` for a provider: the user's key if they gave one, else
    Grafux's.  ``source`` is "user", "grafux" or "" (no key anywhere).  The same
    order the claw/gpu/eda runtimes use: the block's key first, then the server's.
    """
    given = (given or "").strip()
    if given:
        return given, "user"
    for name in _SERVER_KEY_ENV.get(provider, ()):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value, "grafux"
    return "", ""


class MessageRequest(BaseModel):
    text: str
    mode: str = "edit"


class IngestRequest(BaseModel):
    """What the agent job's forwarder POSTs: numbered raw output lines."""
    token: str
    turn: int
    lines: list = Field(default_factory=list)      # [{"n": int, "line": str}]


# ---------------------------------------------------------------------------
# Sealing: workflow inputs are visible to anyone who can read the builds repo
# ---------------------------------------------------------------------------

def wrap_key() -> str:
    return (os.environ.get("GENERATOR_WRAP_KEY") or "").strip()


def seal(payload: Dict[str, Any], key: str = "") -> str:
    """Fernet-encrypt ``payload`` with GENERATOR_WRAP_KEY (also a builds-repo secret)."""
    from cryptography.fernet import Fernet
    return Fernet((key or wrap_key()).encode()).encrypt(json.dumps(payload).encode()).decode()


def unseal(token: str, key: str = "") -> Dict[str, Any]:
    """The inverse of ``seal`` -- what agent.yml does in Python on the runner."""
    from cryptography.fernet import Fernet
    return json.loads(Fernet((key or wrap_key()).encode()).decrypt(token.encode()))


def callback_url() -> str:
    """
    Where the job POSTs live output.  GENERATOR_PUBLIC_URL, else the URL Render
    injects into every web service.  Empty = no live stream; the whole output is
    still read back from the branch when the job ends.
    """
    base = (os.environ.get("GENERATOR_PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").strip()
    return base.rstrip("/")


def missing_configuration() -> str:
    """Why the Generator cannot run on this server, or "" when it can."""
    missing = [name for name in ("GENERATOR_GITHUB_TOKEN", "GENERATOR_BUILDS_REPO", "GENERATOR_WRAP_KEY")
               if not (os.environ.get(name) or "").strip()]
    if not missing:
        return ""
    return ("The Generator is not set up on this server: set " + ", ".join(missing)
            + " on the devices service (see GENERATOR/builds-repo/README.md).")
