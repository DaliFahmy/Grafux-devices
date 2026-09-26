"""
models.py
Request schemas for the Generator, and the sandbox pod's image.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Tuple

from pydantic import BaseModel, Field

from EDA.models import register_kind_image

logger = logging.getLogger("generator.models")

# The sandbox the agents run in: ubuntu + git + node + Claude Code + Codex +
# sshd/start.sh.  Built by .github/workflows/build-generator-image.yml.  PINNED,
# and checked for publication before every session (``image_published``): the
# cpu block shipped a default that had never been pushed, and RunPod answered
# with a bare "Exited by Runpod".  A session must fail fast with the reason.
GENERATOR_IMAGE = os.environ.get("GENERATOR_IMAGE", "ghcr.io/dalifahmy/grafux-generator:v1-20260926")
GENERATOR_INSTANCE = os.environ.get("GENERATOR_INSTANCE", "cpu3c-4")
GENERATOR_DISK_GB = 30

register_kind_image("generator", lambda: GENERATOR_IMAGE, GENERATOR_DISK_GB)


class CreateSessionRequest(BaseModel):
    agent: str = Field("claude_code", description="claude_code | codex")
    model: str = Field("", description="Model id; empty = the agent's default.")
    mode: str = Field("plan", description="plan (propose, write nothing) | edit (build the block)")
    prompt: str = Field(..., description="The user's idea, in words.")
    repo_url: str = Field("", description="Upstream git repository (GitHub tree URLs accepted).")
    owner: str = Field("", description="The user's name or id; namespaces the image tag.")
    # The user's own model keys (the user pays for tokens).  Held in memory for
    # the session only, written to a 0600 file on the sandbox, never logged.
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # RunPod key override (same shapes as every pod-backed block's api_keys port).
    api_keys: str = ""


class MessageRequest(BaseModel):
    text: str
    mode: str = "edit"


# ---------------------------------------------------------------------------
# Is an image actually published (anonymously pullable)?
# ---------------------------------------------------------------------------

_published_cache: Dict[str, Tuple[bool, float]] = {}
_cache_lock = threading.Lock()
_CACHE_S = 600.0


def image_published(image: str, http=None) -> bool:
    """
    True when GHCR serves ``image`` to an anonymous client -- exactly how RunPod
    pulls.  Non-GHCR images are assumed published (we cannot cheaply check).
    Errors reaching GHCR count as published: an outage must not block sessions.
    """
    if not image.startswith("ghcr.io/"):
        return True
    with _cache_lock:
        hit = _published_cache.get(image)
        if hit and time.monotonic() - hit[1] < _CACHE_S:
            return hit[0]
    name, _, tag = image[len("ghcr.io/"):].partition(":")
    tag = tag or "latest"
    try:
        if http is None:
            import httpx
            http = httpx.Client(timeout=15.0)
        tok = http.get("https://ghcr.io/token",
                       params={"scope": f"repository:{name}:pull", "service": "ghcr.io"})
        token = tok.json().get("token") if tok.status_code == 200 else None
        if not token:
            ok = False
        else:
            head = http.head(f"https://ghcr.io/v2/{name}/manifests/{tag}", headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.oci.image.index.v1+json, "
                          "application/vnd.docker.distribution.manifest.list.v2+json, "
                          "application/vnd.docker.distribution.manifest.v2+json, "
                          "application/vnd.oci.image.manifest.v1+json",
            })
            ok = head.status_code == 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not check %s on GHCR (%s); assuming published", image, exc)
        return True
    with _cache_lock:
        _published_cache[image] = (ok, time.monotonic())
    return ok
