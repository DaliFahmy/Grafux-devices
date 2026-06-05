"""
router.py
FastAPI router exposing the OpenClaw runtime over REST.

Mounted by ``devices_server.py`` with ``app.include_router(router)``.  All routes
are under the ``/claw`` prefix:

    POST   /claw/create        -> create a reusable claw, returns {claw_id}
    POST   /claw/{id}/run      -> run an existing claw against a task
    POST   /claw/{id}/create_and_run  (convenience for the block's first run)
    GET    /claw               -> list registered claws (no secrets)
    GET    /claw/{id}          -> a claw's non-secret summary
    DELETE /claw/{id}          -> remove a claw

External app connections (Composio) + inbound messaging channels:

    POST   /claw/{id}/connections/initiate     -> start a Composio OAuth flow
    GET    /claw/{id}/connections              -> list the claw's connected apps
    DELETE /claw/{id}/connections/{app}        -> disconnect an app
    POST   /claw/{id}/channels/{provider}/register -> register an inbound channel
    POST   /claw/{id}/channels/{provider}/webhook  -> inbound message entrypoint

The Grafux "claw" block calls ``/claw/create`` once (caching the returned
``claw_id`` in its output port) and ``/claw/{id}/run`` on every subsequent run.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from . import claw_runtime, connections
from .models import (
    ClawSpec,
    ClawSummary,
    ConnectionSummary,
    CreateClawResponse,
    InitiateConnectionRequest,
    InitiateConnectionResponse,
    RegisterChannelRequest,
    RegisterChannelResponse,
    RunRequest,
    RunResponse,
    ScaffoldRequest,
    ScaffoldResponse,
)
from .registry import registry
from .sessions import sessions

logger = logging.getLogger("openclaw.router")

router = APIRouter(prefix="/claw", tags=["claw"])


@router.post("/create", response_model=CreateClawResponse)
async def create_claw(spec: ClawSpec) -> CreateClawResponse:
    """Provision a reusable claw from its block ports and return its id."""
    claw_id = registry.create(spec)
    return CreateClawResponse(claw_id=claw_id, status="created")


@router.post("/scaffold", response_model=ScaffoldResponse)
async def scaffold_claw(body: ScaffoldRequest) -> ScaffoldResponse:
    """
    Draft a claw's input-port values from a description (used by the create dialog).

    Never errors for AI failures — returns a best-effort object (empty design ports
    + placeholder secrets) so the block can still be created.
    """
    drafted = await claw_runtime.scaffold_claw(body.description, body.name)
    return ScaffoldResponse(**drafted)


@router.get("", response_model=list[ClawSummary])
async def list_claws() -> list[ClawSummary]:
    """List provisioned claws (ids + non-secret summary)."""
    return registry.list()


@router.get("/{claw_id}", response_model=ClawSummary)
async def get_claw(claw_id: str) -> ClawSummary:
    spec = registry.get(claw_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    return ClawSummary(claw_id=claw_id, name=spec.name, agent=spec.agent)


@router.post("/{claw_id}/run", response_model=RunResponse)
async def run_claw(claw_id: str, body: RunRequest) -> RunResponse:
    """Run an existing claw against a task."""
    spec = registry.get(claw_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    result = await claw_runtime.run_claw(spec, body.task, body.memory, body.text_message)
    return RunResponse(claw_id=claw_id, **result)


@router.post("/create_and_run", response_model=RunResponse)
async def create_and_run(spec: ClawSpec) -> RunResponse:
    """
    Convenience endpoint: provision a claw and immediately run it in one call.

    The task/memory are read from the spec-bearing body's sibling fields when the
    caller prefers a single round-trip.  The claw is still registered so later
    runs can reuse it via ``/claw/{id}/run``.
    """
    claw_id = registry.create(spec)
    result = await claw_runtime.run_claw(spec, task="", memory="")
    return RunResponse(claw_id=claw_id, **result)


def _require_spec(claw_id: str) -> ClawSpec:
    spec = registry.get(claw_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    return spec


def _require_composio_key(spec: ClawSpec) -> str:
    key = connections.resolve_composio_key(spec)
    if not key:
        raise HTTPException(
            status_code=400,
            detail="No Composio API key found. Set the claw's api_keys port "
                   "({\"composio\": \"...\"}) or the COMPOSIO_API_KEY env var.",
        )
    return key


# ---------------------------------------------------------------------------
# External app connections (Composio OAuth connected-accounts)
# ---------------------------------------------------------------------------

@router.post("/{claw_id}/connections/initiate", response_model=InitiateConnectionResponse)
async def initiate_connection(claw_id: str, body: InitiateConnectionRequest) -> InitiateConnectionResponse:
    """Start a Composio OAuth flow so the user can link an app to this claw."""
    spec = _require_spec(claw_id)
    key = _require_composio_key(spec)
    try:
        result = await connections.initiate_connection(
            body.app, body.user_id or claw_id, body.redirect_uri, key
        )
    except Exception as exc:  # noqa: BLE001 — surface Composio/network errors to the UI
        logger.warning("connection initiate failed for claw %s: %s", claw_id, exc)
        raise HTTPException(status_code=502, detail=f"Composio initiate failed: {exc}")

    # Record the connection on the live claw spec so its tools are available on the
    # next run without waiting for the frontend to re-provision (registry.get returns
    # the stored ClawSpec by reference, so this mutation sticks for this process).
    conns = connections.parse_connections(spec)
    new_id = result.get("connection_id", "")
    updated = [c for c in conns if c.app.lower() != body.app.lower()]
    updated.append(
        connections.ClawConnection(app=body.app, connection_id=new_id, enabled=True)
    )
    spec.connections = json.dumps([c.model_dump() for c in updated])
    return InitiateConnectionResponse(**result)


@router.get("/{claw_id}/connections", response_model=list[ConnectionSummary])
async def list_claw_connections(claw_id: str) -> list[ConnectionSummary]:
    """
    List the claw's connections, enriched with live Composio status.

    The configured connections (from the ``connections`` port) are the source of
    truth for which apps the claw uses; Composio is queried for their auth status.
    """
    spec = _require_spec(claw_id)
    configured = connections.parse_connections(spec)
    status_by_id: dict[str, str] = {}
    key = connections.resolve_composio_key(spec)
    if key:
        try:
            for c in await connections.list_connections(key):
                status_by_id[c["connection_id"]] = c.get("status", "")
        except Exception as exc:  # noqa: BLE001 — status is best-effort, never fatal
            logger.warning("connection list failed for claw %s: %s", claw_id, exc)
    return [
        ConnectionSummary(
            app=c.app,
            connection_id=c.connection_id,
            account_label=c.account_label,
            status=status_by_id.get(c.connection_id, ""),
            enabled=c.enabled,
            channel=c.channel,
        )
        for c in configured
    ]


@router.delete("/{claw_id}/connections/{app}")
async def delete_claw_connection(claw_id: str, app: str) -> dict:
    """Disconnect ``app``: drop it from the claw spec and revoke it on Composio."""
    spec = _require_spec(claw_id)
    remaining = [c for c in connections.parse_connections(spec) if c.app.lower() != app.lower()]
    removed = [c for c in connections.parse_connections(spec) if c.app.lower() == app.lower()]
    spec.connections = json.dumps([c.model_dump() for c in remaining])
    key = connections.resolve_composio_key(spec)
    if key:
        for c in removed:
            if c.connection_id:
                try:
                    await connections.delete_connection(c.connection_id, key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Composio delete failed for %s: %s", c.connection_id, exc)
    return {"status": "disconnected", "app": app, "claw_id": claw_id}


# ---------------------------------------------------------------------------
# Inbound messaging channels (Telegram / WhatsApp / Slack …)
# ---------------------------------------------------------------------------

@router.post("/{claw_id}/channels/{provider}/register", response_model=RegisterChannelResponse)
async def register_channel(
    claw_id: str, provider: str, body: RegisterChannelRequest
) -> RegisterChannelResponse:
    """
    Mark which connection is this claw's inbound channel and store the webhook URL.

    The webhook URL is whatever public address routes to
    ``/claw/{id}/channels/{provider}/webhook``; the caller (frontend) knows the
    devices-server base URL, so it passes the full URL here for us to record.
    """
    spec = _require_spec(claw_id)
    spec.channel = json.dumps(
        {
            "provider": provider,
            "connection_id": body.connection_id,
            "mode": body.mode,
            "webhook_url": body.webhook_url,
        }
    )
    # Flag the matching connection as the channel so the UI/runtime can show it.
    conns = connections.parse_connections(spec)
    for c in conns:
        c.channel = c.connection_id == body.connection_id and bool(body.connection_id)
    spec.connections = json.dumps([c.model_dump() for c in conns])
    logger.info("claw %s registered on channel %s (mode=%s)", claw_id, provider, body.mode)
    return RegisterChannelResponse(status="registered", provider=provider, webhook_url=body.webhook_url)


async def _process_inbound(claw_id: str, provider: str, payload: dict) -> None:
    """Background worker: run the claw on an inbound message and reply in-chat."""
    spec = registry.get(claw_id)
    if spec is None:
        logger.warning("inbound for unknown claw %s — dropping", claw_id)
        return
    chat_id, text = connections.parse_inbound(provider, payload)
    if not chat_id or not text:
        return  # delivery receipt / non-message event — nothing to do

    history = sessions.get(claw_id, provider, chat_id)
    sessions.append(claw_id, provider, chat_id, "User", text)
    result = await claw_runtime.run_claw(
        spec, task="", memory=history, text_message=text, from_channel=True
    )
    reply = result.get("response", "").strip()
    if result.get("status") != "ok" or not reply:
        logger.warning("claw %s produced no reply (status=%s)", claw_id, result.get("status"))
        return
    sessions.append(claw_id, provider, chat_id, "Assistant", reply)

    key = connections.resolve_composio_key(spec)
    channel = connections._maybe_json(spec.channel) or {}
    connection_id = channel.get("connection_id", "") if isinstance(channel, dict) else ""
    if key:
        try:
            await connections.send_channel_reply(provider, connection_id, chat_id, reply, key)
        except Exception as exc:  # noqa: BLE001 — log, never crash the worker
            logger.warning("send reply failed for claw %s: %s", claw_id, exc)


@router.post("/{claw_id}/channels/{provider}/webhook")
async def channel_webhook(
    claw_id: str, provider: str, request: Request, background: BackgroundTasks
) -> dict:
    """
    Inbound entrypoint for a messaging provider.

    Returns 200 immediately and processes the message in the background so the
    provider (which expects a fast ack) is not blocked on the claw run.
    """
    if registry.get(claw_id) is None:
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 — tolerate empty/non-JSON pings
        payload = {}
    background.add_task(_process_inbound, claw_id, provider, payload)
    return {"status": "accepted"}


@router.delete("/{claw_id}")
async def delete_claw(claw_id: str) -> dict:
    if not registry.delete(claw_id):
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    return {"status": "deleted", "claw_id": claw_id}
