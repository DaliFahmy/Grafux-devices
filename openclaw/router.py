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

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, WebSocket, WebSocketDisconnect

from . import claw_runtime, composio_tools, connections, guidance, qr
from .models import (
    ClawModel,
    ClawModelsResponse,
    ClawSpec,
    ClawSummary,
    ConfigPatchRequest,
    ConnectionSummary,
    CreateClawResponse,
    GuidanceResponse,
    InitiateConnectionRequest,
    InitiateConnectionResponse,
    QrRequest,
    QrResponse,
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


@router.get("/models", response_model=ClawModelsResponse)
async def list_claw_models() -> ClawModelsResponse:
    """
    Return the selectable Claude models (id + price tier) for the creation
    dialog's model dropdown.  Declared before ``/{claw_id}`` so "models" is not
    captured as a claw id.
    """
    return ClawModelsResponse(
        models=[
            ClawModel(
                id=model_id,
                label=info["label"],
                input_per_mtok=info["in"],
                output_per_mtok=info["out"],
            )
            for model_id, info in claw_runtime.MODEL_CATALOG.items()
        ]
    )


@router.post("/qr", response_model=QrResponse)
async def make_qr(body: QrRequest) -> QrResponse:
    """
    Render text (usually an app's authorization URL) as a scannable QR ``data:`` URI.

    Used by the connections dialog to show a scan-to-connect QR and to fill the block's
    ``qr_code`` output port.  Declared before ``/{claw_id}`` so "qr" is not captured as an id.
    """
    return QrResponse(data_uri=qr.qr_data_uri(body.text))


@router.post("/guidance", response_model=GuidanceResponse)
async def claw_guidance(spec: ClawSpec) -> GuidanceResponse:
    """
    Analyze a (possibly un-provisioned) claw spec and return setup guidance.

    Used by the create dialog to show "what to fill in next" before a claw exists, and by
    the block to refresh guidance without a run.  Pure/sync — no API calls.  Declared before
    ``/{claw_id}`` so "guidance" is not captured as a claw id.
    """
    return GuidanceResponse(**guidance.analyze(spec))


def _summarize(claw_id: str, spec: ClawSpec) -> ClawSummary:
    """Build a non-secret summary enriched with the resolved model + connected apps."""
    apps = [c.app for c in connections.parse_connections(spec) if c.enabled and c.app]
    return ClawSummary(
        claw_id=claw_id,
        name=spec.name,
        agent=spec.agent,
        model=claw_runtime._resolve_model_params(spec)["model"],
        apps=apps,
        tool_count=len(apps),  # cheap proxy: one connected app ≈ one toolset
    )


@router.get("/{claw_id}", response_model=ClawSummary)
async def get_claw(claw_id: str) -> ClawSummary:
    spec = registry.get(claw_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    return _summarize(claw_id, spec)


@router.post("/{claw_id}/run", response_model=RunResponse)
async def run_claw(claw_id: str, body: RunRequest) -> RunResponse:
    """
    Run an existing claw against a task.

    When ``session_id`` is set the run is threaded into a rolling block-level
    transcript (provider ``"block"``) so the claw remembers prior turns across Run
    clicks.  Empty ``session_id`` ⇒ a stateless run, byte-identical to before.
    """
    spec = registry.get(claw_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")

    memory = body.memory
    sid = body.session_id.strip()
    if sid:
        history = sessions.get(claw_id, "block", sid)
        if history:
            memory = f"{history}\n\n---\n\n{memory}" if memory else history

    result = await claw_runtime.run_claw(spec, body.task, memory, body.text_message)

    if sid and body.remember and result.get("status") == "ok":
        user_turn = (body.text_message or body.task or "").strip()
        if user_turn:
            sessions.append(claw_id, "block", sid, "User", user_turn)
        reply = (result.get("response") or "").strip()
        if reply:
            sessions.append(claw_id, "block", sid, "Assistant", reply)

    return RunResponse(claw_id=claw_id, **result)


@router.websocket("/{claw_id}/run/stream")
async def run_claw_stream(claw_id: str, websocket: WebSocket) -> None:
    """
    Stream a claw run over a WebSocket so the block renders the answer live.

    Wire protocol: the client sends one JSON message of run params
    ({task, memory, text_message, session_id}); the server then sends
    ``{"type":"delta","text":…}`` frames as tokens arrive, a final
    ``{"type":"done", …usage, "response":<full text>}`` frame, or
    ``{"type":"error","error":…}``.  Session memory is threaded exactly like the
    REST run route, so a streamed turn and a REST turn share one conversation.
    """
    await websocket.accept()
    spec = registry.get(claw_id)
    if spec is None:
        await websocket.send_json({"type": "error", "error": f"No claw with id '{claw_id}'"})
        await websocket.close()
        return

    try:
        params = await websocket.receive_json()
    except Exception:  # noqa: BLE001 — tolerate a bad/empty kickoff frame
        params = {}
    if not isinstance(params, dict):
        params = {}
    task = str(params.get("task", ""))
    text_message = str(params.get("text_message", ""))
    sid = str(params.get("session_id", "")).strip()
    memory = str(params.get("memory", ""))
    if sid:
        history = sessions.get(claw_id, "block", sid)
        if history:
            memory = f"{history}\n\n---\n\n{memory}" if memory else history

    chunks: list[str] = []
    final_status = ""
    try:
        async for kind, payload in claw_runtime.stream_claw(spec, task, memory, text_message):
            if kind == "delta":
                chunks.append(payload)
                await websocket.send_json({"type": "delta", "text": payload})
            elif kind == "done":
                final_status = payload.get("status", "")
                await websocket.send_json(
                    {"type": "done", "response": "".join(chunks), **payload}
                )
            elif kind == "error":
                await websocket.send_json({"type": "error", "error": payload})
    except WebSocketDisconnect:
        return  # client went away mid-stream — nothing to persist beyond what we have
    except Exception as exc:  # noqa: BLE001 — never let the socket handler crash
        logger.exception("claw stream error for %s", claw_id)
        try:
            await websocket.send_json({"type": "error", "error": claw_runtime._describe_exception(exc)})
        except Exception:  # noqa: BLE001
            pass

    if sid and final_status == "ok":
        user_turn = (text_message or task or "").strip()
        if user_turn:
            sessions.append(claw_id, "block", sid, "User", user_turn)
        reply = "".join(chunks).strip()
        if reply:
            sessions.append(claw_id, "block", sid, "Assistant", reply)

    try:
        await websocket.close()
    except Exception:  # noqa: BLE001
        pass


@router.post("/{claw_id}/config", response_model=ClawSummary)
async def patch_claw_config(claw_id: str, body: ConfigPatchRequest) -> ClawSummary:
    """
    Patch the mutable config of an existing claw in place.

    The block calls this on Run so edits to its config ports take effect WITHOUT a full
    Regenerate (which would re-provision and lose the cached claw_id + session).  This now
    includes the key ports (api_keys, credentials) too — otherwise adding e.g. a Composio key
    on an already-provisioned claw would have no effect until the user Regenerated (a footgun).
    """
    spec = _require_spec(claw_id)
    for field in ("soul", "skills", "agent", "tools_config", "connections", "api_keys", "credentials"):
        val = getattr(body, field)
        if val is not None:
            setattr(spec, field, val)
    # A connections change can alter the tool set — drop the cached MCP schemas and Composio
    # action/account lookups so the next run re-discovers them.
    connections.clear_tool_cache()
    composio_tools.clear_cache()
    registry.save(claw_id)  # flush to disk when persistence is enabled
    return _summarize(claw_id, spec)


@router.get("/{claw_id}/composio/probe")
async def composio_probe(claw_id: str) -> dict:
    """
    Diagnostic (LLM-free): report exactly what Composio returns for this claw's apps.

    For each app in the claw's connections it runs the real v3 tools-list + connected-account
    lookup with the claw's Composio key and returns counts / sample tool names / connection status /
    any error.  Open ``/claw/<id>/composio/probe`` in a browser to see why tools do or don't load.
    """
    spec = _require_spec(claw_id)
    composio_tools.clear_cache()  # always probe fresh
    key = connections.resolve_composio_key(spec)
    raw_conn = connections._clean_port(spec.connections)
    parsed_apps = [connections._clean_port(c.app) for c in connections.parse_connections(spec)]
    out: dict = {
        "claw_id": claw_id,
        "claw_name": spec.name,
        "composio_key_present": bool(key),
        "connections_raw": raw_conn[:500],
        "parsed_apps": parsed_apps,
        "apps": [],
    }
    if not raw_conn:
        out["hint"] = ("This claw's 'connections' port is EMPTY on the server. Set it to e.g. "
                       "[\"weathermap\", \"googlecalendar\", \"telegram\"] and click Regenerate "
                       "(then re-open this probe with the NEW claw_id from the block).")
        return out
    if not parsed_apps:
        out["hint"] = ("The 'connections' value could not be parsed as JSON. It must be a JSON list "
                       "of app names, e.g. [\"weathermap\", \"googlecalendar\"] — check for smart "
                       "quotes or typos.")
        return out
    if not key:
        out["hint"] = ("No Composio key found in api_keys/credentials. Add {\"composio\": \"ck_…\"} "
                       "to the api_keys port and Regenerate the block.")
        return out
    for conn in composio_tools._rest_connections(spec):
        app = connections._clean_port(conn.app)
        entry: dict = {"app": app}
        try:
            actions = await composio_tools._list_actions_for_app(key, app)
            entry["tool_count"] = len(actions)
            entry["sample_tools"] = [a["name"] for a in actions[:5]]
        except Exception as exc:  # noqa: BLE001
            entry["list_error"] = str(exc)
        try:
            user_id, account_id = await composio_tools._account_for_app(key, app, conn)
            entry["connected"] = bool(account_id)
            entry["connected_user_id"] = user_id
            entry["connected_account_id"] = account_id
        except Exception as exc:  # noqa: BLE001
            entry["account_error"] = str(exc)
        out["apps"].append(entry)
    return out


@router.delete("/{claw_id}/session/{session_id}")
async def clear_claw_session(claw_id: str, session_id: str) -> dict:
    """Drop a block conversation's transcript (the "reset chat" action)."""
    cleared = sessions.clear(claw_id, "block", session_id)
    return {"status": "cleared" if cleared else "empty", "claw_id": claw_id, "session_id": session_id}


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
    registry.save(claw_id)  # flush the mutation so it survives a restart
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
    registry.save(claw_id)
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
    registry.save(claw_id)
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
