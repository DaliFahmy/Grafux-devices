"""
router.py
REST surface for the Generator (the app's "Generator" panel).

    GET    /generator/agents                       agents, their models, the modes
    POST   /generator/sessions                     start: the first agent turn (a GitHub Actions job)
    GET    /generator/sessions/{id}                summary (state, turns, cost)
    GET    /generator/sessions/{id}/events?after=N events newer than N (poll every ~1.5 s)
    POST   /generator/sessions/{id}/message        a follow-up turn {text, mode}
    GET    /generator/sessions/{id}/result         the verified block: {manifest, image, files}
    POST   /generator/sessions/{id}/stop           stop (cancels the running Actions job)
    DELETE /generator/sessions/{id}                forget the session
    POST   /generator/sessions/{id}/ingest         the agent job's live output (token-checked)

No RunPod pod is ever rented here: pods come only from a block's Regenerate.

Polling rather than a WebSocket on purpose: the WASM build must not do blocking
work on a socket callback (see the WASM WebSocket asyncify note), and every
other long-running block here already speaks create + poll.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from . import agents
from .actions import ActionsAgentRunner
from .builder import BuilderNotConfigured, GitHubBuilder
from .models import (CreateSessionRequest, IngestRequest, MessageRequest, missing_configuration,
                     resolve_agent_key)
from .session import GeneratorSession, sessions

router = APIRouter(prefix="/generator", tags=["generator"])

# Seams the tests replace.
def configuration_error() -> str:
    return missing_configuration()


def make_builder() -> Optional[GitHubBuilder]:
    try:
        return GitHubBuilder.from_env()
    except BuilderNotConfigured:
        return None


def make_runner(builder: GitHubBuilder) -> Any:
    return ActionsAgentRunner(builder)


def _get(sid: str) -> GeneratorSession:
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(status_code=404, detail=f"no generator session '{sid}'")
    return s


@router.get("/agents")
def list_agents() -> Dict[str, Any]:
    return {
        "agents": [
            # server_key: Grafux holds a key for this agent, so the user's is optional.
            # A boolean only -- the key itself never leaves the server this way.
            {"id": "claude_code", "label": "Claude Code", "models": agents.MODELS["claude_code"],
             "key": "anthropic_api_key", "server_key": bool(resolve_agent_key("anthropic")[0])},
            {"id": "codex", "label": "Codex (experimental)", "models": agents.MODELS["codex"],
             "key": "openai_api_key", "server_key": bool(resolve_agent_key("openai")[0])},
        ],
        "modes": list(agents.MODES),
        "builder_configured": make_builder() is not None,
        # Everything a session needs (builds repo, token, wrap key); the panel
        # shows `setup` instead of letting Send fail.
        "configured": not configuration_error(),
        "setup": configuration_error(),
    }


@router.post("/sessions")
def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    if req.agent not in agents.AGENTS:
        raise HTTPException(status_code=422, detail=f"agent must be one of {', '.join(agents.AGENTS)}")
    if req.mode not in agents.MODES:
        raise HTTPException(status_code=422, detail="mode must be plan or edit")
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="describe what the block should do")
    problem = configuration_error()
    builder = make_builder()
    if problem or builder is None:
        raise HTTPException(status_code=503, detail=problem or "the image builder is not configured")
    try:
        s = GeneratorSession(req, runner=make_runner(builder), builder=builder)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sessions.add(s)
    s.start(req.mode)
    return s.summary()


@router.get("/sessions/{sid}")
def get_session(sid: str) -> Dict[str, Any]:
    return _get(sid).summary()


@router.get("/sessions/{sid}/events")
def get_events(sid: str, after: int = Query(0, ge=0)) -> Dict[str, Any]:
    s = _get(sid)
    events = s.events_after(after)
    return {"session_id": sid, "state": s.state, "busy": s.busy, "events": events,
            "next": events[-1]["seq"] if events else after, "verified": bool(s.result)}


@router.post("/sessions/{sid}/message")
def post_message(sid: str, body: MessageRequest) -> Dict[str, Any]:
    s = _get(sid)
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="empty message")
    try:
        s.message(body.text, body.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return s.summary()


@router.get("/sessions/{sid}/result")
def get_result(sid: str) -> Dict[str, Any]:
    s = _get(sid)
    if not s.result:
        raise HTTPException(status_code=409, detail=f"no verified block yet (state: {s.state})")
    return s.result


@router.post("/sessions/{sid}/stop")
def stop_session(sid: str) -> Dict[str, Any]:
    s = _get(sid)
    s.stop()
    return s.summary()


@router.delete("/sessions/{sid}")
def delete_session(sid: str) -> Dict[str, Any]:
    s = sessions.remove(sid)
    if s is None:
        raise HTTPException(status_code=404, detail=f"no generator session '{sid}'")
    if s.state != "stopped":
        s.stop()
    return {"deleted": sid}


@router.post("/sessions/{sid}/ingest")
def ingest(sid: str, body: IngestRequest) -> Dict[str, Any]:
    """Live output from the agent job's forwarder (see builds-repo agent.yml)."""
    s = _get(sid)
    try:
        accepted = s.ingest(body.token, body.turn, body.lines)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="bad token") from exc
    return {"accepted": accepted}
