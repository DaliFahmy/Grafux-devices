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

The Grafux "claw" block calls ``/claw/create`` once (caching the returned
``claw_id`` in its output port) and ``/claw/{id}/run`` on every subsequent run.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from . import claw_runtime
from .models import (
    ClawSpec,
    ClawSummary,
    CreateClawResponse,
    RunRequest,
    RunResponse,
    ScaffoldRequest,
    ScaffoldResponse,
)
from .registry import registry

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
    result = await claw_runtime.run_claw(spec, body.task, body.memory)
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


@router.delete("/{claw_id}")
async def delete_claw(claw_id: str) -> dict:
    if not registry.delete(claw_id):
        raise HTTPException(status_code=404, detail=f"No claw with id '{claw_id}'")
    return {"status": "deleted", "claw_id": claw_id}
