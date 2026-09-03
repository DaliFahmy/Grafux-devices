"""
router_base.py
Builds the REST surface shared by the three EDA block types.

``/verilator``, ``/yosys`` and ``/openroad`` expose an identical set of endpoints
that differ only in the run-request model and which ``start_*_job`` they call, so
one factory produces all three.  They stay separate *prefixes* rather than one
``/eda/{kind}`` route because the Qt client and the orchestrator address block
types by URL, and because a wrong-kind request should 404 at the router rather
than fail deep inside a tool.

Handlers are plain ``def`` (not ``async def``) so FastAPI runs them in a worker
thread — provisioning's blocking poll never stalls the event loop.  Note that
``/create`` blocks for the whole provision (minutes on a cold image) whereas
``/create_async`` returns at once; the Qt client uses the async form and polls.
"""


# NOTE: deliberately NO ``from __future__ import annotations`` here.
#
# The run endpoint's body type is a *runtime* value — ``make_router`` is handed a
# different Pydantic model for each kind — so its annotation must evaluate to the
# real class at def time. With postponed evaluation the annotation stays the
# string "run_request_model", FastAPI cannot resolve it to a model, and it
# silently degrades the request body into a required *query* parameter: every
# POST /{kind}/{id}/run then fails with 422 "Field required: query.body".
# That is the one endpoint the whole client polling flow depends on.

import logging
from typing import Any, Callable, Dict, Type

from fastapi import APIRouter, HTTPException

from . import pod_client, runtime
from .models import (
    PDK_CHOICES,
    CreateEdaResponse,
    DEFAULT_PDK,
    EdaInstance,
    EdaInstancesResponse,
    EdaPdksResponse,
    EdaResultResponse,
    EdaRunAccepted,
    EdaSpec,
    EdaStatusResponse,
    EdaSummary,
    DEFAULT_IMAGE,
    disk_for_kind,
    image_for_kind,
)
from .registry import registry

logger = logging.getLogger("eda.router")

# The EdaSpec default; anything else is a deliberate choice by the caller.
_DEFAULT_DISK_GB = EdaSpec.model_fields["container_disk_gb"].default


def _coerce_kind(spec: EdaSpec, kind: str) -> EdaSpec:
    """
    Force the spec's kind to match the router it arrived on, and give it the image
    and disk that kind actually needs.

    The block sends its whole config blob and may not set ``kind`` at all; taking
    it from the URL means a yosys block can never accidentally provision itself as
    an openroad one and then fail confusingly at run time.

    Image and disk are only filled in when the caller left them at the model
    default -- an explicit value on the block's ``image`` port is the user pinning
    a toolchain, and silently replacing it would be the most confusing bug in this
    file.
    """
    update: dict = {}
    if spec.kind != kind:
        update["kind"] = kind
    if spec.image == DEFAULT_IMAGE:
        wanted = image_for_kind(kind)
        if wanted != spec.image:
            update["image"] = wanted
    if spec.container_disk_gb == _DEFAULT_DISK_GB:
        wanted_disk = disk_for_kind(kind)
        if wanted_disk != spec.container_disk_gb:
            update["container_disk_gb"] = wanted_disk
    return spec.model_copy(update=update) if update else spec


def make_router(
    kind: str,
    run_request_model: Type,
    start_job: Callable[[str, Any], Dict[str, Any]],
) -> APIRouter:
    """Build the REST router for one EDA block type."""
    router = APIRouter(prefix=f"/{kind}", tags=[kind])

    @router.post("/create", response_model=CreateEdaResponse)
    def create(spec: EdaSpec) -> CreateEdaResponse:
        """Provision a pod from the block's config ports (Regenerate, blocking)."""
        return CreateEdaResponse(**runtime.provision_eda(_coerce_kind(spec, kind)))

    @router.post("/create_async", response_model=CreateEdaResponse)
    def create_async(spec: EdaSpec) -> CreateEdaResponse:
        """
        Begin provisioning in the background and return an eda_id immediately.

        The block polls ``GET /{kind}/{id}/status`` until the phase is ``ready``
        (or ``error``), so a multi-minute image pull shows live phases instead of
        a blocking wait.
        """
        return CreateEdaResponse(**runtime.provision_eda_async(_coerce_kind(spec, kind)))

    # Declared before "/{eda_id}" so the literal paths are not swallowed by the
    # path parameter — FastAPI matches in declaration order.
    @router.get("/instances", response_model=EdaInstancesResponse)
    def list_instances() -> EdaInstancesResponse:
        """Selectable machine types (id + label + advisory $/hr) for the dropdown."""
        return EdaInstancesResponse(
            instances=[
                EdaInstance(
                    id=i["id"], label=i["label"],
                    usd_per_hr=float(i.get("usd_per_hr", 0.0)),
                    compute_type=i.get("compute_type", "CPU"),
                )
                for i in pod_client.list_instances()
            ]
        )

    @router.get("/pdks", response_model=EdaPdksResponse)
    def list_pdks() -> EdaPdksResponse:
        """The process design kits the EDA image ships."""
        return EdaPdksResponse(pdks=list(PDK_CHOICES), default=DEFAULT_PDK)

    @router.get("", response_model=list[EdaSummary])
    def list_all() -> list[EdaSummary]:
        """List provisioned pods (ids + non-secret summary incl. phase/stage/cost)."""
        return [s for s in registry.list() if s.kind == kind]

    @router.post("/{eda_id}/run", response_model=EdaRunAccepted)
    def run(eda_id: str, body: run_request_model) -> EdaRunAccepted:  # type: ignore[valid-type]
        """
        START a run and return immediately — this does NOT wait for the result.

        An OpenROAD route outlives any HTTP request, so the job runs in a
        background thread; poll ``/status`` until ``done``, then read ``/result``.
        """
        return EdaRunAccepted(**start_job(eda_id, body))

    @router.get("/{eda_id}/status", response_model=EdaStatusResponse)
    def status(eda_id: str, live: bool = False) -> EdaStatusResponse:
        """
        Live status: provisioning phase, tool stage, log tail, elapsed, cost.

        ``?live=1`` does one RunPod lookup to refine the phase and hourly rate from
        the cloud's own view; without it only the cached record is reported, so a
        fast poll loop cannot hammer the RunPod API.
        """
        result = runtime.eda_status(eda_id, live=live)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No {kind} with id '{eda_id}'")
        return EdaStatusResponse(**result)

    @router.get("/{eda_id}/result", response_model=EdaResultResponse)
    def result(eda_id: str) -> EdaResultResponse:
        """The finished run payload: output ports, artifacts, errors, log."""
        payload = runtime.eda_result(eda_id)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No {kind} with id '{eda_id}'. Ephemeral runs release the pod as "
                    f"soon as the result is delivered, so fetch /result once /status "
                    f"reports done."
                ),
            )
        return EdaResultResponse(**payload)

    @router.get("/{eda_id}", response_model=EdaSummary)
    def get_one(eda_id: str) -> EdaSummary:
        summary = registry.summary(eda_id)
        if summary is None or summary.kind != kind:
            raise HTTPException(status_code=404, detail=f"No {kind} with id '{eda_id}'")
        return summary

    @router.delete("/{eda_id}")
    def delete(eda_id: str) -> dict:
        """Cancel any in-flight run and terminate the pod (Stop / block delete)."""
        if not runtime.cancel_eda(eda_id):
            raise HTTPException(status_code=404, detail=f"No {kind} with id '{eda_id}'")
        return {"status": "deleted", "eda_id": eda_id}

    return router
