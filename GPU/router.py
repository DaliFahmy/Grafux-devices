"""
router.py
FastAPI router exposing the GPU (cloud-GPU) runtime over REST.

Mounted by ``devices_server.py`` with ``app.include_router(router)``.  All routes
are under the ``/gpu`` prefix:

    POST   /gpu/create        -> provision a pod, returns {gpu_id}   (Regenerate)
    POST   /gpu/{id}/run      -> compile + run code on the pod        (Run)
    GET    /gpu               -> list provisioned pods (no secrets)
    GET    /gpu/{id}          -> a pod's non-secret summary
    GET    /gpu/models        -> selectable GPU types for the dialog dropdown
    DELETE /gpu/{id}          -> terminate the pod (block delete / "Stop GPU")

The Grafux "gpu" block calls ``/gpu/create`` on Regenerate (caching the returned
``gpu_id`` in its output port) and ``/gpu/{id}/run`` on every Run.

The handlers are plain ``def`` (not ``async def``) so FastAPI runs them in a
worker thread — provisioning's blocking poll never stalls the event loop.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from . import runpod_client, runtime
from .models import (
    CreateGpuResponse,
    GpuModel,
    GpuModelsResponse,
    GpuRunRequest,
    GpuRunResponse,
    GpuSpec,
    GpuSummary,
)
from .registry import registry

logger = logging.getLogger("gpu.router")

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.post("/create", response_model=CreateGpuResponse)
def create_gpu(spec: GpuSpec) -> CreateGpuResponse:
    """Provision a pod from the block's config ports (Regenerate)."""
    result = runtime.provision_gpu(spec)
    return CreateGpuResponse(**result)


@router.post("/{gpu_id}/run", response_model=GpuRunResponse)
def run_gpu(gpu_id: str, body: GpuRunRequest) -> GpuRunResponse:
    """Compile + run code on an already-provisioned pod (Run)."""
    result = runtime.run_gpu(gpu_id, body)
    return GpuRunResponse(gpu_id=gpu_id, **result)


@router.get("/models", response_model=GpuModelsResponse)
def list_gpu_models() -> GpuModelsResponse:
    """Return the selectable GPU types for the creation-dialog dropdown."""
    return GpuModelsResponse(
        models=[GpuModel(id=g["id"], label=g["label"]) for g in runpod_client.list_gpu_types()]
    )


@router.get("", response_model=list[GpuSummary])
def list_gpus() -> list[GpuSummary]:
    """List provisioned pods (ids + non-secret summary)."""
    return registry.list()


@router.get("/{gpu_id}", response_model=GpuSummary)
def get_gpu(gpu_id: str) -> GpuSummary:
    record = registry.get(gpu_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No gpu with id '{gpu_id}'")
    return GpuSummary(
        gpu_id=gpu_id,
        name=record.spec.name,
        gpu_model=record.spec.gpu_model,
        pod_id=record.pod_id,
        pod_status="running" if record.is_running else "pending",
    )


@router.delete("/{gpu_id}")
def delete_gpu(gpu_id: str) -> dict:
    """Terminate the pod (stopping billing) and remove it."""
    if not runtime.terminate_gpu(gpu_id):
        raise HTTPException(status_code=404, detail=f"No gpu with id '{gpu_id}'")
    return {"status": "deleted", "gpu_id": gpu_id}
