"""
router.py
REST surface for the ``custom`` block: any tool, described by a manifest.

    POST   /custom/create         provision a pod from the manifest's image  (Regenerate)
    POST   /custom/create_async   the same, in the background
    POST   /custom/{id}/run       START a run                                (Run)
    GET    /custom/{id}/status    poll phase / stage / log tail
    GET    /custom/{id}/result    the finished payload
    GET    /custom/instances      machine dropdown
    GET    /custom                list
    DELETE /custom/{id}           cancel + terminate                         (Stop)

    GET    /custom/manifest/schema     the manifest JSON Schema
    POST   /custom/manifest/validate   {manifest} -> {ok, errors, summary}

The lifecycle endpoints come from ``EDA.router_base.make_router`` so the Qt
client's kind-parameterised EDA methods drive a custom block with no new code.
The two manifest endpoints are declared on the SAME router before it is
returned; FastAPI matches in declaration order, and "/manifest/..." has two
segments so it can never be taken for "/{eda_id}".
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from CPU.models import list_cpu_instances
from EDA.router_base import make_router

from .manifest import ManifestError, json_schema, manifest_summary, parse_manifest
from .models import CUSTOM_DEFAULT_PDK, CUSTOM_PDK_CHOICES, CustomRunRequest, require_image
from .runtime import start_custom_job

router = make_router(
    "custom",
    CustomRunRequest,
    start_custom_job,
    pdk_choices=CUSTOM_PDK_CHOICES,
    default_pdk=CUSTOM_DEFAULT_PDK,
    instances=list_cpu_instances,
    check_spec=require_image,
)


class ValidateBody(BaseModel):
    manifest: Any = None


@router.get("/manifest/schema")
def manifest_schema() -> Dict[str, Any]:
    """The manifest v1 JSON Schema."""
    return json_schema()


@router.post("/manifest/validate")
def manifest_validate(body: ValidateBody) -> Dict[str, Any]:
    """Validate a manifest; on success also return the block's full port lists."""
    try:
        m = parse_manifest(body.manifest)
    except ManifestError as exc:
        return {"ok": False, "errors": str(exc), "summary": None}
    return {"ok": True, "errors": "", "summary": manifest_summary(m)}
