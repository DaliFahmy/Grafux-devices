"""
models.py
Request schema for the ``custom`` runtime.

A custom block's own ports are not known to this server in advance -- they are
whatever its manifest declares -- so the run request carries the manifest and a
flat ``inputs`` map instead of one field per port:

    manifest   -> CustomRunRequest.manifest   the block's manifest (JSON text)
    <any port> -> CustomRunRequest.inputs     {port_name: text} for the manifest's inputs
    files      -> _RunBase.input_files        extra files staged before the run
    timeout    -> _RunBase.timeout            0 / unset = the manifest's runtime.timeout_s

The pod itself (image, compute, instance_type) is defined at CREATE time by the
shared ``EdaSpec``, exactly like every other pod-backed block.  The app fills the
spec's ``image`` from ``manifest.runtime.image`` when the block's ``image`` port
is empty; ``require_image`` below refuses a create that arrives without one,
because the EDA fallback would be the 60 GB ORFS image -- a pod that boots fine
and then has no ``run.sh``.
"""

from __future__ import annotations

from typing import Dict

from fastapi import HTTPException
from pydantic import Field

from EDA.models import DEFAULT_IMAGE, EdaSpec, _RunBase, register_kind_image

# Custom images are single-purpose and carry only their own tool; 20 GB is the
# default when the block's create request does not say, and the manifest's
# runtime.disk_gb reaches the spec through the app.
CUSTOM_DISK_GB = 20

# No default image: a custom block's image is its manifest's, never a fallback.
# Registering DEFAULT_IMAGE itself makes ``_coerce_kind`` leave a missing image
# untouched, for ``require_image`` to refuse rather than substitute something
# plausible.
register_kind_image("custom", lambda: DEFAULT_IMAGE, CUSTOM_DISK_GB)

CUSTOM_PDK_CHOICES: tuple = ()
CUSTOM_DEFAULT_PDK = ""


def require_image(spec: EdaSpec) -> EdaSpec:
    """Refuse a custom pod with no image of its own (see the module docstring)."""
    image = (spec.image or "").strip()
    if not image or image == DEFAULT_IMAGE:
        raise HTTPException(
            status_code=422,
            detail=(
                "A custom block needs its own image: set the block's `image` port, "
                "or regenerate the block so its manifest's runtime.image is used."
            ),
        )
    return spec


class CustomRunRequest(_RunBase):
    """Live inputs for one run of a manifest-defined block."""

    timeout: int = Field(0, description="Per-run limit in seconds; 0 = the manifest's runtime.timeout_s.")
    manifest: str = Field("", description="The block's manifest, as JSON text.")
    inputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Values of the manifest's input ports, by port name. Missing = the port's default.",
    )
