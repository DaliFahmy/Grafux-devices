"""
openram_router.py
REST surface for the "openram" block: compile an SRAM macro from parameters.

    POST   /openram/create         provision a pod                    (Regenerate)
    POST   /openram/create_async   provision in the background        (Regenerate)
    POST   /openram/{id}/run       START a memory-compiler job        (Run)
    GET    /openram/{id}/status    poll phase / stage / log tail
    GET    /openram/{id}/result    the finished payload (the generated views)
    GET    /openram/instances      machine dropdown
    GET    /openram/pdks           the TECHNOLOGIES this image ships (not ORFS
                                   platforms -- an OpenRAM technology is a
                                   different thing; it arrives on tech_name)
    GET    /openram                list
    DELETE /openram/{id}           cancel + terminate                 (Stop)

Every endpoint is built by ``router_base.make_router`` -- see that module for why
the EDA block types share one implementation but keep separate prefixes.
"""

from __future__ import annotations

from .models import DEFAULT_OPENRAM_TECH, OPENRAM_TECH_CHOICES, OpenRamRunRequest
from .router_base import make_router
from .runtime import start_openram_job

router = make_router(
    "openram",
    OpenRamRunRequest,
    start_openram_job,
    # The shared /pdks route otherwise answers with the ORFS platform list,
    # which is a set of names this compiler has never heard of.
    pdk_choices=OPENRAM_TECH_CHOICES,
    default_pdk=DEFAULT_OPENRAM_TECH,
)
