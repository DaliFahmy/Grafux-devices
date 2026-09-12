"""
openram_router.py
REST surface for the "openram" block: compile an SRAM macro from parameters.

    POST   /openram/create         provision a pod                    (Regenerate)
    POST   /openram/create_async   provision in the background        (Regenerate)
    POST   /openram/{id}/run       START a memory-compiler job        (Run)
    GET    /openram/{id}/status    poll phase / stage / log tail
    GET    /openram/{id}/result    the finished payload (the generated views)
    GET    /openram/instances      machine dropdown
    GET    /openram/pdks           PDK dropdown (unused here, kept uniform -- an
                                   OpenRAM technology is not an ORFS platform;
                                   the technology arrives on the tech_name port)
    GET    /openram                list
    DELETE /openram/{id}           cancel + terminate                 (Stop)

Every endpoint is built by ``router_base.make_router`` -- see that module for why
the EDA block types share one implementation but keep separate prefixes.
"""

from __future__ import annotations

from .models import OpenRamRunRequest
from .router_base import make_router
from .runtime import start_openram_job

router = make_router("openram", OpenRamRunRequest, start_openram_job)
