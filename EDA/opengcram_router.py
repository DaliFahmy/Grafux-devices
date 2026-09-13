"""
opengcram_router.py
REST surface for the "opengcram" block: compile a GAIN-CELL memory macro with
OpenGCRAM (github.com/xxwang1/OpenGCRAM), the gain-cell fork of OpenRAM.

    POST   /opengcram/create         provision a pod                  (Regenerate)
    POST   /opengcram/create_async   provision in the background      (Regenerate)
    POST   /opengcram/{id}/run       START a gain-cell compile job    (Run)
    GET    /opengcram/{id}/status    poll phase / stage / log tail
    GET    /opengcram/{id}/result    the finished payload (the generated views)
    GET    /opengcram/instances      machine dropdown
    GET    /opengcram/pdks           the technology NAMES a block can target --
                                     not what the image ships: no public
                                     technology has a gain cell, so a real one
                                     arrives on the tech_archive port
    GET    /opengcram                list
    DELETE /opengcram/{id}           cancel + terminate               (Stop)

Every endpoint is built by ``router_base.make_router`` -- see that module for why
the EDA block types share one implementation but keep separate prefixes.
"""

from __future__ import annotations

from .models import DEFAULT_OPENGCRAM_TECH, OPENGCRAM_TECH_CHOICES, OpenGcRamRunRequest
from .router_base import make_router
from .runtime import start_opengcram_job

router = make_router(
    "opengcram",
    OpenGcRamRunRequest,
    start_opengcram_job,
    pdk_choices=OPENGCRAM_TECH_CHOICES,
    default_pdk=DEFAULT_OPENGCRAM_TECH,
)
