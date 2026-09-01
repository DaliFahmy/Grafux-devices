"""
openroad_router.py
REST surface for the "openroad" block: floorplan -> place -> CTS -> route -> GDS.

    POST   /openroad/create         provision a pod                   (Regenerate)
    POST   /openroad/create_async   provision in the background       (Regenerate)
    POST   /openroad/{id}/run       START a physical-design job       (Run)
    GET    /openroad/{id}/status    poll phase / ORFS stage / log tail
    GET    /openroad/{id}/result    the finished payload (GDS, DEF, metrics)
    GET    /openroad/instances      machine dropdown
    GET    /openroad/pdks           PDK / ORFS platform dropdown
    GET    /openroad                list
    DELETE /openroad/{id}           cancel + terminate                (Stop)

This is the kind the asynchronous job protocol exists for: a route runs far
longer than any HTTP request survives, so /run starts a thread and the block
polls /status through the six ORFS stages.  See ``router_base.make_router``.
"""

from __future__ import annotations

from .models import OpenRoadRunRequest
from .router_base import make_router
from .runtime import start_openroad_job

router = make_router("openroad", OpenRoadRunRequest, start_openroad_job)
