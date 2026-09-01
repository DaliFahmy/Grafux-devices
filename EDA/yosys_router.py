"""
yosys_router.py
REST surface for the "yosys" block: synthesize RTL into a gate-level netlist.

    POST   /yosys/create         provision a pod                      (Regenerate)
    POST   /yosys/create_async   provision in the background          (Regenerate)
    POST   /yosys/{id}/run       START a synthesis job                (Run)
    GET    /yosys/{id}/status    poll phase / stage / log tail
    GET    /yosys/{id}/result    the finished payload (netlist + stats)
    GET    /yosys/instances      machine dropdown
    GET    /yosys/pdks           PDK dropdown (which liberty to map onto)
    GET    /yosys                list
    DELETE /yosys/{id}           cancel + terminate                   (Stop)

Every endpoint is built by ``router_base.make_router``.
"""

from __future__ import annotations

from .models import YosysRunRequest
from .router_base import make_router
from .runtime import start_yosys_job

router = make_router("yosys", YosysRunRequest, start_yosys_job)
