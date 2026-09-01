"""
verilator_router.py
REST surface for the "verilator" block: simulate or lint a design.

    POST   /verilator/create         provision a pod                  (Regenerate)
    POST   /verilator/create_async   provision in the background      (Regenerate)
    POST   /verilator/{id}/run       START a lint/sim job             (Run)
    GET    /verilator/{id}/status    poll phase / stage / log tail
    GET    /verilator/{id}/result    the finished payload
    GET    /verilator/instances      machine dropdown
    GET    /verilator/pdks           PDK dropdown (unused here, kept uniform)
    GET    /verilator                list
    DELETE /verilator/{id}           cancel + terminate               (Stop)

Every endpoint is built by ``router_base.make_router`` — see that module for why
the three EDA block types share one implementation but keep separate prefixes.
"""

from __future__ import annotations

from .models import VerilatorRunRequest
from .router_base import make_router
from .runtime import start_verilator_job

router = make_router("verilator", VerilatorRunRequest, start_verilator_job)
