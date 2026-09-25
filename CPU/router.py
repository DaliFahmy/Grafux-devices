"""
router.py
REST surface for the "cpu" block: run a verification case on real silicon.

    POST   /cpu/create         provision a pod                  (Regenerate)
    POST   /cpu/create_async   provision in the background      (Regenerate)
    POST   /cpu/{id}/run       START a verification run         (Run)
    GET    /cpu/{id}/status    poll phase / stage / log tail
    GET    /cpu/{id}/result    the finished payload
    GET    /cpu/instances      machine dropdown (every RunPod CPU flavour)
    GET    /cpu/pdks           empty here — no PDK applies to running a program
    GET    /cpu                list
    DELETE /cpu/{id}           cancel + terminate               (Stop)

Every endpoint is built by ``EDA.router_base.make_router``: this block speaks the
same lifecycle as the EDA kinds, so it gets the same surface rather than a
second, subtly different one for the Qt client to special-case.  (That client
addresses all of them through one kind-parameterised set of methods, so a new
prefix costs it nothing.)

``from __future__ import annotations`` is safe HERE and would not be in
``router_base`` itself — that module annotates an endpoint with a runtime-valued
model, and postponed evaluation would silently turn the request body into a query
parameter.  Nothing in this file has that shape.
"""

from __future__ import annotations

from EDA.router_base import make_router

from .models import CPU_DEFAULT_PDK, CPU_PDK_CHOICES, CpuRunRequest, list_cpu_instances
from .runtime import start_cpu_job

router = make_router(
    "cpu",
    CpuRunRequest,
    start_cpu_job,
    pdk_choices=CPU_PDK_CHOICES,
    default_pdk=CPU_DEFAULT_PDK,
    instances=list_cpu_instances,
)
