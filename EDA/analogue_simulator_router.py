"""
analogue_simulator_router.py
REST surface for the "analogue_simulator" block: simulate a transistor-level
SPICE netlist with ngspice on a RunPod CPU pod.

    POST   /analogue_simulator/create         provision a pod                (Regenerate)
    POST   /analogue_simulator/create_async   provision in the background    (Regenerate)
    POST   /analogue_simulator/{id}/run       START a simulation job         (Run)
    GET    /analogue_simulator/{id}/status    poll phase / stage / log tail
    GET    /analogue_simulator/{id}/result    the finished payload
    GET    /analogue_simulator/instances      machine dropdown
    GET    /analogue_simulator/pdks           the device-model sets the image carries
    GET    /analogue_simulator                list
    DELETE /analogue_simulator/{id}           cancel + terminate              (Stop)

Every endpoint is built by ``router_base.make_router`` -- see that module for why
the EDA block types share one implementation but keep separate prefixes.
"""

from __future__ import annotations

from .models import DEFAULT_NGSPICE_PDK, NGSPICE_PDK_CHOICES, AnalogueSimRunRequest
from .router_base import make_router
from .runtime import start_analogue_simulator_job

router = make_router(
    "analogue_simulator",
    AnalogueSimRunRequest,
    start_analogue_simulator_job,
    pdk_choices=NGSPICE_PDK_CHOICES,
    default_pdk=DEFAULT_NGSPICE_PDK,
)
