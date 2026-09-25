"""
runtime.py
Lifecycle for the ``cpu`` block — which is almost entirely EDA's.

Provisioning, the SSH transport, artifact download, the registry, the idle
reaper, the run watchdog and every cost-safety mechanism are reused from
``EDA.runtime`` unchanged; this module only registers what is CPU-specific
(``flow.run_cpu``) and exposes the ``start_*_job`` entry point the router wants.

See ``CPU/__init__.py`` for why that reuse is a deliberate choice rather than an
accident of convenience.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from EDA import pod_client
from EDA import runtime as eda_runtime
from EDA.registry import registry

from . import flow
from .models import CpuRunRequest  # noqa: F401 — also self-registers the image

logger = logging.getLogger("cpu.runtime")

# Claim the run behaviour for this kind. Done at import, which the router
# triggers, so it has always happened before a request can arrive.
eda_runtime.register_runner("cpu", flow.run_cpu)


def start_cpu_job(eda_id: str, req: CpuRunRequest) -> Dict[str, Any]:
    """Start a post-silicon verification run.  Returns as soon as the job starts."""
    record = registry.get(eda_id)
    if record is not None:
        req._machine_note = pod_client.instance_type_note(record.spec.instance_type)
    return eda_runtime.start_job(eda_id, req, "cpu")
