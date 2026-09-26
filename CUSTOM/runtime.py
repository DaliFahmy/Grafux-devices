"""
runtime.py
Lifecycle for the ``custom`` block -- entirely EDA's, like ``CPU/runtime.py``.

Provisioning, SSH, artifact download, the registry, the reaper, the watchdog and
keep-warm/ephemeral teardown are reused unchanged; this module only registers
the one runner every manifest-defined block shares.
"""

from __future__ import annotations

from typing import Any, Dict

from EDA import runtime as eda_runtime

from . import flow
from .models import CustomRunRequest  # noqa: F401 -- also self-registers the kind's disk

# Claimed at import, which the router triggers, so it has always happened before
# a request can arrive.
eda_runtime.register_runner("custom", flow.run_custom)


def start_custom_job(eda_id: str, req: CustomRunRequest) -> Dict[str, Any]:
    """Start one run of a manifest-defined block.  Returns as soon as the job starts."""
    return eda_runtime.start_job(eda_id, req, "custom")
