"""
runtime.py
Orchestrates the GPU lifecycle: resolve the RunPod key, provision a pod, compile
and run C++/CUDA code on it, and tear it down.

These functions are synchronous (RunPod REST + SSH are blocking) and are called
from the router's plain ``def`` handlers, which FastAPI runs in a worker thread —
so provisioning's polling never blocks the event loop.

Like ``claw_runtime`` the entry points never raise for an operational failure:
they return a result dict with ``status="error"`` and a human-readable ``errors``
string so the block's error port always gets something useful.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from . import runpod_client
from .models import GpuRunRequest, GpuSpec
from .registry import GpuRecord, registry

logger = logging.getLogger("gpu.runtime")

# Sentinels the Grafux frontend writes into a port file when nothing is wired to
# it (see PortDataService::kEmptyPortValue and the "unconnected" literal in the
# block runner).  Treat them as an empty port.
_PLACEHOLDER_VALUES = {"empty", "unconnected"}


def _clean_port(text: Optional[str]) -> str:
    """Return the port's real value, mapping placeholder sentinels to ``""``."""
    text = (text or "").strip()
    if text.lower() in _PLACEHOLDER_VALUES:
        return ""
    return text


def _maybe_json(text: str) -> Optional[Any]:
    """Parse ``text`` as JSON, returning None when it is not valid JSON."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _resolve_runpod_key(spec: GpuSpec) -> Optional[str]:
    """
    Find the RunPod API key.

    Order: the api_keys port (bare 'rp_...' or JSON {"runpod": "..."}), then the
    credentials port (same shapes), then the RUNPOD_API_KEY env var.  The env var
    is the default so connecting "just works" without the user entering a key.
    """
    for raw in (spec.api_keys, spec.credentials):
        raw = _clean_port(raw)
        if not raw:
            continue
        parsed = _maybe_json(raw)
        if isinstance(parsed, dict):
            for key in ("runpod", "runpod_api_key", "RUNPOD_API_KEY", "api_key"):
                if parsed.get(key):
                    return str(parsed[key])
        elif raw.startswith("rp_") or raw.startswith("rpa_"):
            return raw
    return os.environ.get("RUNPOD_API_KEY") or None


def _describe_exception(exc: BaseException) -> str:
    """Render an exception for the ``errors`` port, unwrapping ExceptionGroups."""
    leaves: List[str] = []

    def walk(e: BaseException) -> None:
        sub = getattr(e, "exceptions", None)
        if sub:
            for child in sub:
                walk(child)
        else:
            leaves.append(f"{type(e).__name__}: {e}".strip())

    walk(exc)
    return " | ".join(s for s in leaves if s) or f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Provision (Regenerate) — create a pod and cache it.
# ---------------------------------------------------------------------------

def provision_gpu(spec: GpuSpec) -> Dict[str, Any]:
    """
    Provision a RunPod pod from a GpuSpec and register it.

    Returns {gpu_id, status, pod_id, gpu_model, errors}.  Never raises.
    """
    api_key = _resolve_runpod_key(spec)
    if not api_key:
        return {
            "gpu_id": "",
            "status": "error",
            "pod_id": "",
            "gpu_model": spec.gpu_model,
            "errors": (
                "No RunPod API key found. Set the gpu block's api_keys port "
                "({\"runpod\": \"rp_...\"}) or the RUNPOD_API_KEY env var on the server."
            ),
        }

    pod_id = ""
    try:
        private_pem, public_key = runpod_client.generate_keypair()
        pod_id = runpod_client.create_pod(api_key, spec, public_key)
        public_ip, ssh_port = runpod_client.wait_until_ready(api_key, pod_id)
        record = GpuRecord(
            spec=spec,
            pod_id=pod_id,
            public_ip=public_ip,
            ssh_port=ssh_port,
            private_key_pem=private_pem,
            api_key=api_key,
        )
        gpu_id = registry.create(record)
        return {
            "gpu_id": gpu_id,
            "status": "ok",
            "pod_id": pod_id,
            "gpu_model": spec.gpu_model,
            "errors": "",
        }
    except Exception as exc:  # noqa: BLE001 — surface as an error result, never 500.
        logger.warning("gpu provision failed: %s", exc)
        # If we created a pod but failed to connect, terminate it so it doesn't bill.
        if pod_id:
            runpod_client.terminate_pod(api_key, pod_id)
        return {
            "gpu_id": "",
            "status": "error",
            "pod_id": "",
            "gpu_model": spec.gpu_model,
            "errors": _describe_exception(exc),
        }


# ---------------------------------------------------------------------------
# Run — compile + execute on an already-provisioned pod.
# ---------------------------------------------------------------------------

def run_gpu(gpu_id: str, req: GpuRunRequest) -> Dict[str, Any]:
    """
    Compile and run code on the pod identified by ``gpu_id``.

    Returns {status, response, errors, warnings, benchmark(JSON string)}.
    Never raises.
    """
    record = registry.get(gpu_id)
    if record is None:
        return {
            "status": "error",
            "response": "",
            "errors": f"No GPU with id '{gpu_id}'. Press Regenerate to provision a pod.",
            "warnings": "",
            "benchmark": "",
        }
    if not record.is_running:
        return {
            "status": "error",
            "response": "",
            "errors": "GPU pod is not running. Press Regenerate to provision it.",
            "warnings": "",
            "benchmark": "",
        }

    registry.touch(gpu_id)
    code = _clean_port(req.code)
    if not code:
        return {
            "status": "error",
            "response": "",
            "errors": "The 'code' input port is empty — nothing to compile.",
            "warnings": "",
            "benchmark": "",
        }

    try:
        result = runpod_client.run_remote(
            record.public_ip,
            record.ssh_port,
            record.private_key_pem,
            source=code,
            language=req.language,
            gpu_model=record.spec.gpu_model,
            compile_flags=record.spec.compile_flags,
            args=req.args,
            timeout=int(req.timeout or 120),
        )
    except Exception as exc:  # noqa: BLE001 — connection/SSH failures → error result.
        logger.warning("gpu run failed: %s", exc)
        return {
            "status": "error",
            "response": "",
            "errors": _describe_exception(exc),
            "warnings": "",
            "benchmark": "",
        }

    benchmark = result.get("benchmark") or {}
    return {
        "status": result.get("status", "ok"),
        "response": result.get("response", ""),
        "errors": result.get("errors", ""),
        "warnings": result.get("warnings", ""),
        "benchmark": json.dumps(benchmark) if benchmark else "",
    }


# ---------------------------------------------------------------------------
# Teardown — terminate the pod and drop it from the registry.
# ---------------------------------------------------------------------------

def terminate_gpu(gpu_id: str) -> bool:
    """Terminate a pod (stopping billing) and remove it.  Returns False if unknown."""
    record = registry.get(gpu_id)
    if record is None:
        return False
    if record.pod_id and record.api_key:
        runpod_client.terminate_pod(record.api_key, record.pod_id)
    registry.delete(gpu_id)
    return True
