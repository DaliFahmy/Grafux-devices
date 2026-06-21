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
import time
from typing import Any, Dict, List, Optional

from . import runpod_client
from .models import GpuRunRequest, GpuSpec
from .registry import GpuRecord, registry

logger = logging.getLogger("gpu.runtime")

# Sentinels the Grafux frontend writes into a port file when nothing is wired to
# it (see PortDataService::kEmptyPortValue and the "unconnected" literal in the
# block runner).  Treat them as an empty port.
_PLACEHOLDER_VALUES = {"empty", "unconnected"}

# Ephemeral lifecycle (default ON): terminate the pod the moment a run finishes —
# on success AND on every failure path — so a GPU only costs money while it is
# actually compiling/running.  The run then reports gpu_id="" back to the block,
# which clears its cached gpu_id port so the *next* Run transparently provisions a
# fresh pod (the frontend already auto-provisions on an empty gpu_id).  Idle cost
# is therefore $0.  Set ``GPU_EPHEMERAL=0`` to restore the warm create-once /
# run-many model (pods stay up between runs and are only freed by the idle reaper).
_EPHEMERAL = os.environ.get("GPU_EPHEMERAL", "1").lower() not in ("0", "false", "no")

# Provisioning is best-effort on RunPod: a create can hit transient capacity
# scarcity ("no instances available"), and even with supportPublicIp the pod can
# land on a machine that has no public-IP networking (RUNNING forever with an empty
# publicIp).  Both are *placement* problems a fresh pod usually escapes, so we
# terminate and retry up to this many times before giving up.
_PROVISION_ATTEMPTS = max(1, int(os.environ.get("GPU_PROVISION_ATTEMPTS", "3") or "3"))

# Seconds to wait between retryable provisioning attempts (lets scarce capacity
# free up; a no-public-IP machine is escaped immediately on the next create).
_RETRY_BACKOFF_S = float(os.environ.get("GPU_RETRY_BACKOFF_S", "3") or "3")


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

    last_err = ""
    for attempt in range(1, _PROVISION_ATTEMPTS + 1):
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
            if attempt > 1:
                logger.info(
                    "gpu provisioned on attempt %d/%d", attempt, _PROVISION_ATTEMPTS
                )
            return {
                "gpu_id": gpu_id,
                "status": "ok",
                "pod_id": pod_id,
                "gpu_model": spec.gpu_model,
                "errors": "",
            }
        except runpod_client.ProvisionError as exc:
            # Retryable placement failure (no capacity, or a machine with no public
            # IP).  Free any pod we created so it can't bill, then try a fresh
            # placement — which usually lands on a different, working machine.
            last_err = _describe_exception(exc)
            logger.warning(
                "gpu provision attempt %d/%d failed (retryable): %s",
                attempt, _PROVISION_ATTEMPTS, exc,
            )
            if pod_id:
                runpod_client.terminate_pod(api_key, pod_id)
            if attempt < _PROVISION_ATTEMPTS and _RETRY_BACKOFF_S > 0:
                time.sleep(_RETRY_BACKOFF_S)
            continue
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

    # Every attempt hit a retryable placement failure — surface the last cause plus
    # the levers the user can pull (different GPU, or the more-available SECURE tier).
    return {
        "gpu_id": "",
        "status": "error",
        "pod_id": "",
        "gpu_model": spec.gpu_model,
        "errors": (
            f"GPU provisioning failed after {_PROVISION_ATTEMPTS} attempts. {last_err} "
            "Try a different 'gpu_model', set 'cloud_type' to 'SECURE', or Regenerate "
            "again in a few minutes."
        ),
    }


# ---------------------------------------------------------------------------
# Run — compile + execute on an already-provisioned pod.
# ---------------------------------------------------------------------------

def _teardown_after_run(gpu_id: str, record: GpuRecord) -> None:
    """
    Terminate a pod and drop it from the registry once a run is done.

    Best-effort and never raises — teardown must not turn a successful run into a
    failure.  Called from ``run_gpu``'s ``finally`` in ephemeral mode so a pod
    only ever bills for the duration of an actual run.
    """
    try:
        if record.pod_id and record.api_key:
            runpod_client.terminate_pod(record.api_key, record.pod_id)
    except Exception as exc:  # noqa: BLE001 — teardown is best-effort.
        logger.warning("gpu post-run teardown failed for %s: %s", gpu_id, exc)
    finally:
        registry.delete(gpu_id)


def run_gpu(gpu_id: str, req: GpuRunRequest) -> Dict[str, Any]:
    """
    Compile and run code on the pod identified by ``gpu_id``.

    Returns {gpu_id, status, response, errors, warnings, benchmark(JSON string)}.
    Never raises.

    In the default ephemeral mode (``GPU_EPHEMERAL``) the pod is terminated as soon
    as the run finishes — on success or failure — and the returned ``gpu_id`` is ""
    so the block clears its cached pod id and re-provisions on the next Run.
    """
    record = registry.get(gpu_id)
    if record is None:
        # Unknown / already-torn-down pod: report an empty gpu_id so the next Run
        # provisions a fresh pod instead of repeating this error.
        return {
            "gpu_id": "",
            "status": "error",
            "response": "",
            "errors": (
                f"No GPU with id '{gpu_id}'. A fresh pod will be provisioned on the "
                f"next run."
            ),
            "warnings": "",
            "benchmark": "",
        }

    # In ephemeral mode the pod is gone after this call, so report "" to clear the
    # block's cached gpu_id; otherwise keep reporting the live id (warm reuse).
    report_id = "" if _EPHEMERAL else gpu_id
    try:
        if not record.is_running:
            return {
                "gpu_id": report_id,
                "status": "error",
                "response": "",
                "errors": "GPU pod is not running — a fresh pod will be provisioned on the next run.",
                "warnings": "",
                "benchmark": "",
            }

        registry.touch(gpu_id)
        code = _clean_port(req.code)
        if not code:
            return {
                "gpu_id": report_id,
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
                "gpu_id": report_id,
                "status": "error",
                "response": "",
                "errors": _describe_exception(exc),
                "warnings": "",
                "benchmark": "",
            }

        benchmark = result.get("benchmark") or {}
        return {
            "gpu_id": report_id,
            "status": result.get("status", "ok"),
            "response": result.get("response", ""),
            "errors": result.get("errors", ""),
            "warnings": result.get("warnings", ""),
            "benchmark": json.dumps(benchmark) if benchmark else "",
        }
    finally:
        # The big cost fix: free the pod the instant the run is done so it never
        # bills while idle.  Runs only on a record we actually hold (not the
        # unknown-id early return above).
        if _EPHEMERAL:
            _teardown_after_run(gpu_id, record)


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
