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
import threading
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

# Server-side default keep-warm window (minutes) applied when neither the run
# request nor the block's spec sets one.  0 keeps today's ephemeral behavior.
_DEFAULT_KEEP_WARM_MIN = int(os.environ.get("GPU_DEFAULT_KEEP_WARM_MIN", "0") or "0")


def _keep_warm_minutes(*candidates: Any) -> int:
    """First positive keep-warm window among the candidates, else the env default."""
    for value in candidates:
        try:
            minutes = int(value or 0)
        except (TypeError, ValueError):
            minutes = 0
        if minutes > 0:
            return minutes
    return max(0, _DEFAULT_KEEP_WARM_MIN)


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

def provision_gpu(spec: GpuSpec, *, gpu_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Provision a RunPod pod from a GpuSpec and register it.

    When ``gpu_id`` is given, an already-registered (stub) record is populated in
    place and its ``phase`` is advanced as provisioning progresses — this is how
    ``provision_gpu_async`` exposes live ``creating -> pulling_image -> ready``
    phases.  When omitted, a fresh record is created and its new id returned.

    A positive ``keep_warm_minutes`` on the spec also pre-warms the pod (holds it
    past the idle reaper), so a Regenerate/create doubles as a pre-warm.

    Returns {gpu_id, status, pod_id, gpu_model, errors, usd_per_hr, warm_until}.
    Never raises.
    """
    api_key = _resolve_runpod_key(spec)
    if not api_key:
        msg = (
            "No RunPod API key found. Set the gpu block's api_keys port "
            "({\"runpod\": \"rp_...\"}) or the RUNPOD_API_KEY env var on the server."
        )
        if gpu_id:
            registry.set_phase(gpu_id, "error", msg)
        return {
            "gpu_id": gpu_id or "",
            "status": "error",
            "pod_id": "",
            "gpu_model": spec.gpu_model,
            "errors": msg,
            "usd_per_hr": 0.0,
            "warm_until": 0.0,
        }

    rate = runpod_client.price_for(spec.gpu_model)
    last_err = ""
    for attempt in range(1, _PROVISION_ATTEMPTS + 1):
        pod_id = ""
        try:
            if gpu_id:
                registry.set_phase(gpu_id, "creating", f"placing pod (attempt {attempt})")
            private_pem, public_key = runpod_client.generate_keypair()
            pod_id = runpod_client.create_pod(api_key, spec, public_key)
            if gpu_id:
                registry.set_connection(
                    gpu_id, pod_id=pod_id, private_key_pem=private_pem,
                    api_key=api_key, usd_per_hr=rate,
                )
                registry.set_phase(gpu_id, "pulling_image", "starting container")
            public_ip, ssh_port = runpod_client.wait_until_ready(api_key, pod_id)

            if gpu_id:
                registry.set_connection(gpu_id, public_ip=public_ip, ssh_port=ssh_port)
                registry.set_phase(gpu_id, "ready")
                new_id = gpu_id
            else:
                record = GpuRecord(
                    spec=spec,
                    pod_id=pod_id,
                    public_ip=public_ip,
                    ssh_port=ssh_port,
                    private_key_pem=private_pem,
                    api_key=api_key,
                    usd_per_hr=rate,
                    phase="ready",
                )
                new_id = registry.create(record)

            warm_min = _keep_warm_minutes(spec.keep_warm_minutes)
            warm_until = registry.set_keep_warm(new_id, warm_min) if warm_min > 0 else 0.0
            if attempt > 1:
                logger.info(
                    "gpu provisioned on attempt %d/%d", attempt, _PROVISION_ATTEMPTS
                )
            return {
                "gpu_id": new_id,
                "status": "ok",
                "pod_id": pod_id,
                "gpu_model": spec.gpu_model,
                "errors": "",
                "usd_per_hr": rate,
                "warm_until": warm_until,
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
            err = _describe_exception(exc)
            if gpu_id:
                registry.set_phase(gpu_id, "error", err)
            return {
                "gpu_id": gpu_id or "",
                "status": "error",
                "pod_id": "",
                "gpu_model": spec.gpu_model,
                "errors": err,
                "usd_per_hr": rate,
                "warm_until": 0.0,
            }

    # Every attempt hit a retryable placement failure — surface the last cause plus
    # the levers the user can pull (different GPU, or the more-available SECURE tier).
    msg = (
        f"GPU provisioning failed after {_PROVISION_ATTEMPTS} attempts. {last_err} "
        "Try a different 'gpu_model', set 'cloud_type' to 'SECURE', or Regenerate "
        "again in a few minutes."
    )
    if gpu_id:
        registry.set_phase(gpu_id, "error", msg)
    return {
        "gpu_id": gpu_id or "",
        "status": "error",
        "pod_id": "",
        "gpu_model": spec.gpu_model,
        "errors": msg,
        "usd_per_hr": rate,
        "warm_until": 0.0,
    }


def provision_gpu_async(spec: GpuSpec) -> Dict[str, Any]:
    """
    Begin provisioning in the background and return immediately with a gpu_id.

    Registers a stub record with ``phase="creating"`` and spawns a daemon thread
    (same pattern as the idle reaper) that runs the normal ``provision_gpu`` body,
    advancing the record's phase as it goes.  The block then polls
    ``GET /gpu/{id}/status`` until the phase is ``ready`` (or ``error``) and only
    then issues the Run — so the cold start is visible instead of an opaque wait.

    Returns {gpu_id, status:"creating", pod_id:"", gpu_model, errors}.  An immediate
    failure (no API key) returns status="error" with an empty gpu_id.
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
            "usd_per_hr": 0.0,
            "warm_until": 0.0,
        }

    stub = GpuRecord(spec=spec, phase="creating", usd_per_hr=runpod_client.price_for(spec.gpu_model))
    gpu_id = registry.create(stub)

    def _job() -> None:
        try:
            provision_gpu(spec, gpu_id=gpu_id)
        except Exception as exc:  # noqa: BLE001 — never let the daemon thread die loud.
            logger.warning("background gpu provision crashed for %s: %s", gpu_id, exc)
            registry.set_phase(gpu_id, "error", _describe_exception(exc))

    threading.Thread(target=_job, name=f"gpu-provision-{gpu_id}", daemon=True).start()
    return {
        "gpu_id": gpu_id,
        "status": "creating",
        "pod_id": "",
        "gpu_model": spec.gpu_model,
        "errors": "",
        "usd_per_hr": stub.usd_per_hr,
        "warm_until": 0.0,
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

    Returns {gpu_id, status, response, errors, warnings, benchmark(JSON string),
    artifacts(JSON string), kept_warm, warm_until, usd_per_hr, cost_estimate_usd}.
    Never raises.

    Teardown is decided per run: by default (ephemeral) the pod is terminated the
    instant the run finishes — on success or failure — and the returned ``gpu_id``
    is "" so the block re-provisions on the next Run.  A positive
    ``keep_warm_minutes`` (on the request, else the block's spec, else the env
    default) keeps the pod alive after a *successful* run for instant re-runs: the
    live ``gpu_id`` is reported so the block reuses the warm pod, and the idle
    reaper frees it once the warm window expires.  Failures always tear down (no
    orphan billing).
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

    warm_min = _keep_warm_minutes(req.keep_warm_minutes, record.spec.keep_warm_minutes)
    rate = record.usd_per_hr or runpod_client.price_for(record.spec.gpu_model)
    did_keep_warm = False
    warm_until = 0.0
    # Whether the pod survives this call.  A keep-warm hold is only applied on a
    # clean success (did_keep_warm, set below); failing that, the pod survives only
    # in warm mode (GPU_EPHEMERAL=0).  report_id is the live id when it survives, ""
    # otherwise (so the block re-provisions next Run).  Recomputed after the run once
    # did_keep_warm is known.
    report_id = gpu_id if not _EPHEMERAL else ""

    def _result(**extra: Any) -> Dict[str, Any]:
        base = {
            "gpu_id": report_id,
            "status": "error",
            "response": "",
            "errors": "",
            "warnings": "",
            "benchmark": "",
            "artifacts": "",
            "kept_warm": did_keep_warm,
            "warm_until": warm_until,
            "usd_per_hr": rate,
            "cost_estimate_usd": record.cost_estimate_usd,
        }
        base.update(extra)
        return base

    try:
        if not record.is_running:
            return _result(
                errors="GPU pod is not running — a fresh pod will be provisioned on the next run.",
            )

        registry.touch(gpu_id)
        code = _clean_port(req.code)
        if not code:
            return _result(errors="The 'code' input port is empty — nothing to compile.")

        registry.set_phase(gpu_id, "running", "compiling / executing")
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
                input_files=list(req.input_files or []),
                output_globs=list(req.output_globs or []),
                working_dir=req.working_dir or "/workspace",
            )
        except Exception as exc:  # noqa: BLE001 — connection/SSH failures → error result.
            logger.warning("gpu run failed: %s", exc)
            registry.set_phase(gpu_id, "error", _describe_exception(exc))
            return _result(errors=_describe_exception(exc))

        status = result.get("status", "ok")
        # Keep the pod warm only on a clean success — a broken pod is not worth
        # holding.  This flips report_id to the live id and skips the teardown below.
        if status == "ok" and warm_min > 0:
            warm_until = registry.set_keep_warm(gpu_id, warm_min)
            did_keep_warm = True
            report_id = gpu_id
        registry.set_phase(gpu_id, "ready")

        benchmark = result.get("benchmark") or {}
        artifacts = result.get("artifacts") or []
        return _result(
            status=status,
            response=result.get("response", ""),
            errors=result.get("errors", ""),
            warnings=result.get("warnings", ""),
            benchmark=json.dumps(benchmark) if benchmark else "",
            artifacts=json.dumps(artifacts) if artifacts else "",
        )
    finally:
        # Free the pod the instant the run is done so it never bills while idle —
        # unless this run kept it warm for fast re-runs (did_keep_warm), or warm
        # mode (GPU_EPHEMERAL=0) is keeping it for the idle reaper.
        if _EPHEMERAL and not did_keep_warm:
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


# ---------------------------------------------------------------------------
# Status — live phase / uptime / cost for a pod (polled by the block).
# ---------------------------------------------------------------------------

# Exposing the pod's SSH host/port + private key lets a user open their own
# terminal, but the key is a secret the rest of the API never echoes — so it is
# OFF unless explicitly enabled on a trusted-network deployment.
_EXPOSE_SSH = os.environ.get("GPU_EXPOSE_SSH", "").lower() in ("1", "true", "yes")


def ssh_exposed() -> bool:
    return _EXPOSE_SSH


def _ssh_hint(record: GpuRecord) -> str:
    """A copy-pasteable ssh command for the pod (host/port only, no key)."""
    if not (record.public_ip and record.ssh_port):
        return ""
    return f"ssh -p {record.ssh_port} root@{record.public_ip}"


def gpu_status(gpu_id: str, *, live: bool = False) -> Optional[Dict[str, Any]]:
    """
    Return a live status dict for a pod, or None if the id is unknown.

    With ``live=True`` and a placed pod, do one RunPod ``get_pod`` to refine the
    phase from the cloud's own view (creating/pulling_image/ready/error) and pick up
    the live hourly rate.  Without it, report the cached record phase only (no REST
    call) so a fast frontend poll cannot hammer the RunPod API.
    """
    record = registry.get(gpu_id)
    if record is None:
        return None

    phase, detail = record.phase, record.phase_detail
    rate = record.usd_per_hr or runpod_client.price_for(record.spec.gpu_model)
    if live and record.pod_id and record.api_key:
        try:
            pod = runpod_client.get_pod(record.api_key, record.pod_id)
            phase, detail = runpod_client.phase_from_pod(pod)
            live_rate = runpod_client.cost_per_hr_of(pod)
            if live_rate:
                rate = live_rate
                registry.set_connection(gpu_id, usd_per_hr=live_rate)
        except Exception as exc:  # noqa: BLE001 — status must never raise.
            logger.debug("gpu live status lookup failed for %s: %s", gpu_id, exc)

    return {
        "gpu_id": gpu_id,
        "phase": phase or ("ready" if record.is_running else ""),
        "phase_detail": detail,
        "pod_id": record.pod_id,
        "gpu_model": record.spec.gpu_model,
        "pod_status": "running" if record.is_running else "pending",
        "uptime_s": record.uptime_s,
        "warm_until": record.warm_until,
        "usd_per_hr": rate,
        "cost_estimate_usd": round(rate * record.uptime_s / 3600.0, 4),
        "ssh": _ssh_hint(record) if _EXPOSE_SSH else "",
    }


def gpu_ssh(gpu_id: str) -> Optional[Dict[str, Any]]:
    """
    Return SSH connection details (incl. the private key) for a running pod.

    Gated by ``GPU_EXPOSE_SSH`` (the caller checks ``ssh_exposed()``).  Returns
    None for an unknown or not-yet-running pod.  Touches keep-warm so an
    interactive session is not reaped out from under the user.
    """
    record = registry.get(gpu_id)
    if record is None or not record.is_running:
        return None
    registry.touch(gpu_id)
    return {
        "gpu_id": gpu_id,
        "host": record.public_ip,
        "port": record.ssh_port,
        "username": "root",
        "private_key_pem": record.private_key_pem,
        "connect_hint": _ssh_hint(record),
    }
