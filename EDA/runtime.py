"""
runtime.py
Orchestrates the EDA lifecycle: resolve the RunPod key, provision a pod, run a
tool on it in the background, and tear it down.

Provisioning is synchronous (RunPod REST is blocking) and is called from the
routers' plain ``def`` handlers, which FastAPI runs in a worker thread — so the
polling never blocks the event loop.

Runs are different, and this is the one place the EDA runtime departs sharply
from the gpu block.  ``GPU/runtime.run_gpu`` compiles and runs inside the request
and returns the result; at two minutes that is fine.  An OpenROAD route takes
30-90 minutes, which no HTTP request survives — Render and every proxy in between
would kill it, and it would pin a threadpool worker for the duration.  So
``start_*_job`` spawns a daemon thread, returns immediately, and the block polls
``eda_status`` until ``done`` and then fetches ``eda_result``.  Verilator and
Yosys finish in seconds but use the same protocol deliberately: one code path
here and one in the Qt client is worth more than a special-cased fast path.

Like the claw and gpu runtimes, the entry points never raise for an operational
failure: they return a dict with ``status="error"`` and a human-readable
``errors`` string so the block's error port always gets something useful.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from . import flow, pod_client
from .models import (
    EdaSpec,
    OpenRoadRunRequest,
    VerilatorRunRequest,
    YosysRunRequest,
)
from .registry import EdaRecord, registry

logger = logging.getLogger("eda.runtime")

# Sentinels the Grafux frontend writes into a port file when nothing is wired to
# it (see PortDataService::kEmptyPortValue and the "unconnected" literal in the
# block runner).  Treat them as an empty port.
_PLACEHOLDER_VALUES = {"empty", "unconnected"}

# Ephemeral lifecycle (default ON): terminate the pod once a run finishes — on
# success AND on every failure path — so a pod only costs money while it is
# actually working.  Set ``EDA_EPHEMERAL=0`` to keep pods between runs.
_EPHEMERAL = os.environ.get("EDA_EPHEMERAL", "1").lower() not in ("0", "false", "no")

# Provisioning is best-effort on RunPod: a create can hit transient capacity
# scarcity, and even with supportPublicIp a pod can land on a machine with no
# public-IP networking.  Both are placement problems a fresh pod usually escapes.
_PROVISION_ATTEMPTS = max(1, int(os.environ.get("EDA_PROVISION_ATTEMPTS", "3") or "3"))
_RETRY_BACKOFF_S = float(os.environ.get("EDA_RETRY_BACKOFF_S", "3") or "3")

# Server-side default keep-warm window (minutes).  Higher than the gpu block's 0:
# re-pulling a multi-GB EDA image on every floorplan tweak makes the block
# unusable for the iterative work it exists to support, and 15 minutes of a
# ~$0.12/hr CPU instance is about three cents.
_DEFAULT_KEEP_WARM_MIN = int(os.environ.get("EDA_DEFAULT_KEEP_WARM_MIN", "15") or "0")

# Total wall-clock ceiling for one run, across all stages.  Cost control rather
# than UX: a stage timeout bounds one stage, this bounds the bill.
_MAX_RUN_MINUTES = int(os.environ.get("EDA_MAX_RUN_MINUTES", "180") or "0")

# Development escape hatch: skip RunPod entirely and SSH to a locally-run
# grafux-eda container.  This exercises the whole real path — real verilator,
# real yosys, real ORFS, real artifact download — with no cloud account and no
# spend, which is how nearly all of this package can be tested.  NEVER set it in
# render.yaml.
_LOCAL_SSH = os.environ.get("EDA_LOCAL_SSH", "").lower() in ("1", "true", "yes")
_LOCAL_HOST = os.environ.get("EDA_LOCAL_HOST", "127.0.0.1")
_LOCAL_PORT = int(os.environ.get("EDA_LOCAL_PORT", "2222") or "2222")
_LOCAL_KEY = os.environ.get("EDA_LOCAL_KEY", "")


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
    """Return the port's real value, mapping placeholder sentinels to ''."""
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


def _resolve_runpod_key(spec: EdaSpec) -> Optional[str]:
    """
    Find the RunPod API key.

    Order: the api_keys port (bare 'rp_...' or JSON with a 'runpod' key), then the
    credentials port (same shapes), then the RUNPOD_API_KEY env var.  The env var
    is the default so the block "just works" without the user entering a key.
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


_NO_KEY_MESSAGE = (
    "No RunPod API key found. Set the block's api_keys port to "
    '{"runpod": "rp_..."} or the RUNPOD_API_KEY env var on the devices server. '
    "For local development without a cloud account, set EDA_LOCAL_SSH=1 and run "
    "the grafux-eda image locally on port 2222."
)


def _error_create(spec: EdaSpec, message: str, eda_id: str = "",
                  rate: float = 0.0) -> Dict[str, Any]:
    """A uniform failed-create payload."""
    return {
        "eda_id": eda_id,
        "kind": spec.kind,
        "status": "error",
        "pod_id": "",
        "pdk": spec.pdk,
        "errors": message,
        "usd_per_hr": rate,
        "warm_until": 0.0,
    }


# ---------------------------------------------------------------------------
# Provision (Regenerate) — create a pod and cache it.
# ---------------------------------------------------------------------------

def provision_eda(spec: EdaSpec, *, eda_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Provision a pod from an EdaSpec and register it.

    When ``eda_id`` is given, an already-registered (stub) record is populated in
    place and its ``phase`` advanced as provisioning progresses — this is how
    ``provision_eda_async`` exposes live creating -> pulling_image -> ready phases.

    Returns {eda_id, kind, status, pod_id, pdk, errors, usd_per_hr, warm_until}.
    Never raises.
    """
    if _LOCAL_SSH:
        return _provision_local(spec, eda_id=eda_id)

    api_key = _resolve_runpod_key(spec)
    if not api_key:
        if eda_id:
            registry.set_phase(eda_id, "error", _NO_KEY_MESSAGE)
        return _error_create(spec, _NO_KEY_MESSAGE, eda_id or "")

    rate = pod_client.price_for(spec.instance_type, spec.compute_type)
    last_err = ""
    for attempt in range(1, _PROVISION_ATTEMPTS + 1):
        pod_id = ""
        try:
            if eda_id:
                registry.set_phase(eda_id, "creating", f"placing pod (attempt {attempt})")
            private_pem, public_key = pod_client.generate_keypair()
            pod_id = pod_client.create_pod(api_key, spec, public_key)
            if eda_id:
                registry.set_connection(
                    eda_id, pod_id=pod_id, private_key_pem=private_pem,
                    api_key=api_key, usd_per_hr=rate,
                )
                registry.set_phase(eda_id, "pulling_image", "starting container")
            public_ip, ssh_port = pod_client.wait_until_ready(api_key, pod_id)

            if eda_id:
                registry.set_connection(eda_id, public_ip=public_ip, ssh_port=ssh_port)
                registry.set_phase(eda_id, "ready")
                new_id = eda_id
            else:
                record = EdaRecord(
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
                logger.info("eda provisioned on attempt %d/%d", attempt, _PROVISION_ATTEMPTS)
            return {
                "eda_id": new_id,
                "kind": spec.kind,
                "status": "ok",
                "pod_id": pod_id,
                "pdk": spec.pdk,
                "errors": "",
                "usd_per_hr": rate,
                "warm_until": warm_until,
            }
        except pod_client.ProvisionError as exc:
            # Retryable placement failure.  Free any pod we created so it cannot
            # bill, then try a fresh placement on (usually) a different machine.
            last_err = _describe_exception(exc)
            logger.warning(
                "eda provision attempt %d/%d failed (retryable): %s",
                attempt, _PROVISION_ATTEMPTS, exc,
            )
            if pod_id:
                pod_client.terminate_pod(api_key, pod_id)
            if attempt < _PROVISION_ATTEMPTS and _RETRY_BACKOFF_S > 0:
                time.sleep(_RETRY_BACKOFF_S)
            continue
        except Exception as exc:  # noqa: BLE001 — surface as an error result, never 500.
            logger.warning("eda provision failed: %s", exc)
            if pod_id:
                pod_client.terminate_pod(api_key, pod_id)
            err = _describe_exception(exc)
            if eda_id:
                registry.set_phase(eda_id, "error", err)
            return _error_create(spec, err, eda_id or "", rate)

    msg = (
        f"EDA pod provisioning failed after {_PROVISION_ATTEMPTS} attempts. {last_err} "
        "Try a different 'instance_type', set 'cloud_type' to 'SECURE', or Regenerate "
        "again in a few minutes."
    )
    if eda_id:
        registry.set_phase(eda_id, "error", msg)
    return _error_create(spec, msg, eda_id or "", rate)


def _provision_local(spec: EdaSpec, *, eda_id: Optional[str] = None) -> Dict[str, Any]:
    """
    EDA_LOCAL_SSH: register a record pointing at a locally-run container.

    No cloud, no key, no billing — but every later code path (SSH, staging, the
    stage driver, artifact download) is the real one, which is what makes this
    worth having rather than a mock.
    """
    key_pem = ""
    if _LOCAL_KEY and os.path.isfile(_LOCAL_KEY):
        try:
            with open(_LOCAL_KEY, encoding="utf-8") as fh:
                key_pem = fh.read()
        except OSError as exc:
            return _error_create(spec, f"EDA_LOCAL_KEY could not be read: {exc}", eda_id or "")
    if eda_id:
        registry.set_connection(
            eda_id, pod_id="local", public_ip=_LOCAL_HOST, ssh_port=_LOCAL_PORT,
            private_key_pem=key_pem, api_key="", usd_per_hr=0.0,
        )
        registry.set_phase(eda_id, "ready")
        new_id = eda_id
    else:
        new_id = registry.create(EdaRecord(
            spec=spec, pod_id="local", public_ip=_LOCAL_HOST, ssh_port=_LOCAL_PORT,
            private_key_pem=key_pem, phase="ready",
        ))
    logger.info("eda local-ssh mode: %s -> %s:%s", new_id, _LOCAL_HOST, _LOCAL_PORT)
    return {
        "eda_id": new_id, "kind": spec.kind, "status": "ok", "pod_id": "local",
        "pdk": spec.pdk, "errors": "", "usd_per_hr": 0.0, "warm_until": 0.0,
    }


def provision_eda_async(spec: EdaSpec) -> Dict[str, Any]:
    """
    Begin provisioning in the background and return immediately with an eda_id.

    Registers a stub record with ``phase="creating"`` and spawns a daemon thread
    that runs the normal ``provision_eda`` body, advancing the record's phase as it
    goes.  The block polls ``GET /{kind}/{id}/status`` until ``ready`` (or
    ``error``) and only then issues the Run — so a multi-minute image pull is
    visible progress instead of an opaque wait.
    """
    if not _LOCAL_SSH and not _resolve_runpod_key(spec):
        return _error_create(spec, _NO_KEY_MESSAGE)

    stub = EdaRecord(
        spec=spec,
        phase="creating",
        usd_per_hr=pod_client.price_for(spec.instance_type, spec.compute_type),
    )
    eda_id = registry.create(stub)

    def _job() -> None:
        try:
            provision_eda(spec, eda_id=eda_id)
        except Exception as exc:  # noqa: BLE001 — never let the daemon thread die loud.
            logger.warning("background eda provision crashed for %s: %s", eda_id, exc)
            registry.set_phase(eda_id, "error", _describe_exception(exc))

    threading.Thread(target=_job, name=f"eda-provision-{eda_id}", daemon=True).start()
    return {
        "eda_id": eda_id,
        "kind": spec.kind,
        "status": "creating",
        "pod_id": "",
        "pdk": spec.pdk,
        "errors": "",
        "usd_per_hr": stub.usd_per_hr,
        "warm_until": 0.0,
    }


# ---------------------------------------------------------------------------
# Run — start a job on an already-provisioned pod and poll it.
# ---------------------------------------------------------------------------

def _teardown_after_run(eda_id: str, record: EdaRecord) -> None:
    """
    Terminate a pod and drop it from the registry once a run is done.

    Best-effort and never raises — teardown must not turn a successful run into a
    failure.  Skipped when the run asked to stay warm.
    """
    try:
        if record.pod_id and record.api_key:
            pod_client.terminate_pod(record.api_key, record.pod_id)
    except Exception as exc:  # noqa: BLE001 — teardown is best-effort.
        logger.warning("eda post-run teardown failed for %s: %s", eda_id, exc)


def _start_job(eda_id: str, req, kind: str) -> Dict[str, Any]:
    """
    Shared body of the three ``start_*_job`` entry points.

    Validates the pod, marks the job started, and spawns the worker thread.  All
    the tool-specific behaviour lives in ``flow``; this only handles lifecycle.
    """
    record = registry.get(eda_id)
    if record is None:
        return {"eda_id": eda_id, "kind": kind, "status": "error", "stage": "",
                "errors": f"No {kind} instance with id '{eda_id}'. Press Regenerate "
                          f"to provision one."}
    if not record.is_running:
        return {"eda_id": eda_id, "kind": kind, "status": "error", "stage": "",
                "errors": f"The {kind} pod is not ready yet (phase "
                          f"'{record.phase or 'unknown'}'). Wait for it to finish "
                          f"provisioning, or press Regenerate."}
    if not registry.start_job(eda_id):
        return {"eda_id": eda_id, "kind": kind, "status": "error", "stage": record.stage,
                "errors": "A run is already in progress on this block. Press Stop "
                          "first, or wait for it to finish."}

    def _worker() -> None:
        _run_job(eda_id, req, kind)

    threading.Thread(target=_worker, name=f"eda-run-{eda_id}", daemon=True).start()
    return {"eda_id": eda_id, "kind": kind, "status": "running", "stage": "", "errors": ""}


def _run_job(eda_id: str, req, kind: str) -> None:
    """
    The worker body: connect, dispatch to the right tool, collect, tear down.

    Every exit path finishes the job in the registry — a thread that died without
    doing so would leave the block spinning forever on a poll that never completes.
    """
    record = registry.get(eda_id)
    if record is None:
        return

    def on_stage(stage: str, detail: str) -> None:
        registry.set_stage(eda_id, stage, detail)

    def on_line(line: str) -> None:
        registry.append_log(eda_id, line)

    def should_cancel() -> bool:
        rec = registry.get(eda_id)
        return rec is not None and rec.cancel.is_set()

    client = None
    result: Dict[str, Any]
    try:
        client = pod_client.connect_ssh(
            record.public_ip, record.ssh_port, record.private_key_pem
        )

        # Start from a clean working directory. A warm pod (keep_warm, or the
        # local-SSH dev container) is reused across runs, so last run's outputs
        # would otherwise be picked up by this run's artifact globs -- a lint run
        # reporting the previous run's waveform, or a failed synthesis appearing
        # to have produced the netlist from the attempt before it.
        #
        # Only WORK_DIR is cleared, never $FLOW_HOME/results: ORFS stages are
        # deliberately incremental, and wiping them would break the from_stage
        # port whose whole purpose is resuming a partly-completed flow.
        pod_client.exec_simple(
            client,
            f"rm -rf {pod_client.WORK_DIR} && mkdir -p {pod_client.WORK_DIR}",
            timeout=60,
        )

        if req.input_files:
            sftp = client.open_sftp()
            try:
                pod_client.stage_input_files(sftp, req.input_files)
            finally:
                sftp.close()

        if kind == "verilator":
            outcome = flow.run_verilator(
                client, req, on_stage=on_stage, on_line=on_line,
                should_cancel=should_cancel,
            )
        elif kind == "yosys":
            outcome = flow.run_yosys(
                client, req, pdk=record.spec.pdk, on_stage=on_stage, on_line=on_line,
                should_cancel=should_cancel,
            )
        else:
            outcome = flow.run_orfs(
                client, req, pdk=record.spec.pdk, on_stage=on_stage, on_line=on_line,
                should_cancel=should_cancel, max_run_s=_MAX_RUN_MINUTES * 60,
            )

        # Artifacts are collected even for a failed run: a route that blew up still
        # produced the placement and the reports that explain why.
        registry.set_stage(eda_id, outcome.get("_stage", ""), "downloading")
        artifacts = pod_client.download_artifacts(client, outcome.get("_globs", []))

        outputs = outcome.get("outputs", {})
        outputs["eda_id"] = eda_id
        outputs["artifacts"] = "\n".join(
            a.get("path", "").rsplit("/", 1)[-1] for a in artifacts if a.get("path")
        )
        outputs["cost"] = f"{record.cost_estimate_usd:.4f}"
        result = {
            "eda_id": eda_id,
            "kind": kind,
            "status": outcome.get("_status", "ok"),
            "stage": outcome.get("_stage", ""),
            "done": True,
            "outputs": outputs,
            "artifacts": artifacts,
            "errors": outputs.get("errors", ""),
            "warnings": outputs.get("warnings", ""),
            "log": outputs.get("log", ""),
            "usd_per_hr": record.usd_per_hr,
            "cost_estimate_usd": record.cost_estimate_usd,
        }
    except Exception as exc:  # noqa: BLE001 — surface as a result, never a dead thread.
        logger.warning("eda run failed for %s: %s", eda_id, exc)
        err = _describe_exception(exc)
        result = {
            "eda_id": eda_id, "kind": kind, "status": "error",
            "stage": record.stage, "done": True,
            "outputs": {"status": "error", "errors": err, "eda_id": eda_id},
            "artifacts": [], "errors": err, "warnings": "", "log": "",
            "usd_per_hr": record.usd_per_hr,
            "cost_estimate_usd": record.cost_estimate_usd,
        }
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    registry.finish_job(eda_id, result)

    # Free the pod unless the run asked to stay warm.  Local-SSH mode owns no pod.
    warm_min = _keep_warm_minutes(
        getattr(req, "keep_warm_minutes", 0), record.spec.keep_warm_minutes
    )
    if _LOCAL_SSH:
        return
    if warm_min > 0:
        registry.set_keep_warm(eda_id, warm_min)
    elif _EPHEMERAL:
        _teardown_after_run(eda_id, record)
        registry.delete(eda_id)


def start_verilator_job(eda_id: str, req: VerilatorRunRequest) -> Dict[str, Any]:
    """Start a Verilator lint/simulation run.  Returns as soon as the job starts."""
    return _start_job(eda_id, req, "verilator")


def start_yosys_job(eda_id: str, req: YosysRunRequest) -> Dict[str, Any]:
    """Start a Yosys synthesis run.  Returns as soon as the job starts."""
    return _start_job(eda_id, req, "yosys")


def start_openroad_job(eda_id: str, req: OpenRoadRunRequest) -> Dict[str, Any]:
    """Start an OpenROAD flow run.  Returns as soon as the job starts."""
    return _start_job(eda_id, req, "openroad")


# ---------------------------------------------------------------------------
# Status / result / cancel / teardown
# ---------------------------------------------------------------------------

def eda_status(eda_id: str, *, live: bool = False) -> Optional[Dict[str, Any]]:
    """
    Return a live status dict for a pod and its job, or None if the id is unknown.

    With ``live=True`` and a placed pod, do one RunPod ``get_pod`` to refine the
    phase from the cloud's own view and pick up the live hourly rate.  Without it,
    report the cached record only (no REST call) so a fast frontend poll cannot
    hammer the RunPod API.

    Note the ordering: once a job is in flight the *job* phase wins over the pod
    phase, because "5/6 Routing…" is what the user needs to see, not "ready".
    """
    record = registry.get(eda_id)
    if record is None:
        return None

    phase, detail = record.phase, record.phase_detail
    rate = record.usd_per_hr or pod_client.price_for(
        record.spec.instance_type, record.spec.compute_type
    )
    if live and record.pod_id and record.api_key and not record.job_in_flight:
        try:
            pod = pod_client.get_pod(record.api_key, record.pod_id)
            phase, detail = pod_client.phase_from_pod(pod)
            live_rate = pod_client.cost_per_hr_of(pod)
            if live_rate:
                rate = live_rate
                registry.set_connection(eda_id, usd_per_hr=live_rate)
        except Exception as exc:  # noqa: BLE001 — status must never raise.
            logger.debug("eda live status lookup failed for %s: %s", eda_id, exc)

    if record.job_in_flight:
        phase, detail = "running", record.stage_detail

    return {
        "eda_id": eda_id,
        "kind": record.spec.kind,
        "phase": phase or ("ready" if record.is_running else ""),
        "phase_detail": detail,
        "stage": record.stage,
        "stage_detail": record.stage_detail,
        "stages_done": list(record.stages_done),
        "log_tail": "\n".join(record.log_tail),
        "elapsed_s": record.elapsed_s,
        "done": record.done,
        "pod_id": record.pod_id,
        "pod_status": "running" if record.is_running else "pending",
        "uptime_s": record.uptime_s,
        "warm_until": record.warm_until,
        "usd_per_hr": rate,
        "cost_estimate_usd": round(rate * record.uptime_s / 3600.0, 4),
    }


def eda_result(eda_id: str) -> Optional[Dict[str, Any]]:
    """
    The finished run payload, or None if the id is unknown.

    A job still in flight returns ``status="running"`` with ``done=False`` rather
    than None, so a client that polls ``/result`` directly gets a coherent answer
    instead of a 404 it would have to special-case.

    Ephemeral runs delete the record right after storing the result, so the result
    is fetched from the record *before* teardown in practice; a client that asks
    too late gets a clear "no longer available" rather than silence.
    """
    record = registry.get(eda_id)
    if record is None:
        return None
    if record.result is not None:
        return record.result
    return {
        "eda_id": eda_id,
        "kind": record.spec.kind,
        "status": "running",
        "stage": record.stage,
        "done": False,
        "outputs": {},
        "artifacts": [],
        "errors": "",
        "warnings": "",
        "log": "\n".join(record.log_tail),
        "usd_per_hr": record.usd_per_hr,
        "cost_estimate_usd": record.cost_estimate_usd,
    }


def cancel_eda(eda_id: str) -> bool:
    """
    Stop an in-flight run and terminate the pod.  Returns False if unknown.

    Both halves matter.  Signalling the cancel event stops the tool; terminating
    the pod stops the billing — and the second is the one a user pressing Stop
    actually cares about, because a pod left running after an abandoned route is
    the expensive failure mode this whole package guards against.
    """
    record = registry.get(eda_id)
    if record is None:
        return False
    registry.request_cancel(eda_id)
    if record.pod_id and record.api_key:
        pod_client.terminate_pod(record.api_key, record.pod_id)
    registry.delete(eda_id)
    return True


def terminate_eda(eda_id: str) -> bool:
    """Terminate a pod (stopping billing) and remove it.  Returns False if unknown."""
    return cancel_eda(eda_id)
