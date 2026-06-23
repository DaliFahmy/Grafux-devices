"""
registry.py
In-memory registry of provisioned GPU pods, with optional JSON persistence of the
*spec* (never the private key) so a block's configuration survives a restart.

Each entry pairs a ``GpuSpec`` with the live pod runtime needed to SSH back in
(pod id, public ip, ssh port, the ephemeral private key, the resolved RunPod API
key, and a ``last_used`` timestamp).  A background idle reaper terminates pods
that have been idle past ``GPU_IDLE_TIMEOUT_MIN`` so a forgotten pod cannot bill
forever.

Like the claw registry this is a process-local dict guarded by a lock, adequate
for the single-process uvicorn deployment used by the devices server.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import runpod_client
from .models import GpuSpec, GpuSummary

logger = logging.getLogger("gpu.registry")

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "_gpus")
_PERSIST_ENABLED = os.environ.get("GPU_PERSIST", "").lower() in ("1", "true", "yes")

# Idle reaper: terminate pods unused for this many minutes (0 disables).  This is
# a safety-net backstop only: ephemeral runs (see runtime._EPHEMERAL) already
# self-terminate the instant a run finishes, so the reaper just catches pods that
# were provisioned (Regenerate / auto-provision) but never followed by a Run, plus
# warm pods when GPU_EPHEMERAL=0.  Kept short so a forgotten pod stops billing fast.
_IDLE_TIMEOUT_MIN = int(os.environ.get("GPU_IDLE_TIMEOUT_MIN", "10") or "0")
_REAP_INTERVAL_S = 60


@dataclass
class GpuRecord:
    """A provisioned GPU pod: its spec plus the live connection details."""

    spec: GpuSpec
    pod_id: str = ""
    public_ip: str = ""
    ssh_port: int = 0
    private_key_pem: str = ""
    api_key: str = ""
    last_used: float = field(default_factory=time.monotonic)
    # Provisioning/run phase for live status polling.  Empty until the background
    # provisioner (or run_gpu) sets it: creating -> pulling_image -> ready -> running.
    phase: str = ""
    phase_detail: str = ""
    # When a keep-warm run finishes, the reaper keeps the pod until this monotonic
    # deadline instead of the global idle timeout.  0 = use the global idle timeout.
    keep_warm_deadline: float = 0.0
    # Wall-clock epoch (time.time) the warm hold lasts until — reported to the block
    # so it can show "warm until …".  0 when not warm.  Kept separate from the
    # monotonic reaper deadline above (never mix the two clocks).
    warm_until: float = 0.0
    # The pod's hourly rate (live RunPod costPerHr if known, else the static table).
    usd_per_hr: float = 0.0
    # Wall-clock epoch the record was created, for an uptime/cost estimate.
    created_at: float = field(default_factory=time.time)

    @property
    def is_running(self) -> bool:
        return bool(self.pod_id and self.public_ip and self.ssh_port)

    @property
    def uptime_s(self) -> float:
        return max(0.0, time.time() - self.created_at)

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost of the pod's life so far (rate × uptime)."""
        return round(self.usd_per_hr * self.uptime_s / 3600.0, 4)


class GpuRegistry:
    """Thread-safe store of provisioned GPU pods, with an idle reaper."""

    def __init__(self) -> None:
        self._records: Dict[str, GpuRecord] = {}
        self._lock = threading.Lock()
        if _PERSIST_ENABLED:
            self._load_from_disk()
        self._start_reaper()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, record: GpuRecord) -> str:
        """Register a provisioned pod and return its generated gpu_id."""
        gpu_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._records[gpu_id] = record
        logger.info(
            "created gpu id=%s name=%r pod=%s", gpu_id, record.spec.name, record.pod_id
        )
        if _PERSIST_ENABLED:
            self._save_one(gpu_id, record)
        return gpu_id

    def get(self, gpu_id: str) -> Optional[GpuRecord]:
        with self._lock:
            return self._records.get(gpu_id)

    def touch(self, gpu_id: str) -> None:
        """Mark a pod as just-used so the reaper does not terminate it."""
        with self._lock:
            rec = self._records.get(gpu_id)
            if rec is not None:
                rec.last_used = time.monotonic()

    def set_keep_warm(self, gpu_id: str, minutes: int) -> float:
        """
        Hold a pod warm for ``minutes`` after a run instead of reaping it on idle.

        Sets a monotonic reaper deadline and the wall-clock ``warm_until`` reported
        to the block.  ``minutes <= 0`` clears any warm hold.  Returns the
        wall-clock ``warm_until`` epoch (0 when cleared).
        """
        with self._lock:
            rec = self._records.get(gpu_id)
            if rec is None:
                return 0.0
            now_mono = time.monotonic()
            rec.last_used = now_mono
            if minutes > 0:
                rec.keep_warm_deadline = now_mono + minutes * 60
                rec.warm_until = time.time() + minutes * 60
            else:
                rec.keep_warm_deadline = 0.0
                rec.warm_until = 0.0
            return rec.warm_until

    def set_phase(self, gpu_id: str, phase: str, detail: str = "") -> None:
        """Update a record's provisioning/run phase for live status polling."""
        with self._lock:
            rec = self._records.get(gpu_id)
            if rec is not None:
                rec.phase = phase
                rec.phase_detail = detail

    def set_connection(
        self,
        gpu_id: str,
        *,
        pod_id: Optional[str] = None,
        public_ip: Optional[str] = None,
        ssh_port: Optional[int] = None,
        private_key_pem: Optional[str] = None,
        api_key: Optional[str] = None,
        usd_per_hr: Optional[float] = None,
    ) -> None:
        """Populate a pre-registered record's live connection details (async provision)."""
        with self._lock:
            rec = self._records.get(gpu_id)
            if rec is None:
                return
            if pod_id is not None:
                rec.pod_id = pod_id
            if public_ip is not None:
                rec.public_ip = public_ip
            if ssh_port is not None:
                rec.ssh_port = ssh_port
            if private_key_pem is not None:
                rec.private_key_pem = private_key_pem
            if api_key is not None:
                rec.api_key = api_key
            if usd_per_hr is not None:
                rec.usd_per_hr = usd_per_hr

    def list(self) -> List[GpuSummary]:
        with self._lock:
            return [self._summary(gid, rec) for gid, rec in self._records.items()]

    @staticmethod
    def _summary(gpu_id: str, rec: GpuRecord) -> GpuSummary:
        return GpuSummary(
            gpu_id=gpu_id,
            name=rec.spec.name,
            gpu_model=rec.spec.gpu_model,
            pod_id=rec.pod_id,
            pod_status="running" if rec.is_running else "pending",
            phase=rec.phase,
            uptime_s=rec.uptime_s,
            usd_per_hr=rec.usd_per_hr,
            warm_until=rec.warm_until,
        )

    def summary(self, gpu_id: str) -> Optional[GpuSummary]:
        with self._lock:
            rec = self._records.get(gpu_id)
            return self._summary(gpu_id, rec) if rec is not None else None

    def delete(self, gpu_id: str) -> bool:
        """Remove a pod from the registry (does NOT terminate it — see runtime)."""
        with self._lock:
            existed = self._records.pop(gpu_id, None) is not None
        if existed and _PERSIST_ENABLED:
            self._delete_one(gpu_id)
        return existed

    # ------------------------------------------------------------------
    # Idle reaper — terminate forgotten pods so they stop billing.
    # ------------------------------------------------------------------

    def _start_reaper(self) -> None:
        # Always start the daemon: even with the global idle timeout disabled
        # (GPU_IDLE_TIMEOUT_MIN=0) it must still reap per-record keep-warm pods once
        # their warm window expires, otherwise a warm pod would never be freed.  The
        # loop is a cheap 60s tick that does nothing on an empty registry.
        thread = threading.Thread(target=self._reap_loop, name="gpu-reaper", daemon=True)
        thread.start()
        logger.info(
            "gpu idle reaper started (global timeout=%s)",
            f"{_IDLE_TIMEOUT_MIN}min" if _IDLE_TIMEOUT_MIN > 0 else "off",
        )

    def _select_stale(self, now: float) -> List[tuple[str, GpuRecord]]:
        """
        Return (and remove from the registry) the records that should be reaped.

        A record is stale when either its per-record keep-warm window has expired
        (``keep_warm_deadline > 0`` and ``now`` is past it) or — when no warm window
        is set — it has been idle past the global ``GPU_IDLE_TIMEOUT_MIN``.  All
        times are ``time.monotonic`` (never mix with ``time.time``).
        """
        timeout_s = _IDLE_TIMEOUT_MIN * 60
        stale: List[tuple[str, GpuRecord]] = []
        with self._lock:
            for gid, rec in list(self._records.items()):
                if not rec.is_running:
                    continue
                if rec.keep_warm_deadline > 0:
                    expired = now > rec.keep_warm_deadline
                elif timeout_s > 0:
                    expired = (now - rec.last_used) > timeout_s
                else:
                    expired = False  # no warm window and global timeout disabled
                if expired:
                    stale.append((gid, rec))
                    self._records.pop(gid, None)
        return stale

    def _reap_loop(self) -> None:
        while True:
            time.sleep(_REAP_INTERVAL_S)
            for gid, rec in self._select_stale(time.monotonic()):
                logger.info("reaping idle gpu id=%s pod=%s", gid, rec.pod_id)
                runpod_client.terminate_pod(rec.api_key, rec.pod_id)
                if _PERSIST_ENABLED:
                    self._delete_one(gid)

    # ------------------------------------------------------------------
    # Persistence (spec only; never the private/api keys).  Best-effort.
    # ------------------------------------------------------------------

    def _save_one(self, gpu_id: str, record: GpuRecord) -> None:
        try:
            os.makedirs(_PERSIST_DIR, exist_ok=True)
            path = os.path.join(_PERSIST_DIR, f"{gpu_id}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"spec": record.spec.model_dump(), "pod_id": record.pod_id}, fh)
        except OSError as exc:
            logger.warning("failed to persist gpu %s: %s", gpu_id, exc)

    def _delete_one(self, gpu_id: str) -> None:
        try:
            os.remove(os.path.join(_PERSIST_DIR, f"{gpu_id}.json"))
        except OSError:
            pass

    def _load_from_disk(self) -> None:
        if not os.path.isdir(_PERSIST_DIR):
            return
        for fname in os.listdir(_PERSIST_DIR):
            if not fname.endswith(".json"):
                continue
            gpu_id = fname[:-len(".json")]
            try:
                with open(os.path.join(_PERSIST_DIR, fname), encoding="utf-8") as fh:
                    data = json.load(fh)
                # The live pod/keys are gone after a restart — keep only the spec so
                # the block can re-provision via Regenerate.
                self._records[gpu_id] = GpuRecord(spec=GpuSpec(**data.get("spec", {})))
            except (OSError, ValueError) as exc:
                logger.warning("failed to load persisted gpu %s: %s", fname, exc)
        logger.info("loaded %d persisted gpu spec(s)", len(self._records))


# Module-level singleton shared by the router.
registry = GpuRegistry()
