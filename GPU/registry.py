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

    @property
    def is_running(self) -> bool:
        return bool(self.pod_id and self.public_ip and self.ssh_port)


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

    def list(self) -> List[GpuSummary]:
        with self._lock:
            return [
                GpuSummary(
                    gpu_id=gid,
                    name=rec.spec.name,
                    gpu_model=rec.spec.gpu_model,
                    pod_id=rec.pod_id,
                    pod_status="running" if rec.is_running else "pending",
                )
                for gid, rec in self._records.items()
            ]

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
        if _IDLE_TIMEOUT_MIN <= 0:
            return
        thread = threading.Thread(target=self._reap_loop, name="gpu-reaper", daemon=True)
        thread.start()
        logger.info("gpu idle reaper started (timeout=%dmin)", _IDLE_TIMEOUT_MIN)

    def _reap_loop(self) -> None:
        timeout_s = _IDLE_TIMEOUT_MIN * 60
        while True:
            time.sleep(_REAP_INTERVAL_S)
            now = time.monotonic()
            stale: List[tuple[str, GpuRecord]] = []
            with self._lock:
                for gid, rec in list(self._records.items()):
                    if rec.is_running and (now - rec.last_used) > timeout_s:
                        stale.append((gid, rec))
                        self._records.pop(gid, None)
            for gid, rec in stale:
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
