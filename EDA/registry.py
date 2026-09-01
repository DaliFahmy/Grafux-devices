"""
registry.py
In-memory registry of provisioned EDA pods and their in-flight jobs, with optional
JSON persistence of the *spec* (never the private key) so a block's configuration
survives a restart.

Each entry pairs an ``EdaSpec`` with the live pod runtime needed to SSH back in
(pod id, public ip, ssh port, the ephemeral private key, the resolved RunPod API
key) *and* the state of the job running on it (stage, log tail, cancel event,
finished result).  A background idle reaper terminates pods that have been idle
past ``EDA_IDLE_TIMEOUT_MIN`` so a forgotten pod cannot bill forever.

The one substantive difference from ``GPU/registry.py`` is in ``_select_stale``.
The GPU reaper decides staleness from ``last_used``, which is only touched when a
request arrives — fine for a two-minute compile, fatal here.  An OpenROAD route
runs for 45 minutes inside a single SSH call, touching nothing, so that reaper
would terminate a perfectly healthy pod mid-route and the user would see an
inexplicable failure.  A record with a job in flight is therefore never reaped;
the job thread also touches the record on every stage transition as a second line
of defence.

Like the GPU and claw registries this is a process-local dict guarded by a lock,
adequate for the single-process uvicorn deployment used by the devices server.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from . import pod_client
from .models import EdaSpec, EdaSummary

logger = logging.getLogger("eda.registry")

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "_eda")
_PERSIST_ENABLED = os.environ.get("EDA_PERSIST", "").lower() in ("1", "true", "yes")

# Idle reaper: terminate pods unused for this many minutes (0 disables the global
# timeout, but the reaper still runs to expire keep-warm holds).  Longer than the
# GPU block's 10 minutes because the EDA keep-warm default is longer.
_IDLE_TIMEOUT_MIN = int(os.environ.get("EDA_IDLE_TIMEOUT_MIN", "20") or "0")
_REAP_INTERVAL_S = 60

# How many lines of tool output to retain for the live status poll.  Enough to see
# what routing is doing; bounded so a chatty run cannot grow without limit.
_LOG_TAIL_LINES = 200


@dataclass
class EdaRecord:
    """A provisioned EDA pod: its spec, live connection details, and job state."""

    spec: EdaSpec
    pod_id: str = ""
    public_ip: str = ""
    ssh_port: int = 0
    private_key_pem: str = ""
    api_key: str = ""
    last_used: float = field(default_factory=time.monotonic)
    # Provisioning phase: creating -> pulling_image -> ready -> running -> done/error.
    phase: str = ""
    phase_detail: str = ""
    keep_warm_deadline: float = 0.0
    warm_until: float = 0.0
    usd_per_hr: float = 0.0
    created_at: float = field(default_factory=time.time)

    # --- job state -----------------------------------------------------
    # The current tool stage ("synth", "route", "sim", …) and whether it is
    # running/done/failed.  Together these drive the block's live substatus.
    stage: str = ""
    stage_detail: str = ""
    stages_done: List[str] = field(default_factory=list)
    log_tail: Deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_TAIL_LINES))
    # Monotonic time the current job started; 0 when no job is in flight.  This is
    # also the reaper's "do not touch" flag — see _select_stale.
    job_started_at: float = 0.0
    done: bool = False
    result: Optional[Dict[str, Any]] = None
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def is_running(self) -> bool:
        return bool(self.pod_id and self.public_ip and self.ssh_port)

    @property
    def job_in_flight(self) -> bool:
        """True while a tool run is executing on this pod."""
        return self.job_started_at > 0 and not self.done

    @property
    def uptime_s(self) -> float:
        return max(0.0, time.time() - self.created_at)

    @property
    def elapsed_s(self) -> float:
        """Seconds the current job has been running (0 when idle)."""
        return max(0.0, time.monotonic() - self.job_started_at) if self.job_started_at else 0.0

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost of the pod's life so far (rate x uptime)."""
        return round(self.usd_per_hr * self.uptime_s / 3600.0, 4)


class EdaRegistry:
    """Thread-safe store of provisioned EDA pods, with an idle reaper."""

    def __init__(self) -> None:
        self._records: Dict[str, EdaRecord] = {}
        self._lock = threading.Lock()
        if _PERSIST_ENABLED:
            self._load_from_disk()
        self._start_reaper()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, record: EdaRecord) -> str:
        """Register a provisioned pod and return its generated eda_id."""
        eda_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._records[eda_id] = record
        logger.info(
            "created eda id=%s kind=%s name=%r pod=%s",
            eda_id, record.spec.kind, record.spec.name, record.pod_id,
        )
        if _PERSIST_ENABLED:
            self._save_one(eda_id, record)
        return eda_id

    def get(self, eda_id: str) -> Optional[EdaRecord]:
        with self._lock:
            return self._records.get(eda_id)

    def touch(self, eda_id: str) -> None:
        """Mark a pod as just-used so the reaper does not terminate it."""
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is not None:
                rec.last_used = time.monotonic()

    def set_keep_warm(self, eda_id: str, minutes: int) -> float:
        """
        Hold a pod warm for ``minutes`` after a run instead of reaping it on idle.

        Sets a monotonic reaper deadline and the wall-clock ``warm_until`` reported
        to the block.  ``minutes <= 0`` clears any warm hold.  Returns the
        wall-clock ``warm_until`` epoch (0 when cleared).
        """
        with self._lock:
            rec = self._records.get(eda_id)
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

    def set_phase(self, eda_id: str, phase: str, detail: str = "") -> None:
        """Update a record's provisioning phase for live status polling."""
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is not None:
                rec.phase = phase
                rec.phase_detail = detail

    def set_stage(self, eda_id: str, stage: str, detail: str = "") -> None:
        """
        Record the tool stage a job has reached.

        Also refreshes ``last_used``: a long stage makes no requests, so without
        this the record would look idle to the reaper for the whole of it.
        """
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is None:
                return
            rec.stage = stage
            rec.stage_detail = detail
            rec.last_used = time.monotonic()
            if detail == "done" and stage and stage not in rec.stages_done:
                rec.stages_done.append(stage)

    def append_log(self, eda_id: str, line: str) -> None:
        """Add one line of tool output to the record's bounded tail."""
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is not None and line:
                rec.log_tail.append(line)

    def start_job(self, eda_id: str) -> bool:
        """
        Mark a job as started.  Returns False if one is already in flight.

        Refusing a concurrent run matters: two ``make`` invocations in the same ORFS
        working directory would corrupt each other's intermediate results in ways
        that surface much later as an inscrutable tool error.
        """
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is None or rec.job_in_flight:
                return False
            rec.job_started_at = time.monotonic()
            rec.done = False
            rec.result = None
            rec.stage = ""
            rec.stage_detail = ""
            rec.stages_done = []
            rec.log_tail.clear()
            rec.cancel = threading.Event()
            rec.phase = "running"
            rec.last_used = time.monotonic()
            return True

    def finish_job(self, eda_id: str, result: Dict[str, Any]) -> None:
        """Store a finished job's result and mark the record collectable again."""
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is None:
                return
            rec.result = result
            rec.done = True
            rec.job_started_at = 0.0
            rec.phase = "done" if result.get("status") == "ok" else "error"
            rec.last_used = time.monotonic()

    def request_cancel(self, eda_id: str) -> bool:
        """Signal an in-flight job to stop.  Returns False for an unknown id."""
        with self._lock:
            rec = self._records.get(eda_id)
            if rec is None:
                return False
            rec.cancel.set()
            return True

    def set_connection(
        self,
        eda_id: str,
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
            rec = self._records.get(eda_id)
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

    def list(self) -> List[EdaSummary]:
        with self._lock:
            return [self._summary(eid, rec) for eid, rec in self._records.items()]

    @staticmethod
    def _summary(eda_id: str, rec: EdaRecord) -> EdaSummary:
        return EdaSummary(
            eda_id=eda_id,
            kind=rec.spec.kind,
            name=rec.spec.name,
            pdk=rec.spec.pdk,
            instance_type=rec.spec.instance_type,
            pod_id=rec.pod_id,
            pod_status="running" if rec.is_running else "pending",
            phase=rec.phase,
            stage=rec.stage,
            uptime_s=rec.uptime_s,
            usd_per_hr=rec.usd_per_hr,
            warm_until=rec.warm_until,
        )

    def summary(self, eda_id: str) -> Optional[EdaSummary]:
        with self._lock:
            rec = self._records.get(eda_id)
            return self._summary(eda_id, rec) if rec is not None else None

    def delete(self, eda_id: str) -> bool:
        """Remove a pod from the registry (does NOT terminate it — see runtime)."""
        with self._lock:
            existed = self._records.pop(eda_id, None) is not None
        if existed and _PERSIST_ENABLED:
            self._delete_one(eda_id)
        return existed

    # ------------------------------------------------------------------
    # Idle reaper — terminate forgotten pods so they stop billing.
    # ------------------------------------------------------------------

    def _start_reaper(self) -> None:
        # Always start the daemon: even with the global idle timeout disabled it
        # must still expire per-record keep-warm holds, otherwise a warm pod would
        # never be freed.  A cheap 60s tick that does nothing on an empty registry.
        thread = threading.Thread(target=self._reap_loop, name="eda-reaper", daemon=True)
        thread.start()
        logger.info(
            "eda idle reaper started (global timeout=%s)",
            f"{_IDLE_TIMEOUT_MIN}min" if _IDLE_TIMEOUT_MIN > 0 else "off",
        )

    def _select_stale(self, now: float) -> List[tuple]:
        """
        Return (and remove from the registry) the records that should be reaped.

        A record with a job in flight is NEVER stale, however long it has been
        since anything touched it — that is the whole point of the check.  An
        OpenROAD route occupies a single SSH call for the better part of an hour;
        reaping on idle time alone would terminate the pod out from under it.
        """
        timeout_s = _IDLE_TIMEOUT_MIN * 60
        stale: List[tuple] = []
        with self._lock:
            for eid, rec in list(self._records.items()):
                if not rec.is_running:
                    continue
                if rec.job_in_flight:
                    continue
                if rec.keep_warm_deadline > 0:
                    expired = now > rec.keep_warm_deadline
                elif timeout_s > 0:
                    expired = (now - rec.last_used) > timeout_s
                else:
                    expired = False  # no warm window and global timeout disabled
                if expired:
                    stale.append((eid, rec))
                    self._records.pop(eid, None)
        return stale

    def _reap_loop(self) -> None:
        while True:
            time.sleep(_REAP_INTERVAL_S)
            try:
                for eid, rec in self._select_stale(time.monotonic()):
                    logger.info("reaping idle eda id=%s pod=%s", eid, rec.pod_id)
                    pod_client.terminate_pod(rec.api_key, rec.pod_id)
                    if _PERSIST_ENABLED:
                        self._delete_one(eid)
            except Exception as exc:  # noqa: BLE001 — the reaper must never die
                logger.warning("eda reaper tick failed: %s", exc)

    # ------------------------------------------------------------------
    # Persistence (spec only; never the private/api keys).  Best-effort.
    # ------------------------------------------------------------------

    def _save_one(self, eda_id: str, record: EdaRecord) -> None:
        try:
            os.makedirs(_PERSIST_DIR, exist_ok=True)
            path = os.path.join(_PERSIST_DIR, f"{eda_id}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"spec": record.spec.model_dump(), "pod_id": record.pod_id}, fh)
        except OSError as exc:
            logger.warning("failed to persist eda %s: %s", eda_id, exc)

    def _delete_one(self, eda_id: str) -> None:
        try:
            os.remove(os.path.join(_PERSIST_DIR, f"{eda_id}.json"))
        except OSError:
            pass

    def _load_from_disk(self) -> None:
        if not os.path.isdir(_PERSIST_DIR):
            return
        for fname in os.listdir(_PERSIST_DIR):
            if not fname.endswith(".json"):
                continue
            eda_id = fname[:-len(".json")]
            try:
                with open(os.path.join(_PERSIST_DIR, fname), encoding="utf-8") as fh:
                    data = json.load(fh)
                # The live pod/keys are gone after a restart — keep only the spec so
                # the block can re-provision via Regenerate.
                self._records[eda_id] = EdaRecord(spec=EdaSpec(**data.get("spec", {})))
            except (OSError, ValueError) as exc:
                logger.warning("failed to load persisted eda %s: %s", fname, exc)
        logger.info("loaded %d persisted eda spec(s)", len(self._records))


# Module-level singleton shared by the routers.
registry = EdaRegistry()
