"""
registry.py
In-memory registry of provisioned claws, with optional JSON persistence so claws
survive a server restart.

The registry stores ``ClawSpec`` objects keyed by a generated ``claw_id``.  It is
deliberately simple — a process-local dict guarded by a lock — which is adequate
for the single-process uvicorn deployment used by the devices server.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from typing import Dict, List, Optional

from .models import ClawSpec, ClawSummary

logger = logging.getLogger("openclaw.registry")

# Directory used for optional persistence.  Disabled unless OPENCLAW_PERSIST is set
# to a truthy value, because the default Render deployment has an ephemeral disk.
_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "_claws")
_PERSIST_ENABLED = os.environ.get("OPENCLAW_PERSIST", "").lower() in ("1", "true", "yes")


class ClawRegistry:
    """Thread-safe store of provisioned claws."""

    def __init__(self) -> None:
        self._claws: Dict[str, ClawSpec] = {}
        self._lock = threading.Lock()
        if _PERSIST_ENABLED:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, spec: ClawSpec) -> str:
        """Register a new claw and return its generated id."""
        claw_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._claws[claw_id] = spec
        logger.info("created claw id=%s name=%r", claw_id, spec.name)
        if _PERSIST_ENABLED:
            self._save_one(claw_id, spec)
        return claw_id

    def get(self, claw_id: str) -> Optional[ClawSpec]:
        with self._lock:
            return self._claws.get(claw_id)

    def list(self) -> List[ClawSummary]:
        with self._lock:
            return [
                ClawSummary(claw_id=cid, name=spec.name, agent=spec.agent)
                for cid, spec in self._claws.items()
            ]

    def delete(self, claw_id: str) -> bool:
        with self._lock:
            existed = self._claws.pop(claw_id, None) is not None
        if existed and _PERSIST_ENABLED:
            self._delete_one(claw_id)
        return existed

    # ------------------------------------------------------------------
    # Persistence (best-effort; failures are logged, never fatal)
    # ------------------------------------------------------------------

    def _save_one(self, claw_id: str, spec: ClawSpec) -> None:
        try:
            os.makedirs(_PERSIST_DIR, exist_ok=True)
            path = os.path.join(_PERSIST_DIR, f"{claw_id}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(spec.model_dump(), fh)
        except OSError as exc:
            logger.warning("failed to persist claw %s: %s", claw_id, exc)

    def _delete_one(self, claw_id: str) -> None:
        try:
            os.remove(os.path.join(_PERSIST_DIR, f"{claw_id}.json"))
        except OSError:
            pass

    def _load_from_disk(self) -> None:
        if not os.path.isdir(_PERSIST_DIR):
            return
        for fname in os.listdir(_PERSIST_DIR):
            if not fname.endswith(".json"):
                continue
            claw_id = fname[:-len(".json")]
            try:
                with open(os.path.join(_PERSIST_DIR, fname), encoding="utf-8") as fh:
                    self._claws[claw_id] = ClawSpec(**json.load(fh))
            except (OSError, ValueError) as exc:
                logger.warning("failed to load persisted claw %s: %s", fname, exc)
        logger.info("loaded %d persisted claw(s)", len(self._claws))


# Module-level singleton shared by the router.
registry = ClawRegistry()
