"""
sessions.py
Per-conversation memory for channel-driven claws.

When a claw is wired to an inbound channel (Telegram/WhatsApp/Slack), each chat is
a long-running conversation.  This registry keeps a rolling transcript keyed by
``(claw_id, provider, chat_id)`` so successive messages share context — it is fed
into the claw's ``memory`` on the next run.

Deliberately mirrors ``registry.py``: a process-local dict guarded by a lock, with
optional JSON persistence gated by the same ``OPENCLAW_PERSIST`` env var.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Dict, Tuple

logger = logging.getLogger("openclaw.sessions")

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "_sessions")
_PERSIST_ENABLED = os.environ.get("OPENCLAW_PERSIST", "").lower() in ("1", "true", "yes")

# Cap the stored transcript so a busy chat cannot grow without bound.  The most
# recent characters are kept (older context is dropped from the front).
_MAX_CHARS = 8000

_Key = Tuple[str, str, str]


class SessionStore:
    """Thread-safe rolling transcripts for channel conversations."""

    def __init__(self) -> None:
        self._sessions: Dict[_Key, str] = {}
        self._lock = threading.Lock()
        if _PERSIST_ENABLED:
            self._load_from_disk()

    @staticmethod
    def _fname(key: _Key) -> str:
        claw_id, provider, chat_id = key
        safe = f"{claw_id}__{provider}__{chat_id}".replace(os.sep, "_").replace("/", "_")
        return safe + ".txt"

    def get(self, claw_id: str, provider: str, chat_id: str) -> str:
        with self._lock:
            return self._sessions.get((claw_id, provider, chat_id), "")

    def append(self, claw_id: str, provider: str, chat_id: str, role: str, text: str) -> None:
        """Append a ``role: text`` line, trimming the transcript to the size cap."""
        text = (text or "").strip()
        if not text:
            return
        key = (claw_id, provider, chat_id)
        line = f"{role}: {text}"
        with self._lock:
            existing = self._sessions.get(key, "")
            combined = (existing + "\n" + line) if existing else line
            if len(combined) > _MAX_CHARS:
                combined = combined[-_MAX_CHARS:]
            self._sessions[key] = combined
            transcript = combined
        if _PERSIST_ENABLED:
            self._save_one(key, transcript)

    # -- persistence (best-effort) ---------------------------------------

    def _save_one(self, key: _Key, transcript: str) -> None:
        try:
            os.makedirs(_PERSIST_DIR, exist_ok=True)
            with open(os.path.join(_PERSIST_DIR, self._fname(key)), "w", encoding="utf-8") as fh:
                fh.write(transcript)
        except OSError as exc:
            logger.warning("failed to persist session %s: %s", key, exc)

    def _load_from_disk(self) -> None:
        if not os.path.isdir(_PERSIST_DIR):
            return
        for fname in os.listdir(_PERSIST_DIR):
            if not fname.endswith(".txt"):
                continue
            stem = fname[: -len(".txt")]
            parts = stem.split("__", 2)
            if len(parts) != 3:
                continue
            try:
                with open(os.path.join(_PERSIST_DIR, fname), encoding="utf-8") as fh:
                    self._sessions[(parts[0], parts[1], parts[2])] = fh.read()
            except OSError as exc:
                logger.warning("failed to load session %s: %s", fname, exc)


# Module-level singleton shared by the router.
sessions = SessionStore()
