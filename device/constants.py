"""
device/constants.py
Single source of truth for the device-block server's tunables.

Centralizes the config that was previously read inline in several modules
(``AGENT_TOKEN`` in router.py and ws_server/server.py, magic timeouts in
router.py / results.py).  Every value is overridable via an environment
variable so deployments can tune behavior without code changes.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

#: Shared secret every agent must supply in the /ws URL.  Override in production.
AGENT_TOKEN: str = os.environ.get("AGENT_TOKEN", "changeme")

# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------

#: Seconds an uncollected device result is retained before the sweeper drops it.
RESULT_TTL_S: int = int(os.environ.get("DEVICE_RESULT_TTL_S", "120"))

# ---------------------------------------------------------------------------
# Request/reply wait headroom
# ---------------------------------------------------------------------------
# How much longer than the command's own ``timeout`` the hub waits for the
# device's reply, to cover transport + scheduling latency (and, for
# download_and_run, the download + compile time that precedes execution).

#: Extra wait headroom for run_code / compile_and_run (seconds).
RUN_WAIT_BUFFER_S: int = int(os.environ.get("DEVICE_RUN_WAIT_BUFFER_S", "10"))

#: Extra wait headroom for download_and_run — adds download + compile time.
DOWNLOAD_WAIT_BUFFER_S: int = int(os.environ.get("DEVICE_DOWNLOAD_WAIT_BUFFER_S", "30"))
