"""
device/app.py
Composition root and entrypoint for the Grafux devices service.

This builds the FastAPI app, wires CORS, starts the result-store sweeper, and
mounts every block-type router that shares this host/port:

    device  (this package)  — the hardware device hub: /ws + /devices/… + /broadcast
    robot                   — MELFA robot shortcut endpoints
    GPU                     — cloud-GPU runtime: /gpu/…
    openclaw                — AI-agent runtime: /claw/…

Run it with::

    uvicorn device.app:app --host 0.0.0.0 --port $PORT

Environment variables
---------------------
PORT         Port to bind (set automatically by Render).
AGENT_TOKEN  Shared secret every agent must supply in the /ws URL.
             Default "changeme" — always override in production!
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import router as device_router_mod
from . import robot as robot_router_mod
from .router import manager, results  # re-exported for tests / external scripts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("devices.app")

PORT: int = int(os.environ.get("PORT", "8000"))

if device_router_mod.AGENT_TOKEN == "changeme":
    logger.warning(
        "AGENT_TOKEN is set to the default value 'changeme'. "
        "Set the AGENT_TOKEN environment variable in production!"
    )


# ---------------------------------------------------------------------------
# Lifespan — run the result-store background sweeper for the app's lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    results.start_sweeper()
    logger.info("Result-store sweeper started")
    try:
        yield
    finally:
        await results.stop_sweeper()
        logger.info("Result-store sweeper stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Device Agent Server",
    description="WebSocket hub for managing remote agents (Raspberry Pi, etc.)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device block — the hardware hub (this package).
app.include_router(device_router_mod.router)
app.include_router(robot_router_mod.router)

# OpenClaw — server-side claw runtime (software AI agents; no device_id needed).
try:
    from openclaw.router import router as claw_router

    app.include_router(claw_router)
    logger.info("OpenClaw router mounted at /claw")
except Exception as exc:  # noqa: BLE001 — never let claw imports break device serving
    logger.warning("OpenClaw router not mounted: %s", exc)

# GPU — server-side cloud-GPU runtime (provision a RunPod GPU, compile + run on it).
try:
    from GPU.router import router as gpu_router

    app.include_router(gpu_router)
    logger.info("GPU router mounted at /gpu")
except Exception as exc:  # noqa: BLE001 — never let GPU imports break device serving
    logger.warning("GPU router not mounted: %s", exc)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Device Agent Server on 0.0.0.0:%d", PORT)
    uvicorn.run("device.app:app", host="0.0.0.0", port=PORT, reload=False)
