"""
devices_server.py
Backward-compatibility shim.

The devices service has moved into the self-contained ``device`` package.  The
ASGI app now lives in ``device.app``; deploy it with::

    uvicorn device.app:app --host 0.0.0.0 --port $PORT

This module is kept only so existing imports (``from devices_server import app``)
and any old ``uvicorn devices_server:app`` invocations keep working.
"""

from device.app import app, manager  # noqa: F401  (re-exported for compatibility)

if __name__ == "__main__":
    import uvicorn

    from device.app import PORT

    uvicorn.run("device.app:app", host="0.0.0.0", port=PORT, reload=False)
