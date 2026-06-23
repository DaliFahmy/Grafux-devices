"""
device/ws_server/discovery.py
Optional LAN mDNS/Zeroconf advertisement for the device-side WebSocket server.

Advertises the service type ``_grafux-device._tcp.local.`` so the Grafux app can
discover this device on the local network and offer it in the device picker —
instead of the user hand-typing a ws:// URL.

This is intentionally best-effort and fully optional:
  * if the ``zeroconf`` package is not installed, or
  * if the ``DEVICE_MDNS_DISABLE`` environment variable is set,
``start_advertiser`` returns ``None`` and the server runs exactly as before.
Headless/cloud deployments therefore need no extra dependency.
"""

from __future__ import annotations

import logging
import os
import socket

logger = logging.getLogger("device.ws.discovery")

SERVICE_TYPE = "_grafux-device._tcp.local."


def _primary_ipv4() -> str:
    """Best-effort local IPv4 (the address used to reach the default route)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # no packets sent; just picks the iface
        return s.getsockname()[0]
    except Exception:                        # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def start_advertiser(device_id: str, port: int, version: str = "1",
                     device_type: str = "generic"):
    """Register the mDNS service and return an opaque handle, or ``None``.

    The handle is passed back to :func:`stop_advertiser` for clean shutdown.
    """
    if os.environ.get("DEVICE_MDNS_DISABLE"):
        logger.info("mDNS advertisement disabled via DEVICE_MDNS_DISABLE")
        return None
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        logger.info("zeroconf not installed — skipping LAN mDNS advertisement "
                    "(pip install zeroconf to enable device discovery)")
        return None

    try:
        ip = _primary_ipv4()
        safe_id = "".join(c for c in device_id if c.isalnum() or c in "-_") or "device"
        info = ServiceInfo(
            SERVICE_TYPE,
            f"{safe_id}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=int(port),
            properties={
                "device_id": device_id,
                "type":      device_type,
                "version":   version,
            },
            server=f"{safe_id}.local.",
        )
        zc = Zeroconf()
        zc.register_service(info)
        logger.info("Advertising %s on %s:%d via mDNS", device_id, ip, port)
        return (zc, info)
    except Exception as exc:                 # noqa: BLE001
        logger.warning("mDNS advertisement failed: %s", exc)
        return None


def stop_advertiser(handle) -> None:
    """Unregister and close a handle returned by :func:`start_advertiser`."""
    if not handle:
        return
    zc, info = handle
    try:
        zc.unregister_service(info)
    except Exception:                        # noqa: BLE001
        pass
    try:
        zc.close()
    except Exception:                        # noqa: BLE001
        pass
