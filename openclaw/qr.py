"""
qr.py
Render a QR code as a PNG ``data:`` URI for the claw block's ``qr_code`` port.

A claw's ``qr_code`` output port carries a scannable QR of an app's authorization
link (Composio connect / OAuth redirect URL).  The user scans it with their phone
to authorize WhatsApp / Telegram / Slack / … without leaving the canvas.

``segno`` is a pure-python QR encoder (no Pillow / native deps), imported lazily —
like ``anthropic`` / ``httpx`` elsewhere in OpenClaw — so the devices server still
boots where it is absent (``qr_data_uri`` then returns "").
"""

from __future__ import annotations

import logging

logger = logging.getLogger("openclaw.qr")


def qr_data_uri(text: str, scale: int = 5, border: int = 2) -> str:
    """
    Encode ``text`` as a QR code and return it as a ``data:image/png;base64,…`` URI.

    Returns "" for empty input or when ``segno`` is not installed — callers treat an
    empty result as "no QR" and fall back to showing the raw link in guidance.
    """
    text = (text or "").strip()
    if not text:
        return ""
    try:
        import segno  # lazy — see module docstring
    except ImportError:
        logger.warning("qr: 'segno' not installed — returning no QR (pip install segno)")
        return ""
    try:
        return segno.make(text, error="m").png_data_uri(scale=scale, border=border)
    except Exception as exc:  # noqa: BLE001 — never let QR rendering break a run
        logger.warning("qr: failed to render QR for %r: %s", text[:60], exc)
        return ""
