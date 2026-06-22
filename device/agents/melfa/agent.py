"""
agent.py
Mitsubishi MELFA ROS2 Grafux Agent — rosbridge proxy.

Runs on the robot machine (no ROS2 environment needed) and bridges the Grafux
device hub to the MELFA ROS2 driver via rosbridge_server.

The Grafux WebSocket loop (connect / reconnect / keepalive / dispatch / result
envelope) is shared with every other agent via
``device.agents.base.BaseAgent``.  This agent adds the second connection to
rosbridge and exposes async robot handlers.

How it works
------------
1. Connects outbound to the Grafux device hub (BaseAgent).
2. Connects locally to rosbridge_server (default ws://localhost:9090).
3. Subscribes to /joint_states on rosbridge (continuous cache).
4. Robot command blocks are dispatched to async handlers in ``handlers.py``.
5. Both connections reconnect automatically; rosbridge is closed cleanly on exit.

Usage
-----
    pip install -r requirements.txt
    python agent.py --host wss://... --device-id melfa-001 --token ... \\
        --rosbridge ws://localhost:9090 --move-group melfa_arm

Environment variables mirror the flags: AGENT_HOST, DEVICE_ID, AGENT_TOKEN,
ROSBRIDGE_URL, MOVE_GROUP, BASE_LINK, EE_LINK, JOINT_NAMES.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from device.agents.base import BaseAgent

from . import config as cfg
from .handlers import HANDLERS, AgentConfig
from .ros_bridge import RosBridgeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("melfa.agent")

DEFAULT_HOST = "ws://localhost:8000"
DEFAULT_DEVICE_ID = "melfa-001"
DEFAULT_TOKEN = "changeme"
DEFAULT_ROSBRIDGE = "ws://localhost:9090"

#: How long to wait for rosbridge at startup before continuing without it.
ROSBRIDGE_CONNECT_TIMEOUT_S = float(getattr(cfg, "ROSBRIDGE_CONNECT_TIMEOUT_S", 10.0))


class MelfaAgent(BaseAgent):
    """BaseAgent + a rosbridge side-channel for robot control."""

    def __init__(self, args: argparse.Namespace) -> None:
        joint_names = (
            [n.strip() for n in args.joint_names.split(",")]
            if args.joint_names else list(cfg.JOINT_NAMES)
        )
        self.agent_cfg = AgentConfig(
            rosbridge_url=args.rosbridge,
            move_group=args.move_group,
            base_link=args.base_link,
            ee_link=args.ee_link,
            joint_names=joint_names,
        )
        self.rb = RosBridgeClient(args.rosbridge)

        # Bind the (payload, rb, agent_cfg) robot handlers to this agent so they
        # match BaseAgent's (payload) -> dict handler contract.
        handlers = {name: self._bind(fn) for name, fn in HANDLERS.items()}
        super().__init__(args.host, args.device_id, args.token, handlers=handlers, logger=logger)

    def _bind(self, fn):
        async def wrapped(payload: dict) -> dict:
            return await fn(payload, self.rb, self.agent_cfg)
        return wrapped

    # ------------------------------------------------------------------
    # rosbridge connection
    # ------------------------------------------------------------------

    async def _connect_rosbridge(self) -> None:
        """Connect to rosbridge and subscribe to essential topics (with backoff)."""
        delay = cfg.ROSBRIDGE_RECONNECT_DELAY_S
        while True:
            try:
                await self.rb.connect()
                await self.rb.subscribe(
                    cfg.TOPIC_JOINT_STATES,
                    "sensor_msgs/msg/JointState",
                    callback=None,        # cache-only; handlers read via rb.get_cached()
                    throttle_rate=100,    # at most one message per 100 ms
                )
                logger.info("rosbridge ready — subscribed to %s", cfg.TOPIC_JOINT_STATES)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("rosbridge connection failed (%s) — retrying in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, cfg.ROSBRIDGE_RECONNECT_MAX_S)

    async def _rosbridge_keepalive(self) -> None:
        """Reconnect to rosbridge whenever it drops."""
        while True:
            await asyncio.sleep(5)
            if not self.rb.is_connected:
                logger.info("rosbridge disconnected — reconnecting…")
                await self._connect_rosbridge()

    # ------------------------------------------------------------------
    # BaseAgent hooks
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        logger.info("MELFA Grafux Agent starting…")
        logger.info("  Grafux server : %s", self.host)
        logger.info("  Device ID     : %s", self.device_id)
        logger.info("  rosbridge     : %s", self.agent_cfg.rosbridge_url)
        logger.info("  Move group    : %s", self.agent_cfg.move_group)
        logger.info("  Base / EE     : %s / %s", self.agent_cfg.base_link, self.agent_cfg.ee_link)
        logger.info("  Joint names   : %s", self.agent_cfg.joint_names)
        # Connect to rosbridge but don't block forever — the agent still serves
        # Grafux (returning robot errors) if the robot is offline.
        try:
            await asyncio.wait_for(self._connect_rosbridge(), timeout=ROSBRIDGE_CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "rosbridge not reachable at %s — continuing; robot commands will error "
                "until rosbridge is available.", self.agent_cfg.rosbridge_url,
            )

    async def extra_tasks(self):
        return [self._rosbridge_keepalive()]

    async def on_shutdown(self) -> None:
        # Close rosbridge cleanly (cancel subscriptions / pending goals).
        try:
            await self.rb.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.debug("rosbridge disconnect error: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mitsubishi MELFA ROS2 Grafux Agent (rosbridge proxy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
                        help="Grafux device server WebSocket URL")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
                        help="Unique ID to register this robot with the Grafux server")
    parser.add_argument("--token", default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
                        help="Shared secret token (must match server AGENT_TOKEN)")
    parser.add_argument("--rosbridge", default=os.environ.get("ROSBRIDGE_URL", DEFAULT_ROSBRIDGE),
                        help="rosbridge_server WebSocket URL")
    parser.add_argument("--move-group", default=os.environ.get("MOVE_GROUP", cfg.DEFAULT_MOVE_GROUP),
                        help="MoveIt2 planning group name")
    parser.add_argument("--base-link", default=os.environ.get("BASE_LINK", cfg.DEFAULT_BASE_LINK),
                        help="Robot base frame name")
    parser.add_argument("--ee-link", default=os.environ.get("EE_LINK", cfg.DEFAULT_EE_LINK),
                        help="End-effector frame name")
    parser.add_argument("--joint-names", default=os.environ.get("JOINT_NAMES", ",".join(cfg.JOINT_NAMES)),
                        help="Comma-separated joint names matching the URDF")
    return parser.parse_args()


if __name__ == "__main__":
    MelfaAgent(_parse_args()).run_forever()
