"""
agent.py
Mitsubishi MELFA ROS2 Grafux Agent — rosbridge proxy (Option B).

This script runs on the robot machine (no ROS2 environment needed) and
acts as a bridge between the Grafux device server and the MELFA ROS2
driver via rosbridge_server.

How it works
------------
1. Connects outbound to the Grafux device server WebSocket.
2. Connects locally to rosbridge_server (default ws://localhost:9090).
3. On startup, subscribes to /joint_states on rosbridge (continuous cache).
4. When the Grafux server sends a command block, the handler is looked up
   in HANDLERS and awaited; the result is sent back as a WebSocket message.
5. Both connections reconnect automatically on disconnect.

Usage
-----
Install dependencies (no ROS2 environment required):
    pip install -r requirements.txt

Run:
    python agent.py \\
        --host      wss://your-devices-server.onrender.com \\
        --device-id melfa-001 \\
        --token     YOUR_AGENT_TOKEN \\
        --rosbridge ws://localhost:9090 \\
        --move-group melfa_arm \\
        --base-link  base_link \\
        --ee-link    link_6

Environment variables (alternative to flags):
    AGENT_HOST       Grafux device server WebSocket URL
    DEVICE_ID        Unique ID for this robot (e.g. melfa-001)
    AGENT_TOKEN      Shared secret (must match server AGENT_TOKEN env var)
    ROSBRIDGE_URL    rosbridge WebSocket URL (default ws://localhost:9090)
    MOVE_GROUP       MoveIt2 planning group (default melfa_arm)
    BASE_LINK        Robot base frame (default base_link)
    EE_LINK          End-effector frame (default link_6)
    JOINT_NAMES      Comma-separated joint names (default j1,j2,j3,j4,j5,j6)

Robot machine setup
-------------------
# Terminal 1 — MELFA driver + MoveIt2
source /opt/ros/humble/setup.bash
source ~/melfa_ws/install/setup.bash
ros2 launch melfa_bringup melfa_bringup.launch.py

# Terminal 2 — rosbridge
ros2 run rosbridge_server rosbridge_websocket --ros-args -p port:=9090

# Terminal 3 — this agent (no ROS2 env needed)
python agent.py --host wss://... --device-id melfa-001 --token ...
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

try:
    import websockets
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except ImportError:
    print("ERROR: websockets is not installed.  Run: pip install websockets")
    sys.exit(1)

from .ros_bridge import RosBridgeClient
from .handlers import HANDLERS, AgentConfig
from . import config as cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("melfa.agent")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "ws://localhost:8000"
DEFAULT_DEVICE_ID = "melfa-001"
DEFAULT_TOKEN = "changeme"
DEFAULT_ROSBRIDGE = "ws://localhost:9090"


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class MelfaAgent:
    """
    Dual-connection agent:
      - Grafux WebSocket: receives command blocks, sends results
      - rosbridge WebSocket: controls the ROS2 robot
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.grafux_host = args.host
        self.device_id = args.device_id
        self.token = args.token

        joint_names = (
            [n.strip() for n in args.joint_names.split(",")]
            if args.joint_names
            else list(cfg.JOINT_NAMES)
        )

        self.agent_cfg = AgentConfig(
            rosbridge_url=args.rosbridge,
            move_group=args.move_group,
            base_link=args.base_link,
            ee_link=args.ee_link,
            joint_names=joint_names,
        )

        self.rb = RosBridgeClient(args.rosbridge)
        self._grafux_ws = None

    # ------------------------------------------------------------------
    # rosbridge connection
    # ------------------------------------------------------------------

    async def _connect_rosbridge(self) -> None:
        """Connect to rosbridge and subscribe to essential topics."""
        delay = cfg.ROSBRIDGE_RECONNECT_DELAY_S
        while True:
            try:
                await self.rb.connect()
                # Subscribe to joint states for instant polling
                await self.rb.subscribe(
                    cfg.TOPIC_JOINT_STATES,
                    "sensor_msgs/msg/JointState",
                    callback=None,   # cache-only; handlers read via rb.get_cached()
                    throttle_rate=100,  # at most one message per 100 ms
                )
                logger.info("rosbridge ready — subscribed to %s", cfg.TOPIC_JOINT_STATES)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "rosbridge connection failed (%s) — retrying in %.0fs", exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, cfg.ROSBRIDGE_RECONNECT_MAX_S)

    async def _rosbridge_keepalive(self) -> None:
        """Resubscribe to topics whenever rosbridge reconnects after a drop."""
        while True:
            await asyncio.sleep(5)
            if not self.rb.is_connected:
                logger.info("rosbridge disconnected — reconnecting…")
                await self._connect_rosbridge()

    # ------------------------------------------------------------------
    # Grafux WebSocket loop
    # ------------------------------------------------------------------

    async def _grafux_loop(self) -> None:
        url = f"{self.grafux_host}/ws?device_id={self.device_id}&token={self.token}"

        while True:
            try:
                logger.info("Connecting to Grafux device server at %s …", url)
                async with websockets.connect(url) as ws:
                    self._grafux_ws = ws
                    logger.info(
                        "Connected to Grafux as device_id='%s'", self.device_id
                    )

                    async for raw_message in ws:
                        await self._handle_message(ws, raw_message)

            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                logger.warning("Grafux WebSocket closed: %s", exc)
            except OSError as exc:
                logger.error("Grafux connection failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Grafux unexpected error: %s", exc)
            finally:
                self._grafux_ws = None

            logger.info(
                "Reconnecting to Grafux in %.0fs…", cfg.GRAFUX_RECONNECT_DELAY_S
            )
            await asyncio.sleep(cfg.GRAFUX_RECONNECT_DELAY_S)

    async def _handle_message(self, ws, raw_message: str) -> None:
        """Parse a Grafux command block and dispatch to the appropriate handler."""
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning("Non-JSON message received from Grafux, ignoring.")
            return

        command_type = message.get("type", "unknown")
        command_id = message.get("id")
        payload = message.get("payload", {})

        logger.info("← command  type=%-30s  id=%s", command_type, command_id)

        handler = HANDLERS.get(command_type)
        if handler:
            try:
                result = await handler(payload, self.rb, self.agent_cfg)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Handler %s raised an exception", command_type)
                result = {"type": "error", "error": str(exc)}
        else:
            result = {
                "type": "unknown_command",
                "received_type": command_type,
                "available_commands": list(HANDLERS.keys()),
            }

        # Attach correlation fields so the Grafux server can match the result
        result["command_id"] = command_id
        result["device_id"] = self.device_id
        result["timestamp"] = time.time()

        await ws.send(json.dumps(result))
        logger.info("→ result   type=%-30s  success=%s", result.get("type"), result.get("success", "n/a"))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("MELFA Grafux Agent starting…")
        logger.info("  Grafux server : %s", self.grafux_host)
        logger.info("  Device ID     : %s", self.device_id)
        logger.info("  rosbridge     : %s", self.agent_cfg.rosbridge_url)
        logger.info("  Move group    : %s", self.agent_cfg.move_group)
        logger.info("  Base / EE     : %s / %s", self.agent_cfg.base_link, self.agent_cfg.ee_link)
        logger.info("  Joint names   : %s", self.agent_cfg.joint_names)

        # Connect to rosbridge first (non-blocking — agent still runs if robot is offline)
        try:
            await asyncio.wait_for(self._connect_rosbridge(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(
                "rosbridge not reachable at %s — continuing anyway. "
                "Robot commands will return errors until rosbridge is available.",
                self.agent_cfg.rosbridge_url,
            )

        # Run rosbridge keepalive and Grafux loop concurrently
        await asyncio.gather(
            self._rosbridge_keepalive(),
            self._grafux_loop(),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mitsubishi MELFA ROS2 Grafux Agent (rosbridge proxy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
        help="Grafux device server WebSocket URL",
    )
    parser.add_argument(
        "--device-id",
        default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
        help="Unique ID to register this robot with the Grafux server",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
        help="Shared secret token (must match server AGENT_TOKEN)",
    )
    parser.add_argument(
        "--rosbridge",
        default=os.environ.get("ROSBRIDGE_URL", DEFAULT_ROSBRIDGE),
        help="rosbridge_server WebSocket URL",
    )
    parser.add_argument(
        "--move-group",
        default=os.environ.get("MOVE_GROUP", cfg.DEFAULT_MOVE_GROUP),
        help="MoveIt2 planning group name",
    )
    parser.add_argument(
        "--base-link",
        default=os.environ.get("BASE_LINK", cfg.DEFAULT_BASE_LINK),
        help="Robot base frame name",
    )
    parser.add_argument(
        "--ee-link",
        default=os.environ.get("EE_LINK", cfg.DEFAULT_EE_LINK),
        help="End-effector frame name",
    )
    parser.add_argument(
        "--joint-names",
        default=os.environ.get("JOINT_NAMES", ",".join(cfg.JOINT_NAMES)),
        help="Comma-separated joint names matching the URDF (e.g. j1,j2,j3,j4,j5,j6)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    agent = MelfaAgent(args)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
