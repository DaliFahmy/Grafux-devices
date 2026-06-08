"""
websocket/client_example.py
Reference client / manual test harness for the device-side WebSocket server.

Connects to a running ``device_ws_server.py``, sends an inline C++ program with
``action=compile_and_run`` (the primary Grafux flow), then a quick Python
``run_code``, and prints the port-mapped results.

Usage
-----
    # Terminal 1 — on the device:
    python device_ws_server.py --token test

    # Terminal 2:
    python client_example.py --url ws://localhost:8765 --token test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import websockets

# Make output safe on consoles with a non-UTF-8 default encoding (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

CPP_SOURCE = """#include <iostream>
int main() {
    std::cout << 6 * 7;   // -> 42
    return 0;
}
"""


def _summarize(result: dict) -> str:
    return (
        f"  status   : {result.get('status')}\n"
        f"  output   : {result.get('output')!r}\n"
        f"  errors   : {result.get('errors')!r}\n"
        f"  warnings : {result.get('warnings')!r}\n"
        f"  response : {result.get('response')}"
    )


async def _request(ws, message: dict) -> dict:
    message.setdefault("id", str(uuid.uuid4()))
    await ws.send(json.dumps(message))
    return json.loads(await ws.recv())


async def main(url: str, token: str) -> None:
    async with websockets.connect(f"{url}/?token={token}") as ws:
        print("→ compile_and_run (C++, inline source)")
        r = await _request(ws, {
            "action": "compile_and_run", "language": "cpp",
            "code": CPP_SOURCE, "timeout": 60,
        })
        print(_summarize(r))

        print("\n→ run_code (Python)")
        r = await _request(ws, {"action": "run_code", "code": "print(6 * 7)"})
        print(_summarize(r))

        print("\n→ status")
        r = await _request(ws, {"action": "status"})
        print(_summarize(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Device WebSocket server test client")
    parser.add_argument("--url", default="ws://localhost:8765", help="Server URL")
    parser.add_argument("--token", default="changeme", help="Shared secret token")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.token))
