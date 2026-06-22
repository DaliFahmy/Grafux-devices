# device — the Grafux device block, end to end.
#
# A "device" block represents a real piece of hardware (Raspberry Pi, Windows
# workstation, MELFA robot, Jetson, Arduino, …) that Grafux can send code and
# commands to and read results back from.  Unlike the GPU and OpenClaw blocks —
# which run entirely *inside* the devices server — the device block has two
# halves that talk over a WebSocket:
#
#     server-side hub  (this package, mounted in device.app)
#         The FastAPI app that Grafux-app talks to.  It tracks every connected
#         device, relays command blocks to them over the /ws socket, and hands
#         their results back to the REST callers as Grafux output ports.
#
#     device-side agents  (device.ws_server, device.agents.*)
#         The code that runs *on the hardware*.  ``device.ws_server`` is a
#         standalone WebSocket server a device can host directly; the per-device
#         agents in ``device.agents`` (raspberry_pi, windows, melfa, …) connect
#         out to the hub and execute compile/run/shell/robot commands locally.
#
# Layout (mirrors the GPU/ and openclaw/ packages):
#
#     app.py       — composition root + entrypoint (``uvicorn device.app:app``)
#     router.py    — the hub REST + /ws endpoints
#     robot.py     — the MELFA robot shortcut endpoints (a sub-router)
#     runtime.py   — orchestration: command send/wait + output-port mapping
#     registry.py  — ConnectionManager: live device sockets + pending waiters
#     results.py   — ResultStore: TTL result cache for the poll/fire-and-forget path
#     commands.py  — command-block builders sent to devices
#     models.py    — typed request bodies for the REST endpoints
