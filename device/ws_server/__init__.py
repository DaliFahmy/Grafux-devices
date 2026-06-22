# device.ws_server — the device-side WebSocket server.
#
# Runs *on the device* and lets Grafux connect to it directly (no central hub):
# it receives compile/run/shell requests over a WebSocket, executes them locally
# via ``handlers.dispatch``, and sends the results back.  See server.py.
