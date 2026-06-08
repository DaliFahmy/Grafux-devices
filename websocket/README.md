# Device-side WebSocket connection

Connect an external device to Grafux **directly** over WebSocket — no central
hub. You run a small server on the device; the Grafux `devices` block connects
straight to it, sends the source code you wrote, the device **compiles and runs
it locally**, and the program's output / errors / warnings show up in the block's
output ports.

```
Grafux app  ──ws://device-host:8765/?token=…──►  device_ws_server.py (your device)
   write C++ here                                    compile + run here
        └──────────────── output ports ◄──────────── results back
```

## What's here

| File                   | Purpose                                                         |
| ---------------------- | -------------------------------------------------------------- |
| `device_ws_server.py`  | The WebSocket server you run **on the device**.                |
| `handlers.py`          | Compile/run logic (cpp, c, python, shell) → output-port fields.|
| `client_example.py`    | Reference client / manual test harness.                        |
| `requirements.txt`     | `websockets` (compilers come from the OS).                      |

## Run the server on the device

```bash
pip install -r requirements.txt

# C/C++ support needs a toolchain on the device:
#   Debian/Ubuntu/Raspberry Pi OS:  sudo apt install build-essential
#   macOS:                          xcode-select --install

python device_ws_server.py --port 8765 --token <your-secret> --device-id mydev
```

The server binds `0.0.0.0:8765` by default. Clients must supply the same token as
a query parameter (`ws://host:8765/?token=<your-secret>`); a mismatch is rejected
with close code `1008`.

Environment variables (alternatives to flags): `DEVICE_WS_PORT`, `AGENT_TOKEN`,
`DEVICE_ID`.

## Quick test

```bash
# Terminal 1 (device):
python device_ws_server.py --token test

# Terminal 2:
python client_example.py --url ws://localhost:8765 --token test
# → compile_and_run (C++, inline source)
#   status   : ok
#   output   : '42'
```

## Wire up a Grafux `devices` block

1. Start `device_ws_server.py` on the device and note its address, e.g.
   `ws://192.168.1.50:8765`.
2. Add a `devices` block in Grafux and set its input ports:
   - **`device_id`** → the device address, e.g. `ws://192.168.1.50:8765`
   - **`code`** → the source you want to run (e.g. a full C++ program)
   - **`language`** → `cpp` | `c` | `python` | `shell` (defaults to `cpp`)
   - **`args`** *(optional)* → command-line arguments
   - **`command`** *(optional)* → action override (`compile_and_run` default,
     or `shell` / `status`)
3. Run the block. The compiled program's stdout lands in `output`/`results`,
   compiler/runtime errors in `errors`, warnings in `warnings`, plus a `status`
   and a human-readable `response`. Any files the program writes are returned in
   `files`.

## Message protocol (clean direct API)

Client → device (one JSON text frame per request; `id` correlates the reply):

```json
{ "id": "<uuid>", "action": "compile_and_run",
  "language": "cpp",
  "code": "#include <iostream>\nint main(){ std::cout << 6*7; }",
  "args": "", "timeout": 120 }
```

Other actions: `run_code` (inline Python), `shell` (`command`), `status`, `ping`.

Device → device after compiling/running:

```json
{ "id": "<uuid>", "action": "compile_and_run", "status": "ok",
  "output": "42", "errors": "", "warnings": "",
  "response": "ok — 1 line output, 0 errors, 0 warnings",
  "stdout": ["42"], "stderr": [], "compile_stderr": [], "files": [],
  "device_id": "mydev", "timestamp": 1699999999.0 }
```

`status` is `ok` | `compile_error` | `runtime_error` | `timeout` | `error`. For
`compile_error`, the compiler diagnostics are split into `errors` vs `warnings`.

## Notes & limitations

- **Desktop app first.** A browser page served over HTTPS (the Grafux web/WASM
  build) cannot open an insecure `ws://` to a LAN device (mixed-content + the
  device has no TLS cert). Use the desktop app, or front the device with `wss://`
  / a relay for the web build.
- This direct path is **independent** of the central-hub stack
  (`devices_server.py`, `raspberry_pi/agent.py`), which still works for devices
  that connect outward over WebSocket.
- Security: code is compiled and executed on the device with the server process's
  privileges. Run it only on trusted networks with a strong `--token`.
