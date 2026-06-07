# Windows Device Agent

This is the device agent that runs **on the Windows machine itself**.  
It connects to the Device Agent WebSocket server, receives command blocks, and executes them locally — including downloading user code from S3, running scripts, capturing camera images, and taking screenshots.

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** `opencv-python` requires a camera to use `capture_image`.  
> `mss` and `Pillow` are required for `screenshot`.  
> `psutil` is required for `get_system_info` and `list_processes`.  
> If you only use S3 downloads, make sure `boto3` has valid AWS credentials (see below).

### 2. Set environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENT_HOST` | Yes | WebSocket URL of the devices server, e.g. `wss://grafux.onrender.com` |
| `DEVICE_ID` | Yes | Unique identifier for this Windows machine, e.g. `win-001` |
| `AGENT_TOKEN` | Yes | Shared secret — must match the server's `AGENT_TOKEN` |
| `AWS_ACCESS_KEY_ID` | For S3 | AWS credentials for downloading code files |
| `AWS_SECRET_ACCESS_KEY` | For S3 | AWS credentials for downloading code files |
| `AWS_S3_BUCKET` | For S3 | Default bucket name, e.g. `grafux-user-files` |
| `AWS_REGION` | For S3 | AWS region (default: `us-east-1`) |
| `WORKSPACE_DIR` | No | Local folder for downloaded/run files (default: `%USERPROFILE%\win_workspace`) |

Create a `.env` file or set them in PowerShell:

```powershell
$env:AGENT_HOST     = "wss://grafux.onrender.com"
$env:DEVICE_ID      = "win-001"
$env:AGENT_TOKEN    = "YOUR_AGENT_TOKEN"
$env:AWS_ACCESS_KEY_ID     = "AKIA..."
$env:AWS_SECRET_ACCESS_KEY = "..."
$env:AWS_S3_BUCKET  = "grafux-user-files"
$env:AWS_REGION     = "us-east-1"
```

### 3. Run the agent

```powershell
python agent.py
```

Or with explicit flags:

```powershell
python agent.py `
  --host wss://grafux.onrender.com `
  --device-id win-001 `
  --token YOUR_AGENT_TOKEN
```

The agent reconnects automatically if the connection drops.

---

## Supported Commands

### Standard Commands

| Command | Description | Key Payload Fields |
|---------|-------------|-------------------|
| `ping` | Liveness check | — |
| `get_status` | System info (platform, uptime, CPU, RAM) | — |
| `run_code` | Execute an inline Python snippet | `code`, `timeout` |
| `shell` | Run a PowerShell command (or `cmd.exe` with `use_powershell: false`) | `command`, `timeout`, `use_powershell` |
| `set_config` | Push a config key/value | `key`, `value` |
| `restart` | Restart the agent process | — |
| `download_from_s3` | Download a file from S3 to the workspace | `s3_key`, `bucket`, `filename`; or `file_url` |
| `compile_and_run` | Run a local file (Python, batch, PowerShell, or C/C++ with MinGW) | `file_path`, `args`, `timeout` |
| `download_and_run` | Download from S3 then run | `s3_key`, `filename`, `args`, `timeout`; or `file_url` |

### Windows-Specific Commands

| Command | Description | Key Payload Fields |
|---------|-------------|-------------------|
| `capture_image` | Open a camera and capture a PNG frame | `camera_index`, `filename`, `compress_level`, `warmup_frames` |
| `screenshot` | Capture the screen or a region as PNG | `filename`, `region` (`top`, `left`, `width`, `height`), `monitor` |
| `get_system_info` | Detailed CPU, RAM, disk, and network stats | — |
| `list_processes` | List top running processes | `limit`, `sort_by` (`cpu` or `memory`) |
| `run_powershell` | Execute a PowerShell script string | `script`, `timeout` |

---

## Using the Diagram Device Block

1. In the Grafux diagram editor, create a **Device** block (New Block → Device).
2. Set the input ports:
   - **`device_id`** — the ID of your Windows machine (e.g. `win-001`)
   - **`command`** — one of the supported commands above
   - **`file`** — *(for download_and_run)* S3 key of your code file
   - **`text`** — *(optional)* CLI arguments or script content
3. Run the block. The agent executes the command and the output appears in the output ports.

### Output Ports

| Port | Content |
|------|---------|
| `output` | Standard output text, or image capture summary |
| `errors` | Error messages |
| `warnings` | Compiler / runtime warnings |
| `status` | `ok`, `error`, `timeout`, `compile_error`, `runtime_error` |
| `response` | Human-readable summary (e.g. `ok — 1920x1080 JPEG, 84320 bytes`) |
| `file` | Filename of the captured image (set by the Grafux client after upload) |
| `files` | List of base64-encoded output files (images, generated files) |

### Where captured images are stored

| Location | Path | Who writes it |
|----------|------|---------------|
| Windows machine (agent) | `WORKSPACE_DIR\<filename>` e.g. `C:\Users\ahmed\win_workspace\capture.png` | Windows agent (saved immediately on capture) |
| Grafux client machine | `<blockFolder>\outputs\<filename>` | Grafux client (`blockrunner.cpp`) |
| S3 | `users/<id>/<user>/<project>/devices/<cat>/<block>/outputs/<filename>` e.g. `users/27/dali/test1/devices/general/win_001/outputs/capture.jpg` | Grafux client (auto-synced from the block folder) |

The agent always saves to its local workspace first, then returns the image as base64 in `files[0]`. The Grafux client receives this, saves it to the device block's `outputs/` folder, and uploads to S3 at the correct project path — identical to how all other device block output files are stored.

### Camera Capture Example

```
device_id  →  win-001
command    →  capture_image
```

The captured PNG is saved to `WORKSPACE_DIR\capture.png` on the Windows machine. The Grafux client stores it at the device block's `outputs/capture.png` and the **`file`** output port is set to `capture.png`. The full image is also available via the **`files`** output port.

### Screenshot Example

```
device_id  →  win-001
command    →  screenshot
```

To capture a specific region, send the command via the generic `command` endpoint:

```json
{
  "type": "screenshot",
  "payload": { "region": { "top": 0, "left": 0, "width": 1280, "height": 720 } }
}
```

---

## Supported Languages for `compile_and_run`

| Extension | Action |
|-----------|--------|
| `.py` | Run with `python` |
| `.bat` / `.cmd` | Run with `cmd.exe /c` |
| `.ps1` | Run with `powershell.exe -File` |
| `.c` | Compile with `gcc -lm`, then run *(requires MinGW on PATH)* |
| `.cpp` / `.cc` / `.cxx` | Compile with `g++ -lm -lstdc++`, then run *(requires MinGW on PATH)* |

> To install MinGW on Windows, use [winget](https://learn.microsoft.com/windows/package-manager/winget/):
> ```powershell
> winget install --id=MSYS2.MSYS2
> ```
> Then add `C:\msys64\mingw64\bin` to your PATH.

---

## Workspace Directory

Downloaded and run files are stored in `%USERPROFILE%\win_workspace` by default.  
Override with the `WORKSPACE_DIR` environment variable or `--workspace` flag.

---

## Run as a Windows Service (optional)

Use [NSSM](https://nssm.cc) (Non-Sucking Service Manager) to run the agent as a background Windows Service that starts automatically on boot.

### 1. Download NSSM

```powershell
winget install NSSM.NSSM
```

### 2. Install the service

Open an elevated (Administrator) PowerShell:

```powershell
nssm install GrafuxWindowsAgent python.exe
nssm set GrafuxWindowsAgent AppDirectory "C:\path\to\windows"
nssm set GrafuxWindowsAgent AppParameters "agent.py --host wss://grafux.onrender.com --device-id win-001 --token YOUR_TOKEN"
nssm set GrafuxWindowsAgent AppEnvironmentExtra `
  "AGENT_HOST=wss://grafux.onrender.com" `
  "DEVICE_ID=win-001" `
  "AGENT_TOKEN=YOUR_TOKEN" `
  "AWS_ACCESS_KEY_ID=AKIA..." `
  "AWS_SECRET_ACCESS_KEY=..." `
  "AWS_S3_BUCKET=grafux-user-files"
nssm set GrafuxWindowsAgent Start SERVICE_AUTO_START
nssm start GrafuxWindowsAgent
```

### 3. Check service status

```powershell
nssm status GrafuxWindowsAgent
# View logs written by NSSM:
Get-Content "$env:APPDATA\nssm\GrafuxWindowsAgent\*.log" -Tail 50
```

### 4. Stop / remove the service

```powershell
nssm stop    GrafuxWindowsAgent
nssm remove  GrafuxWindowsAgent confirm
```

---

## Quick Test (PowerShell)

```powershell
# Ping the agent
Invoke-RestMethod -Method POST `
  -Uri "https://grafux.onrender.com/devices/win-001/ping"

# Capture a camera image
Invoke-RestMethod -Method POST `
  -Uri "https://grafux.onrender.com/devices/win-001/command" `
  -ContentType "application/json" `
  -Body '{"type":"capture_image","payload":{"camera_index":0,"quality":90}}'

# Take a screenshot
Invoke-RestMethod -Method POST `
  -Uri "https://grafux.onrender.com/devices/win-001/command" `
  -ContentType "application/json" `
  -Body '{"type":"screenshot","payload":{}}'

# Get system info
Invoke-RestMethod -Method POST `
  -Uri "https://grafux.onrender.com/devices/win-001/command" `
  -ContentType "application/json" `
  -Body '{"type":"get_system_info","payload":{}}'

# Run a PowerShell snippet
Invoke-RestMethod -Method POST `
  -Uri "https://grafux.onrender.com/devices/win-001/command" `
  -ContentType "application/json" `
  -Body '{"type":"run_powershell","payload":{"script":"Get-Date"}}'
```
