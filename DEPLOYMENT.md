# Device Agent Server — Deployment Guide

## Overview

This service is a FastAPI WebSocket hub that remote agents (Raspberry Pi, etc.) connect to.
Once connected, you can send typed command blocks to any device and receive results back in real time.

---

## Project Structure

```
Grafux-devices/
├── devices_server.py     # FastAPI app — WebSocket endpoint + REST API
├── connection_manager.py # Tracks active device connections
├── commands.py           # Command block builders
├── requirements.txt      # Python dependencies
├── render.yaml           # Render deployment config
├── client_example.py     # Agent client (runs on the Raspberry Pi)
└── DEPLOYMENT.md         # This file
```

---

## 1. Deploy to Render

### Step 1 — Create a new Web Service

1. Go to [https://render.com](https://render.com) and log in.
2. Click **New → Web Service**.
3. Connect your GitHub/GitLab repository.
4. Leave **Root Directory** empty (the repo root contains `devices_server.py` directly).

### Step 2 — Configure build & start commands

| Setting | Value |
|---------|-------|
| Environment | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn devices_server:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

> **Tip:** You can also use `render.yaml` (already in this folder) for infrastructure-as-code deployment. Link it in the Render dashboard under **Blueprint**.

### Step 3 — Set environment variables

In the Render dashboard, go to your service → **Environment**:

| Variable | Value | Notes |
|----------|-------|-------|
| `AGENT_TOKEN` | *(generate a strong random string)* | Every device must use this token |
| `PYTHON_VERSION` | `3.11.0` | Ensures consistent runtime |
| `ANTHROPIC_API_KEY` | *(your Anthropic key, `sk-ant-…`)* | **Required for the OpenClaw runtime** — used to run and AI-scaffold claws (`/claw/*`). Declared as `sync: false` in `render.yaml`, so set the value here in the dashboard. |

> **Generate a secure token:**
> ```bash
> python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```
>
> **OpenClaw note:** the `anthropic` dependency is already in `requirements.txt` and the
> `/claw/*` endpoints are mounted automatically by `devices_server.py`. If `ANTHROPIC_API_KEY`
> is unset, the rest of the server still runs, but claw run/scaffold calls return a graceful
> "no API key" result instead of an AI response.

### Step 4 — Deploy

Click **Deploy**. Render will build and start the service. Copy the public URL shown in the dashboard (e.g. `https://devices-agent-server.onrender.com`).

---

## 2. Run the Agent on a Raspberry Pi (or any device)

### Install the dependency

```bash
pip install websockets
```

### Connect the agent

```bash
python client_example.py \
  --host wss://devices-agent-server.onrender.com \
  --device-id pi-001 \
  --token YOUR_AGENT_TOKEN
```

You can also use environment variables instead of flags:

```bash
export AGENT_HOST=wss://devices-agent-server.onrender.com
export DEVICE_ID=pi-001
export AGENT_TOKEN=YOUR_AGENT_TOKEN
python client_example.py
```

The client reconnects automatically if the connection drops.

---

## 3. Send Commands to a Device

All REST endpoints are available at your Render URL.

### List connected devices

```bash
curl https://devices-agent-server.onrender.com/devices
```

```json
{"devices": ["pi-001", "pi-002"]}
```

### Ping a device

```bash
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/ping
```

### Get device status

```bash
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/status
```

### Run Python code on a device

```bash
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/run_code \
     -H "Content-Type: application/json" \
     -d '{"code": "print(1 + 1)", "timeout": 10}'
```

### Run a shell command on a device

```bash
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/shell \
     -H "Content-Type: application/json" \
     -d '{"command": "uptime", "timeout": 5}'
```

### Send a raw command block

```bash
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/command \
     -H "Content-Type: application/json" \
     -d '{"type": "shell", "payload": {"command": "hostname -I"}}'
```

### Broadcast to all devices

```bash
curl -X POST https://devices-agent-server.onrender.com/broadcast \
     -H "Content-Type: application/json" \
     -d '{"type": "ping"}'
```

---

## 4. Supported Command Types

| Type | Description | Payload fields |
|------|-------------|---------------|
| `ping` | Liveness check | *(none)* |
| `get_status` | System info (platform, uptime, etc.) | *(none)* |
| `run_code` | Execute Python code | `code`, `timeout` |
| `shell` | Run a shell command | `command`, `timeout` |
| `set_config` | Push a config key/value | `key`, `value` |
| `restart` | Restart the agent process | *(none)* |
| `download_from_s3` | Download a file from S3 to the Pi workspace | `s3_key`, `filename`, `bucket`; or `file_url` |
| `compile_and_run` | Compile (C/C++) and run a workspace file | `file_path`, `args`, `timeout` |
| `download_and_run` | Download from S3 then compile and run | `s3_key`, `filename`, `args`, `timeout`; or `file_url` |

> The `download_from_s3`, `compile_and_run`, and `download_and_run` commands require the Raspberry Pi agent (`raspberry_pi/agent.py`) instead of the generic `client_example.py`.

### Raspberry Pi — download and run example

```bash
# Download a Python file from S3 and run it on pi-001
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/download_and_run \
     -H "Content-Type: application/json" \
     -d '{
       "s3_key": "users/42/alice/myproject/logic.py",
       "bucket": "grafux-user-files",
       "args": "--mode test",
       "timeout": 60
     }'
```

```bash
# Download a C program from a pre-signed URL and run it
curl -X POST https://devices-agent-server.onrender.com/devices/pi-001/download_and_run \
     -H "Content-Type: application/json" \
     -d '{
       "file_url": "https://grafux-user-files.s3.amazonaws.com/users/42/alice/myproject/sort.c?X-Amz-...",
       "filename": "sort.c",
       "timeout": 90
     }'
```

---

## 5. Interactive API Docs

FastAPI generates interactive docs automatically:

- **Swagger UI:** `https://devices-agent-server.onrender.com/docs`
- **ReDoc:** `https://devices-agent-server.onrender.com/redoc`

---

## 6. Run Locally

```bash
pip install -r requirements.txt
AGENT_TOKEN=mysecret uvicorn devices_server:app --reload
```

Then connect a test client:

```bash
python client_example.py --host ws://localhost:8000 --device-id test-device --token mysecret
```

---

## 7. Security Notes

- The `AGENT_TOKEN` is a shared secret. **Never commit it to version control.**
- Every device must supply the correct token in the WebSocket URL query string.
  Connections with an invalid or missing token are rejected immediately (WebSocket close code 1008).
- In a multi-tenant setup, consider per-device tokens stored in a database and validated in the WebSocket handshake.
- Render services use HTTPS/WSS by default — your WebSocket traffic is encrypted in transit.
