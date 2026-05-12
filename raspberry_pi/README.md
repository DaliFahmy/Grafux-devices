# Raspberry Pi Agent

This is the device agent that runs **on the Raspberry Pi** itself.  
It connects to the Device Agent WebSocket server, receives command blocks, and executes them locally — including downloading user code from S3 and compiling / running it.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you plan to use S3 downloads, make sure `boto3` has valid AWS credentials (see below).  
> If you only use direct/pre-signed URLs, `boto3` credentials are not required.

### 2. Set environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENT_HOST` | Yes | WebSocket URL of the devices server, e.g. `wss://devices-agent-server.onrender.com` |
| `DEVICE_ID` | Yes | Unique identifier for this Pi, e.g. `pi-001` |
| `AGENT_TOKEN` | Yes | Shared secret — must match the server's `AGENT_TOKEN` |
| `AWS_ACCESS_KEY_ID` | For S3 | AWS credentials for downloading code files |
| `AWS_SECRET_ACCESS_KEY` | For S3 | AWS credentials for downloading code files |
| `AWS_S3_BUCKET` | For S3 | Default bucket name, e.g. `grafux-user-files` |
| `AWS_REGION` | For S3 | AWS region (default: `us-east-1`) |
| `WORKSPACE_DIR` | No | Local folder for downloaded/compiled files (default: `~/pi_workspace`) |

Create a `.env` file or export them in your shell:

```bash
export AGENT_HOST=wss://devices-agent-server.onrender.com
export DEVICE_ID=pi-001
export AGENT_TOKEN=YOUR_AGENT_TOKEN
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_S3_BUCKET=grafux-user-files
export AWS_REGION=us-east-1
```

### 3. Run the agent

```bash
python agent.py
```

Or with flags:

```bash
python agent.py \
  --host wss://devices-agent-server.onrender.com \
  --device-id pi-001 \
  --token YOUR_AGENT_TOKEN
```

The agent reconnects automatically if the connection drops.

---

## Supported Commands

| Command | Description | Key Payload Fields |
|---------|-------------|-------------------|
| `ping` | Liveness check | — |
| `get_status` | System info (platform, uptime, workspace) | — |
| `run_code` | Execute an inline Python snippet | `code`, `timeout` |
| `shell` | Run a shell command | `command`, `timeout` |
| `set_config` | Push a config key/value | `key`, `value` |
| `restart` | Restart the agent process | — |
| `download_from_s3` | Download a file from S3 to the workspace | `s3_key`, `bucket`, `filename`; or `file_url` |
| `compile_and_run` | Compile (C/C++) and run a local file | `file_path`, `args`, `timeout` |
| `download_and_run` | Download from S3 then compile and run | `s3_key`, `filename`, `args`, `timeout`; or `file_url` |

---

## Using the Diagram Device Block

1. In the Grafux diagram editor, create a **Device** block (New Block → Device).
2. Set the input ports:
   - **`device_id`** — the ID of your rented Pi (e.g. `pi-001`)
   - **`command`** — `download_and_run`
   - **`file`** — S3 key of your code file (e.g. `users/42/alice/myproject/logic.py`)
   - **`text`** — *(optional)* CLI arguments to pass to your program
3. Run the block.  The Pi downloads the file, compiles it (if C/C++), runs it, and the output appears in the **`response`** and **`results`** output ports.

---

## Supported Languages

| Extension | Action |
|-----------|--------|
| `.py` | Run with `python3` |
| `.c` | Compile with `gcc -lm`, then run |
| `.cpp` / `.cc` / `.cxx` | Compile with `g++ -lm -lstdc++`, then run |
| `.sh` / `.bash` | Run with `/bin/bash` |

> Make sure `gcc` and `g++` are installed on the Pi:  
> `sudo apt-get install -y build-essential`

---

## Workspace Directory

Downloaded and compiled files are stored in `~/pi_workspace` by default.  
Override with the `WORKSPACE_DIR` environment variable or `--workspace` flag.

---

## Run as a systemd Service (optional)

Create `/etc/systemd/system/pi-agent.service`:

```ini
[Unit]
Description=Raspberry Pi Device Agent
After=network-online.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/raspberry_pi
ExecStart=/usr/bin/python3 /home/pi/raspberry_pi/agent.py
Restart=always
RestartSec=5
EnvironmentFile=/home/pi/.env

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable pi-agent
sudo systemctl start pi-agent
sudo journalctl -u pi-agent -f
```

curl -X POST https://grafux.onrender.com/devices/pi-001/run_code ^
  -H "Content-Type: application/json" ^
  -d "{\"code\":\"print(2+2)\",\"timeout\":10}"

  