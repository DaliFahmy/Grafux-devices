"""
raspberry_pi/agent.py
Raspberry Pi device agent — runs on the Pi itself.

Connects to the Device Agent WebSocket server, listens for command blocks,
executes them locally, and sends results back.  Supports downloading user code
files from S3 and compiling / running them directly on the Pi.

Usage
-----
Install dependencies:
    pip install -r requirements.txt

Run:
    python agent.py

Or with explicit flags:
    python agent.py --host wss://grafux.onrender.com --device-id pi-001 --token topsecret

Environment variables (all optional when flags are used):

    AGENT_HOST              WebSocket server URL
    DEVICE_ID               Unique ID for this Pi
    AGENT_TOKEN             Shared secret (must match server AGENT_TOKEN)
    AWS_ACCESS_KEY_ID       AWS credentials for S3 downloads
    AWS_SECRET_ACCESS_KEY   AWS credentials for S3 downloads
    AWS_S3_BUCKET           Default S3 bucket name
    AWS_REGION              AWS region (default: us-east-1)
    WORKSPACE_DIR           Local directory for downloaded/compiled files
                            (default: ~/pi_workspace)
                            
    curl -X POST wss://grafux.onrender.com/devices/pi-001/status
"""

import argparse
import asyncio
import io
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

try:
    import websockets
except ImportError:
    print("ERROR: websockets is not installed.  Run: pip install websockets")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pi.agent")

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST      = "ws://localhost:8000"
DEFAULT_DEVICE_ID = "pi-001"
DEFAULT_TOKEN     = "changeme"
RECONNECT_DELAY   = 5  # seconds between reconnect attempts

WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR",
    os.path.join(os.path.expanduser("~"), "pi_workspace"),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_workspace() -> str:
    """Create the workspace directory if it does not exist and return its path."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR


def _split_diagnostics(lines: list) -> tuple:
    """Split compiler output lines into (warnings, errors).

    A line is a warning if it contains 'warning:' and an error if it contains
    'error:' or 'fatal:'.  Context lines (notes, carets, etc.) are ignored so
    that the port values stay concise.
    """
    warnings, errors = [], []
    for line in lines:
        ll = line.lower()
        if "warning:" in ll:
            warnings.append(line)
        elif "error:" in ll or "fatal:" in ll:
            errors.append(line)
    return warnings, errors


def _detect_language(filename: str) -> str:
    """Return a language tag based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".py":  "python",
        ".c":   "c",
        ".cpp": "cpp",
        ".cc":  "cpp",
        ".cxx": "cpp",
        ".sh":  "shell",
        ".bash":"shell",
    }.get(ext, "unknown")


# ---------------------------------------------------------------------------
# Command handlers — standard (same as client_example.py)
# ---------------------------------------------------------------------------


def handle_ping(payload: dict) -> dict:
    return {"type": "pong", "message": "pong"}


def handle_get_status(payload: dict) -> dict:
    info = {
        "platform": platform.platform(),
        "python":   sys.version,
        "hostname": platform.node(),
        "uptime_seconds": None,
        "workspace_dir": WORKSPACE_DIR,
    }
    try:
        with open("/proc/uptime") as f:
            info["uptime_seconds"] = float(f.read().split()[0])
    except Exception:
        pass
    return {"type": "status_report", "data": info}


def _collect_files(directory: str) -> list:
    """Scan a directory and return a list of base64-encoded file objects."""
    import base64
    import mimetypes
    files = []
    try:
        for fname in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "rb") as f:
                    content = base64.b64encode(f.read()).decode("utf-8")
                mime_type, _ = mimetypes.guess_type(fname)
                files.append({
                    "name":      fname,
                    "content":   content,
                    "mime_type": mime_type or "application/octet-stream",
                })
            except Exception as exc:
                logger.warning("Could not encode file %s: %s", fpath, exc)
    except Exception as exc:
        logger.warning("Could not scan output directory %s: %s", directory, exc)
    return files


def handle_run_code(payload: dict) -> dict:
    """Execute an inline Python snippet."""
    import contextlib
    import tempfile

    code    = payload.get("code", "")
    timeout = int(payload.get("timeout", 30))
    stdout_lines: list = []
    stderr_lines: list = []

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(compile(code, "<device>", "exec"), {})  # noqa: S102
                stdout_lines = stdout_buf.getvalue().splitlines()
                stderr_lines = stderr_buf.getvalue().splitlines()
                status = "ok"
            except Exception as exc:
                stdout_lines = stdout_buf.getvalue().splitlines()
                stderr_lines = stderr_buf.getvalue().splitlines() or [str(exc)]
                status = "error"
        finally:
            os.chdir(old_cwd)

        files = _collect_files(tmpdir)

    # Classify stderr lines: runtime exceptions land here as errors
    _warnings, _errors = _split_diagnostics(stderr_lines)
    # Any stderr line that isn't a structured warning/error is still an error
    _error_str = "\n".join(stderr_lines) if stderr_lines else ""

    _n_out = len(stdout_lines)
    _n_err = len(_errors) or (1 if status == "error" else 0)

    return {
        "type":     "run_code_result",
        "status":   status,
        # --- Grafux output ports ---
        "output":   "\n".join(stdout_lines),
        "errors":   _error_str,
        "warnings": "\n".join(_warnings),
        "response": f"{status} \u2014 {_n_out} line{'s' if _n_out != 1 else ''} output, {_n_err} error{'s' if _n_err != 1 else ''}",
        # --- raw lists kept for debugging ---
        "stdout":   stdout_lines,
        "stderr":   stderr_lines,
        "files":    files,
    }


def handle_shell(payload: dict) -> dict:
    """Run an arbitrary shell command."""
    command = payload.get("command", "")
    timeout = int(payload.get("timeout", 30))
    try:
        result = subprocess.run(
            command,
            shell=True,          # noqa: S602
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "type":       "shell_result",
            "status":     "ok",
            "returncode": result.returncode,
            "stdout":     result.stdout.splitlines(),
            "stderr":     result.stderr.splitlines(),
        }
    except subprocess.TimeoutExpired:
        return {"type": "shell_result", "status": "timeout", "command": command}
    except Exception as exc:
        return {"type": "shell_result", "status": "error", "error": str(exc)}


def handle_set_config(payload: dict) -> dict:
    key   = payload.get("key")
    value = payload.get("value")
    logger.info("Config update: %s = %s", key, value)
    return {"type": "config_ack", "key": key, "value": value}


def handle_restart(payload: dict) -> dict:
    logger.info("Restart requested — restarting agent in 2 s …")
    asyncio.get_event_loop().call_later(
        2, lambda: os.execv(sys.executable, [sys.executable] + sys.argv)
    )
    return {"type": "restart_ack", "message": "Restarting agent..."}


# ---------------------------------------------------------------------------
# Command handlers — Raspberry Pi specific
# ---------------------------------------------------------------------------


def handle_download_from_s3(payload: dict) -> dict:
    """
    Download a file from S3 onto the Pi's local workspace.

    Payload fields
    --------------
    s3_key   (str)           : Key inside the S3 bucket (e.g. users/42/alice/logic.py).
    filename (str, optional) : Local filename to save as. Defaults to the basename of s3_key.
    bucket   (str, optional) : S3 bucket name. Falls back to AWS_S3_BUCKET env var.
    file_url (str, optional) : Direct / pre-signed URL. Used instead of s3_key when provided.
    """
    workspace = _ensure_workspace()
    file_url  = payload.get("file_url", "").strip()
    s3_key    = payload.get("s3_key",   "").strip()
    filename  = payload.get("filename", "").strip()
    bucket    = payload.get("bucket",   "").strip() or os.environ.get("AWS_S3_BUCKET", "")

    if not filename:
        source = file_url or s3_key
        filename = os.path.basename(source.rstrip("/")) if source else "downloaded_file"

    local_path = os.path.join(workspace, filename)

    try:
        if file_url:
            # Download from a direct / pre-signed URL using requests
            try:
                import requests  # noqa: PLC0415
            except ImportError:
                return {
                    "type":   "download_result",
                    "status": "error",
                    "error":  "requests package not installed — run: pip install requests",
                }
            resp = requests.get(file_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
        elif s3_key:
            if not bucket:
                return {
                    "type":   "download_result",
                    "status": "error",
                    "error":  "No S3 bucket specified. Set AWS_S3_BUCKET env var or pass 'bucket' in payload.",
                }
            try:
                import boto3  # noqa: PLC0415
            except ImportError:
                return {
                    "type":   "download_result",
                    "status": "error",
                    "error":  "boto3 not installed — run: pip install boto3",
                }
            s3 = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
            s3.download_file(bucket, s3_key, local_path)
        else:
            return {
                "type":   "download_result",
                "status": "error",
                "error":  "Payload must include either 's3_key' or 'file_url'.",
            }

        file_size = os.path.getsize(local_path)
        logger.info("Downloaded '%s' → %s (%d bytes)", s3_key or file_url, local_path, file_size)
        return {
            "type":       "download_result",
            "status":     "ok",
            "local_path": local_path,
            "filename":   filename,
            "size_bytes": file_size,
        }

    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return {"type": "download_result", "status": "error", "error": str(exc)}


def handle_compile_and_run(payload: dict) -> dict:
    """
    Compile (if needed) and run a file that already exists in the workspace.

    Payload fields
    --------------
    file_path (str)           : Absolute or workspace-relative path to the file.
    args      (str, optional) : Command-line arguments string passed to the program.
    timeout   (int, optional) : Max execution seconds (default 120).
    """
    workspace = _ensure_workspace()
    file_path = payload.get("file_path", "").strip()
    args      = payload.get("args",      "").strip()
    timeout   = int(payload.get("timeout", 120))

    if not file_path:
        return {"type": "compile_run_result", "status": "error", "error": "file_path is required"}

    # Resolve relative paths against the workspace
    if not os.path.isabs(file_path):
        file_path = os.path.join(workspace, file_path)

    if not os.path.exists(file_path):
        return {
            "type":   "compile_run_result",
            "status": "error",
            "error":  f"File not found: {file_path}",
        }

    language = _detect_language(file_path)
    logger.info("compile_and_run: %s (%s)", file_path, language)

    compile_stdout: list = []
    compile_stderr: list = []
    compile_returncode: int = 0

    try:
        if language == "python":
            run_cmd = [sys.executable, file_path] + (args.split() if args else [])

        elif language == "c":
            binary = file_path.replace(".c", "")
            compile_result = subprocess.run(
                ["gcc", "-o", binary, file_path, "-lm"],
                capture_output=True, text=True, timeout=60,
            )
            compile_stdout      = compile_result.stdout.splitlines()
            compile_stderr      = compile_result.stderr.splitlines()
            compile_returncode  = compile_result.returncode
            if compile_returncode != 0:
                _w, _e = _split_diagnostics(compile_stderr)
                return {
                    "type":               "compile_run_result",
                    "status":             "compile_error",
                    # --- Grafux output ports ---
                    "output":             "",
                    "errors":             "\n".join(_e or compile_stderr),
                    "warnings":           "\n".join(_w),
                    "response":           f"compile_error \u2014 {len(_e)} error{'s' if len(_e) != 1 else ''}, {len(_w)} warning{'s' if len(_w) != 1 else ''}",
                    # --- raw ---
                    "compile_stdout":     compile_stdout,
                    "compile_stderr":     compile_stderr,
                    "compile_returncode": compile_returncode,
                }
            run_cmd = [binary] + (args.split() if args else [])

        elif language == "cpp":
            binary = os.path.splitext(file_path)[0]
            compile_result = subprocess.run(
                ["g++", "-o", binary, file_path, "-lm", "-lstdc++"],
                capture_output=True, text=True, timeout=60,
            )
            compile_stdout      = compile_result.stdout.splitlines()
            compile_stderr      = compile_result.stderr.splitlines()
            compile_returncode  = compile_result.returncode
            if compile_returncode != 0:
                _w, _e = _split_diagnostics(compile_stderr)
                return {
                    "type":               "compile_run_result",
                    "status":             "compile_error",
                    # --- Grafux output ports ---
                    "output":             "",
                    "errors":             "\n".join(_e or compile_stderr),
                    "warnings":           "\n".join(_w),
                    "response":           f"compile_error \u2014 {len(_e)} error{'s' if len(_e) != 1 else ''}, {len(_w)} warning{'s' if len(_w) != 1 else ''}",
                    # --- raw ---
                    "compile_stdout":     compile_stdout,
                    "compile_stderr":     compile_stderr,
                    "compile_returncode": compile_returncode,
                }
            run_cmd = [binary] + (args.split() if args else [])

        elif language == "shell":
            os.chmod(file_path, 0o755)
            run_cmd = ["/bin/bash", file_path] + (args.split() if args else [])

        else:
            return {
                "type":   "compile_run_result",
                "status": "error",
                "error":  f"Unsupported file type: {os.path.basename(file_path)}. "
                          "Supported: .py, .c, .cpp, .cc, .cxx, .sh",
            }

        # Snapshot files in workspace before running so we can detect new ones
        import tempfile
        with tempfile.TemporaryDirectory() as run_outdir:
            before_files = set(os.listdir(workspace))
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_outdir,
            )
            # Collect any files the program wrote to its CWD
            output_files = _collect_files(run_outdir)
            # Also pick up any new files written directly to the workspace
            after_files = set(os.listdir(workspace))
            new_workspace_files = after_files - before_files
            for fname in sorted(new_workspace_files):
                fpath = os.path.join(workspace, fname)
                if os.path.isfile(fpath):
                    import base64, mimetypes as _mt
                    try:
                        with open(fpath, "rb") as f:
                            content = base64.b64encode(f.read()).decode("utf-8")
                        mime_type, _ = _mt.guess_type(fname)
                        output_files.append({
                            "name":      fname,
                            "content":   content,
                            "mime_type": mime_type or "application/octet-stream",
                        })
                    except Exception as exc:
                        logger.warning("Could not encode workspace file %s: %s", fpath, exc)

        _run_stdout = run_result.stdout.splitlines()
        _run_stderr = run_result.stderr.splitlines()
        _run_status = "ok" if run_result.returncode == 0 else "runtime_error"

        # Split compile diagnostics into warnings vs errors for output ports
        _c_warnings, _c_errors = _split_diagnostics(compile_stderr)
        # Runtime stderr are always errors
        _rt_warnings, _rt_errors = _split_diagnostics(_run_stderr)
        _all_errors   = _c_errors   + _rt_errors
        _all_warnings = _c_warnings + _rt_warnings

        _n_out  = len(_run_stdout)
        _n_err  = len(_all_errors)
        _n_warn = len(_all_warnings)

        return {
            "type":     "compile_run_result",
            "status":   _run_status,
            "language": language,
            # --- Grafux output ports ---
            "output":   "\n".join(_run_stdout),
            "errors":   "\n".join(_all_errors   + ([s for s in _run_stderr if s not in _all_errors]   if _run_status == "runtime_error" else [])),
            "warnings": "\n".join(_all_warnings),
            "response": f"{_run_status} \u2014 {_n_out} line{'s' if _n_out != 1 else ''} output, {_n_err} error{'s' if _n_err != 1 else ''}, {_n_warn} warning{'s' if _n_warn != 1 else ''}",
            # --- raw ---
            "compile_stdout":     compile_stdout,
            "compile_stderr":     compile_stderr,
            "compile_returncode": compile_returncode,
            "stdout":             _run_stdout,
            "stderr":             _run_stderr,
            "returncode":         run_result.returncode,
            "files":              output_files,
        }

    except subprocess.TimeoutExpired:
        return {"type": "compile_run_result", "status": "timeout", "language": language}
    except FileNotFoundError as exc:
        # Compiler not found
        return {
            "type":   "compile_run_result",
            "status": "error",
            "error":  f"Tool not found: {exc}. Install gcc/g++ on the Pi.",
        }
    except Exception as exc:
        logger.error("compile_and_run failed: %s", exc)
        return {"type": "compile_run_result", "status": "error", "error": str(exc)}


def handle_download_and_run(payload: dict) -> dict:
    """
    Download a file from S3 (or URL) and immediately compile / run it.

    Payload fields
    --------------
    s3_key   (str)           : S3 key of the code file.
    filename (str, optional) : Local filename to save as.
    bucket   (str, optional) : S3 bucket name. Falls back to AWS_S3_BUCKET env var.
    file_url (str, optional) : Pre-signed / direct URL (alternative to s3_key).
    args     (str, optional) : CLI args string for the program.
    timeout  (int, optional) : Execution timeout in seconds (default 120).
    """
    # Step 1 — download
    dl_result = handle_download_from_s3(payload)
    if dl_result.get("status") != "ok":
        dl_result["type"] = "download_and_run_result"
        return dl_result

    # Step 2 — compile + run
    run_payload = {
        "file_path": dl_result["local_path"],
        "args":      payload.get("args", ""),
        "timeout":   payload.get("timeout", 120),
    }
    run_result = handle_compile_and_run(run_payload)

    # Merge download info into the run result
    run_result["type"]       = "download_and_run_result"
    run_result["local_path"] = dl_result["local_path"]
    run_result["filename"]   = dl_result["filename"]
    run_result["size_bytes"] = dl_result["size_bytes"]
    return run_result


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: dict = {
    "ping":               handle_ping,
    "get_status":         handle_get_status,
    "run_code":           handle_run_code,
    "shell":              handle_shell,
    "set_config":         handle_set_config,
    "restart":            handle_restart,
    "download_from_s3":   handle_download_from_s3,
    "compile_and_run":    handle_compile_and_run,
    "download_and_run":   handle_download_and_run,
}

# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


async def run_agent(host: str, device_id: str, token: str) -> None:
    url = f"{host}/ws?device_id={device_id}&token={token}"

    while True:
        try:
            logger.info("Connecting to %s …", url)
            async with websockets.connect(url) as ws:
                logger.info("Connected as device_id='%s'  workspace='%s'", device_id, WORKSPACE_DIR)

                async for raw_message in ws:
                    try:
                        message = json.loads(raw_message)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message received, ignoring.")
                        continue

                    command_type = message.get("type", "unknown")
                    command_id   = message.get("id")
                    payload      = message.get("payload", {})

                    logger.info("← received command type=%s id=%s", command_type, command_id)

                    handler = HANDLERS.get(command_type)
                    if handler:
                        try:
                            result = handler(payload)
                        except Exception as exc:
                            result = {"type": "error", "error": str(exc)}
                    else:
                        result = {"type": "unknown_command", "received_type": command_type}

                    result["command_id"] = command_id
                    result["device_id"]  = device_id
                    result["timestamp"]  = time.time()

                    await ws.send(json.dumps(result))
                    logger.info("→ sent result type=%s status=%s",
                                result.get("type"), result.get("status", "—"))

        except websockets.exceptions.ConnectionClosedError as exc:
            logger.warning("Connection closed: %s", exc)
        except OSError as exc:
            logger.error("Connection failed: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)

        logger.info("Reconnecting in %d s …", RECONNECT_DELAY)
        await asyncio.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi device agent")
    parser.add_argument(
        "--host",
        default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
        help="Server WebSocket URL (e.g. wss://your-app.onrender.com)",
    )
    parser.add_argument(
        "--device-id",
        default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
        help="Unique ID for this Pi",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
        help="Shared secret token (must match server AGENT_TOKEN)",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("WORKSPACE_DIR", WORKSPACE_DIR),
        help="Local directory for downloaded / compiled files",
    )
    args = parser.parse_args()

    # Allow --workspace to override the module-level default
    WORKSPACE_DIR = args.workspace
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    try:
        asyncio.run(run_agent(args.host, args.device_id, args.token))
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
