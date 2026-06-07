"""
windows/agent.py
Windows device agent — runs on the Windows machine itself.

Connects to the Device Agent WebSocket server, listens for command blocks,
executes them locally, and sends results back.  Supports downloading user code
files from S3 and running them, plus Windows-specific features like camera
capture, screenshots, and system info.

Usage
-----
Install dependencies:
    pip install -r requirements.txt

Run:
    python agent.py

Or with explicit flags:
    python agent.py --host wss://grafux.onrender.com --device-id win-001 --token topsecret

Environment variables:

    AGENT_HOST=wss://grafux.onrender.com
    AGENT_TOKEN
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_S3_BUCKET
    AWS_REGION

    AGENT_HOST              WebSocket server URL
    DEVICE_ID               Unique ID for this Windows machine
    AGENT_TOKEN             Shared secret (must match server AGENT_TOKEN)
    AWS_ACCESS_KEY_ID       AWS credentials for S3 downloads
    AWS_SECRET_ACCESS_KEY   AWS credentials for S3 downloads
    AWS_S3_BUCKET           Default S3 bucket name (e.g. grafux-user-files)
    AWS_REGION              AWS region (default: us-east-1)
    WORKSPACE_DIR           Local directory for downloaded/run files
                            (default: %USERPROFILE%\\win_workspace)
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import platform
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
logger = logging.getLogger("windows.agent")

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST      = "wss://grafux.onrender.com"
DEFAULT_DEVICE_ID = "win-001"
DEFAULT_TOKEN     = "changeme"
RECONNECT_DELAY   = 5  # seconds between reconnect attempts

WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR",
    os.path.join(os.path.expanduser("~"), "win_workspace"),
)

DEVICE_ID: str = os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_workspace() -> str:
    """Create the workspace directory if it does not exist and return its path."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR


def _split_diagnostics(lines: list) -> tuple:
    """Split compiler/runtime output lines into (warnings, errors)."""
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
        ".py":   "python",
        ".c":    "c",
        ".cpp":  "cpp",
        ".cc":   "cpp",
        ".cxx":  "cpp",
        ".bat":  "batch",
        ".cmd":  "batch",
        ".ps1":  "powershell",
    }.get(ext, "unknown")


def _collect_files(directory: str) -> list:
    """Scan a directory and return a list of base64-encoded file objects."""
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


# ---------------------------------------------------------------------------
# Command handlers — standard (shared with Pi, adapted for Windows)
# ---------------------------------------------------------------------------


def handle_ping(payload: dict) -> dict:
    return {"type": "pong", "message": "pong"}


def handle_get_status(payload: dict) -> dict:
    info = {
        "platform":       platform.platform(),
        "python":         sys.version,
        "hostname":       platform.node(),
        "uptime_seconds": None,
        "workspace_dir":  WORKSPACE_DIR,
    }
    try:
        import psutil  # noqa: PLC0415
        boot_ts = psutil.boot_time()
        info["uptime_seconds"] = time.time() - boot_ts
        vm = psutil.virtual_memory()
        info["ram_total_mb"]  = round(vm.total  / 1024 / 1024)
        info["ram_used_mb"]   = round(vm.used   / 1024 / 1024)
        info["ram_percent"]   = vm.percent
        info["cpu_count"]     = psutil.cpu_count(logical=True)
        info["cpu_percent"]   = psutil.cpu_percent(interval=0.5)
    except ImportError:
        pass
    return {"type": "status_report", "data": info}


def handle_run_code(payload: dict) -> dict:
    """Execute an inline Python snippet."""
    import contextlib  # noqa: PLC0415

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

    _warnings, _errors = _split_diagnostics(stderr_lines)
    _error_str = "\n".join(stderr_lines) if stderr_lines else ""
    _n_out = len(stdout_lines)
    _n_err = len(_errors) or (1 if status == "error" else 0)

    return {
        "type":     "run_code_result",
        "status":   status,
        "output":   "\n".join(stdout_lines),
        "errors":   _error_str,
        "warnings": "\n".join(_warnings),
        "response": f"{status} \u2014 {_n_out} line{'s' if _n_out != 1 else ''} output, {_n_err} error{'s' if _n_err != 1 else ''}",
        "stdout":   stdout_lines,
        "stderr":   stderr_lines,
        "files":    files,
    }


def handle_shell(payload: dict) -> dict:
    """Run an arbitrary shell command via PowerShell (falls back to cmd.exe)."""
    command = payload.get("command", "")
    timeout = int(payload.get("timeout", 30))
    use_powershell = payload.get("use_powershell", True)

    if use_powershell:
        full_cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        full_cmd = ["cmd.exe", "/c", command]

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        _n_out = len(stdout_lines)
        _run_status = "ok" if result.returncode == 0 else "error"
        return {
            "type":       "shell_result",
            "status":     _run_status,
            "returncode": result.returncode,
            "output":     result.stdout,
            "errors":     result.stderr,
            "warnings":   "",
            "response":   f"{_run_status} \u2014 {_n_out} line{'s' if _n_out != 1 else ''} output, returncode={result.returncode}",
            "stdout":     stdout_lines,
            "stderr":     stderr_lines,
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
# Command handlers — S3 / file download (shared with Pi)
# ---------------------------------------------------------------------------


def handle_download_from_s3(payload: dict) -> dict:
    """
    Download a file from S3 into the local workspace.

    Payload fields
    --------------
    s3_key   (str)           : Key inside the S3 bucket.
    filename (str, optional) : Local filename to save as.
    bucket   (str, optional) : S3 bucket name. Falls back to AWS_S3_BUCKET env var.
    file_url (str, optional) : Direct / pre-signed URL (used instead of s3_key).
    """
    workspace = _ensure_workspace()
    file_url  = payload.get("file_url", "").strip()
    s3_key    = payload.get("s3_key",   "").strip()
    filename  = payload.get("filename", "").strip()
    bucket    = payload.get("bucket",   "").strip() or os.environ.get("AWS_S3_BUCKET", "")

    # If s3_key is a bare filename (no path separators) it means the Grafux client
    # failed to resolve it to a full S3 path.  Fall back to a local workspace copy
    # so the user can place scripts directly in WORKSPACE_DIR without needing S3.
    if s3_key and "/" not in s3_key and "\\" not in s3_key and not file_url:
        local_name = filename or s3_key
        workspace_copy = os.path.join(workspace, local_name)
        if os.path.isfile(workspace_copy):
            logger.info(
                "download_from_s3: bare key '%s' — found local copy in workspace: %s",
                s3_key, workspace_copy,
            )
            return {
                "type":       "download_result",
                "status":     "ok",
                "local_path": workspace_copy,
                "filename":   local_name,
                "size_bytes": os.path.getsize(workspace_copy),
                "source":     "workspace_local",
            }
        # Not in workspace either — return a clear diagnostic
        return {
            "type":   "download_result",
            "status": "error",
            "error":  (
                f"'{s3_key}' is a bare filename — no full S3 path was provided. "
                f"To fix this: either (a) place '{s3_key}' in {workspace} on this "
                f"machine and run again, or (b) enable auto-sync in the Grafux "
                f"desktop app so the file is uploaded to S3 with a full project path."
            ),
        }

    if not filename:
        source   = file_url or s3_key
        filename = os.path.basename(source.rstrip("/")) if source else "downloaded_file"

    local_path = os.path.join(workspace, filename)

    try:
        if file_url:
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

    Supported extensions on Windows:
    - .py          — run with Python
    - .bat / .cmd  — run with cmd.exe
    - .ps1         — run with PowerShell
    - .c / .cpp    — compile with gcc/g++ (MinGW required on PATH)

    Payload fields
    --------------
    file_path (str)           : Absolute or workspace-relative path to the file.
    args      (str, optional) : CLI arguments string passed to the program.
    timeout   (int, optional) : Max execution seconds (default 120).
    """
    workspace = _ensure_workspace()
    file_path = payload.get("file_path", "").strip()
    args      = payload.get("args",      "").strip()
    timeout   = int(payload.get("timeout", 120))

    if not file_path:
        return {"type": "compile_run_result", "status": "error", "error": "file_path is required"}

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

        elif language == "batch":
            run_cmd = ["cmd.exe", "/c", file_path] + (args.split() if args else [])

        elif language == "powershell":
            run_cmd = [
                "powershell.exe", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-File", file_path,
            ] + (args.split() if args else [])

        elif language in ("c", "cpp"):
            compiler = "gcc" if language == "c" else "g++"
            binary   = os.path.splitext(file_path)[0] + ".exe"
            compile_args = (
                [compiler, "-o", binary, file_path, "-lm"]
                if language == "c"
                else [compiler, "-o", binary, file_path, "-lm", "-lstdc++"]
            )
            compile_result = subprocess.run(
                compile_args,
                capture_output=True, text=True, timeout=60,
            )
            compile_stdout     = compile_result.stdout.splitlines()
            compile_stderr     = compile_result.stderr.splitlines()
            compile_returncode = compile_result.returncode
            if compile_returncode != 0:
                _w, _e = _split_diagnostics(compile_stderr)
                return {
                    "type":               "compile_run_result",
                    "status":             "compile_error",
                    "output":             "",
                    "errors":             "\n".join(_e or compile_stderr),
                    "warnings":           "\n".join(_w),
                    "response":           f"compile_error \u2014 {len(_e)} error{'s' if len(_e) != 1 else ''}, {len(_w)} warning{'s' if len(_w) != 1 else ''}",
                    "compile_stdout":     compile_stdout,
                    "compile_stderr":     compile_stderr,
                    "compile_returncode": compile_returncode,
                }
            run_cmd = [binary] + (args.split() if args else [])

        else:
            return {
                "type":   "compile_run_result",
                "status": "error",
                "error":  f"Unsupported file type: {os.path.basename(file_path)}. "
                          "Supported: .py, .bat, .cmd, .ps1, .c, .cpp",
            }

        with tempfile.TemporaryDirectory() as run_outdir:
            before_files = set(os.listdir(workspace))
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_outdir,
            )
            output_files = _collect_files(run_outdir)
            after_files  = set(os.listdir(workspace))
            for fname in sorted(after_files - before_files):
                fpath = os.path.join(workspace, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, "rb") as f:
                            content = base64.b64encode(f.read()).decode("utf-8")
                        mime_type, _ = mimetypes.guess_type(fname)
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

        _c_warnings, _c_errors = _split_diagnostics(compile_stderr)
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
            "output":   "\n".join(_run_stdout),
            "errors":   "\n".join(
                _all_errors + ([s for s in _run_stderr if s not in _all_errors]
                               if _run_status == "runtime_error" else [])
            ),
            "warnings": "\n".join(_all_warnings),
            "response": (
                f"{_run_status} \u2014 {_n_out} line{'s' if _n_out != 1 else ''} output, "
                f"{_n_err} error{'s' if _n_err != 1 else ''}, "
                f"{_n_warn} warning{'s' if _n_warn != 1 else ''}"
            ),
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
        return {
            "type":   "compile_run_result",
            "status": "error",
            "error":  f"Tool not found: {exc}. Install MinGW (gcc/g++) and add it to PATH.",
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
    bucket   (str, optional) : S3 bucket name.
    file_url (str, optional) : Pre-signed / direct URL.
    args     (str, optional) : CLI args string.
    timeout  (int, optional) : Execution timeout in seconds (default 120).
    """
    dl_result = handle_download_from_s3(payload)
    if dl_result.get("status") != "ok":
        dl_result["type"] = "download_and_run_result"
        return dl_result

    run_payload = {
        "file_path": dl_result["local_path"],
        "args":      payload.get("args", ""),
        "timeout":   payload.get("timeout", 120),
    }
    run_result = handle_compile_and_run(run_payload)
    run_result["type"]       = "download_and_run_result"
    run_result["local_path"] = dl_result["local_path"]
    run_result["filename"]   = dl_result["filename"]
    run_result["size_bytes"] = dl_result["size_bytes"]
    return run_result


# ---------------------------------------------------------------------------
# Command handlers — Windows-specific
# ---------------------------------------------------------------------------


def handle_capture_image(payload: dict) -> dict:
    """
    Open a camera and capture a single frame as a PNG image.

    Payload fields
    --------------
    camera_index  (int, optional) : Camera device index (default 0).
    filename      (str, optional) : Output filename (default 'capture.png').
    compress_level (int, optional): PNG compression level 0–9 (default 3; 0 = no compression).
    warmup_frames (int, optional) : Frames to discard before capture for auto-exposure (default 5).
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return {
            "type":   "capture_image_result",
            "status": "error",
            "error":  "opencv-python not installed — run: pip install opencv-python",
        }

    camera_index   = int(payload.get("camera_index",   0))
    filename       = payload.get("filename", "capture.png")
    compress_level = int(payload.get("compress_level", 3))
    warmup_frames  = int(payload.get("warmup_frames",  5))

    # Ensure the output filename has a .png extension
    if not filename.lower().endswith(".png"):
        filename = os.path.splitext(filename)[0] + ".png"

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return {
            "type":   "capture_image_result",
            "status": "error",
            "error":  f"Cannot open camera at index {camera_index}. Check that a camera is connected.",
        }

    try:
        # Discard warmup frames so auto-exposure can settle
        for _ in range(warmup_frames):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            return {
                "type":   "capture_image_result",
                "status": "error",
                "error":  "Camera opened but failed to capture a frame.",
            }

        height, width = frame.shape[:2]

        # Encode frame to PNG bytes
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
        success, buffer = cv2.imencode(".png", frame, encode_params)
        if not success:
            return {
                "type":   "capture_image_result",
                "status": "error",
                "error":  "Failed to encode frame as PNG.",
            }

        img_bytes  = buffer.tobytes()
        img_b64    = base64.b64encode(img_bytes).decode("utf-8")
        size_bytes = len(img_bytes)

        # Save locally to workspace before returning
        local_path = os.path.join(_ensure_workspace(), filename)
        with open(local_path, "wb") as _f:
            _f.write(img_bytes)

        logger.info(
            "capture_image: %dx%d px, %d bytes, camera_index=%d, saved → %s",
            width, height, size_bytes, camera_index, local_path,
        )

        return {
            "type":   "capture_image_result",
            "status": "ok",
            "output":   f"Captured {width}x{height} image ({size_bytes} bytes) from camera {camera_index}",
            "errors":   "",
            "warnings": "",
            "response": f"ok \u2014 {width}x{height} PNG, {size_bytes} bytes",
            "files": [
                {
                    "name":      filename,
                    "content":   img_b64,
                    "mime_type": "image/png",
                }
            ],
            "width":        width,
            "height":       height,
            "size_bytes":   size_bytes,
            "camera_index": camera_index,
            "local_path":   local_path,
        }

    finally:
        cap.release()


def handle_screenshot(payload: dict) -> dict:
    """
    Capture the screen (or a region of it) as a PNG image.

    Payload fields
    --------------
    filename (str, optional) : Output filename (default 'screenshot.png').
    region   (dict, optional): Dict with keys top, left, width, height (pixels).
                               Omit to capture the entire primary monitor.
    monitor  (int, optional) : Monitor index for mss (default 1 = primary).
    """
    try:
        import mss  # noqa: PLC0415
        import mss.tools  # noqa: PLC0415
    except ImportError:
        return {
            "type":   "screenshot_result",
            "status": "error",
            "error":  "mss not installed — run: pip install mss",
        }

    filename = payload.get("filename", "screenshot.png")
    region   = payload.get("region")
    monitor  = int(payload.get("monitor", 1))

    try:
        with mss.mss() as sct:
            if region:
                grab_area = {
                    "top":    int(region.get("top",    0)),
                    "left":   int(region.get("left",   0)),
                    "width":  int(region.get("width",  800)),
                    "height": int(region.get("height", 600)),
                }
            else:
                grab_area = sct.monitors[monitor]

            screenshot = sct.grab(grab_area)
            width  = screenshot.width
            height = screenshot.height

            # Convert to PNG bytes using Pillow
            try:
                from PIL import Image  # noqa: PLC0415
                img = Image.frombytes("RGB", (width, height), screenshot.rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
            except ImportError:
                # Fall back to mss built-in PNG writer
                png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        img_b64    = base64.b64encode(png_bytes).decode("utf-8")
        size_bytes = len(png_bytes)

        # Save locally to workspace before returning
        local_path = os.path.join(_ensure_workspace(), filename)
        with open(local_path, "wb") as _f:
            _f.write(png_bytes)

        logger.info(
            "screenshot: %dx%d px, %d bytes, saved → %s",
            width, height, size_bytes, local_path,
        )

        return {
            "type":   "screenshot_result",
            "status": "ok",
            "output":   f"Screenshot {width}x{height} ({size_bytes} bytes)",
            "errors":   "",
            "warnings": "",
            "response": f"ok \u2014 {width}x{height} PNG, {size_bytes} bytes",
            "files": [
                {
                    "name":      filename,
                    "content":   img_b64,
                    "mime_type": "image/png",
                }
            ],
            "width":      width,
            "height":     height,
            "size_bytes": size_bytes,
            "local_path": local_path,
        }

    except Exception as exc:
        logger.error("screenshot failed: %s", exc)
        return {"type": "screenshot_result", "status": "error", "error": str(exc)}


def handle_get_system_info(payload: dict) -> dict:
    """
    Return detailed Windows system information via psutil.

    Returns CPU, RAM, disk, and network stats.
    """
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return {
            "type":   "system_info_result",
            "status": "error",
            "error":  "psutil not installed — run: pip install psutil",
        }

    try:
        vm      = psutil.virtual_memory()
        swap    = psutil.swap_memory()
        cpu_pct = psutil.cpu_percent(interval=1.0)
        cpu_freq = psutil.cpu_freq()
        boot_ts  = psutil.boot_time()

        disks = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append({
                    "device":     part.device,
                    "mountpoint": part.mountpoint,
                    "fstype":     part.fstype,
                    "total_gb":   round(usage.total / 1024**3, 2),
                    "used_gb":    round(usage.used  / 1024**3, 2),
                    "free_gb":    round(usage.free  / 1024**3, 2),
                    "percent":    usage.percent,
                })
            except PermissionError:
                continue

        net_io = psutil.net_io_counters()

        info = {
            "platform":        platform.platform(),
            "hostname":        platform.node(),
            "processor":       platform.processor(),
            "python":          sys.version,
            "uptime_seconds":  round(time.time() - boot_ts),
            "cpu_count_logical":   psutil.cpu_count(logical=True),
            "cpu_count_physical":  psutil.cpu_count(logical=False),
            "cpu_percent":         cpu_pct,
            "cpu_freq_mhz":        round(cpu_freq.current) if cpu_freq else None,
            "ram_total_mb":        round(vm.total  / 1024**2),
            "ram_used_mb":         round(vm.used   / 1024**2),
            "ram_available_mb":    round(vm.available / 1024**2),
            "ram_percent":         vm.percent,
            "swap_total_mb":       round(swap.total / 1024**2),
            "swap_used_mb":        round(swap.used  / 1024**2),
            "swap_percent":        swap.percent,
            "disks":               disks,
            "net_bytes_sent_mb":   round(net_io.bytes_sent / 1024**2, 2),
            "net_bytes_recv_mb":   round(net_io.bytes_recv / 1024**2, 2),
        }

        summary_lines = [
            f"CPU:  {cpu_pct}% ({info['cpu_count_logical']} logical cores)",
            f"RAM:  {info['ram_used_mb']} / {info['ram_total_mb']} MB ({vm.percent}%)",
            f"Disk: " + " | ".join(f"{d['mountpoint']} {d['percent']}%" for d in disks),
            f"Net:  sent {info['net_bytes_sent_mb']} MB, recv {info['net_bytes_recv_mb']} MB",
        ]

        return {
            "type":     "system_info_result",
            "status":   "ok",
            "output":   "\n".join(summary_lines),
            "errors":   "",
            "warnings": "",
            "response": f"ok \u2014 CPU {cpu_pct}%, RAM {vm.percent}%",
            "data":     info,
        }

    except Exception as exc:
        logger.error("get_system_info failed: %s", exc)
        return {"type": "system_info_result", "status": "error", "error": str(exc)}


def handle_list_processes(payload: dict) -> dict:
    """
    List running Windows processes sorted by CPU usage.

    Payload fields
    --------------
    limit (int, optional) : Maximum number of processes to return (default 20).
    sort_by (str, optional): Sort field — 'cpu' or 'memory' (default 'cpu').
    """
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return {
            "type":   "list_processes_result",
            "status": "error",
            "error":  "psutil not installed — run: pip install psutil",
        }

    limit   = int(payload.get("limit",   20))
    sort_by = payload.get("sort_by", "cpu")

    try:
        procs = []
        for proc in psutil.process_iter(["pid", "name", "status", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                mem_mb = round(info["memory_info"].rss / 1024**2, 1) if info["memory_info"] else 0
                procs.append({
                    "pid":       info["pid"],
                    "name":      info["name"],
                    "status":    info["status"],
                    "cpu_pct":   info["cpu_percent"] or 0.0,
                    "mem_mb":    mem_mb,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sort_key = "cpu_pct" if sort_by == "cpu" else "mem_mb"
        procs.sort(key=lambda p: p[sort_key], reverse=True)
        procs = procs[:limit]

        lines = [f"{'PID':>7}  {'CPU%':>6}  {'MEM MB':>8}  {'STATUS':<12}  NAME"]
        lines.append("-" * 60)
        for p in procs:
            lines.append(
                f"{p['pid']:>7}  {p['cpu_pct']:>6.1f}  {p['mem_mb']:>8.1f}  {p['status']:<12}  {p['name']}"
            )

        return {
            "type":      "list_processes_result",
            "status":    "ok",
            "output":    "\n".join(lines),
            "errors":    "",
            "warnings":  "",
            "response":  f"ok \u2014 {len(procs)} processes (sorted by {sort_by})",
            "processes": procs,
        }

    except Exception as exc:
        logger.error("list_processes failed: %s", exc)
        return {"type": "list_processes_result", "status": "error", "error": str(exc)}


def handle_run_powershell(payload: dict) -> dict:
    """
    Run a PowerShell script string or file on this Windows machine.

    Payload fields
    --------------
    script  (str)           : PowerShell script text to execute.
    timeout (int, optional) : Max execution seconds (default 60).
    """
    script  = payload.get("script", "")
    timeout = int(payload.get("timeout", 60))

    if not script.strip():
        return {"type": "powershell_result", "status": "error", "error": "script is required"}

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command", script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        _n_out  = len(stdout_lines)
        _status = "ok" if result.returncode == 0 else "error"

        return {
            "type":       "powershell_result",
            "status":     _status,
            "returncode": result.returncode,
            "output":     result.stdout,
            "errors":     result.stderr,
            "warnings":   "",
            "response":   f"{_status} \u2014 {_n_out} line{'s' if _n_out != 1 else ''} output, returncode={result.returncode}",
            "stdout":     stdout_lines,
            "stderr":     stderr_lines,
        }

    except subprocess.TimeoutExpired:
        return {"type": "powershell_result", "status": "timeout", "script_length": len(script)}
    except Exception as exc:
        logger.error("run_powershell failed: %s", exc)
        return {"type": "powershell_result", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: dict = {
    # Standard (shared with Pi)
    "ping":               handle_ping,
    "get_status":         handle_get_status,
    "run_code":           handle_run_code,
    "shell":              handle_shell,
    "set_config":         handle_set_config,
    "restart":            handle_restart,
    "download_from_s3":   handle_download_from_s3,
    "compile_and_run":    handle_compile_and_run,
    "download_and_run":   handle_download_and_run,
    # Windows-specific
    "capture_image":      handle_capture_image,
    "screenshot":         handle_screenshot,
    "get_system_info":    handle_get_system_info,
    "list_processes":     handle_list_processes,
    "run_powershell":     handle_run_powershell,
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
                logger.info(
                    "Connected as device_id='%s'  workspace='%s'",
                    device_id, WORKSPACE_DIR,
                )

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
                        result = {
                            "type":          "unknown_command",
                            "received_type": command_type,
                        }

                    result["command_id"] = command_id
                    result["device_id"]  = device_id
                    result["timestamp"]  = time.time()

                    await ws.send(json.dumps(result))
                    logger.info(
                        "→ sent result type=%s status=%s",
                        result.get("type"), result.get("status", "—"),
                    )

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
    parser = argparse.ArgumentParser(description="Windows device agent")
    parser.add_argument(
        "--host",
        default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
        help="Server WebSocket URL (e.g. wss://your-app.onrender.com)",
    )
    parser.add_argument(
        "--device-id",
        default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
        help="Unique ID for this Windows machine",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
        help="Shared secret token (must match server AGENT_TOKEN)",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("WORKSPACE_DIR", WORKSPACE_DIR),
        help="Local directory for downloaded / run files",
    )
    args = parser.parse_args()

    WORKSPACE_DIR = args.workspace
    DEVICE_ID     = args.device_id
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    try:
        asyncio.run(run_agent(args.host, args.device_id, args.token))
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
