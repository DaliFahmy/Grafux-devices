"""
windows/agent.py
Windows device agent — runs on the Windows machine itself.

Connects to the Grafux device hub, runs command blocks locally, and sends
results back.  Supports S3/URL download + compile/run, plus Windows-specific
features: camera capture, screenshots, system info, process listing, PowerShell.

The connection loop, reconnect/keepalive, dispatch and result envelope live in
``device.agents.base.BaseAgent``; the cross-platform handlers/helpers live in
``device.agents.common``.  This module contributes the Windows-specific handlers
and the Windows compile/run command resolution.

Usage
-----
    pip install -r requirements.txt
    python agent.py --host wss://grafux.onrender.com --device-id win-001 --token topsecret

Environment variables (all optional when flags are used):
    AGENT_HOST, DEVICE_ID, AGENT_TOKEN, WORKSPACE_DIR,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_REGION
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import platform
import subprocess
import sys
import time

# Make the shared ``device`` package importable when this file is run directly
# as ``python agent.py`` (deploy the device/agents/ tree alongside this script).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from device.agents import common  # noqa: E402
from device.agents.base import BaseAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("windows.agent")

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "wss://grafux.onrender.com"
DEFAULT_DEVICE_ID = "win-001"
DEFAULT_TOKEN = "changeme"

WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR", os.path.join(os.path.expanduser("~"), "win_workspace")
)

# Extensions beyond common's base map (.py/.c/.cpp/.cc/.cxx).
_LANGUAGE_EXTRA = {".bat": "batch", ".cmd": "batch", ".ps1": "powershell"}


def _ensure_workspace() -> str:
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR


# ---------------------------------------------------------------------------
# Standard handlers (Windows variants)
# ---------------------------------------------------------------------------

def handle_get_status(payload: dict) -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
        "uptime_seconds": None,
        "workspace_dir": WORKSPACE_DIR,
    }
    try:
        import psutil  # noqa: PLC0415
        info["uptime_seconds"] = time.time() - psutil.boot_time()
        vm = psutil.virtual_memory()
        info["ram_total_mb"] = round(vm.total / 1024 / 1024)
        info["ram_used_mb"] = round(vm.used / 1024 / 1024)
        info["ram_percent"] = vm.percent
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.5)
    except ImportError:
        pass
    return {"type": "status_report", "status": "ok", "data": info}


def handle_shell(payload: dict) -> dict:
    """Run a shell command via PowerShell (or cmd.exe when use_powershell=False)."""
    command = payload.get("command", "")
    timeout = common.safe_int(payload.get("timeout"), 30)
    use_powershell = payload.get("use_powershell", True)

    if use_powershell:
        full_cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        full_cmd = ["cmd.exe", "/c", command]

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        _n_out = len(stdout_lines)
        _status = "ok" if result.returncode == 0 else "error"
        return {
            "type": "shell_result",
            "status": _status,
            "returncode": result.returncode,
            "output": result.stdout,
            "errors": result.stderr,
            "warnings": "",
            "response": f"{_status} — {_n_out} line{'s' if _n_out != 1 else ''} output, returncode={result.returncode}",
            "stdout": stdout_lines,
            "stderr": stderr_lines,
        }
    except subprocess.TimeoutExpired:
        return {"type": "shell_result", "status": "timeout", "command": command}
    except Exception as exc:  # noqa: BLE001
        return {"type": "shell_result", "status": "error", "error": str(exc)}


def _resolve_run_cmd(language, file_path, args_list, workspace):
    """Map a detected language to its run command, compiling C/C++ as needed."""
    if language == "python":
        return [sys.executable, file_path] + args_list, None, None
    if language == "batch":
        return ["cmd.exe", "/c", file_path] + args_list, None, None
    if language == "powershell":
        return [
            "powershell.exe", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", file_path,
        ] + args_list, None, None
    if language in ("c", "cpp"):
        compiler = "gcc" if language == "c" else "g++"
        binary = os.path.splitext(file_path)[0] + ".exe"
        compile_args = (
            [compiler, "-o", binary, file_path, "-lm"] if language == "c"
            else [compiler, "-o", binary, file_path, "-lm", "-lstdc++"]
        )
        cp = common.compile_source(compile_args)
        diag = (cp.stdout.splitlines(), cp.stderr.splitlines(), cp.returncode)
        if cp.returncode != 0:
            return None, common.compile_error_result(*diag), None
        return [binary] + args_list, None, diag
    return None, {
        "type": "compile_run_result",
        "status": "error",
        "error": f"Unsupported file type: {os.path.basename(file_path)}. "
                 "Supported: .py, .bat, .cmd, .ps1, .c, .cpp",
    }, None


def handle_compile_and_run(payload: dict) -> dict:
    return common.compile_and_run(
        payload,
        workspace=_ensure_workspace(),
        resolve=_resolve_run_cmd,
        language_extra=_LANGUAGE_EXTRA,
        tool_hint="Install MinGW (gcc/g++) and add it to PATH.",
    )


def handle_download_from_s3(payload: dict) -> dict:
    return common.download_from_s3(
        payload, workspace=_ensure_workspace(), bare_key_workspace_fallback=True
    )


def handle_download_and_run(payload: dict) -> dict:
    return common.download_and_run(
        payload,
        workspace=_ensure_workspace(),
        compile_and_run_fn=handle_compile_and_run,
        bare_key_workspace_fallback=True,
    )


# ---------------------------------------------------------------------------
# Windows-specific handlers
# ---------------------------------------------------------------------------

def handle_capture_image(payload: dict) -> dict:
    """Open a camera and capture a single frame as a PNG image."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return {"type": "capture_image_result", "status": "error",
                "error": "opencv-python not installed — run: pip install opencv-python"}

    camera_index = common.safe_int(payload.get("camera_index"), 0)
    filename = payload.get("filename", "capture.png")
    compress_level = common.safe_int(payload.get("compress_level"), 3)
    warmup_frames = common.safe_int(payload.get("warmup_frames"), 5)

    if not filename.lower().endswith(".png"):
        filename = os.path.splitext(filename)[0] + ".png"

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return {"type": "capture_image_result", "status": "error",
                "error": f"Cannot open camera at index {camera_index}. Check that a camera is connected."}
    try:
        for _ in range(warmup_frames):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            return {"type": "capture_image_result", "status": "error",
                    "error": "Camera opened but failed to capture a frame."}
        height, width = frame.shape[:2]
        success, buffer = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        if not success:
            return {"type": "capture_image_result", "status": "error",
                    "error": "Failed to encode frame as PNG."}
        img_bytes = buffer.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        size_bytes = len(img_bytes)
        local_path = os.path.join(_ensure_workspace(), filename)
        with open(local_path, "wb") as _f:
            _f.write(img_bytes)
        logger.info("capture_image: %dx%d px, %d bytes, camera_index=%d, saved → %s",
                    width, height, size_bytes, camera_index, local_path)
        return {
            "type": "capture_image_result",
            "status": "ok",
            "output": f"Captured {width}x{height} image ({size_bytes} bytes) from camera {camera_index}",
            "errors": "",
            "warnings": "",
            "response": f"ok — {width}x{height} PNG, {size_bytes} bytes",
            "files": [{"name": filename, "content": img_b64, "mime_type": "image/png"}],
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
            "camera_index": camera_index,
            "local_path": local_path,
        }
    finally:
        cap.release()


def handle_screenshot(payload: dict) -> dict:
    """Capture the screen (or a region) as a PNG image."""
    try:
        import mss  # noqa: PLC0415
        import mss.tools  # noqa: PLC0415
    except ImportError:
        return {"type": "screenshot_result", "status": "error",
                "error": "mss not installed — run: pip install mss"}

    filename = payload.get("filename", "screenshot.png")
    region = payload.get("region")
    monitor = common.safe_int(payload.get("monitor"), 1)

    try:
        with mss.mss() as sct:
            if region:
                grab_area = {
                    "top": common.safe_int(region.get("top"), 0),
                    "left": common.safe_int(region.get("left"), 0),
                    "width": common.safe_int(region.get("width"), 800),
                    "height": common.safe_int(region.get("height"), 600),
                }
            else:
                grab_area = sct.monitors[monitor]
            screenshot = sct.grab(grab_area)
            width, height = screenshot.width, screenshot.height
            try:
                from PIL import Image  # noqa: PLC0415
                img = Image.frombytes("RGB", (width, height), screenshot.rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
            except ImportError:
                png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        img_b64 = base64.b64encode(png_bytes).decode("utf-8")
        size_bytes = len(png_bytes)
        local_path = os.path.join(_ensure_workspace(), filename)
        with open(local_path, "wb") as _f:
            _f.write(png_bytes)
        logger.info("screenshot: %dx%d px, %d bytes, saved → %s", width, height, size_bytes, local_path)
        return {
            "type": "screenshot_result",
            "status": "ok",
            "output": f"Screenshot {width}x{height} ({size_bytes} bytes)",
            "errors": "",
            "warnings": "",
            "response": f"ok — {width}x{height} PNG, {size_bytes} bytes",
            "files": [{"name": filename, "content": img_b64, "mime_type": "image/png"}],
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
            "local_path": local_path,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("screenshot failed: %s", exc)
        return {"type": "screenshot_result", "status": "error", "error": str(exc)}


def handle_get_system_info(payload: dict) -> dict:
    """Return detailed Windows system information via psutil."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return {"type": "system_info_result", "status": "error",
                "error": "psutil not installed — run: pip install psutil"}
    try:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        cpu_pct = psutil.cpu_percent(interval=1.0)
        cpu_freq = psutil.cpu_freq()
        boot_ts = psutil.boot_time()

        disks = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append({
                    "device": part.device, "mountpoint": part.mountpoint, "fstype": part.fstype,
                    "total_gb": round(usage.total / 1024**3, 2),
                    "used_gb": round(usage.used / 1024**3, 2),
                    "free_gb": round(usage.free / 1024**3, 2),
                    "percent": usage.percent,
                })
            except PermissionError:
                continue

        net_io = psutil.net_io_counters()
        info = {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "processor": platform.processor(),
            "python": sys.version,
            "uptime_seconds": round(time.time() - boot_ts),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_percent": cpu_pct,
            "cpu_freq_mhz": round(cpu_freq.current) if cpu_freq else None,
            "ram_total_mb": round(vm.total / 1024**2),
            "ram_used_mb": round(vm.used / 1024**2),
            "ram_available_mb": round(vm.available / 1024**2),
            "ram_percent": vm.percent,
            "swap_total_mb": round(swap.total / 1024**2),
            "swap_used_mb": round(swap.used / 1024**2),
            "swap_percent": swap.percent,
            "disks": disks,
            "net_bytes_sent_mb": round(net_io.bytes_sent / 1024**2, 2),
            "net_bytes_recv_mb": round(net_io.bytes_recv / 1024**2, 2),
        }
        summary_lines = [
            f"CPU:  {cpu_pct}% ({info['cpu_count_logical']} logical cores)",
            f"RAM:  {info['ram_used_mb']} / {info['ram_total_mb']} MB ({vm.percent}%)",
            "Disk: " + " | ".join(f"{d['mountpoint']} {d['percent']}%" for d in disks),
            f"Net:  sent {info['net_bytes_sent_mb']} MB, recv {info['net_bytes_recv_mb']} MB",
        ]
        return {
            "type": "system_info_result",
            "status": "ok",
            "output": "\n".join(summary_lines),
            "errors": "",
            "warnings": "",
            "response": f"ok — CPU {cpu_pct}%, RAM {vm.percent}%",
            "data": info,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("get_system_info failed: %s", exc)
        return {"type": "system_info_result", "status": "error", "error": str(exc)}


def handle_list_processes(payload: dict) -> dict:
    """List running Windows processes sorted by CPU or memory usage."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return {"type": "list_processes_result", "status": "error",
                "error": "psutil not installed — run: pip install psutil"}

    limit = common.safe_int(payload.get("limit"), 20)
    sort_by = payload.get("sort_by", "cpu")
    try:
        procs = []
        for proc in psutil.process_iter(["pid", "name", "status", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                mem_mb = round(info["memory_info"].rss / 1024**2, 1) if info["memory_info"] else 0
                procs.append({
                    "pid": info["pid"], "name": info["name"], "status": info["status"],
                    "cpu_pct": info["cpu_percent"] or 0.0, "mem_mb": mem_mb,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        sort_key = "cpu_pct" if sort_by == "cpu" else "mem_mb"
        procs.sort(key=lambda p: p[sort_key], reverse=True)
        procs = procs[:limit]
        lines = [f"{'PID':>7}  {'CPU%':>6}  {'MEM MB':>8}  {'STATUS':<12}  NAME", "-" * 60]
        for p in procs:
            lines.append(f"{p['pid']:>7}  {p['cpu_pct']:>6.1f}  {p['mem_mb']:>8.1f}  {p['status']:<12}  {p['name']}")
        return {
            "type": "list_processes_result",
            "status": "ok",
            "output": "\n".join(lines),
            "errors": "",
            "warnings": "",
            "response": f"ok — {len(procs)} processes (sorted by {sort_by})",
            "processes": procs,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("list_processes failed: %s", exc)
        return {"type": "list_processes_result", "status": "error", "error": str(exc)}


def handle_run_powershell(payload: dict) -> dict:
    """Run a PowerShell script string on this Windows machine."""
    script = payload.get("script", "")
    timeout = common.safe_int(payload.get("timeout"), 60)
    if not script.strip():
        return {"type": "powershell_result", "status": "error", "error": "script is required"}
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout_lines = result.stdout.splitlines()
        stderr_lines = result.stderr.splitlines()
        _n_out = len(stdout_lines)
        _status = "ok" if result.returncode == 0 else "error"
        return {
            "type": "powershell_result",
            "status": _status,
            "returncode": result.returncode,
            "output": result.stdout,
            "errors": result.stderr,
            "warnings": "",
            "response": f"{_status} — {_n_out} line{'s' if _n_out != 1 else ''} output, returncode={result.returncode}",
            "stdout": stdout_lines,
            "stderr": stderr_lines,
        }
    except subprocess.TimeoutExpired:
        return {"type": "powershell_result", "status": "timeout", "script_length": len(script)}
    except Exception as exc:  # noqa: BLE001
        logger.error("run_powershell failed: %s", exc)
        return {"type": "powershell_result", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: dict = {
    # Standard
    "ping": common.handle_ping,
    "get_status": handle_get_status,
    "run_code": common.run_code,
    "shell": handle_shell,
    "set_config": common.handle_set_config,
    "restart": common.handle_restart,
    "download_from_s3": handle_download_from_s3,
    "compile_and_run": handle_compile_and_run,
    "download_and_run": handle_download_and_run,
    # Windows-specific
    "capture_image": handle_capture_image,
    "screenshot": handle_screenshot,
    "get_system_info": handle_get_system_info,
    "list_processes": handle_list_processes,
    "run_powershell": handle_run_powershell,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows device agent")
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", DEFAULT_HOST),
                        help="Server WebSocket URL (e.g. wss://your-app.onrender.com)")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", DEFAULT_DEVICE_ID),
                        help="Unique ID for this Windows machine")
    parser.add_argument("--token", default=os.environ.get("AGENT_TOKEN", DEFAULT_TOKEN),
                        help="Shared secret token (must match server AGENT_TOKEN)")
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE_DIR", WORKSPACE_DIR),
                        help="Local directory for downloaded / run files")
    args = parser.parse_args()

    WORKSPACE_DIR = args.workspace
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    logger.info("workspace = %s", WORKSPACE_DIR)
    BaseAgent(args.host, args.device_id, args.token, handlers=HANDLERS, logger=logger).run_forever()
