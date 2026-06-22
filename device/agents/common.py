"""
device/agents/common.py
Shared handler logic and helpers used by the raspberry_pi and windows agents.

These were previously duplicated almost verbatim across both agents.  The
functions here preserve the exact result-dict shapes (``type``/``status`` and
the Grafux output-port keys ``output``/``errors``/``warnings``/``response``/
``files``/``stdout``/``stderr``) so the hub and UI behave identically.

Reliability improvements baked in:
  * ``run_code`` executes in a *subprocess* with an enforced timeout, so the
    ``timeout`` field finally works — an infinite loop is killed instead of
    hanging the agent forever (in-process ``exec`` could not be interrupted).
  * ``compile_and_run`` confines ``file_path`` to the workspace by default
    (set ``AGENT_ALLOW_ABSOLUTE_PATHS=1`` to opt out), closing an arbitrary
    file-read path-traversal hole.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import subprocess
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("device.agent.common")

# ---------------------------------------------------------------------------
# Tunables (overridable via environment)
# ---------------------------------------------------------------------------

DEFAULT_RUN_CODE_TIMEOUT_S: int = 30
DEFAULT_COMPILE_RUN_TIMEOUT_S: int = 120
COMPILE_TIMEOUT_S: int = int(os.environ.get("AGENT_COMPILE_TIMEOUT_S", "60"))
ALLOW_ABSOLUTE_PATHS: bool = os.environ.get("AGENT_ALLOW_ABSOLUTE_PATHS", "0") == "1"

_BASE_LANGUAGES = {
    ".py": "python",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def safe_int(value, default: int) -> int:
    """Coerce *value* to int, falling back to *default* on bad input.

    Payload fields arrive as free-form JSON; ``int("abc")`` would crash a
    handler, so unparseable values degrade to the default instead.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def split_diagnostics(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Split compiler/runtime output lines into (warnings, errors)."""
    warnings, errors = [], []
    for line in lines:
        ll = line.lower()
        if "warning:" in ll:
            warnings.append(line)
        elif "error:" in ll or "fatal:" in ll:
            errors.append(line)
    return warnings, errors


def detect_language(filename: str, extra: Optional[dict] = None) -> str:
    """Return a language tag from the file extension (base map + *extra*)."""
    table = dict(_BASE_LANGUAGES)
    if extra:
        table.update(extra)
    ext = os.path.splitext(filename)[1].lower()
    return table.get(ext, "unknown")


def confine_path(file_path: str, workspace: str) -> Optional[str]:
    """Resolve *file_path* and ensure it stays inside *workspace*.

    Relative paths are joined to the workspace.  Absolute paths are allowed only
    if they live under the workspace (unless ``AGENT_ALLOW_ABSOLUTE_PATHS=1``).
    Returns the absolute path, or ``None`` if it escapes the workspace.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(workspace, file_path)
    abs_path = os.path.abspath(file_path)
    if ALLOW_ABSOLUTE_PATHS:
        return abs_path
    ws_abs = os.path.abspath(workspace)
    try:
        if os.path.commonpath([abs_path, ws_abs]) == ws_abs:
            return abs_path
    except ValueError:
        # Different drives (Windows) — definitely outside the workspace.
        pass
    return None


def collect_files(directory: str) -> list:
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
                    "name": fname,
                    "content": content,
                    "mime_type": mime_type or "application/octet-stream",
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not encode file %s: %s", fpath, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not scan output directory %s: %s", directory, exc)
    return files


# ---------------------------------------------------------------------------
# Standard handlers (identical across agents)
# ---------------------------------------------------------------------------

def handle_ping(payload: dict) -> dict:
    return {"type": "pong", "status": "ok", "message": "pong"}


def handle_set_config(payload: dict) -> dict:
    key = payload.get("key")
    value = payload.get("value")
    logger.info("Config update: %s = %s", key, value)
    return {"type": "config_ack", "status": "ok", "key": key, "value": value}


def handle_restart(payload: dict) -> dict:
    """Re-exec the agent process shortly after replying.

    Uses a ``threading.Timer`` (not the event loop) so it works regardless of
    which thread the handler runs in.
    """
    logger.info("Restart requested — restarting agent in 2 s …")
    threading.Timer(2.0, lambda: os.execv(sys.executable, [sys.executable] + sys.argv)).start()
    return {"type": "restart_ack", "status": "ok", "message": "Restarting agent..."}


def run_code(payload: dict) -> dict:
    """Execute an inline Python snippet in an isolated subprocess with a timeout.

    Running as a subprocess (rather than in-process ``exec``) means the
    ``timeout`` is actually enforceable and user code cannot touch the agent's
    own state.  Any files the snippet writes to its working directory are
    collected and returned.
    """
    code = payload.get("code", "")
    timeout = safe_int(payload.get("timeout"), DEFAULT_RUN_CODE_TIMEOUT_S)

    import tempfile  # local import keeps module import light

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {
                "type": "run_code_result",
                "status": "timeout",
                "output": "",
                "errors": f"Execution exceeded {timeout}s and was terminated.",
                "warnings": "",
                "response": f"timeout — killed after {timeout}s",
                "stdout": [],
                "stderr": [],
                "files": [],
            }
        stdout_lines = proc.stdout.splitlines()
        stderr_lines = proc.stderr.splitlines()
        status = "ok" if proc.returncode == 0 else "error"
        files = collect_files(tmpdir)

    _warnings, _errors = split_diagnostics(stderr_lines)
    _error_str = "\n".join(stderr_lines) if stderr_lines else ""
    _n_out = len(stdout_lines)
    _n_err = len(_errors) or (1 if status == "error" else 0)

    return {
        "type": "run_code_result",
        "status": status,
        "output": "\n".join(stdout_lines),
        "errors": _error_str,
        "warnings": "\n".join(_warnings),
        "response": f"{status} — {_n_out} line{'s' if _n_out != 1 else ''} output, {_n_err} error{'s' if _n_err != 1 else ''}",
        "stdout": stdout_lines,
        "stderr": stderr_lines,
        "files": files,
    }


# ---------------------------------------------------------------------------
# compile_and_run — shared core, platform-specific run-command resolution
# ---------------------------------------------------------------------------

# resolve(language, file_path, args_list, workspace) ->
#     (run_cmd | None, early_result | None, compile_diag | None)
# where compile_diag is (compile_stdout_lines, compile_stderr_lines, returncode).
# Exactly one of run_cmd / early_result is non-None.  compile_diag is supplied
# when a compile step ran (so its warnings can be merged into the run result).
CompileDiag = Tuple[List[str], List[str], int]
ResolveFn = Callable[
    [str, str, List[str], str],
    Tuple[Optional[List[str]], Optional[dict], Optional[CompileDiag]],
]


def compile_source(compile_cmd: List[str]):
    """Run a compiler command with the standard compile timeout."""
    return subprocess.run(compile_cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)


def compile_error_result(compile_stdout: List[str], compile_stderr: List[str], returncode: int) -> dict:
    """Build the standard ``compile_error`` result dict."""
    _w, _e = split_diagnostics(compile_stderr)
    return {
        "type": "compile_run_result",
        "status": "compile_error",
        "output": "",
        "errors": "\n".join(_e or compile_stderr),
        "warnings": "\n".join(_w),
        "response": f"compile_error — {len(_e)} error{'s' if len(_e) != 1 else ''}, {len(_w)} warning{'s' if len(_w) != 1 else ''}",
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
        "compile_returncode": returncode,
    }


def compile_and_run(
    payload: dict,
    *,
    workspace: str,
    resolve: ResolveFn,
    language_extra: Optional[dict] = None,
    tool_hint: str = "Install the required compiler/runtime.",
) -> dict:
    """Compile (if needed) and run a workspace file — shared by Pi and Windows.

    *resolve* maps a detected language to the actual run command (handling any
    compile step) and returns ``(run_cmd, early_result)``: exactly one is
    non-None.  An ``early_result`` short-circuits (e.g. a compile error or an
    unsupported language); otherwise *run_cmd* is executed.
    """
    file_path = (payload.get("file_path") or "").strip()
    args = (payload.get("args") or "").strip()
    timeout = safe_int(payload.get("timeout"), DEFAULT_COMPILE_RUN_TIMEOUT_S)

    if not file_path:
        return {"type": "compile_run_result", "status": "error", "error": "file_path is required"}

    resolved = confine_path(file_path, workspace)
    if resolved is None:
        return {
            "type": "compile_run_result",
            "status": "error",
            "error": f"file_path must be inside the workspace ({workspace}). "
                     "Set AGENT_ALLOW_ABSOLUTE_PATHS=1 to allow absolute paths.",
        }
    file_path = resolved

    if not os.path.exists(file_path):
        return {"type": "compile_run_result", "status": "error", "error": f"File not found: {file_path}"}

    language = detect_language(file_path, language_extra)
    args_list = args.split() if args else []
    logger.info("compile_and_run: %s (%s)", file_path, language)

    import tempfile

    try:
        run_cmd, early, compile_diag = resolve(language, file_path, args_list, workspace)
        if early is not None:
            return early

        with tempfile.TemporaryDirectory() as run_outdir:
            before_files = set(os.listdir(workspace))
            run_result = subprocess.run(
                run_cmd, capture_output=True, text=True, timeout=timeout, cwd=run_outdir,
            )
            output_files = collect_files(run_outdir)
            after_files = set(os.listdir(workspace))
            for fname in sorted(after_files - before_files):
                fpath = os.path.join(workspace, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, "rb") as f:
                            content = base64.b64encode(f.read()).decode("utf-8")
                        mime_type, _ = mimetypes.guess_type(fname)
                        output_files.append({
                            "name": fname,
                            "content": content,
                            "mime_type": mime_type or "application/octet-stream",
                        })
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not encode workspace file %s: %s", fpath, exc)

        # Merge compile diagnostics (if a compile step ran) with runtime ones,
        # matching the original per-agent behaviour.
        compile_stdout, compile_stderr, compile_returncode = compile_diag or ([], [], 0)

        _run_stdout = run_result.stdout.splitlines()
        _run_stderr = run_result.stderr.splitlines()
        _run_status = "ok" if run_result.returncode == 0 else "runtime_error"

        _c_warnings, _c_errors = split_diagnostics(compile_stderr)
        _rt_warnings, _rt_errors = split_diagnostics(_run_stderr)
        _all_errors = _c_errors + _rt_errors
        _all_warnings = _c_warnings + _rt_warnings
        _n_out = len(_run_stdout)
        _n_err = len(_all_errors)
        _n_warn = len(_all_warnings)

        return {
            "type": "compile_run_result",
            "status": _run_status,
            "language": language,
            "output": "\n".join(_run_stdout),
            "errors": "\n".join(
                _all_errors + ([s for s in _run_stderr if s not in _all_errors]
                               if _run_status == "runtime_error" else [])
            ),
            "warnings": "\n".join(_all_warnings),
            "response": (
                f"{_run_status} — {_n_out} line{'s' if _n_out != 1 else ''} output, "
                f"{_n_err} error{'s' if _n_err != 1 else ''}, "
                f"{_n_warn} warning{'s' if _n_warn != 1 else ''}"
            ),
            "compile_stdout": compile_stdout,
            "compile_stderr": compile_stderr,
            "compile_returncode": compile_returncode,
            "stdout": _run_stdout,
            "stderr": _run_stderr,
            "returncode": run_result.returncode,
            "files": output_files,
        }

    except subprocess.TimeoutExpired:
        return {"type": "compile_run_result", "status": "timeout", "language": language}
    except FileNotFoundError as exc:
        return {"type": "compile_run_result", "status": "error", "error": f"Tool not found: {exc}. {tool_hint}"}
    except Exception as exc:  # noqa: BLE001
        logger.error("compile_and_run failed: %s", exc)
        return {"type": "compile_run_result", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# S3 / URL download (shared)
# ---------------------------------------------------------------------------

def download_from_s3(payload: dict, *, workspace: str, bare_key_workspace_fallback: bool = False) -> dict:
    """Download a file from S3 or a URL into the workspace.

    When *bare_key_workspace_fallback* is set (Windows behaviour), a bare
    filename ``s3_key`` with no path falls back to a copy already sitting in the
    workspace, with a clear diagnostic if it is missing.
    """
    file_url = payload.get("file_url", "").strip()
    s3_key = payload.get("s3_key", "").strip()
    filename = payload.get("filename", "").strip()
    bucket = payload.get("bucket", "").strip() or os.environ.get("AWS_S3_BUCKET", "")

    if bare_key_workspace_fallback and s3_key and "/" not in s3_key and "\\" not in s3_key and not file_url:
        local_name = filename or s3_key
        workspace_copy = os.path.join(workspace, local_name)
        if os.path.isfile(workspace_copy):
            logger.info("download_from_s3: bare key '%s' — using workspace copy %s", s3_key, workspace_copy)
            return {
                "type": "download_result",
                "status": "ok",
                "local_path": workspace_copy,
                "filename": local_name,
                "size_bytes": os.path.getsize(workspace_copy),
                "source": "workspace_local",
            }
        return {
            "type": "download_result",
            "status": "error",
            "error": (
                f"'{s3_key}' is a bare filename — no full S3 path was provided. "
                f"Place '{s3_key}' in {workspace} on this machine, or enable auto-sync "
                f"in the Grafux app so it is uploaded to S3 with a full project path."
            ),
        }

    if not filename:
        source = file_url or s3_key
        filename = os.path.basename(source.rstrip("/")) if source else "downloaded_file"

    local_path = os.path.join(workspace, os.path.basename(filename))

    try:
        if file_url:
            try:
                import requests  # noqa: PLC0415
            except ImportError:
                return {"type": "download_result", "status": "error",
                        "error": "requests package not installed — run: pip install requests"}
            resp = requests.get(file_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
        elif s3_key:
            if not bucket:
                return {"type": "download_result", "status": "error",
                        "error": "No S3 bucket specified. Set AWS_S3_BUCKET env var or pass 'bucket' in payload."}
            try:
                import boto3  # noqa: PLC0415
            except ImportError:
                return {"type": "download_result", "status": "error",
                        "error": "boto3 not installed — run: pip install boto3"}
            s3 = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
            s3.download_file(bucket, s3_key, local_path)
        else:
            return {"type": "download_result", "status": "error",
                    "error": "Payload must include either 's3_key' or 'file_url'."}

        file_size = os.path.getsize(local_path)
        logger.info("Downloaded '%s' → %s (%d bytes)", s3_key or file_url, local_path, file_size)
        return {
            "type": "download_result",
            "status": "ok",
            "local_path": local_path,
            "filename": os.path.basename(local_path),
            "size_bytes": file_size,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed: %s", exc)
        return {"type": "download_result", "status": "error", "error": str(exc)}


def download_and_run(payload: dict, *, workspace: str, compile_and_run_fn, bare_key_workspace_fallback: bool = False) -> dict:
    """Download a file then immediately compile/run it (shared two-step flow)."""
    dl_result = download_from_s3(payload, workspace=workspace, bare_key_workspace_fallback=bare_key_workspace_fallback)
    if dl_result.get("status") != "ok":
        dl_result["type"] = "download_and_run_result"
        return dl_result

    run_result = compile_and_run_fn({
        "file_path": dl_result["local_path"],
        "args": payload.get("args", ""),
        "timeout": payload.get("timeout", DEFAULT_COMPILE_RUN_TIMEOUT_S),
    })
    run_result["type"] = "download_and_run_result"
    run_result["local_path"] = dl_result["local_path"]
    run_result["filename"] = dl_result["filename"]
    run_result["size_bytes"] = dl_result["size_bytes"]
    return run_result
