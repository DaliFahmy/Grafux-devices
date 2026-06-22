"""
websocket/handlers.py
Transport-agnostic command handlers for the device-side WebSocket server.

These functions execute a command and return a plain dict whose fields map
directly onto the Grafux "devices" block output ports:

    output    program stdout
    errors    compile + runtime errors
    warnings  compiler/runtime warnings
    response  one-line human summary
    status    ok | compile_error | runtime_error | timeout | error
    files     list of base64-encoded artifacts the program produced

The compile / run / diagnostic logic is transferred from
``raspberry_pi/agent.py`` and adapted so the *exact source the user wrote* is
sent inline (the ``code`` field) and compiled on the device, instead of being
downloaded from S3.  No transport (WebSocket / HTTP / stdin) is assumed here.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import platform
import subprocess
import sys
import tempfile

logger = logging.getLogger("device.handlers")

#: Wall-clock cap on a compile step (seconds).
COMPILE_TIMEOUT_S = int(os.environ.get("AGENT_COMPILE_TIMEOUT_S", "60"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(value, default: int) -> int:
    """Coerce *value* to int, falling back to *default* on bad input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _split_diagnostics(lines: list) -> tuple:
    """Split compiler/runtime output lines into (warnings, errors).

    A line is a warning if it contains 'warning:' and an error if it contains
    'error:' or 'fatal:'.  Context lines (notes, carets, etc.) are ignored so
    the port values stay concise.
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
        ".py":   "python",
        ".c":    "c",
        ".cpp":  "cpp",
        ".cc":   "cpp",
        ".cxx":  "cpp",
        ".sh":   "shell",
        ".bash": "shell",
    }.get(ext, "unknown")


def _ext_for_language(language: str) -> str:
    """Return a source-file extension for a normalized language tag."""
    return {
        "python": ".py",
        "c":      ".c",
        "cpp":    ".cpp",
        "shell":  ".sh",
    }.get(language, "")


def _normalize_language(language: str, filename: str) -> str:
    """Resolve the effective language from an explicit tag or a filename.

    The explicit ``language`` wins; otherwise it is inferred from ``filename``.
    Common aliases (c++, py, bash) are accepted.
    """
    lang = (language or "").strip().lower()
    aliases = {
        "c++": "cpp", "cxx": "cpp", "cc": "cpp",
        "py": "python", "python3": "python",
        "bash": "shell", "sh": "shell",
    }
    lang = aliases.get(lang, lang)
    if lang in ("python", "c", "cpp", "shell"):
        return lang
    if filename:
        return _detect_language(filename)
    return "unknown"


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
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not encode file %s: %s", fpath, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not scan output directory %s: %s", directory, exc)
    return files


# ---------------------------------------------------------------------------
# Primary handler — compile (if needed) and run inline source
# ---------------------------------------------------------------------------


def handle_compile_and_run(payload: dict) -> dict:
    """Compile (if needed) and run the *inline source the user wrote*.

    Payload fields
    --------------
    code      (str)            : The exact source text to compile/run.
    language  (str, optional)  : cpp | c | python | shell. Inferred from
                                 ``filename`` when omitted.
    filename  (str, optional)  : Used to name the temp source file and to
                                 infer the language when ``language`` is absent.
    args      (str, optional)  : Command-line arguments for the program.
    timeout   (int, optional)  : Max execution seconds (default 120).
    file_path (str, optional)  : Run a file already present on the device
                                 instead of inline ``code``.
    """
    code      = payload.get("code", "")
    filename  = (payload.get("filename") or "").strip()
    language  = _normalize_language(payload.get("language", ""), filename)
    args      = (payload.get("args") or "").strip()
    timeout   = _safe_int(payload.get("timeout"), 120)
    file_path = (payload.get("file_path") or "").strip()

    # Run an existing on-device file when file_path is given (no inline code).
    if file_path and not code:
        if not os.path.exists(file_path):
            return _error("compile_run_result", f"File not found: {file_path}")
        language = language if language != "unknown" else _detect_language(file_path)
        return _compile_and_run_path(file_path, language, args, timeout)

    if not code:
        return _error("compile_run_result", "code is required (the source to compile/run)")

    if language == "unknown":
        return _error(
            "compile_run_result",
            "Could not determine language. Set 'language' to cpp|c|python|shell "
            "or pass a 'filename' with a known extension.",
        )

    # Write the inline source into an isolated temp directory and compile/run there.
    with tempfile.TemporaryDirectory() as tmpdir:
        src_name = filename or ("main" + _ext_for_language(language))
        src_path = os.path.join(tmpdir, os.path.basename(src_name))
        try:
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(code)
        except Exception as exc:  # noqa: BLE001
            return _error("compile_run_result", f"Could not write source file: {exc}")

        return _compile_and_run_path(src_path, language, args, timeout)


def _compile_and_run_path(file_path: str, language: str, args: str, timeout: int) -> dict:
    """Compile ``file_path`` for ``language`` (if needed) and run the result.

    Mirrors the compile/run/diagnostic flow from raspberry_pi/agent.py, returning
    the same enriched result shape the Grafux block already understands.
    """
    logger.info("compile_and_run: %s (%s)", file_path, language)

    compile_stdout: list = []
    compile_stderr: list = []
    compile_returncode: int = 0
    workdir = os.path.dirname(file_path) or "."

    try:
        if language == "python":
            run_cmd = [sys.executable, file_path] + (args.split() if args else [])

        elif language == "c":
            binary = os.path.splitext(file_path)[0]
            compile_result = subprocess.run(
                ["gcc", "-o", binary, file_path, "-lm"],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
            )
            compile_stdout     = compile_result.stdout.splitlines()
            compile_stderr     = compile_result.stderr.splitlines()
            compile_returncode = compile_result.returncode
            if compile_returncode != 0:
                return _compile_error(compile_stdout, compile_stderr, compile_returncode)
            run_cmd = [binary] + (args.split() if args else [])

        elif language == "cpp":
            binary = os.path.splitext(file_path)[0]
            compile_result = subprocess.run(
                ["g++", "-o", binary, file_path, "-lm", "-lstdc++"],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
            )
            compile_stdout     = compile_result.stdout.splitlines()
            compile_stderr     = compile_result.stderr.splitlines()
            compile_returncode = compile_result.returncode
            if compile_returncode != 0:
                return _compile_error(compile_stdout, compile_stderr, compile_returncode)
            run_cmd = [binary] + (args.split() if args else [])

        elif language == "shell":
            os.chmod(file_path, 0o755)
            run_cmd = ["/bin/bash", file_path] + (args.split() if args else [])

        else:
            return _error(
                "compile_run_result",
                f"Unsupported language: {language}. Supported: python, c, cpp, shell",
            )

        # Run in an isolated dir so we can collect any files the program writes.
        with tempfile.TemporaryDirectory() as run_outdir:
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_outdir,
            )
            output_files = _collect_files(run_outdir)

        _run_stdout = run_result.stdout.splitlines()
        _run_stderr = run_result.stderr.splitlines()
        _run_status = "ok" if run_result.returncode == 0 else "runtime_error"

        _c_warnings, _c_errors   = _split_diagnostics(compile_stderr)
        _rt_warnings, _rt_errors = _split_diagnostics(_run_stderr)
        _all_errors   = _c_errors + _rt_errors
        _all_warnings = _c_warnings + _rt_warnings
        # Any remaining runtime stderr lines that aren't structured warnings are errors.
        if _run_status == "runtime_error":
            _all_errors += [s for s in _run_stderr if s not in _all_errors]

        _n_out, _n_err, _n_warn = len(_run_stdout), len(_all_errors), len(_all_warnings)

        return {
            "type":     "compile_run_result",
            "status":   _run_status,
            "language": language,
            # --- Grafux output ports ---
            "output":   "\n".join(_run_stdout),
            "errors":   "\n".join(_all_errors),
            "warnings": "\n".join(_all_warnings),
            "response": (
                f"{_run_status} — {_n_out} line{'s' if _n_out != 1 else ''} output, "
                f"{_n_err} error{'s' if _n_err != 1 else ''}, "
                f"{_n_warn} warning{'s' if _n_warn != 1 else ''}"
            ),
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
        return {"type": "compile_run_result", "status": "timeout",
                "language": language, "errors": f"Execution exceeded {timeout}s",
                "response": "timeout"}
    except FileNotFoundError as exc:
        return _error(
            "compile_run_result",
            f"Tool not found: {exc}. Install the compiler/interpreter on the device "
            f"(gcc/g++ for C/C++, python3, bash).",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("compile_and_run failed: %s", exc)
        return _error("compile_run_result", str(exc))


def _compile_error(compile_stdout: list, compile_stderr: list, returncode: int) -> dict:
    """Build a compile-error result with diagnostics split into errors/warnings."""
    _w, _e = _split_diagnostics(compile_stderr)
    return {
        "type":               "compile_run_result",
        "status":             "compile_error",
        # --- Grafux output ports ---
        "output":             "",
        "errors":             "\n".join(_e or compile_stderr),
        "warnings":           "\n".join(_w),
        "response": (
            f"compile_error — {len(_e)} error{'s' if len(_e) != 1 else ''}, "
            f"{len(_w)} warning{'s' if len(_w) != 1 else ''}"
        ),
        # --- raw ---
        "compile_stdout":     compile_stdout,
        "compile_stderr":     compile_stderr,
        "compile_returncode": returncode,
    }


# ---------------------------------------------------------------------------
# Convenience / utility handlers
# ---------------------------------------------------------------------------


def handle_run_code(payload: dict) -> dict:
    """Execute an inline Python snippet in an isolated subprocess with a timeout.

    Running as a subprocess (not in-process ``exec``) means the ``timeout`` is
    actually enforceable — an infinite loop is killed instead of hanging the
    device server forever — and user code cannot reach the server's own state.
    """
    code    = payload.get("code", "")
    timeout = _safe_int(payload.get("timeout"), 30)
    if not code:
        return _error("run_code_result", "code is required")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {
                "type": "run_code_result", "status": "timeout",
                "output": "", "errors": f"Execution exceeded {timeout}s and was terminated.",
                "warnings": "", "response": f"timeout — killed after {timeout}s",
                "stdout": [], "stderr": [], "files": [],
            }
        stdout_lines = proc.stdout.splitlines()
        stderr_lines = proc.stderr.splitlines()
        status = "ok" if proc.returncode == 0 else "error"
        files = _collect_files(tmpdir)

    _warnings, _ = _split_diagnostics(stderr_lines)
    _error_str = "\n".join(stderr_lines) if stderr_lines else ""
    _n_out = len(stdout_lines)
    _n_err = 1 if status == "error" else 0

    return {
        "type":     "run_code_result",
        "status":   status,
        "output":   "\n".join(stdout_lines),
        "errors":   _error_str,
        "warnings": "\n".join(_warnings),
        "response": f"{status} — {_n_out} line{'s' if _n_out != 1 else ''} output, "
                    f"{_n_err} error{'s' if _n_err != 1 else ''}",
        "stdout":   stdout_lines,
        "stderr":   stderr_lines,
        "files":    files,
    }


def handle_shell(payload: dict) -> dict:
    """Run an arbitrary shell command."""
    command = payload.get("command", "")
    timeout = _safe_int(payload.get("timeout"), 30)
    if not command:
        return _error("shell_result", "command is required")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,  # noqa: S602
        )
        _stdout = result.stdout.splitlines()
        _stderr = result.stderr.splitlines()
        return {
            "type":       "shell_result",
            "status":     "ok" if result.returncode == 0 else "runtime_error",
            "output":     "\n".join(_stdout),
            "errors":     "\n".join(_stderr) if result.returncode != 0 else "",
            "warnings":   "",
            "response":   f"exit {result.returncode}",
            "returncode": result.returncode,
            "stdout":     _stdout,
            "stderr":     _stderr,
        }
    except subprocess.TimeoutExpired:
        return {"type": "shell_result", "status": "timeout", "command": command,
                "errors": f"Command exceeded {timeout}s", "response": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return _error("shell_result", str(exc))


def handle_get_status(payload: dict) -> dict:
    """Report basic system information about the device."""
    info = {
        "platform": platform.platform(),
        "python":   sys.version,
        "hostname": platform.node(),
    }
    return {
        "type":     "status_report",
        "status":   "ok",
        "output":   "\n".join(f"{k}: {v}" for k, v in info.items()),
        "errors":   "",
        "warnings": "",
        "response": f"{info['hostname']} ({platform.system()})",
        "data":     info,
    }


def handle_ping(payload: dict) -> dict:
    return {"type": "pong", "status": "ok", "output": "pong",
            "errors": "", "warnings": "", "response": "pong"}


# ---------------------------------------------------------------------------
# Shared error shape + dispatch table
# ---------------------------------------------------------------------------


def _error(result_type: str, message: str) -> dict:
    """Build a uniform error result that still fills the output ports."""
    return {
        "type":     result_type,
        "status":   "error",
        "output":   "",
        "errors":   message,
        "warnings": "",
        "response": f"error — {message}",
        "error":    message,
    }


HANDLERS = {
    "compile_and_run": handle_compile_and_run,
    "run_code":        handle_run_code,
    "shell":           handle_shell,
    "status":          handle_get_status,
    "get_status":      handle_get_status,
    "ping":            handle_ping,
}


def dispatch(action: str, payload: dict) -> dict:
    """Run the handler for ``action`` and return its result dict."""
    handler = HANDLERS.get(action)
    if handler is None:
        return _error("unknown_command", f"unknown action: {action}")
    try:
        return handler(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("handler for '%s' raised: %s", action, exc)
        return _error(f"{action}_result", str(exc))
