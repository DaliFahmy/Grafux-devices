"""
flow.py
The one runner every ``custom`` block shares.

Three stages, the same for every manifest:

    stage    write one file per input port into $GRAFUX_IN
    run      bash -lc <runtime.entry>, streaming its output to the block face
    collect  read text outputs from $GRAFUX_OUT/<port>, match artifact globs,
             and judge the run

The verdict is NOT the exit code alone -- the rule openram taught: a tool can
exit 0 having built nothing.  A run is ``ok`` only if the entry exited 0 AND
every port in ``status.require_outputs`` came back non-empty.

What the entry may write besides its ports: ``$GRAFUX_OUT/errors`` and
``$GRAFUX_OUT/warnings`` are appended to the standard ports of those names, so an
adapter can explain a failure in words instead of leaving the user a log tail.

This module is pure apart from the transport functions it imports by name
(``exec_simple``/``exec_stream``/``_read_pod_text``), which the tests patch.
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any, Callable, Dict, List, Optional, Tuple

from EDA.flow import _read_pod_text
from EDA.pod_client import WORK_DIR, exec_simple, exec_stream, sftp_makedirs

from .manifest import InputPort, Manifest, ManifestError, parse_manifest

logger = logging.getLogger("custom.flow")

IN_SUBDIR = "in"
OUT_SUBDIR = "out"

# The sentinels the app writes into an unconnected port (PortDataService::
# kEmptyPortValue and the "unconnected" literal) -- they mean "no value".
_PLACEHOLDER_VALUES = {"empty", "unconnected"}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

LOG_TAIL_CHARS = 20000


def _is_placeholder(text: str) -> bool:
    return (text or "").strip().lower() in _PLACEHOLDER_VALUES


def coerce_input(port: InputPort, raw: str) -> Tuple[str, str]:
    """
    Normalise one input value against its declared type.

    Returns ``(value, problem)``; a non-empty problem refuses the run BEFORE the
    pod does any work.  Refusing is right here (unlike the cpu block's language
    port) because the adapter was written against the declared type -- handing
    ``run.sh`` "abc" for an int is a failure an hour into a run, not a guess.
    """
    value = raw
    t = port.type
    stripped = value.strip()
    if t == "int":
        if not stripped:
            return "", ""
        try:
            return str(int(stripped)), ""
        except ValueError:
            return value, f"input '{port.name}' must be an integer, got {stripped[:40]!r}"
    if t == "float":
        if not stripped:
            return "", ""
        try:
            float(stripped)
            return stripped, ""
        except ValueError:
            return value, f"input '{port.name}' must be a number, got {stripped[:40]!r}"
    if t == "bool":
        low = stripped.lower()
        if low in _TRUE:
            return "true", ""
        if low in _FALSE:
            return "false", ""
        return value, f"input '{port.name}' must be true or false, got {stripped[:40]!r}"
    if t == "enum":
        if stripped and stripped not in port.choices:
            return value, (f"input '{port.name}' must be one of {', '.join(port.choices)}, "
                           f"got {stripped[:40]!r}")
        return stripped, ""
    if t == "json":
        if not stripped:
            return "", ""
        try:
            json.loads(stripped)
        except json.JSONDecodeError as exc:
            return value, f"input '{port.name}' must be JSON: {exc.msg} at line {exc.lineno}"
        return stripped, ""
    # text / file: passed through byte for byte -- a program's leading
    # whitespace is part of the program.
    return value, ""


def resolve_inputs(manifest: Manifest, given: Dict[str, Any]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Decide every input's value: the block's port, else the manifest default.

    Returns ``(values, problems, notes)``.  ``notes`` lists values the block sent
    for ports the manifest does not declare -- ignored, but said out loud, since
    it usually means the block's manifest is stale.
    """
    given = {str(k): ("" if v is None else str(v)) for k, v in (given or {}).items()}
    values: Dict[str, str] = {}
    problems: List[str] = []
    for port in manifest.inputs:
        raw = given.get(port.name, "")
        if _is_placeholder(raw) or not raw.strip():
            raw = port.default
        value, problem = coerce_input(port, raw)
        if problem:
            problems.append(problem)
        elif port.required and not value.strip():
            problems.append(f"input '{port.name}' is required and has no value or default")
        values[port.name] = value
    known = set(manifest.input_names())
    extra = sorted(k for k in given if k not in known and not _is_placeholder(given[k]) and given[k].strip())
    notes = ([f"ignored input(s) the manifest does not declare: {', '.join(extra)}"] if extra else [])
    return values, problems, notes


def build_run_command(manifest: Manifest, work: str = WORK_DIR) -> str:
    """The one command the run stage executes, under a login shell."""
    in_dir = f"{work}/{IN_SUBDIR}"
    out_dir = f"{work}/{OUT_SUBDIR}"
    script = (
        f"export GRAFUX_WORK={shlex.quote(work)} GRAFUX_IN={shlex.quote(in_dir)} "
        f"GRAFUX_OUT={shlex.quote(out_dir)}; "
        'mkdir -p "$GRAFUX_OUT/files" && cd "$GRAFUX_WORK" && '
        f"{shlex.quote(manifest.runtime.entry)}"
    )
    return "bash -lc " + shlex.quote(script)


def artifact_globs(manifest: Manifest, work: str = WORK_DIR) -> List[str]:
    """Absolute on-pod globs for every artifact port, for the shared downloader."""
    out_dir = f"{work}/{OUT_SUBDIR}"
    return [f"{out_dir}/{p.glob}" for p in manifest.outputs if p.kind == "artifact" and p.glob]


def _list_matches(client, out_dir: str, glob: str) -> List[str]:
    """Paths (relative to out_dir) the glob matches on the pod, sorted."""
    script = (f"cd {shlex.quote(out_dir)} 2>/dev/null && "
              f'for f in {glob}; do [ -f "$f" ] && echo "$f"; done')
    _code, out, _err = exec_simple(client, "bash -lc " + shlex.quote(script), timeout=60)
    return sorted({ln.strip() for ln in (out or "").splitlines() if ln.strip()})


def run_custom(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run one manifest-defined block on the pod.  Never raises for a user error."""
    work = WORK_DIR
    in_dir = f"{work}/{IN_SUBDIR}"
    out_dir = f"{work}/{OUT_SUBDIR}"
    notes: List[str] = []

    def _result(outputs: Dict[str, str], status: str, stage: str, globs: List[str]) -> Dict[str, Any]:
        return {"outputs": outputs, "_status": status, "_stage": stage, "_globs": globs}

    try:
        manifest = parse_manifest(getattr(req, "manifest", ""))
    except ManifestError as exc:
        on_stage("stage", "failed")
        return _result({"status": "error", "errors": str(exc), "warnings": "", "log": ""},
                       "error", "stage", [])

    def _base(**over: str) -> Dict[str, str]:
        base = {name: "" for name in manifest.output_names()}
        base.update({"status": "error", "errors": "", "warnings": "", "log": ""})
        base.update(over)
        return base

    globs = artifact_globs(manifest, work)

    values, problems, extra_notes = resolve_inputs(manifest, getattr(req, "inputs", {}) or {})
    notes.extend(extra_notes)
    if problems:
        on_stage("stage", "failed")
        return _result(_base(errors="The run was refused before it started:\n" + "\n".join(problems),
                             warnings="\n".join(notes)), "error", "stage", [])

    # ---- stage: one file per input port ------------------------------------
    on_stage("stage", "running")
    sftp = client.open_sftp()
    try:
        sftp_makedirs(sftp, in_dir)
        sftp_makedirs(sftp, f"{out_dir}/files")
        for name, value in values.items():
            with sftp.open(f"{in_dir}/{name}", "wb") as fh:
                fh.write(value.encode("utf-8"))
    finally:
        sftp.close()
    on_stage("stage", "done")

    # ---- run ---------------------------------------------------------------
    on_stage("run", "running")
    timeout = int(getattr(req, "timeout", 0) or 0) or manifest.runtime.timeout_s
    code, out, err = exec_stream(
        client, build_run_command(manifest, work), timeout=timeout,
        on_line=on_line, should_cancel=should_cancel,
    )
    log_text = "\n".join(t for t in ((out or "").strip(), (err or "").strip()) if t)[-LOG_TAIL_CHARS:]
    on_stage("run", "done" if code == 0 else "failed")

    # ---- collect (also after a failure: partial outputs explain it) --------
    on_stage("collect", "running")
    outputs = _base(log=log_text)
    produced: Dict[str, bool] = {}
    for port in manifest.outputs:
        if port.kind == "text":
            text, too_big = _read_pod_text(client, f"{out_dir}/{port.name}", limit=port.inline_max)
            if too_big:
                notes.append(f"output '{port.name}' exceeded {port.inline_max} bytes and was "
                             f"attached as the artifact '{port.name}' instead.")
                globs.append(f"{out_dir}/{port.name}")
                produced[port.name] = True
            else:
                outputs[port.name] = text
                produced[port.name] = bool(text.strip())
        else:
            matches = _list_matches(client, out_dir, port.glob)
            outputs[port.name] = "\n".join(m.rsplit("/", 1)[-1] for m in matches)
            produced[port.name] = bool(matches)

    adapter_errors, _ = _read_pod_text(client, f"{out_dir}/errors", limit=64 * 1024)
    adapter_warnings, _ = _read_pod_text(client, f"{out_dir}/warnings", limit=64 * 1024)
    if adapter_warnings.strip():
        notes.append(adapter_warnings.strip())

    missing = [n for n in manifest.status.require_outputs if not produced.get(n)]
    errors: List[str] = []
    if code == -1:
        errors.append("The run was stopped.")
    elif code == -2:
        errors.append(f"The run exceeded its {timeout}s timeout. Raise the block's `timeout` port "
                      f"or the manifest's runtime.timeout_s.")
    elif code != 0:
        errors.append(f"{manifest.runtime.entry} exited with code {code}.")
        if code == 127:
            errors.append("Exit 127 means a command was not found: the image may not contain the "
                          "entry script or a tool it calls. Regenerate the block to rebuild its image.")
    if code == 0 and missing:
        errors.append("The run finished but produced no " + ", ".join(f"`{m}`" for m in missing)
                      + " -- a run that exits 0 without its required outputs has failed.")
    if adapter_errors.strip():
        errors.append(adapter_errors.strip())
    if code != 0 and (err or "").strip():
        errors.append((err or "").strip()[-8000:])

    ok = code == 0 and not missing
    outputs["status"] = "ok" if ok else "error"
    outputs["errors"] = "\n\n".join(errors) if not ok or adapter_errors.strip() else ""
    outputs["warnings"] = "\n".join(n for n in notes if n)
    on_stage("collect", "done")
    return _result(outputs, outputs["status"], "collect", globs)
