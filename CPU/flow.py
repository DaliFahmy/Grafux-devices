"""
flow.py
Turns a ``cpu`` block's ports into a compile, a benchmark and a verdict.

The module splits in two, deliberately, the same way ``EDA/flow.py`` does:

* **Pure functions** (``normalize_language``, ``build_compile_cmd``,
  ``run_target_for``, ``parse_check_lines``, ``benchmark_stats``,
  ``build_bench_script``, ``globs_for``) are string-in/string-out.  They hold
  the real knowledge — what compiles what, how to read a verdict out of program
  output, how to turn a list of samples into an honest measurement — and are
  unit-testable with no cloud, no container and no SSH.
* **The runner** (``run_cpu``) takes an already-connected paramiko client and
  does I/O.

HOW THE MEASUREMENT IS TAKEN, and why it is shaped this way:

* Timing happens ON THE POD, around the process, with ``date +%s%N``.  Nothing
  in the number is network latency or SSH scheduling — the same discipline the
  gpu block uses.
* The case runs ``warmup`` times discarded, then ``repetitions`` times measured.
  One sample on a shared cloud vCPU is noise; page faults and a cold instruction
  cache land in the warmup instead of in the first reported figure.
* ``/usr/bin/time -v`` runs ONCE MORE, separately and unmeasured, purely for peak
  RSS.  Folding it into a measured run would put the wrapper's own overhead into
  the timing it is standing next to.
* The reported ``duration`` is the MEAN, and ``benchmark`` carries min, max and
  stddev alongside it.  A mean with no spread beside it invites a comparison the
  data does not support.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from EDA.pod_client import WORK_DIR, exec_simple, exec_stream

logger = logging.getLogger("cpu.flow")

# Written into the pod's work directory, which EDA's runner clears before every
# run — so a warm pod never reports the previous run's output as this one's.
SOURCE_STEMS = {"c": "case.c", "cpp": "case.cpp", "python": "case.py"}
BINARY_NAME = "case"
BENCH_SCRIPT = "bench.sh"
STDOUT_FILE = "run_out.txt"
STDERR_FILE = "run_err.txt"
TIME_FILE = "time_v.txt"

# Markers the on-pod script prints.  Parsed rather than timed from this side,
# because this side is on the other end of an SSH connection.
MS_MARKER = "GRAFUX_MS:"
RC_MARKER = "GRAFUX_RC:"
EXIT_MARKER = "GRAFUX_EXIT:"

# Ceilings so a runaway case cannot hold a pod open until the run watchdog fires.
COMPILE_TIMEOUT_S = 300
PROBE_TIMEOUT_S = 60

_LANG_ALIASES = {
    "c": "c",
    "c99": "c", "c11": "c", "c17": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp",
    "cplusplus": "cpp", "c++17": "cpp", "c++20": "cpp",
    "python": "python", "py": "python", "python3": "python",
}


def normalize_language(text: str) -> Tuple[str, str]:
    """
    Resolve a ``language`` port into one of c / cpp / python.

    Returns ``(language, note)``; ``note`` is non-empty when the value was not
    recognised and the default was used instead.  An unknown language does NOT
    fail the run: the port is free text, the cost of guessing wrong is one wasted
    compile, and the cost of refusing is a user staring at an error because they
    typed "C++17".
    """
    raw = (text or "").strip().lower()
    if not raw:
        return "cpp", ""
    resolved = _LANG_ALIASES.get(raw)
    if resolved:
        return resolved, ""
    return "cpp", (
        f"Language '{text.strip()}' is not one of c, cpp or python; the case was "
        f"built as C++. Set the language port to c, cpp or python."
    )


def _positive_int(text: str, default: int, *, maximum: int) -> int:
    """
    Read a count off a text port.

    Empty, non-numeric and non-positive all mean "use the default" — an unwired
    port arrives as "" or as the frontend's "empty" sentinel, and neither should
    be read as "run it zero times".
    """
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        return default
    if value < 1:
        return default
    return min(value, maximum)


def source_name_for(language: str) -> str:
    """The filename the case is written to, which is what picks the compiler."""
    return SOURCE_STEMS.get(language, SOURCE_STEMS["cpp"])


def build_compile_cmd(
    language: str,
    *,
    source: str,
    binary: str,
    build_flags: str = "",
    defines: str = "",
    include_dirs: str = "",
) -> Optional[str]:
    """
    The compile command, or None for a language that does not compile.

    ``build_flags`` go AFTER the source and ``-o`` output.  That is not cosmetic:
    ``-l`` libraries must follow the object that references them or the linker
    reports undefined symbols, and a user adding ``-lm`` to the port would
    otherwise get a link error with no visible cause.  The gpu block learned this
    one the hard way.
    """
    if language == "python":
        return None
    compiler = "gcc" if language == "c" else "g++"
    parts = [compiler]
    for token in (defines or "").split():
        parts.append(f"-D{token}" if not token.startswith("-D") else token)
    for token in (include_dirs or "").replace("\n", " ").split():
        parts.append(f"-I{token}" if not token.startswith("-I") else token)
    parts += [source, "-o", binary]
    parts += (build_flags or "").split()
    return " ".join(shlex.quote(p) if " " in p else p for p in parts)


def run_target_for(language: str, *, binary: str, source: str, args: str = "") -> str:
    """The command that executes the case, argv included."""
    base = f"python3 {source}" if language == "python" else f"./{binary}"
    argv = (args or "").strip()
    return f"{base} {argv}" if argv else base


def build_bench_script(
    run_target: str,
    *,
    warmup: int,
    repetitions: int,
    work_dir: str = WORK_DIR,
) -> str:
    """
    The benchmark driver that runs ON THE POD.

    Everything about the measurement that matters is in here: the clock is the
    pod's, the warmup runs are discarded, each measured run is timed
    individually so a spread can be reported, and the stdout kept is the LAST
    measured run's so it corresponds to a run that was actually timed.

    ``set -u`` but deliberately NOT ``set -e``: a verification case that fails is
    an ordinary outcome this block must report, not a reason to abandon the
    remaining repetitions and lose the measurement.
    """
    return "\n".join([
        "#!/bin/bash",
        "set -u",
        f"cd {shlex.quote(work_dir)}",
        "worst_rc=0",
        f"for _ in $(seq 1 {warmup}); do",
        f"  {run_target} >/dev/null 2>&1 || true",
        "done",
        f"for _ in $(seq 1 {repetitions}); do",
        "  s=$(date +%s%N)",
        f"  {run_target} > {STDOUT_FILE} 2> {STDERR_FILE}",
        "  r=$?",
        "  e=$(date +%s%N)",
        f'  echo "{MS_MARKER}$(( (e - s) / 1000000 ))"',
        f'  echo "{RC_MARKER}$r"',
        '  if [ "$r" -ne 0 ]; then worst_rc=$r; fi',
        "done",
        # Peak RSS only. A separate, unmeasured run so the wrapper's overhead
        # never lands in a number reported as the case's duration.
        f"/usr/bin/time -v {run_target} > /dev/null 2> {TIME_FILE} || true",
        f'echo "{EXIT_MARKER}$worst_rc"',
        "",
    ])


def parse_bench_output(text: str) -> Tuple[List[int], List[int], Optional[int]]:
    """Pull the per-run timings, per-run exit codes and final verdict off the log."""
    samples: List[int] = []
    codes: List[int] = []
    worst: Optional[int] = None
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(MS_MARKER):
            try:
                samples.append(int(line[len(MS_MARKER):]))
            except ValueError:
                continue
        elif line.startswith(RC_MARKER):
            try:
                codes.append(int(line[len(RC_MARKER):]))
            except ValueError:
                continue
        elif line.startswith(EXIT_MARKER):
            try:
                worst = int(line[len(EXIT_MARKER):])
            except ValueError:
                continue
    return samples, codes, worst


def benchmark_stats(samples_ms: List[int]) -> Dict[str, Any]:
    """
    Summarise the measured runs.

    ``stddev`` needs two samples; with one it is reported as 0.0 rather than
    omitted, so a consumer never has to branch on a missing key — but `min` and
    `max` being equal is the honest signal that there is no spread to speak of.
    """
    clean = [s for s in samples_ms if s >= 0]
    if not clean:
        return {"exec_ms_mean": 0.0, "exec_ms_min": 0, "exec_ms_max": 0,
                "exec_ms_stddev": 0.0, "repetitions": 0}
    return {
        "exec_ms_mean": round(statistics.fmean(clean), 3),
        "exec_ms_min": min(clean),
        "exec_ms_max": max(clean),
        "exec_ms_stddev": round(statistics.stdev(clean), 3) if len(clean) > 1 else 0.0,
        "repetitions": len(clean),
    }


# A verification case reports its own checks. These are the shapes that actually
# show up in hand-written bring-up code, in rough order of how common they are.
_CHECK_PATTERNS = (
    re.compile(r"^\s*(?P<verdict>PASS|FAIL|OK|ERROR)\s*[:\-]\s*(?P<name>.+?)\s*$", re.I),
    re.compile(r"^\s*\[\s*(?P<verdict>PASS|FAIL|OK)\s*\]\s*(?P<name>.+?)\s*$", re.I),
    re.compile(r"^\s*(?P<name>.+?)\s*[:\-]{1,2}\s*(?P<verdict>PASS|FAIL|OK|ERROR)\s*$", re.I),
)
_FAIL_WORDS = {"fail", "error"}


def parse_check_lines(text: str) -> Dict[str, str]:
    """
    Read a case's own PASS/FAIL lines out of its stdout.

    Best-effort by design: this is a convenience on top of the exit code, never a
    substitute for it.  A case that prints nothing recognisable is judged purely
    on its exit status, which is why ``run_cpu`` requires BOTH a zero exit and no
    failing line before it reports ``passed``.
    """
    checks: Dict[str, str] = {}
    for line in (text or "").splitlines():
        if len(line) > 400:
            continue
        for pattern in _CHECK_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group("name").strip()
            verdict = match.group("verdict").strip().lower()
            if not name or len(name) > 200:
                break
            checks[name] = "fail" if verdict in _FAIL_WORDS else "pass"
            break
    return checks


def parse_max_rss_kb(time_v_output: str) -> int:
    """Peak resident set size in KB from ``/usr/bin/time -v``; 0 when absent."""
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                      time_v_output or "")
    return int(match.group(1)) if match else 0


def parse_machine(lscpu_output: str) -> Dict[str, str]:
    """The few ``lscpu`` fields worth putting in front of a user."""
    wanted = {
        "Model name": "cpu_model",
        "CPU(s)": "vcpus",
        "Architecture": "arch",
        "CPU max MHz": "cpu_max_mhz",
        "Flags": "",  # matched then skipped; kept here so the intent is visible
    }
    out: Dict[str, str] = {}
    for line in (lscpu_output or "").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        target = wanted.get(key.strip())
        if target:
            out[target] = value.strip()
    return out


def globs_for(kind: str = "cpu", *, work_dir: str = WORK_DIR) -> List[str]:
    """
    Artifact globs pulled back after a run.

    Collected even when the case FAILED — a failing run's output files are
    usually the whole reason to look at it.  The compiled binary is deliberately
    not globbed: it is large, opaque, and reproducible from the source port.
    """
    return [f"{work_dir}/*.txt", f"{work_dir}/*.log", f"{work_dir}/*.json",
            f"{work_dir}/*.csv"]


def _sh(command: str) -> str:
    """
    Run a command under a LOGIN shell.

    ``-l`` is load-bearing: a non-login SSH exec session does not inherit the
    image's Docker ENV PATH, so without it a perfectly good toolchain reports
    "gcc: command not found" with exit 127.
    """
    return "bash -lc " + shlex.quote(command)


def _read_pod_file(client, path: str, *, limit: int = 200_000) -> str:
    """Read a file the run produced; empty string when it does not exist."""
    code, out, _err = exec_simple(
        client, _sh(f"head -c {limit} {shlex.quote(path)} 2>/dev/null || true"),
        timeout=PROBE_TIMEOUT_S,
    )
    return out if code == 0 else ""


def run_cpu(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Compile, benchmark and judge a verification case on the pod.

    Four stages — ``build``, ``warmup``, ``bench``, ``collect`` — so the block
    face says something true while a multi-repetition run is in flight.

    The verdict is NOT the exit code alone: a case that returns 0 while printing
    "FAIL: cache writeback" has failed, and one that prints nothing recognisable
    is judged on its exit code.  Both must be clean for ``passed`` to be true.
    """
    work = WORK_DIR
    language, lang_note = normalize_language(getattr(req, "language", ""))
    source = source_name_for(language)
    warmup = _positive_int(getattr(req, "warmup", ""), 1, maximum=50)
    repetitions = _positive_int(getattr(req, "repetitions", ""), 5, maximum=200)
    timeout = int(getattr(req, "timeout", 900) or 900)
    notes: List[str] = [n for n in (lang_note,) if n]

    def _outputs(**over: Any) -> Dict[str, Any]:
        base = {
            "status": "error", "passed": "false", "response": "", "results": "{}",
            "benchmark": "{}", "duration": "", "machine": "{}",
            "errors": "", "warnings": "\n".join(notes), "log": "",
        }
        base.update(over)
        return base

    def _fail(stage: str, message: str, **over: Any) -> Dict[str, Any]:
        on_stage(stage, "failed")
        return {"outputs": _outputs(errors=message, **over),
                "_status": "error", "_stage": stage, "_globs": globs_for()}

    code_text = (getattr(req, "code", "") or "").strip()
    if not code_text:
        return _fail("build", (
            "No verification case to run: the `code` port is empty. Wire a "
            "post_silicon_verification block's `code` output into it, or paste a "
            "program in yourself."
        ))

    # ---- stage 1: build ----------------------------------------------------
    on_stage("build", "running")
    sftp = client.open_sftp()
    try:
        with sftp.open(f"{work}/{source}", "wb") as fh:
            fh.write(code_text.encode("utf-8"))
    finally:
        sftp.close()

    compile_ms = 0
    compile_cmd = build_compile_cmd(
        language,
        source=source,
        binary=BINARY_NAME,
        build_flags=getattr(req, "build_flags", ""),
        defines=getattr(req, "defines", ""),
        include_dirs=getattr(req, "include_dirs", ""),
    )
    if compile_cmd is not None:
        import time as _time
        started = _time.monotonic()
        code, out, err = exec_stream(
            client, _sh(f"cd {shlex.quote(work)} && {compile_cmd}"),
            timeout=min(timeout, COMPILE_TIMEOUT_S),
            on_line=on_line, should_cancel=should_cancel,
        )
        compile_ms = int((_time.monotonic() - started) * 1000)
        diagnostics = "\n".join(t for t in (out, err) if t).strip()
        if code == -1:
            return _fail("build", "The run was stopped before the case finished building.")
        if code != 0:
            if code == 127:
                return _fail("build", (
                    "The compiler was not found on the pod (exit 127). Leave the "
                    "`image` port empty to use the default post-silicon image, which "
                    "carries gcc and g++."
                ), log=diagnostics)
            return _fail("build", (
                f"The verification case did not compile ({language}):\n{diagnostics}"
            ), log=diagnostics)
        if diagnostics:
            # Warnings on a successful build are worth surfacing -- in a
            # verification case, an unused-result or sign-compare warning is often
            # the bug the case was written to find.
            notes.append(diagnostics)
    on_stage("build", "done")

    if should_cancel is not None and should_cancel():
        return _fail("build", "The run was stopped before the benchmark started.")

    # ---- stages 2+3: warmup and bench --------------------------------------
    on_stage("warmup" if warmup else "bench", "running")
    run_target = run_target_for(
        language, binary=BINARY_NAME, source=source, args=getattr(req, "args", ""),
    )
    script = build_bench_script(
        run_target, warmup=warmup, repetitions=repetitions, work_dir=work,
    )
    sftp = client.open_sftp()
    try:
        with sftp.open(f"{work}/{BENCH_SCRIPT}", "wb") as fh:
            fh.write(script.encode("utf-8"))
    finally:
        sftp.close()

    on_stage("bench", f"{repetitions} runs")
    code, out, err = exec_stream(
        client, _sh(f"bash {shlex.quote(work + '/' + BENCH_SCRIPT)}"),
        timeout=timeout, on_line=on_line, should_cancel=should_cancel,
    )
    if code == -1:
        return _fail("bench", "The run was stopped.")
    if code == -2:
        return _fail("bench", (
            f"The case did not finish within {timeout}s across {repetitions} "
            f"repetitions. Raise the `timeout` port, or lower `repetitions`."
        ), log=(out or "")[-8000:])

    samples, per_run_codes, worst_rc = parse_bench_output(out)
    if not samples:
        return _fail("bench", (
            "The case produced no timing samples — it may have failed to start. "
            "Check the log below."
        ), log="\n".join(t for t in (out, err) if t)[-8000:])

    # ---- stage 4: collect --------------------------------------------------
    on_stage("collect", "running")
    stdout_text = _read_pod_file(client, f"{work}/{STDOUT_FILE}")
    stderr_text = _read_pod_file(client, f"{work}/{STDERR_FILE}")
    time_v_text = _read_pod_file(client, f"{work}/{TIME_FILE}", limit=20_000)
    _code, lscpu_text, _err = exec_simple(
        client, _sh("lscpu 2>/dev/null || true"), timeout=PROBE_TIMEOUT_S,
    )

    machine = parse_machine(lscpu_text)
    checks = parse_check_lines(stdout_text)
    exit_code = worst_rc if worst_rc is not None else (
        per_run_codes[-1] if per_run_codes else 0
    )
    has_failing_check = any(v == "fail" for v in checks.values())
    passed = exit_code == 0 and not has_failing_check

    stats = benchmark_stats(samples)
    benchmark = {
        "compile_ms": compile_ms,
        **stats,
        "warmup": warmup,
        "samples_ms": samples,
        "max_rss_kb": parse_max_rss_kb(time_v_text),
        "exit_code": exit_code,
        "language": language,
        **machine,
    }

    errors = ""
    if not passed:
        failing = [name for name, verdict in checks.items() if verdict == "fail"]
        if failing:
            errors = "The case reported failing checks: " + ", ".join(failing[:20])
            if len(failing) > 20:
                errors += f" (+{len(failing) - 20} more)"
        else:
            errors = (
                f"The case exited with code {exit_code} and reported no PASS/FAIL "
                f"lines, so the verdict is its exit status."
            )
        if stderr_text.strip():
            errors += "\n\n" + stderr_text.strip()[:8000]

    on_stage("collect", "done")
    outputs = _outputs(
        status="ok" if passed else "failed",
        passed="true" if passed else "false",
        response=stdout_text,
        results=json.dumps(checks),
        benchmark=json.dumps(benchmark),
        duration=f"{stats['exec_ms_mean']:.3f}",
        machine=json.dumps(machine),
        errors=errors,
        log="\n".join(t for t in (out, err) if t)[-20000:],
    )
    return {"outputs": outputs,
            "_status": "ok" if passed else "failed",
            "_stage": "collect",
            "_globs": globs_for()}
