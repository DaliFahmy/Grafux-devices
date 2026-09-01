"""
flow.py
Turns block ports into tool invocations, drives them stage by stage over an open
SSH session, and parses what comes back.

The module splits cleanly in two, and the split is deliberate:

* **Pure functions** (``build_*``, ``parse_*``, ``stages_between``,
  ``infer_top_module``, ``globs_for``) are string-in/string-out.  They hold the
  real domain knowledge — what a yosys script must say to map onto a liberty file,
  what an ORFS ``config.mk`` needs, how to read a cell count out of ``stat`` — and
  they are unit-testable with no cloud, no container and no SSH.
* **Runner functions** (``run_verilator``, ``run_yosys``, ``run_orfs``) take an
  already-connected paramiko client and do I/O.

ORFS stages are driven one ``make`` target at a time rather than as a single
``make final``.  That is what makes "3/6 Placement…" exact instead of scraped out
of a log, and it means a failed route still leaves the placement artifacts
downloadable — which is usually exactly what the user needs to see.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .models import ORFS_STAGES
from .pod_client import WORK_DIR, _EDA_ENV, exec_simple, exec_stream

logger = logging.getLogger("eda.flow")

# Per-stage wall-clock ceilings (seconds).  Routing dominates everything else by an
# order of magnitude, which is why it gets its own generous budget.  Seeded from
# the M0 timing table; every one is env-overridable because they are design-size
# dependent and a big design will blow through any fixed default.
_STAGE_TIMEOUTS: Dict[str, int] = {
    "synth": int(os.environ.get("EDA_STAGE_TIMEOUT_SYNTH", "900") or "900"),
    "floorplan": int(os.environ.get("EDA_STAGE_TIMEOUT_FLOORPLAN", "600") or "600"),
    "place": int(os.environ.get("EDA_STAGE_TIMEOUT_PLACE", "1800") or "1800"),
    "cts": int(os.environ.get("EDA_STAGE_TIMEOUT_CTS", "900") or "900"),
    "route": int(os.environ.get("EDA_STAGE_TIMEOUT_ROUTE", "7200") or "7200"),
    "final": int(os.environ.get("EDA_STAGE_TIMEOUT_FINAL", "1800") or "1800"),
}

# Inline text ports are strings in a .txt file on the client, so an enormous
# netlist has no business being echoed into one.  Past this it is truncated and the
# full file is handed over as an artifact instead (the openroad block accepts
# either form in its netlist port).
NETLIST_INLINE_MAX = int(os.environ.get("EDA_NETLIST_INLINE_MAX", str(1024 * 1024)))

_DEFAULT_TOP = "top"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def stages_between(from_stage: str, to_stage: str) -> Tuple[str, ...]:
    """
    The ORFS stages to run, inclusive, clamped to the known sequence.

    An unknown or empty bound falls back to the full flow rather than raising —
    a typo in a port should not fail the run before a single tool has started.
    Reversed bounds collapse to the single ``from`` stage.
    """
    order = list(ORFS_STAGES)
    start = order.index(from_stage) if from_stage in order else 0
    end = order.index(to_stage) if to_stage in order else len(order) - 1
    if end < start:
        end = start
    return tuple(order[start:end + 1])


def stage_timeout(stage: str) -> int:
    """Wall-clock ceiling for one ORFS stage, in seconds."""
    return _STAGE_TIMEOUTS.get(stage, 1800)


_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", re.MULTILINE)


def infer_top_module(rtl: str, fallback: str = _DEFAULT_TOP) -> str:
    """
    Guess the top module from Verilog source: the LAST module declared.

    Last rather than first because Verilog convention puts submodules above the
    thing that instantiates them, and an AI-written file follows that convention
    more often than not.  Callers should always prefer an explicit ``top`` port;
    this only exists so a block with just RTL wired up still runs.
    """
    names = _MODULE_RE.findall(rtl or "")
    return names[-1] if names else fallback


def _split_tokens(text: str) -> List[str]:
    """Split a space- or newline-separated port value into tokens."""
    return [t for t in re.split(r"[\s,]+", (text or "").strip()) if t]


def define_flags(defines: str, prefix: str = "-D") -> List[str]:
    """Turn a 'WIDTH=8 DEBUG' port into compiler define flags."""
    return [f"{prefix}{tok}" for tok in _split_tokens(defines)]


def incdir_flags(include_dirs: str, prefix: str = "-I") -> List[str]:
    """Turn an include-dirs port into compiler include flags."""
    return [f"{prefix}{tok}" for tok in _split_tokens(include_dirs)]


def liberty_glob(pdk: str) -> str:
    """Shell glob for a platform's liberty files inside the ORFS tree."""
    return f"$FLOW_HOME/platforms/{pdk}/lib/*.lib"


def default_sdc(clock_port: str, clock_period: str) -> str:
    """
    A minimal SDC when the user wired nothing into the sdc port.

    One create_clock is enough for ORFS to run end to end, and it is the constraint
    a user would otherwise have to look up. Anything more opinionated would be
    guessing at their design.
    """
    port = (clock_port or "clk").strip() or "clk"
    period = (clock_period or "10").strip() or "10"
    return (
        f"create_clock -name core_clock -period {period} [get_ports {{{port}}}]\n"
        f"set_clock_uncertainty 0.1 [get_clocks core_clock]\n"
    )


def build_verilator_cmd(
    *,
    top: str,
    mode: str,
    sources: Sequence[str],
    testbench_file: str = "",
    defines: str = "",
    include_dirs: str = "",
    trace: bool = True,
    extra_flags: str = "",
) -> str:
    """
    Build the verilator command line.

    ``lint`` mode stops after elaboration checks; ``sim`` mode compiles the design
    plus the C++ harness into a native binary. ``-Wall`` is on in lint mode because
    lint output is the whole point there, but off in sim mode where a wall of style
    warnings would bury the actual simulation result.
    """
    parts = ["verilator"]
    if (mode or "sim").lower() == "lint":
        parts += ["--lint-only", "-Wall"]
    else:
        parts += ["--cc", "--exe", "--build", "-o", "sim"]
        if trace:
            parts.append("--trace")
    if top:
        parts += ["--top-module", top]
    parts += define_flags(defines, "+define+")
    parts += incdir_flags(include_dirs, "+incdir+")
    if extra_flags:
        parts += _split_tokens(extra_flags)
    parts += list(sources)
    if testbench_file and (mode or "sim").lower() != "lint":
        parts.append(testbench_file)
    return " ".join(parts)


def default_testbench(top: str, trace: bool = True) -> str:
    """
    A C++ harness for a design the user gave no testbench for.

    It does not verify behaviour — it cannot know what correct means — but it does
    elaborate, reset and clock the design for a few hundred cycles, which catches
    the failures that actually matter at this stage: a design that will not build,
    will not elaborate, or blows an assertion. It also emits a waveform so the user
    has something to look at while writing a real testbench.
    """
    cls = f"V{top}"
    trace_decl = (
        f'#include "verilated_vcd_c.h"\n' if trace else ""
    )
    trace_open = (
        "    Verilated::traceEverOn(true);\n"
        "    VerilatedVcdC* tfp = new VerilatedVcdC;\n"
        "    dut->trace(tfp, 99);\n"
        '    tfp->open("sim.vcd");\n'
        if trace else ""
    )
    trace_dump = "        tfp->dump(main_time);\n" if trace else ""
    trace_close = "    tfp->close();\n" if trace else ""
    return f"""// Auto-generated by Grafux — replace this by wiring your own testbench
// into the verilator block's `testbench` port.
#include <verilated.h>
{trace_decl}#include "{cls}.h"

static vluint64_t main_time = 0;
double sc_time_stamp() {{ return main_time; }}

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    {cls}* dut = new {cls};
{trace_open}
    // Drive reset low for a few cycles if the design has one, then free-run.
    for (int cycle = 0; cycle < 200; ++cycle) {{
        dut->eval();
{trace_dump}        ++main_time;
    }}

    dut->final();
{trace_close}    printf("GRAFUX_SIM_DONE cycles=%llu\\n", (unsigned long long)main_time);
    delete dut;
    return 0;
}}
"""


def build_yosys_script(
    *,
    top: str,
    sources: Sequence[str],
    liberty: str,
    netlist_out: str,
    stat_out: str = "",
    defines: str = "",
    include_dirs: str = "",
    synth_flags: str = "",
) -> str:
    """
    Build the yosys ``.ys`` script for an ASIC synthesis run.

    Standard flow: read, elaborate to the given top, ``synth``, technology-map
    flip-flops and combinational logic onto the liberty cells, clean up, then write
    the gate-level netlist and a machine-readable ``stat``.

    ``liberty`` may be empty, in which case the script stops at generic synthesis
    rather than failing — a netlist of generic cells is still useful feedback, and
    it keeps the block working on a platform whose liberty file we could not find.
    """
    read_flags = define_flags(defines, "-D") + incdir_flags(include_dirs, "-I")
    lines: List[str] = ["# Generated by Grafux — yosys block"]
    for src in sources:
        # Joined from a list so an empty flag set does not leave a double space.
        lines.append(" ".join(["read_verilog", "-sv", *read_flags, src]))
    lines.append(f"hierarchy -check -top {top}")
    lines.append(f"synth -top {top} {synth_flags}".rstrip())
    if liberty:
        lines.append(f"dfflibmap -liberty {liberty}")
        lines.append(f"abc -liberty {liberty}")
        lines.append("setundef -zero")
        lines.append("splitnets")
    lines.append("opt_clean -purge")
    lines.append("check")
    lines.append(f"write_verilog -noattr {netlist_out}")
    # stat goes to the console (which is what parse_yosys_stats reads) and, when a
    # path is given, to a file as well so it survives as a downloadable artifact.
    # `tee` echoes rather than redirects, so one command serves both.
    stat_cmd = "stat" + (f" -liberty {liberty}" if liberty else "")
    lines.append(f"tee -o {stat_out} {stat_cmd}" if stat_out else stat_cmd)
    return "\n".join(lines) + "\n"


def build_orfs_config(
    *,
    design: str,
    platform: str,
    verilog_files: Sequence[str],
    sdc_file: str,
    netlist_file: str = "",
    clock_period: str = "",
    core_utilization: str = "",
    aspect_ratio: str = "",
    die_area: str = "",
    core_area: str = "",
    place_density: str = "",
    extra: str = "",
) -> str:
    """
    Build an OpenROAD-flow-scripts ``config.mk``.

    When ``netlist_file`` is set the flow starts from a gate-level netlist (the
    normal path when a yosys block is wired upstream); otherwise ORFS runs its own
    synthesis from ``verilog_files``.  Explicit ``die_area``/``core_area`` override
    ``core_utilization`` — ORFS honours whichever pair it is given, and setting both
    is how a user pins an exact floorplan.
    """
    lines = [
        "# Generated by Grafux — openroad block",
        f"export DESIGN_NAME     = {design}",
        f"export PLATFORM        = {platform}",
    ]
    if netlist_file:
        # SYNTH_NETLIST short-circuits ORFS' own synthesis and starts at floorplan.
        lines.append(f"export SYNTH_NETLIST   = {netlist_file}")
    if verilog_files:
        lines.append("export VERILOG_FILES   = " + " ".join(verilog_files))
    lines.append(f"export SDC_FILE        = {sdc_file}")
    if die_area:
        lines.append(f"export DIE_AREA        = {die_area}")
    if core_area:
        lines.append(f"export CORE_AREA       = {core_area}")
    if not die_area and core_utilization:
        lines.append(f"export CORE_UTILIZATION = {core_utilization}")
        lines.append(f"export CORE_ASPECT_RATIO = {aspect_ratio or '1'}")
        lines.append("export CORE_MARGIN     = 2")
    if place_density:
        lines.append(f"export PLACE_DENSITY   = {place_density}")
    if clock_period:
        lines.append(f"export ABC_CLOCK_PERIOD_IN_PS = {_ns_to_ps(clock_period)}")
    if extra.strip():
        lines.append("")
        lines.append("# --- extra_config port ---")
        lines.append(extra.strip())
    return "\n".join(lines) + "\n"


def _ns_to_ps(period_ns: str) -> str:
    """Convert a nanosecond clock period to integer picoseconds for ABC."""
    try:
        return str(int(round(float(period_ns) * 1000)))
    except (TypeError, ValueError):
        return "10000"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_STAT_CELLS_RE = re.compile(r"Number of cells:\s+(\d+)")
_STAT_WIRES_RE = re.compile(r"Number of wires:\s+(\d+)")
_STAT_AREA_RE = re.compile(r"Chip area for(?: module)?[^:]*:\s*([0-9.]+)")
_STAT_CELL_LINE_RE = re.compile(r"^\s{4,}(\S+)\s+(\d+)\s*$", re.MULTILINE)
# Standard-cell libraries name flip-flops with a "df" infix (sky130 __dfxtp_,
# nangate DFF_X1, asap7 DFF...), which is the only portable way to count state.
_SEQ_CELL_RE = re.compile(r"(?:^|_)d?ff|dlxtp|dfxtp|_df", re.IGNORECASE)


def parse_yosys_stats(text: str) -> Dict[str, Any]:
    """
    Extract a structured summary from yosys ``stat`` output.

    Returns {cell_count, wire_count, area_um2, sequential_cells, by_cell_type}.
    Every field is best-effort: yosys' stat format varies with the flags used, and
    a missing number should degrade the report, not fail the run.
    """
    text = text or ""
    out: Dict[str, Any] = {
        "cell_count": 0,
        "wire_count": 0,
        "area_um2": 0.0,
        "sequential_cells": 0,
        "by_cell_type": {},
    }
    cells = _STAT_CELLS_RE.search(text)
    if cells:
        out["cell_count"] = int(cells.group(1))
    wires = _STAT_WIRES_RE.search(text)
    if wires:
        out["wire_count"] = int(wires.group(1))
    area = _STAT_AREA_RE.search(text)
    if area:
        try:
            out["area_um2"] = float(area.group(1))
        except ValueError:
            pass
    by_type: Dict[str, int] = {}
    seq = 0
    for name, count in _STAT_CELL_LINE_RE.findall(text):
        # The summary lines ("Number of cells") also match the indent pattern.
        if name.lower().startswith("number"):
            continue
        n = int(count)
        by_type[name] = by_type.get(name, 0) + n
        if _SEQ_CELL_RE.search(name):
            seq += n
    out["by_cell_type"] = by_type
    out["sequential_cells"] = seq
    return out


# ORFS metrics keys vary across releases, so each logical metric lists the source
# keys it accepts, most-preferred first.
_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "wns_ns": ("finish__timing__setup__ws", "route__timing__setup__ws", "timing__setup__ws"),
    "tns_ns": ("finish__timing__setup__tns", "route__timing__setup__tns", "timing__setup__tns"),
    "area_um2": ("finish__design__instance__area", "design__instance__area"),
    "utilization": ("finish__design__instance__utilization", "design__instance__utilization"),
    "num_instances": ("finish__design__instance__count", "design__instance__count"),
    "num_nets": ("design__nets__count", "route__net"),
    "drc_violations": ("finish__design__violations", "route__drc_errors", "detailedroute__route__drc_errors"),
    "power_mw": ("finish__power__total", "power__total"),
    "clock_period_ns": ("constraints__clocks__count", "clock__period"),
}


def parse_orfs_metrics(raw: Any) -> Dict[str, Any]:
    """
    Normalize an ORFS metrics/metadata JSON blob into the block's ``metrics`` port.

    ``raw`` may be the JSON text or an already-parsed dict.  Unknown or absent keys
    are simply omitted rather than defaulted, so a caller can tell "the flow did not
    report this" apart from "the flow reported zero" — which for a DRC count is a
    distinction that matters a great deal.
    """
    data: Dict[str, Any]
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return {}
    if not isinstance(data, dict):
        return {}

    out: Dict[str, Any] = {}
    for metric, aliases in _METRIC_ALIASES.items():
        for key in aliases:
            if key in data and data[key] is not None:
                value = data[key]
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                out[metric] = value
                break

    # Per-stage runtimes, when ORFS recorded them.
    runtimes = {
        key.split("__")[0]: value
        for key, value in data.items()
        if key.endswith("__runtime__total") and value is not None
    }
    if runtimes:
        out["runtime_s"] = runtimes
    return out


def globs_for(kind: str, *, work_dir: str = WORK_DIR, platform: str = "",
              design: str = "") -> List[str]:
    """
    Artifact globs to pull back after a run.

    Collected even when a stage FAILED, so a route that blew up still returns the
    placement DEF and the reports that explain why.
    """
    if kind == "verilator":
        return [f"{work_dir}/*.vcd", f"{work_dir}/*.log", f"{work_dir}/obj_dir/*.log"]
    if kind == "yosys":
        return [f"{work_dir}/*.v", f"{work_dir}/*.log", f"{work_dir}/*.json",
                f"{work_dir}/*.txt"]
    results = f"$FLOW_HOME/results/{platform}/{design}/base"
    reports = f"$FLOW_HOME/reports/{platform}/{design}/base"
    return [
        f"{results}/*.gds", f"{results}/*.def", f"{results}/*.v",
        f"{results}/*.spef", f"{results}/*.sdc",
        f"{reports}/*.json", f"{reports}/*.rpt", f"{reports}/*.log",
        f"{work_dir}/*.png",
    ]


# ---------------------------------------------------------------------------
# Runners — these take a connected paramiko client and do I/O.
# ---------------------------------------------------------------------------

def _write_file(sftp, path: str, content: str) -> None:
    """Write a text file into the pod, creating parent dirs as needed."""
    from .pod_client import sftp_makedirs
    parent = path.rsplit("/", 1)[0]
    if parent and parent != path:
        sftp_makedirs(sftp, parent)
    with sftp.open(path, "wb") as fh:
        fh.write(content.encode("utf-8"))


def _sh(command: str) -> str:
    """Wrap a command so it runs under bash with the EDA environment exported."""
    return "bash -lc " + shlex.quote(_EDA_ENV + command)


def _resolve_liberty(client, pdk: str, override: str = "") -> str:
    """
    Find a liberty file for the platform inside the container.

    Globbed on-device rather than hardcoded because the filename encodes process
    corner and voltage (sky130_fd_sc_hd__tt_025C_1v80.lib) and differs per platform.
    Returns "" when nothing matches, which callers treat as "synthesize generically".
    """
    if override.strip():
        return override.strip()
    cmd = _sh(f"ls -1 {liberty_glob(pdk)} 2>/dev/null | head -1")
    _code, out, _err = exec_simple(client, cmd, timeout=60)
    return out.strip().splitlines()[0].strip() if out.strip() else ""


def run_verilator(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Lint or simulate a design with Verilator.

    Returns the ``outputs`` map for the block's ports plus bookkeeping keys
    (``_globs``, ``_status``, ``_stage``).  Never raises for a tool failure — a
    design that does not compile is a normal, expected result here, and the whole
    point of the block is to report it clearly.
    """
    top = (req.top or "").strip() or infer_top_module(req.rtl)
    mode = (req.mode or "sim").strip().lower()
    trace = str(req.trace or "").strip().lower() in ("1", "true", "yes", "on")
    sftp = client.open_sftp()
    try:
        _write_file(sftp, f"{WORK_DIR}/design.v", req.rtl or "")
        tb = req.testbench or ""
        if mode != "lint" and not tb.strip():
            tb = default_testbench(top, trace)
        if tb.strip():
            _write_file(sftp, f"{WORK_DIR}/tb.cpp", tb)
    finally:
        sftp.close()

    outputs: Dict[str, str] = {"top": top, "rtl": req.rtl or ""}
    log_parts: List[str] = []

    on_stage("verilate", "running")
    build_cmd = build_verilator_cmd(
        top=top,
        mode=mode,
        sources=["design.v"],
        testbench_file="tb.cpp" if (mode != "lint" and tb.strip()) else "",
        defines=req.defines,
        include_dirs=req.include_dirs,
        trace=trace,
        extra_flags=req.verilator_flags,
    )
    code, out, err = exec_stream(
        client, _sh(f"cd {WORK_DIR} && {build_cmd}"),
        timeout=min(int(req.timeout or 900), 1800),
        on_line=on_line, should_cancel=should_cancel,
    )
    log_parts.append(f"$ {build_cmd}\n{out}\n{err}")
    # Verilator writes diagnostics to stderr; both lint findings and build errors
    # land there, and which one it is depends only on the exit code.
    outputs["lint"] = err.strip()
    on_stage("verilate", "done" if code == 0 else "failed")

    if code != 0:
        outputs["status"] = "error"
        outputs["passed"] = "false"
        outputs["errors"] = err.strip() or out.strip() or (
            "Verilator was cancelled" if code == -1 else
            "Verilator exceeded its timeout" if code == -2 else
            f"Verilator exited with code {code}"
        )
        outputs["warnings"] = ""
        outputs["sim_output"] = ""
        outputs["log"] = "\n".join(log_parts)
        return {"outputs": outputs, "_status": "error", "_stage": "verilate",
                "_globs": globs_for("verilator")}

    # Warnings are the stderr of a build that nonetheless succeeded.
    outputs["warnings"] = err.strip()

    if mode == "lint":
        outputs["status"] = "ok"
        outputs["passed"] = "true"
        outputs["sim_output"] = ""
        outputs["errors"] = ""
        outputs["log"] = "\n".join(log_parts)
        return {"outputs": outputs, "_status": "ok", "_stage": "verilate",
                "_globs": globs_for("verilator")}

    on_stage("sim", "running")
    sim_cmd = f"./obj_dir/sim {req.sim_args or ''}".strip()
    code, out, err = exec_stream(
        client, _sh(f"cd {WORK_DIR} && {sim_cmd}"),
        timeout=int(req.timeout or 900),
        on_line=on_line, should_cancel=should_cancel,
    )
    log_parts.append(f"$ {sim_cmd}\n{out}\n{err}")
    on_stage("sim", "done" if code == 0 else "failed")

    outputs["sim_output"] = out.strip()
    outputs["log"] = "\n".join(log_parts)
    # A non-zero exit is the near-universal convention for a failing testbench, and
    # $fatal/assertion failures surface that way too.
    passed = code == 0
    outputs["passed"] = "true" if passed else "false"
    outputs["status"] = "ok" if passed else "error"
    outputs["errors"] = "" if passed else (
        err.strip() or f"Simulation exited with code {code}"
    )
    if err.strip() and passed:
        outputs["warnings"] = (outputs.get("warnings", "") + "\n" + err.strip()).strip()
    return {"outputs": outputs, "_status": "ok" if passed else "error", "_stage": "sim",
            "_globs": globs_for("verilator")}


def run_yosys(
    client,
    req,
    *,
    pdk: str,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Synthesize RTL to a gate-level netlist and report cell statistics."""
    top = (req.top or "").strip() or infer_top_module(req.rtl)
    platform = (req.pdk or "").strip() or pdk

    on_stage("synth", "running")
    liberty = _resolve_liberty(client, platform, req.liberty)
    script = build_yosys_script(
        top=top,
        sources=["design.v"],
        liberty=liberty,
        netlist_out="netlist.v",
        stat_out="stat.txt",
        defines=req.defines,
        include_dirs=req.include_dirs,
        synth_flags=req.synth_flags,
    )
    sftp = client.open_sftp()
    try:
        _write_file(sftp, f"{WORK_DIR}/design.v", req.rtl or "")
        _write_file(sftp, f"{WORK_DIR}/synth.ys", script)
    finally:
        sftp.close()

    code, out, err = exec_stream(
        client, _sh(f"cd {WORK_DIR} && yosys -s synth.ys 2>&1 | tee yosys.log"),
        timeout=int(req.timeout or 900),
        on_line=on_line, should_cancel=should_cancel,
    )
    on_stage("synth", "done" if code == 0 else "failed")

    outputs: Dict[str, str] = {
        "top": top,
        "pdk": platform,
        "log": out,
        "report": out,
    }
    if code != 0:
        outputs["status"] = "error"
        outputs["netlist"] = ""
        outputs["stats"] = "{}"
        outputs["errors"] = err.strip() or out.strip() or (
            "Yosys was cancelled" if code == -1 else
            "Yosys exceeded its timeout" if code == -2 else
            f"Yosys exited with code {code}"
        )
        outputs["warnings"] = ""
        return {"outputs": outputs, "_status": "error", "_stage": "synth",
                "_globs": globs_for("yosys")}

    # Read the netlist back inline when it is small enough for a port file; a large
    # one still travels as an artifact (globs_for picks up *.v).
    netlist = ""
    truncated = False
    sftp = client.open_sftp()
    try:
        with sftp.open(f"{WORK_DIR}/netlist.v", "rb") as fh:
            data = fh.read(NETLIST_INLINE_MAX + 1)
        truncated = len(data) > NETLIST_INLINE_MAX
        netlist = data[:NETLIST_INLINE_MAX].decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — a missing netlist is reported, not raised
        logger.warning("could not read back netlist: %s", exc)
    finally:
        sftp.close()

    stats = parse_yosys_stats(out)
    warnings = "\n".join(
        line for line in out.splitlines() if line.lstrip().lower().startswith("warning")
    )
    if truncated:
        netlist = ""
        warnings = (
            warnings + "\nNetlist exceeded the inline limit and was attached as the "
            "artifact netlist.v — wire it into the openroad block's netlist port as usual."
        ).strip()

    outputs["status"] = "ok" if netlist or truncated else "error"
    outputs["netlist"] = netlist
    outputs["stats"] = json.dumps(stats)
    outputs["errors"] = "" if (netlist or truncated) else "Yosys produced no netlist."
    outputs["warnings"] = warnings
    return {"outputs": outputs, "_status": outputs["status"], "_stage": "synth",
            "_globs": globs_for("yosys")}


def run_orfs(
    client,
    req,
    *,
    pdk: str,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    max_run_s: int = 0,
) -> Dict[str, Any]:
    """
    Run the OpenROAD flow, one ``make`` stage at a time.

    Stage-by-stage is what makes progress reporting exact and lets a failure stop
    the flow while still returning everything produced up to that point.
    """
    import time as _time

    platform = (req.pdk or "").strip() or pdk
    design = (req.top or "").strip() or infer_top_module(req.rtl) or _DEFAULT_TOP
    netlist = (req.netlist or "").strip()
    stages = stages_between(req.from_stage, req.to_stage)

    # A wired netlist means synthesis is already done upstream — running ORFS' own
    # synth would silently discard the yosys block's result.
    if netlist and "synth" in stages:
        stages = tuple(s for s in stages if s != "synth") or ("floorplan",)

    design_dir = f"$FLOW_HOME/designs/{platform}/{design}"
    src_dir = f"$FLOW_HOME/designs/src/{design}"
    # config.mk is read by make, which does not expand $FLOW_HOME inside the file
    # the same way the shell does, so paths inside it use make's own variable.
    cfg_design_dir = f"./designs/{platform}/{design}"
    cfg_src_dir = f"./designs/src/{design}"

    sdc = req.sdc.strip() or default_sdc(req.clock_port, req.clock_period)
    config = build_orfs_config(
        design=design,
        platform=platform,
        verilog_files=[f"{cfg_src_dir}/design.v"] if not netlist else [],
        netlist_file=f"{cfg_design_dir}/netlist.v" if netlist else "",
        sdc_file=f"{cfg_design_dir}/constraint.sdc",
        clock_period=req.clock_period,
        core_utilization=req.core_utilization,
        aspect_ratio=req.aspect_ratio,
        die_area=req.die_area,
        core_area=req.core_area,
        place_density=req.place_density,
        extra=req.extra_config,
    )

    # Resolve $FLOW_HOME once so SFTP (which has no shell) can write real paths.
    _c, flow_home, _e = exec_simple(client, _sh("echo $FLOW_HOME"), timeout=30)
    flow_home = flow_home.strip() or "/OpenROAD-flow-scripts/flow"
    abs_design_dir = f"{flow_home}/designs/{platform}/{design}"
    abs_src_dir = f"{flow_home}/designs/src/{design}"

    sftp = client.open_sftp()
    try:
        _write_file(sftp, f"{abs_design_dir}/config.mk", config)
        _write_file(sftp, f"{abs_design_dir}/constraint.sdc", sdc)
        if netlist:
            _write_file(sftp, f"{abs_design_dir}/netlist.v", netlist)
        if req.rtl.strip():
            _write_file(sftp, f"{abs_src_dir}/design.v", req.rtl)
    finally:
        sftp.close()

    outputs: Dict[str, str] = {"top": design, "pdk": platform}
    log_parts: List[str] = []
    stages_done: List[str] = []
    reached = ""
    failed = False
    started = _time.monotonic()

    for stage in stages:
        if should_cancel and should_cancel():
            failed = True
            outputs["errors"] = "Run cancelled."
            break
        # The watchdog is checked between stages as well as inside each one: a
        # single stage's timeout says nothing about total spend, and an orphaned
        # pod over a weekend is real money.
        if max_run_s and (_time.monotonic() - started) > max_run_s:
            failed = True
            outputs["errors"] = (
                f"Run exceeded EDA_MAX_RUN_MINUTES ({max_run_s // 60} min) and was stopped "
                f"after the {reached or 'first'} stage."
            )
            break

        on_stage(stage, "running")
        cmd = f"cd $FLOW_HOME && make DESIGN_CONFIG={cfg_design_dir}/config.mk {stage}"
        code, out, err = exec_stream(
            client, _sh(cmd),
            timeout=stage_timeout(stage),
            on_line=on_line, should_cancel=should_cancel,
        )
        log_parts.append(f"$ make {stage}\n{out}\n{err}")
        reached = stage
        if code == 0:
            stages_done.append(stage)
            on_stage(stage, "done")
            continue

        on_stage(stage, "failed")
        failed = True
        outputs["errors"] = (
            f"Stage '{stage}' was cancelled." if code == -1 else
            f"Stage '{stage}' exceeded its {stage_timeout(stage)}s timeout." if code == -2 else
            f"Stage '{stage}' failed (exit {code}).\n" + (err.strip() or out.strip()[-4000:])
        )
        break

    # Metrics and reports are collected whether or not the flow completed — a
    # failed route is exactly when the user most needs to see the numbers.
    metrics: Dict[str, Any] = {}
    _c, meta, _e = exec_simple(
        client,
        _sh(f"cat $FLOW_HOME/reports/{platform}/{design}/base/metadata.json 2>/dev/null || true"),
        timeout=60,
    )
    if meta.strip():
        metrics = parse_orfs_metrics(meta)

    if "final" in stages_done:
        _render_layout_png(client, flow_home, platform, design, on_line)

    outputs["stage"] = reached
    outputs["status"] = "error" if failed else "ok"
    outputs["metrics"] = json.dumps(metrics)
    outputs["log"] = "\n".join(log_parts)
    outputs["warnings"] = "\n".join(
        line for part in log_parts for line in part.splitlines()
        if "warning" in line.lower()
    )[:20000]
    outputs.setdefault("errors", "")
    return {
        "outputs": outputs,
        "_status": outputs["status"],
        "_stage": reached,
        "_stages_done": stages_done,
        "_globs": globs_for("openroad", platform=platform, design=design),
    }


def _render_layout_png(client, flow_home: str, platform: str, design: str,
                       on_line: Optional[Callable[[str], None]]) -> None:
    """
    Render the finished GDS to a PNG for the in-block preview.

    Entirely best-effort: KLayout's batch rendering depends on the image build, and
    a missing picture must never turn a successful tapeout-ready run into a failure.
    """
    gds = f"{flow_home}/results/{platform}/{design}/base/6_final.gds"
    script = (
        "layout = RBA::Layout::new\n"
        f'layout.read("{gds}")\n'
        "view = RBA::LayoutView.new\n"
        f'view.load_layout("{gds}", 0)\n'
        "view.max_hier\n"
        "view.zoom_fit\n"
        f'view.save_image("{WORK_DIR}/layout.png", 1200, 1200)\n'
    )
    try:
        sftp = client.open_sftp()
        try:
            _write_file(sftp, f"{WORK_DIR}/render.rb", script)
        finally:
            sftp.close()
        code, _out, err = exec_simple(
            client, _sh(f"cd {WORK_DIR} && klayout -zz -rm render.rb"), timeout=300
        )
        if code != 0 and on_line:
            on_line(f"(layout preview unavailable: {err.strip()[:200]})")
    except Exception as exc:  # noqa: BLE001 — preview is cosmetic
        logger.debug("layout png render failed: %s", exc)
