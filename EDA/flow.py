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

import ast
import difflib
import json
import logging
import os
import re
import shlex
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import ngspice
from .models import DEFAULT_OPENGCRAM_TECH, DEFAULT_OPENRAM_TECH, ORFS_STAGES
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


def _allow_empty_netlist() -> bool:
    """
    True when the post-synthesis empty-netlist gate in ``run_orfs`` is disabled.

    Read per call rather than at import so a test (or an operator) can flip it
    without reloading the module. The gate is on by default because a netlist
    with no cells is never routable — but the escape hatch exists for anyone
    deliberately exercising the later stages on an empty design.
    """
    return (os.environ.get("EDA_ALLOW_EMPTY_NETLIST", "") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }


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


# Verilog port lists come in two flavours and BOTH must be handled, because the
# RTL arrives from whoever — or whatever — filled the block's `rtl` port:
#
#   ANSI:      module calc #(parameter W=8) (input wire clk, output reg [W-1:0] q);
#   non-ANSI:  module calc(a, b, y);  input a, b;  output y;
#
# Keeping the LAST identifier of each comma-separated item covers both: it drops
# the direction/type/range prefix of the ANSI form and is a no-op on the bare
# names of the non-ANSI form.
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")

# Words that can be the last identifier of a port item without being its name.
_PORT_NOISE = frozenset({
    "input", "output", "inout", "wire", "reg", "logic", "signed", "unsigned",
    "bit", "byte", "int", "integer", "real", "time", "tri", "wand", "wor",
    "supply0", "supply1", "parameter", "localparam", "var",
})


def _strip_comments(text: str) -> str:
    """Drop // and /* */ comments so they cannot hide or fake a port name."""
    return _COMMENT_RE.sub(" ", text or "")


def _balanced(text: str, start: int) -> str:
    """
    Contents of the parenthesised group opening at ``text[start]``.

    Returns "" when the parentheses never close, which is how a truncated or
    malformed source gets rejected rather than half-parsed.
    """
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return ""


def _split_top_level(text: str) -> List[str]:
    """Split on commas that are not nested inside (), [] or {}."""
    items: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [item for item in items if item.strip()]


def module_ports(rtl: str, top: str) -> List[str]:
    """
    The ports declared in module ``top``'s header, in declaration order.

    An empty list means "could not tell", NOT "this module has no ports" —
    callers must never act on it as though the design were portless.  Rewriting a
    user's clock constraint on the strength of a failed parse would be worse than
    leaving it alone.
    """
    source = _strip_comments(rtl)
    name = (top or "").strip()
    if not name:
        return []
    match = re.search(r"\bmodule\s+" + re.escape(name) + r"\b", source)
    if not match:
        return []

    pos = match.end()

    def skip_space(index: int) -> int:
        while index < len(source) and source[index].isspace():
            index += 1
        return index

    pos = skip_space(pos)
    # An optional #( ... ) parameter list sits between the name and the ports.
    if pos < len(source) and source[pos] == "#":
        pos = skip_space(pos + 1)
        if pos >= len(source) or source[pos] != "(":
            return []
        params = _balanced(source, pos)
        if not params:
            return []
        pos = skip_space(pos + len(params) + 2)
    if pos >= len(source) or source[pos] != "(":
        return []

    ports: List[str] = []
    for item in _split_top_level(_balanced(source, pos)):
        names = [n for n in _IDENT_RE.findall(item) if n not in _PORT_NOISE]
        if names:
            ports.append(names[-1])
    return ports


# Port names that mean "clock" in practice, most canonical first.  An exact match
# here beats the substring sweep below, so `clk` wins over `clk_div_out`.
_CLOCK_NAMES = (
    "clk", "clock", "clk_i", "i_clk", "clk_in", "sys_clk", "sysclk",
    "clock_i", "i_clock", "core_clk", "clk_core", "aclk", "hclk",
)


def resolve_clock_port(rtl: str, top: str, requested: str) -> Tuple[str, str]:
    """
    Reconcile the block's ``clock_port`` with the ports the RTL actually declares.

    Returns ``(port, warning)``.  An empty ``port`` means the design has no clock
    and the caller should reach for ``virtual_clock_sdc`` instead.

    Why this is worth doing: ``default_sdc`` writes ``create_clock ...
    [get_ports {clk}]`` with no idea whether ``clk`` exists.  When it does not,
    OpenSTA does not fail — it quietly substitutes a VIRTUAL clock ([WARNING
    STA-0366] followed later by [WARNING STA-0450]), CTS then finds no clock nets,
    and the design sails through four more stages with no clock tree.
    """
    wanted = (requested or "").strip() or "clk"
    ports = module_ports(rtl, top)
    if not ports:
        # Header unparseable: trust the user's setting rather than guess.
        return wanted, ""
    if wanted in ports:
        return wanted, ""

    swap = (
        "clock_port '{wanted}' is not a port of module '{top}'; using '{found}' "
        "instead. Set the block's clock_port to silence this."
    )
    lowered = {port.lower(): port for port in ports}
    for candidate in _CLOCK_NAMES:
        if candidate in lowered:
            found = lowered[candidate]
            return found, swap.format(wanted=wanted, top=top, found=found)
    for port in ports:
        if "clk" in port.lower() or "clock" in port.lower():
            return port, swap.format(wanted=wanted, top=top, found=port)

    return "", (
        f"Module '{top}' declares no clock port (ports: {', '.join(ports)}). "
        "Constraining with a virtual clock so timing analysis stays valid. If "
        "this design is meant to be sequential, its clock is missing from the "
        "module header."
    )


def _safe_filename(name: str) -> str:
    """
    Reduce a module name to something safe to use as a filename.

    Verilog identifiers may contain ``$`` (and an escaped identifier almost
    anything), which is legal in a module name but awkward in a path.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    return cleaned or _DEFAULT_TOP


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


# A platform's lib/ directory holds more than the standard-cell library: I/O pad
# stubs, RAM macros, fill and antenna cells. Those contain no flip-flops, so
# mapping onto one makes ``dfflibmap`` fail with the thoroughly unhelpful "dffs
# with async set or reset are not supported" — which reads like a problem with the
# user's RTL rather than with the library we chose for them.
#
# sky130hd is the case that bites: its lib/ sorts ``sky130_dummy_io.lib`` ahead of
# the real ``sky130_fd_sc_hd__tt_025C_1v80.lib``, so picking the first file
# alphabetically picks the useless one every time.
_SUPPORT_LIB_MARKERS = ("dummy", "_io", "io_", "ram", "sram", "fill", "antenna", "pad")


def _is_support_lib(path: str) -> bool:
    """True for an I/O / macro / fill library rather than the standard cells."""
    name = path.rsplit("/", 1)[-1].lower()
    return any(marker in name for marker in _SUPPORT_LIB_MARKERS)


def pick_liberty(paths: Sequence[str]) -> str:
    """
    Choose the standard-cell liberty file from a platform's lib/ listing.

    ``paths`` must be ordered LARGEST FIRST (the caller runs ``ls -S``): across
    every ORFS platform the standard-cell library is by far the biggest file in
    the directory, which is a far more portable signal than any naming convention.
    Support libraries are filtered out first; if that leaves nothing we fall back
    to the largest file rather than giving up, since an unfamiliar platform's
    naming should not block synthesis entirely.
    """
    candidates = [p.strip() for p in paths if p and p.strip()]
    if not candidates:
        return ""
    preferred = [p for p in candidates if not _is_support_lib(p)]
    return (preferred or candidates)[0]


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


def virtual_clock_sdc(clock_period: str) -> str:
    """
    An SDC for a design that has no clock port at all.

    ``create_clock`` with no ``[get_ports]`` declares a VIRTUAL clock, which is
    the correct constraint for purely combinational logic.  It is also, notably,
    what OpenSTA falls back to on its own when ``create_clock`` names a port that
    does not exist — except that it then also warns the clock cannot be
    propagated, and the user is left reading STA-0366/STA-0450 rather than being
    told their design has no clock.  Saying it deliberately is quieter and honest.

    The I/O delays give the combinational paths something to be timed against.
    Zero rather than an invented budget: any real number here would be a guess
    about a surrounding system we know nothing about.
    """
    period = (clock_period or "10").strip() or "10"
    return (
        f"create_clock -name core_clock -period {period}\n"
        f"set_clock_uncertainty 0.1 [get_clocks core_clock]\n"
        f"set_input_delay 0 -clock core_clock [all_inputs]\n"
        f"set_output_delay 0 -clock core_clock [all_outputs]\n"
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


# ---------------------------------------------------------------------------
# cocotb: build the runner script that drives a Python testbench
# ---------------------------------------------------------------------------
#
# WHY A GENERATED SCRIPT rather than a Makefile.  cocotb's classic entry point is
# a Makefile that pulls in `$(shell cocotb-config --makefiles)/Makefile.sim`, which
# hard-codes assumptions about the build directory and is awkward to drive over a
# non-login SSH exec.  cocotb 2.x ships a first-class Python runner instead, so a
# ~30-line script gives exact control over the build flags, the results path and
# the stage markers, and it fails in Python where the traceback is readable.

# cocotb's runner puts the compiled simulation and verilator's coverage.dat here.
COCOTB_BUILD_DIR = "sim_build"
COCOTB_RESULTS_XML = "results.xml"
COCOTB_COVERAGE_DAT = f"{COCOTB_BUILD_DIR}/coverage.dat"
COCOTB_COVERAGE_INFO = "coverage.info"
COCOTB_LOG = "cocotb.log"
# `tail -c` on the way back IS the size bound: a runaway log cannot blow up
# the run result, and the tail is the half that explains the outcome.
COCOTB_LOG_MAX_BYTES = 512 * 1024

# A testbench arrives as a lump of text on a port, with nothing to say what
# language it is written in.  Classifying it is what lets `mode=sim` — the default
# every verilator block already carries — keep working the moment a testbench
# block is wired in: the cocotb tests are recognised and run as cocotb instead of
# being handed to a C++ compiler that would reject them as syntax errors.
_COCOTB_MARKERS = ("import cocotb", "from cocotb", "@cocotb.test")
_CPP_MARKERS = ("#include", "Verilated", "sc_time_stamp", "int main(")


def testbench_kind(testbench: str) -> str:
    """
    Classify a testbench as ``"python"`` (cocotb), ``"cpp"``, or ``""``.

    ``""`` means "cannot tell", never "there is no testbench", so callers fall back
    to the mode they were explicitly asked for rather than guessing.
    """
    text = testbench or ""
    if not text.strip():
        return ""
    if any(marker in text for marker in _COCOTB_MARKERS):
        return "python"
    if any(marker in text for marker in _CPP_MARKERS):
        return "cpp"
    # Last resort: a file that parses as Python is Python, whatever it imports.
    # Handing a testbench with no cocotb import to a C++ compiler produces a wall
    # of syntax errors that reads as the user's fault rather than the tool's.
    try:
        ast.parse(text)
    except SyntaxError:
        return ""
    return "python"


def resolve_verilator_mode(mode: str, testbench: str) -> Tuple[str, str]:
    """
    The mode a verilator run should actually use, and why, if it differs.

    Every verilator block created before cocotb existed carries ``mode="sim"``, and
    so does the creation dialog's default — so wiring a testbench block into one
    has to simply work rather than feeding Python to a C++ compiler.  A Python
    testbench therefore promotes "sim" to "cocotb", and the returned note goes onto
    the warnings port so the user can see why the run took that path.

    ``lint`` is NEVER promoted: it is a deliberate "do not simulate" request, and
    quietly turning it into a simulation would spend pod-minutes nobody asked for.
    """
    requested = (mode or "sim").strip().lower()
    kind = testbench_kind(testbench)
    if requested == "lint":
        return "lint", ""
    if requested == "cocotb":
        if kind == "cpp":
            return "cocotb", (
                "mode=cocotb was requested but the testbench looks like a C++ "
                "harness; it is being run as cocotb anyway.")
        if not kind:
            return "cocotb", "mode=cocotb was requested with no Python testbench."
        return "cocotb", ""
    if kind == "python":
        return "cocotb", (
            f"The testbench is a cocotb (Python) testbench, so it was run with "
            f"mode=cocotb rather than the requested mode={requested}.")
    return requested, ""


def cocotb_build_args(
    simulator: str,
    *,
    trace: bool = True,
    coverage: bool = True,
    assertions: bool = False,
    extra_flags: str = "",
) -> List[str]:
    """
    Simulator-specific build flags for a cocotb run.

    ``--timing`` is not optional for Verilator: cocotb 2.x drives the design from
    Python coroutines, and without the timing-aware scheduler a testbench that
    awaits a clock edge hangs until the timeout instead of failing loudly.

    ``-Wno-fatal`` is deliberate.  Verilator promotes most warnings to errors by
    default, so a single width mismatch in AI-drafted RTL would kill the build
    before a single test ran — turning "your FIFO drops the last entry", which is
    what the user needs to hear, into "verilator exited 1".  Lint is its own mode
    for exactly this reason; here the tests are the verdict.
    """
    sim = (simulator or "verilator").strip().lower()
    args: List[str] = []
    if sim == "verilator":
        # --public-flat-rw is what makes the design's ports visible over VPI;
        # without it every dut.<signal> in the testbench fails with "not found",
        # which reads as a broken testbench rather than a missing flag. cocotb's
        # own runner adds it in most versions — passing it is harmless if so.
        args += ["--timing", "--public-flat-rw", "-Wno-fatal"]
        if trace:
            args.append("--trace")
        if coverage:
            args.append("--coverage")
        if assertions:
            # Only when the user actually wired SVA in: --assert makes Verilator
            # honour `assert property`, and turning it on unconditionally would
            # change how an ordinary design's own assertions behave.
            args.append("--assert")
    elif sim == "icarus":
        # cocotb needs the 2012 dialect for anything past plain Verilog-2001.
        args.append("-g2012")
    args += _split_tokens(extra_flags)
    return args


def normalize_simulator(simulator: str) -> str:
    """Map the simulator port value onto a name ``cocotb_tools.runner`` knows."""
    sim = (simulator or "").strip().lower()
    if sim in ("iverilog", "icarus"):
        return "icarus"
    return "verilator"


def sva_binding_problem(sva: str) -> str:
    """
    Why this SVA text cannot be compiled in, or "" when it can.

    An assertion module that is never ``bind``-ed compiles cleanly, runs nothing,
    and reports success — a false green, which is the one outcome a verification
    block must never produce.  Without a bind statement we would have to guess the
    port mapping, so it is refused and the reason goes on the warnings port.
    """
    text = (sva or "").strip()
    if not text:
        return ""
    if not re.search(r"\bbind\b", text):
        return ("the SVA text has no `bind` statement, so its assertions would "
                "never be attached to the design and would report success without "
                "checking anything; it was skipped")
    return ""


def build_cocotb_runner_script(
    *,
    top: str,
    sources: Sequence[str],
    test_module: str,
    simulator: str = "verilator",
    trace: bool = True,
    coverage: bool = True,
    assertions: bool = False,
    seed: str = "",
    tests: str = "",
    extra_flags: str = "",
    build_dir: str = COCOTB_BUILD_DIR,
) -> str:
    """
    The ``run_cocotb.py`` text: build the design, run the tests, write results.xml.

    ``cocotb_tools.runner`` is the cocotb **2.x** location; ``cocotb.runner`` is the
    1.x one and importing it here fails only inside the pod, at run time.

    The script prints ``GRAFUX_STAGE <name>`` markers so one invocation still
    reports build and simulation as separate stages, and it swallows the non-zero
    exit a failing test produces — a failing test is a normal result for this
    block, and ``results.xml`` is what the caller judges on.
    """
    sim = normalize_simulator(simulator)
    build_args = cocotb_build_args(
        sim, trace=trace, coverage=coverage, assertions=assertions,
        extra_flags=extra_flags)
    testcases = _split_tokens(tests)
    # cocotb wants an int seed; anything else is dropped rather than crashing the
    # run inside the pod over a stray character in a port value.
    seed_val = (seed or "").strip()
    if not seed_val.isdigit():
        seed_val = ""
    return f'''# Auto-generated by Grafux - do not edit; regenerate from EDA/flow.py.
import os
import sys
import traceback

# Everything this run prints has to survive to be PARSED afterwards, and the SSH
# reader upstream keeps only the last 400 lines of each stream. So the script
# re-executes itself once and mirrors both streams into cocotb.log, which is
# fetched back whole.
#
# Why not `python3 run_cocotb.py | tee cocotb.log` in the shell: a pipeline's
# exit code is the LAST command's, so the caller would read tee's success and a
# failing build would look clean. Why not stderr=STDOUT: the caller reads `err`
# separately to decide build_failed, libpython_missing and what lands on
# `warnings`, and merging the streams would silently empty all three.
GRAFUX_LOG_FILE = os.path.abspath("cocotb.log")
if os.environ.get("GRAFUX_TEE") != "1":
    import subprocess
    import threading

    child_env = dict(os.environ)
    child_env["GRAFUX_TEE"] = "1"
    log_handle = open(GRAFUX_LOG_FILE, "wb")
    log_lock = threading.Lock()
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=child_env)

    def grafux_pump(source, sink):
        # Line at a time, flushed every line: the parent is streaming these to a
        # live UI, and a 4 KB buffer would make every GRAFUX_STAGE marker late.
        for chunk in iter(source.readline, b""):
            sink.buffer.write(chunk)
            sink.flush()
            with log_lock:
                log_handle.write(chunk)
                log_handle.flush()
        source.close()

    pumps = [threading.Thread(target=grafux_pump, args=(child.stdout, sys.stdout)),
             threading.Thread(target=grafux_pump, args=(child.stderr, sys.stderr))]
    for pump in pumps:
        pump.daemon = True
        pump.start()
    grafux_code = child.wait()
    for pump in pumps:
        pump.join(timeout=30)
    log_handle.close()
    sys.exit(grafux_code)


# cocotb moved its runner from `cocotb.runner` (1.x) to `cocotb_tools.runner`
# (2.x). Trying both, and saying which one answered, turns a version mismatch
# into one readable line in the log instead of an ImportError inside a pod.
try:
    from cocotb_tools.runner import get_runner
    print("GRAFUX_COCOTB_RUNNER cocotb_tools.runner", flush=True)
except ImportError:
    from cocotb.runner import get_runner
    print("GRAFUX_COCOTB_RUNNER cocotb.runner", flush=True)

TOP = {json.dumps(top)}
SOURCES = {json.dumps(list(sources))}
TEST_MODULE = {json.dumps(test_module)}
SIMULATOR = {json.dumps(sim)}
BUILD_ARGS = {json.dumps(build_args)}
BUILD_DIR = {json.dumps(build_dir)}
TESTCASES = {json.dumps(testcases)}
SEED = {json.dumps(seed_val)}
WAVES = {bool(trace)!r}

# An absolute path is used as-is by the runner; a relative one would land inside
# the build directory, where the caller does not look for it. A results file that
# cannot be found is indistinguishable from tests that never ran, so it is pinned
# three ways: this path, TEST_DIR below, and a glob on the way back out.
TEST_DIR = os.getcwd()
RESULTS_XML = os.path.abspath({json.dumps(COCOTB_RESULTS_XML)})

# Both spellings: cocotb 2.0 renamed these, and setting only one silently no-ops.
ENV = dict(COCOTB_ANSI_OUTPUT="0", COCOTB_REDUCED_LOG_FMT="1",
           COCOTB_RESULTS_FILE=RESULTS_XML)
if SEED:
    ENV["COCOTB_RANDOM_SEED"] = SEED
    ENV["RANDOM_SEED"] = SEED


# cocotb's VPI shim EMBEDS CPython inside the simulator binary, so it needs the
# SHARED libpython, not merely a working `python3`. Debian and Ubuntu link the
# interpreter statically and ship libpython3.x.so in a separate package, so an
# image with python3 and cocotb installed still reaches this point and then dies
# inside the runner with "Unable to find libpython" -- after a full, successful
# Verilator build, which is what makes it read like a simulation bug.
#
# The image installs it (see EDA/docker/Dockerfile.verify). This resolver is the
# belt-and-braces for an image that carries the library somewhere cocotb's own
# find_libpython does not look, and it turns "genuinely not installed" into one
# named marker the caller can explain instead of a traceback out of cocotb.
def resolve_libpython():
    loc = os.environ.get("LIBPYTHON_LOC", "")
    if loc and os.path.exists(loc):
        return loc
    try:
        import find_libpython
        found = find_libpython.find_libpython()
        if found:
            return found
    except Exception:
        pass
    import glob
    import sysconfig
    tag = "%d.%d" % sys.version_info[:2]
    # LIBDIR/LIBPL point at the BASE interpreter even from inside a virtualenv,
    # which is exactly where the distro puts the library.
    dirs = [sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LIBPL"),
            "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/local/lib"]
    for directory in dirs:
        if not directory:
            continue
        for pattern in ("libpython%s*.so" % tag, "libpython%s*.so.*" % tag):
            hits = sorted(glob.glob(os.path.join(directory, pattern)))
            if hits:
                return hits[0]
    return ""


LIBPYTHON = resolve_libpython()
if LIBPYTHON:
    # os.environ, not just extra_env: the runner copies os.environ over its own
    # env dict, so setting only extra_env is the version-dependent half of this.
    os.environ["LIBPYTHON_LOC"] = LIBPYTHON
    ENV["LIBPYTHON_LOC"] = LIBPYTHON
    print("GRAFUX_LIBPYTHON %s" % (LIBPYTHON,), flush=True)
else:
    print("GRAFUX_LIBPYTHON_MISSING", flush=True)

runner = get_runner(SIMULATOR)

print("GRAFUX_STAGE build", flush=True)
try:
    runner.build(verilog_sources=SOURCES, hdl_toplevel=TOP,
                 build_args=BUILD_ARGS, build_dir=BUILD_DIR,
                 waves=WAVES, always=True)
except Exception:
    traceback.print_exc()
    print("GRAFUX_BUILD_FAILED", flush=True)
    sys.exit(2)

print("GRAFUX_STAGE sim", flush=True)
kwargs = dict(hdl_toplevel=TOP, test_module=TEST_MODULE, build_dir=BUILD_DIR,
              test_dir=TEST_DIR, results_xml=RESULTS_XML, waves=WAVES,
              extra_env=ENV)
if TESTCASES:
    kwargs["testcase"] = TESTCASES
if SEED:
    kwargs["seed"] = int(SEED)
try:
    runner.test(**kwargs)
except SystemExit as exc:
    # A failing test makes the runner exit non-zero. That is an expected outcome
    # here, not a tool crash: results.xml carries the verdict.
    print("GRAFUX_TESTS_FAILED %s" % (exc.code,), flush=True)
except TypeError:
    # An older/newer runner that does not accept one of the optional kwargs;
    # retry with the minimum that every version has ever supported rather than
    # failing a run over a keyword name.
    traceback.print_exc()
    print("GRAFUX_SIM_RETRY_MINIMAL", flush=True)
    try:
        runner.test(hdl_toplevel=TOP, test_module=TEST_MODULE,
                    build_dir=BUILD_DIR, results_xml=RESULTS_XML)
    except SystemExit as exc:
        print("GRAFUX_TESTS_FAILED %s" % (exc.code,), flush=True)
except Exception:
    traceback.print_exc()
    print("GRAFUX_SIM_ERROR", flush=True)
    sys.exit(3)
print("GRAFUX_STAGE report", flush=True)
'''


def build_coverage_cmd(out: str = COCOTB_COVERAGE_INFO) -> str:
    """
    Turn Verilator's raw ``coverage.dat`` into an lcov ``.info`` report.

    Verilator writes ``coverage.dat`` into whatever directory the simulation ran
    in, and which directory that is depends on the cocotb version — so the file is
    located at run time rather than assumed.  Guessing wrong here would report
    "no coverage" for a run that measured it perfectly well.
    """
    candidates = f"coverage.dat {COCOTB_BUILD_DIR}/coverage.dat"
    return (
        f'DAT=$(ls -1 {candidates} 2>/dev/null | head -1); '
        f'if [ -n "$DAT" ]; then verilator_coverage --write-info {out} "$DAT"; '
        f'else echo "no coverage.dat was produced" >&2; exit 1; fi'
    )


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

    ``verilog_files`` is the design's single source.  There is deliberately no
    "start from a pre-synthesized netlist" option: OpenROAD-flow-scripts has no
    such variable — its ``1_synth.odb`` target depends on the yosys chain
    unconditionally, so a config claiming otherwise is silently ignored and the
    flow synthesizes anyway.  ``run_orfs`` therefore hands it source and lets it
    do its own synthesis; see that function for what the upstream yosys block is
    then for.

    Explicit ``die_area``/``core_area`` override ``core_utilization`` — ORFS
    honours whichever pair it is given, and setting both is how a user pins an
    exact floorplan.
    """
    lines = [
        "# Generated by Grafux — openroad block",
        f"export DESIGN_NAME     = {design}",
        f"export PLATFORM        = {platform}",
    ]
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
# OpenRAM (memory compiler)
# ---------------------------------------------------------------------------

# Where a run's generated views land, relative to WORK_DIR.  A directory of its
# own rather than WORK_DIR itself because ``classify_openram_outputs`` reads the
# whole listing and classifies by extension -- a stray ``.v`` staged through the
# ``files`` port would otherwise be mistaken for the generated model.
OPENRAM_OUT_DIR = "openram_out"

# The config file written into the pod.  Named, not inlined, because OpenRAM
# copies it into the output directory as ``<output_name>.py`` and that copy --
# with every default it filled in -- is what the ``config`` output port reports.
OPENRAM_CONFIG_FILE = "openram_config.py"

# Roughly the point past which a compile stops being interactive.  Not a hard
# limit: ``EDA_MAX_RUN_MINUTES`` and the run timeout are, and this only warns.
OPENRAM_WARN_BITS = 256 * 1024
# Past this a run is hours, and the pod bills for every one of them.  A run this
# size is almost always a typo in ``num_words``, so it is refused rather than
# started -- the one place in this file where a request is turned away before a
# tool sees it.
OPENRAM_MAX_BITS = 4 * 1024 * 1024


def _int_or(text: str, fallback: int) -> int:
    """Parse a port value as an int, falling back when it is empty or junk."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return fallback


def _py_list(text: str, *, quote: bool) -> str:
    """
    Render a comma- or space-separated port value as a Python list literal.

    OpenRAM's corner settings are lists (``process_corners = ["TT"]``), and the
    ports carry them as plain text, so this is the one translation between the
    two.  Returns "" for an empty port, which is the signal to omit the
    assignment entirely and let the technology's own default stand.
    """
    tokens = _split_tokens(text)
    if not tokens:
        return ""
    if quote:
        return "[" + ", ".join(json.dumps(t) for t in tokens) + "]"
    return "[" + ", ".join(tokens) + "]"


def openram_output_name(req, tech: str) -> str:
    """
    The macro's module name, and the base name of every file it produces.

    Derived from the geometry when the port is empty so two differently-sized
    macros in one project never collide on disk.
    """
    explicit = (getattr(req, "output_name", "") or "").strip()
    if explicit:
        return _safe_filename(explicit)
    word = _int_or(getattr(req, "word_size", ""), 8)
    words = _int_or(getattr(req, "num_words", ""), 64)
    return _safe_filename("sram_{0}x{1}_{2}".format(word, words, tech))


def openram_size_warnings(req) -> Tuple[List[str], str]:
    """
    Advisory notes about the requested size, and a refusal when it is absurd.

    Returns ``(notes, refusal)``.  A non-empty refusal means do not start the
    run at all.

    This exists because nothing else in the system bounds the cost: the block
    face shows two numbers, and the difference between them is the difference
    between a thirty-second run and a four-hour bill on a rented machine.  The
    user finding that out from an invoice is the failure this prevents.
    """
    word = _int_or(getattr(req, "word_size", ""), 0)
    words = _int_or(getattr(req, "num_words", ""), 0)
    banks = _int_or(getattr(req, "num_banks", ""), 1)
    bits = word * words
    notes: List[str] = []
    if bits <= 0:
        return notes, ""
    if bits > OPENRAM_MAX_BITS:
        return notes, (
            "Refusing to start: {0} x {1} is {2} bits ({3:.1f} Mbit), past the "
            "{4:.0f} Mbit ceiling this block will attempt. A macro that size "
            "compiles for hours on a pod that bills the whole time, and is "
            "almost always a typo in num_words. Reduce num_words, or split the "
            "memory across several blocks."
        ).format(word, words, bits, bits / 1024 / 1024,
                 OPENRAM_MAX_BITS / 1024 / 1024)
    if bits > OPENRAM_WARN_BITS:
        notes.append(
            "This is a {0:.0f} kbit macro ({1} x {2}); expect a long compile. "
            "More banks (num_banks is {3}) shortens the bitlines and usually "
            "helps both the run time and the aspect ratio."
            .format(bits / 1024, word, words, banks)
        )
    if words > 0 and (words & (words - 1)) != 0:
        notes.append(
            "num_words {0} is not a power of two; OpenRAM rounds the array up, "
            "so the macro is bigger than the word count suggests.".format(words)
        )
    return notes, ""


def build_openram_config(req, *, tech: str, output_name: str, out_dir: str,
                         gain_cell: bool = False) -> str:
    """
    Build the OpenRAM configuration file for a run.

    OpenRAM takes no command-line parameters worth the name: every knob is a
    module-level assignment in a Python file it executes.  So this IS the
    interface, and it is a pure string builder for exactly that reason -- the
    image's CI smoke test compiles the output of THIS function, so an option
    OpenRAM no longer accepts fails the image build rather than a user's pod.

    An empty port omits its assignment rather than writing a guess, so the
    technology's own default stands.  The exceptions are the three settings
    below, which are written on every run because leaving them to the default
    would fail late and expensively:

    * ``check_lvsdrc`` -- DRC and LVS need magic and netgen, which this image
      does not carry.  Off unless the block asked, in which case the run fails
      with a message naming the missing tools instead of silently skipping them.
    * ``analytical_delay`` -- the alternative is a SPICE simulation, and the
      image ships no simulator.  With it off, characterization dies at the very
      end, after all the layout work is already paid for.
    * ``nominal_corner_only`` -- one corner means one ``.lib``, and the ``lib``
      output port holds one filename.  Only forced when the block did not ask
      for corners of its own.

    ``gain_cell=True`` builds the config for OpenGCRAM, the gain-cell fork, which
    reads the same variables plus ``gc_type`` and ``vddio``.  Two of its defaults
    differ from OpenRAM's and would sink a run, so they are written explicitly:
    ``analytical_delay`` is False upstream (HSpice) and ``use_pex`` is True
    (Calibre extraction).  Its port counts default to one read + one write --
    see GAIN_CELL_PORT_DEFAULTS.  ``extra_config`` is appended last, so a user
    with the tools can still turn either back on.

    ``output_path`` and ``output_name`` are forced even when the block supplied
    a whole ``config``: the server has to know where to read the results back
    from, and a user config pointing elsewhere produces a run whose outputs are
    invisible.  Note OpenRAM concatenates the two with no separator, so the path
    MUST end in a slash.
    """
    out = out_dir if out_dir.endswith("/") else out_dir + "/"
    forced = [
        "# --- forced by Grafux: the server reads the results back from here ---",
        "output_path = {0}".format(json.dumps(out)),
        "output_name = {0}".format(json.dumps(output_name)),
    ]

    override = (getattr(req, "config", "") or "").strip()
    if override:
        # The `config` port WINS, entirely.  A half-merge -- our parameters plus
        # the user's file -- is the worst of both: the block face shows
        # word_size 8 while the macro comes out 32 wide, and nothing on screen
        # says which won.
        return override.rstrip() + "\n\n" + "\n".join(forced) + "\n"

    corners = _py_list(getattr(req, "process_corners", ""), quote=True)
    volts = _py_list(getattr(req, "supply_voltages", ""), quote=False)
    temps = _py_list(getattr(req, "temperatures", ""), quote=False)
    drc = (getattr(req, "check_lvsdrc", "") or "").strip() in ("1", "true", "yes", "on")
    netlist_only = (getattr(req, "netlist_only", "") or "").strip() in ("1", "true", "yes", "on")

    lines = ["# Generated by Grafux for the {0} block. Do not edit in the pod.".format(
        "opengcram" if gain_cell else "openram")]
    for field in ("word_size", "num_words", "num_banks", "num_rw_ports",
                  "num_r_ports", "num_w_ports", "write_size"):
        raw = (getattr(req, field, "") or "").strip()
        if raw:
            lines.append("{0} = {1}".format(field, _int_or(raw, 0)))
        elif gain_cell and field in GAIN_CELL_PORT_DEFAULTS:
            lines.append("{0} = {1}".format(field, GAIN_CELL_PORT_DEFAULTS[field]))
    lines.append("tech_name = {0}".format(json.dumps(tech)))
    if gain_cell:
        gc_type, _ = normalize_gc_type(getattr(req, "gc_type", ""))
        lines.append("gc_type = {0}".format(json.dumps(gc_type or "OS")))
        vddio = (getattr(req, "vddio", "") or "").strip()
        if vddio:
            try:
                lines.append("vddio = {0}".format(float(vddio)))
            except ValueError:
                pass
        lines.append("use_pex = False   # upstream default True needs Calibre extraction")
    if corners:
        lines.append("process_corners = {0}".format(corners))
    if volts:
        lines.append("supply_voltages = {0}".format(volts))
    if temps:
        lines.append("temperatures = {0}".format(temps))
    if not corners and not volts and not temps:
        # One corner means one .lib, and the `lib` port holds one filename.
        lines.append("nominal_corner_only = True")
    lines.append("analytical_delay = True   # no SPICE simulator in this image")
    lines.append("check_lvsdrc = {0}".format(bool(drc)))
    if netlist_only:
        lines.append("netlist_only = True   # no layout: no GDS and no LEF")
    lines.extend(forced)

    extra = (getattr(req, "extra_config", "") or "").strip()
    if extra:
        lines.append("")
        lines.append("# --- extra_config port ---")
        lines.append(extra)
    return "\n".join(lines) + "\n"


# Extension -> output port.  This is the ONLY place a generated file is matched,
# and it is matched by EXTENSION rather than by a predicted filename on purpose:
# OpenRAM encodes the process corner into the Liberty name
# (``sram_8x64_TT_1p8V_25C.lib``) and its naming has moved between releases, so
# any filename spelled out here would be a guess that fails on a rented machine.
# Same reasoning as ``pick_liberty`` above.
_OPENRAM_EXT_PORTS = (
    (".gds", "gds"), (".gds.gz", "gds"),
    (".lef", "lef"),
    (".lib", "lib"),
    (".sp", "spice"), (".spice", "spice"),
    (".v", "verilog_model"),
    (".html", "datasheet"), (".htm", "datasheet"),
    (".py", "config"),
    (".log", "log_file"),
)


def classify_openram_outputs(names: Sequence[str],
                             output_name: str = "") -> Dict[str, str]:
    """
    Map an output directory listing onto the block's ports, by extension.

    Three passes per extension, in this order, and the order is the whole point:

    1. the file named exactly ``<output_name><ext>``;
    2. any file whose name starts with ``<output_name>``;
    3. the first remaining match in ``ls -S`` (largest-first) order.

    Pass 1 exists because a real run's output directory is NOT one file per
    extension.  A 2x16 scn4m_subm macro writes EIGHT ``.sp`` files -- the SRAM
    netlist plus ``functional_stim.sp``, ``sram.sp``, ``trimmed.sp``,
    ``delay_meas.sp`` and friends -- and the stimulus files are LARGER than the
    netlist, so size alone picks a test stimulus for the ``spice`` port.  This
    was found by the image's CI smoke test against a real compile, which is
    exactly what that step is for; the listing is captured in
    tests/fixtures/eda/openram_listing.txt so it cannot regress.

    Pass 2 catches the Liberty file, which is the opposite case: OpenRAM encodes
    the corner into its name (``<output_name>_TT_5p0V_25C.lib``), so an exact
    match never hits and a prefix match is the most specific rule available.

    Pass 3 is the fallback for anything OpenRAM did not name after the macro.
    It keeps ``ls -S`` order load-bearing, so the caller must not sort the
    listing itself.

    Unknown extensions are simply absent from the result; they still reach the
    user through the artifact download.
    """
    cleaned: List[str] = []
    for raw in names:
        name = (raw or "").strip()
        if name and not name.endswith("/"):
            cleaned.append(name)

    stem = (output_name or "").strip()
    found: Dict[str, str] = {}
    for ext, port in _OPENRAM_EXT_PORTS:
        if port in found:
            continue
        matches = [n for n in cleaned if n.lower().endswith(ext)]
        if not matches:
            continue
        exact = [n for n in matches if stem and n == stem + ext]
        prefixed = [n for n in matches if stem and n.startswith(stem)]
        found[port] = (exact or prefixed or matches)[0]
    return found


_OPENRAM_AREA_RE = re.compile(
    r"(?:total\s+)?area[^0-9\n]*?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_OPENRAM_WIDTH_RE = re.compile(
    r"\bwidth[^0-9\n]*?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_OPENRAM_HEIGHT_RE = re.compile(
    r"\bheight[^0-9\n]*?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_openram_summary(req, *, tech: str, output_name: str,
                          log_text: str, found: Dict[str, str]) -> Dict[str, Any]:
    """
    The `stats` port: what was built, as JSON.

    Deliberately tolerant -- anything it cannot find is simply absent from the
    dict rather than reported as zero, because a zero area reads as a broken
    macro.  It must never raise: the compile has already succeeded by the time
    this runs, and a summary that throws would turn a good run red.
    """
    word = _int_or(getattr(req, "word_size", ""), 0)
    words = _int_or(getattr(req, "num_words", ""), 0)
    summary: Dict[str, Any] = {
        "top": output_name,
        "tech_name": tech,
        "views": sorted(k for k in found if k not in ("config", "log_file")),
    }
    if word:
        summary["word_size"] = word
    if words:
        summary["num_words"] = words
    if word and words:
        summary["total_bits"] = word * words
    for field in ("num_banks", "num_rw_ports", "num_r_ports", "num_w_ports",
                  "write_size"):
        raw = (getattr(req, field, "") or "").strip()
        if raw:
            summary[field] = _int_or(raw, 0)
    corners = _split_tokens(getattr(req, "process_corners", ""))
    if corners:
        summary["process_corners"] = corners
    text = log_text or ""
    for key, pattern in (("area_um2", _OPENRAM_AREA_RE),
                         ("width_um", _OPENRAM_WIDTH_RE),
                         ("height_um", _OPENRAM_HEIGHT_RE)):
        match = pattern.search(text)
        if match:
            try:
                summary[key] = float(match.group(1))
            except ValueError:
                pass
    return summary


# ---------------------------------------------------------------------------
# OpenGCRAM (gain-cell memory compiler)
# ---------------------------------------------------------------------------
#
# A fork of OpenRAM, so the config builder, the output classifier and the summary
# above are shared.  What is here is only what a gain-cell run adds -- and the
# largest part of that is refusing, early and in words, the runs that upstream
# would fail cryptically.  Facts read from OpenGCRAM @ cbdc35b, not assumed:
#
# * The gain cell's file name depends on gc_type
#   (compiler/drc/custom_cell_properties.py): OS -> os_gc, Si -> si_gc,
#   hybrid -> hybrid_gc.  Any other value leaves the name unset and the compiler
#   dies with a KeyError.
# * Only ``gain_cell_2port`` exists (compiler/modules/); setup_gain_cell imports
#   ``gain_cell_{N}port`` OUTSIDE its try block, so a port total other than 2 is
#   an ImportError, not the parameterized fallback its comment promises.
# * No technology in the public repository contains one: tsmcN40 is an empty
#   NDA placeholder and freepdk45 has only the 6T SRAM cells.

# Where an uploaded technology is unpacked, relative to WORK_DIR.  Prepended to
# OPENRAM_TECH by _OPENGCRAM_ENV, so it wins over anything the image carries.
OPENGCRAM_TECH_DIR = "gcram_tech"

GC_TYPE_CELLS = {"OS": "os_gc", "Si": "si_gc", "hybrid": "hybrid_gc"}

# OpenGCRAM's only cell is a 2-port one: a read port and a write port.  Upstream's
# own defaults (1 rw, 0 r, 0 w) add up to ONE port and cannot compile, so a blank
# port means these values, not the compiler's.
GAIN_CELL_PORT_DEFAULTS = {"num_rw_ports": 0, "num_r_ports": 1, "num_w_ports": 1}

_TECH_ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tar", ".zip")


def normalize_gc_type(text: str) -> Tuple[str, str]:
    """
    ``(canonical, error)`` for the gc_type port.  Empty means OS, upstream's
    default.  Case-insensitive, because the compiler compares exact strings and
    "os" would otherwise silently name no cell at all.
    """
    raw = (text or "").strip()
    if not raw:
        return "OS", ""
    for canonical in GC_TYPE_CELLS:
        if raw.lower() == canonical.lower():
            return canonical, ""
    return "", ("gc_type '{0}' is not one OpenGCRAM knows. Use OS (oxide "
                "semiconductor), Si or hybrid.".format(raw))


def gain_cell_ports(req) -> Tuple[int, int, int]:
    """``(rw, r, w)`` with GAIN_CELL_PORT_DEFAULTS filling every blank port."""
    values = []
    for field in ("num_rw_ports", "num_r_ports", "num_w_ports"):
        raw = (getattr(req, field, "") or "").strip()
        values.append(_int_or(raw, GAIN_CELL_PORT_DEFAULTS[field]) if raw
                      else GAIN_CELL_PORT_DEFAULTS[field])
    return values[0], values[1], values[2]


def opengcram_output_name(req, tech: str) -> str:
    """The macro's module name; derived from geometry, cell flavour and tech when blank."""
    explicit = (getattr(req, "output_name", "") or "").strip()
    if explicit:
        return _safe_filename(explicit)
    word = _int_or(getattr(req, "word_size", ""), 8)
    words = _int_or(getattr(req, "num_words", ""), 32)
    gc_type, _ = normalize_gc_type(getattr(req, "gc_type", ""))
    return _safe_filename("gcram_{0}x{1}_{2}_{3}".format(
        word, words, (gc_type or "OS").lower(), tech))


def opengcram_preflight(req) -> Tuple[List[str], str]:
    """
    ``(notes, refusal)`` from the request alone -- no pod needed.

    Size limits are OpenRAM's (same array generator); on top of them the two
    gain-cell rules that upstream turns into tracebacks.  A block with its own
    ``config`` owns those choices, so only the size check applies to it.
    """
    notes, refusal = openram_size_warnings(req)
    if refusal:
        return notes, refusal
    if (getattr(req, "config", "") or "").strip():
        return notes, ""
    _gc, gc_error = normalize_gc_type(getattr(req, "gc_type", ""))
    if gc_error:
        return notes, gc_error
    rw, r, w = gain_cell_ports(req)
    if rw + r + w != 2 or w < 1 or (r + rw) < 1:
        return notes, (
            "OpenGCRAM ships only a two-port gain cell, so the ports must be one "
            "read and one write (num_rw_ports 0, num_r_ports 1, num_w_ports 1); "
            "this block asks for {0} rw + {1} r + {2} w.".format(rw, r, w))
    vddio = (getattr(req, "vddio", "") or "").strip()
    if vddio:
        try:
            float(vddio)
        except ValueError:
            return notes, "vddio '{0}' is not a voltage; use a number such as 1.2.".format(vddio)
    return notes, ""


def archive_stem(path: str) -> str:
    """``/w/tsmcN40.tar.gz`` -> ``tsmcN40``."""
    base = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    lower = base.lower()
    for ext in _TECH_ARCHIVE_EXTS:
        if lower.endswith(ext):
            return base[: -len(ext)]
    return base


def find_staged_archive(req) -> Tuple[str, str]:
    """
    ``(pod_path, error)`` of the technology archive among the staged input files.

    The block uploads the archive through ``input_files`` like any other file and
    names it on ``tech_archive``; the port holds the CLIENT's path, so only its
    base name is comparable.  ``("", "")`` when the block supplied no archive.
    """
    wanted = (getattr(req, "tech_archive", "") or "").strip()
    if not wanted:
        return "", ""
    base = wanted.replace("\\", "/").rsplit("/", 1)[-1]
    if not base.lower().endswith(_TECH_ARCHIVE_EXTS):
        return "", ("tech_archive '{0}' is not an archive this block can unpack. "
                    "Use .tar.gz, .tgz, .tar or .zip.".format(base))
    for item in getattr(req, "input_files", None) or []:
        if not isinstance(item, dict):
            continue
        path = (item.get("path") or "").strip()
        if path and path.rsplit("/", 1)[-1] == base:
            return path, ""
    return "", ("tech_archive names {0}, but no such file reached the pod. The "
                "block could not read it -- check the path exists on this "
                "machine (a browser session cannot read local paths).".format(base))


def resolve_tech_layout(entries: Sequence[str], *, tech_port: str,
                        stem: str) -> Tuple[str, str, str]:
    """
    ``(tech_name, subdir, refusal)`` from the top-level listing of an unpacked archive.

    Two layouts are common and both are accepted: the technology directory itself
    (``tsmcN40/tech/tech.py``), or its CONTENTS (``tech/``, ``gds_lib/`` at the
    root).  ``subdir`` is what to move into place ("" = the root is the tech).
    The name must be a Python identifier because OpenRAM imports it as a module.
    """
    names = [n.strip().rstrip("/") for n in entries if n and n.strip().rstrip("/")]
    tech_port = (tech_port or "").strip()
    if not names:
        return "", "", "The technology archive is empty."
    if any(n in ("__init__.py", "tech", "gds_lib", "sp_lib") for n in names):
        tech, subdir = tech_port or stem, ""
    elif len(names) == 1:
        tech, subdir = tech_port or names[0], names[0]
    elif tech_port and tech_port in names:
        tech, subdir = tech_port, tech_port
    else:
        return "", "", (
            "The technology archive holds several top-level entries ({0}) and "
            "none is named by tech_name. Archive a single technology directory, "
            "or set tech_name to the one to use.".format(", ".join(names[:8])))
    if not tech or not tech.isidentifier():
        return "", "", (
            "'{0}' cannot be a technology name: OpenGCRAM imports the technology "
            "as a Python module, so it must be a valid identifier (letters, "
            "digits, underscores). Set tech_name.".format(tech))
    return tech, subdir, ""


def gain_cell_probe_command(tech: str) -> str:
    """
    List the gain-cell libraries of the first ``tech`` on OPENRAM_TECH.

    Emits ``DIR:<path>``, then ``GDS:<file>``/``SP:<file>`` lines; nothing at all
    when no directory on the search path carries the technology.  Run under
    ``_sh_opengcram`` so the search path is the one the compiler will use.
    """
    q = shlex.quote(tech)
    return (
        'for d in $(printf %s "$OPENRAM_TECH" | tr ":" " "); do '
        'if [ -d "$d"/' + q + ' ]; then echo "DIR:$d/"' + q + '; '
        'ls -1 "$d"/' + q + '/gds_lib 2>/dev/null | sed "s/^/GDS:/"; '
        'ls -1 "$d"/' + q + '/sp_lib 2>/dev/null | sed "s/^/SP:/"; '
        'break; fi; done'
    )


def check_gain_cell_listing(listing: str, *, tech: str, gc_type: str,
                            netlist_only: bool) -> str:
    """
    The refusal for a technology that cannot build this gain cell, or "".

    A netlist-only run needs the SPICE cell; a layout run also needs the GDS.
    When the technology carries a DIFFERENT flavour, the message names it --
    "set gc_type to Si" is a one-edit fix, where "file not found" is a search.
    """
    lines = [ln.strip() for ln in (listing or "").splitlines() if ln.strip()]
    if not any(ln.startswith("DIR:") for ln in lines):
        return (
            "Technology '{0}' is not installed on the pod. OpenGCRAM ships no "
            "technology that contains a gain cell (its reference tsmcN40 is an "
            "empty NDA placeholder), so supply one on the tech_archive port: an "
            "archive of the technology directory whose gds_lib and sp_lib hold "
            "os_gc / si_gc / hybrid_gc.".format(tech))
    gds = {ln[4:] for ln in lines if ln.startswith("GDS:")}
    sp = {ln[3:] for ln in lines if ln.startswith("SP:")}
    cell = GC_TYPE_CELLS.get(gc_type, "")
    missing = []
    if cell + ".sp" not in sp:
        missing.append("sp_lib/{0}.sp".format(cell))
    if not netlist_only and cell + ".gds" not in gds:
        missing.append("gds_lib/{0}.gds".format(cell))
    if not missing:
        return ""
    present = sorted(t for t, c in GC_TYPE_CELLS.items()
                     if t != gc_type and (c + ".sp" in sp or c + ".gds" in gds))
    hint = (" It does carry the {0} cell -- set gc_type to {1}.".format(
        present[0], present[0]) if present else
        " It carries no gain cell of any flavour (freepdk45 in the image is an "
        "SRAM-only technology).")
    return ("Technology '{0}' cannot build a {1} gain-cell memory: missing {2}.{3}"
            .format(tech, gc_type, " and ".join(missing), hint))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# Yosys' ``stat`` output comes in two shapes and BOTH must be handled, because the
# image's yosys version is not ours to choose:
#
#   Legacy (yosys <= ~0.4x), a labelled list:
#       Number of wires:                 12
#       Number of cells:                 24
#         sky130_fd_sc_hd__dfxtp_1        8
#
#   Modern (yosys 0.68, what openroad/orfs ships), a right-aligned table:
#             10        - wires
#             14  152.646 cells
#              4  100.096   sky130_fd_sc_hd__dfrtp_1
#
# Parsing only the legacy shape silently yields zeros everywhere except the chip
# area, which is exactly what the block first reported: a "successful" synthesis
# claiming zero cells.
_STAT_CELLS_LEGACY_RE = re.compile(r"Number of cells:\s+(\d+)")
_STAT_WIRES_LEGACY_RE = re.compile(r"Number of wires:\s+(\d+)")
_STAT_CELL_LINE_LEGACY_RE = re.compile(r"^\s{4,}(\S+)\s+(\d+)\s*$", re.MULTILINE)

_STAT_CELLS_MODERN_RE = re.compile(r"^\s*(\d+)\s+[\d.]+\s+cells\s*$", re.MULTILINE)
_STAT_WIRES_MODERN_RE = re.compile(r"^\s*(\d+)\s+-\s+wires\s*$", re.MULTILINE)
# count, area, cell name — the per-type rows of the modern table.
_STAT_CELL_LINE_MODERN_RE = re.compile(
    r"^\s*(\d+)\s+[\d.]+\s+([A-Za-z_$][\w$]*)\s*$", re.MULTILINE)

_STAT_AREA_RE = re.compile(r"Chip area for(?: module)?[^:]*:\s*([0-9.]+)")
_STAT_SEQ_AREA_RE = re.compile(
    r"used for sequential elements:\s*([0-9.]+)", re.IGNORECASE)

# Words that appear in the cell-name column of a summary row rather than being a
# real cell type.
_STAT_SUMMARY_WORDS = {"cells", "wires", "processes", "memories", "bits", "ports"}

# Standard-cell libraries name flip-flops with a "df" infix (sky130 __dfxtp_/
# __dfrtp_, nangate DFF_X1, asap7 DFF...), which is the only portable way to count
# state across PDKs.
_SEQ_CELL_RE = re.compile(r"(?:^|_)d?ff|dlxtp|dfxtp|dfrtp|_df", re.IGNORECASE)


def parse_yosys_stats(text: str) -> Dict[str, Any]:
    """
    Extract a structured summary from yosys ``stat`` output.

    Returns {cell_count, wire_count, area_um2, sequential_cells,
    sequential_area_um2, by_cell_type}.  Every field is best-effort: the format
    varies with yosys version and flags, and a missing number should degrade the
    report rather than fail a synthesis that actually succeeded.
    """
    text = text or ""
    out: Dict[str, Any] = {
        "cell_count": 0,
        "wire_count": 0,
        "area_um2": 0.0,
        "sequential_cells": 0,
        "sequential_area_um2": 0.0,
        "by_cell_type": {},
    }

    def _first_int(*matches) -> int:
        for m in matches:
            if m:
                return int(m.group(1))
        return 0

    out["cell_count"] = _first_int(_STAT_CELLS_MODERN_RE.search(text),
                                   _STAT_CELLS_LEGACY_RE.search(text))
    out["wire_count"] = _first_int(_STAT_WIRES_MODERN_RE.search(text),
                                   _STAT_WIRES_LEGACY_RE.search(text))

    for regex, key in ((_STAT_AREA_RE, "area_um2"),
                       (_STAT_SEQ_AREA_RE, "sequential_area_um2")):
        m = regex.search(text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass

    by_type: Dict[str, int] = {}
    # Modern table first; fall back to the legacy shape only if it found nothing,
    # so a log containing both never double-counts.
    pairs = [(name, int(count))
             for count, name in _STAT_CELL_LINE_MODERN_RE.findall(text)]
    if not pairs:
        pairs = [(name, int(count))
                 for name, count in _STAT_CELL_LINE_LEGACY_RE.findall(text)
                 if not name.lower().startswith("number")]

    seq = 0
    for name, n in pairs:
        if name.lower() in _STAT_SUMMARY_WORDS:
            continue
        by_type[name] = by_type.get(name, 0) + n
        if _SEQ_CELL_RE.search(name):
            seq += n
    out["by_cell_type"] = by_type
    out["sequential_cells"] = seq
    # A modern log without a "N ... cells" summary line still yields a total.
    if not out["cell_count"] and by_type:
        out["cell_count"] = sum(by_type.values())
    return out


# ORFS metrics keys vary across releases, so each logical metric lists the source
# keys it accepts, most-preferred first.
_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "wns_ns": ("finish__timing__setup__ws", "route__timing__setup__ws", "timing__setup__ws"),
    "tns_ns": ("finish__timing__setup__tns", "route__timing__setup__tns", "timing__setup__tns"),
    # Hold slack matters as much as setup at signoff: a design can meet its clock
    # target and still be unmanufacturable on hold.
    "hold_wns_ns": ("finish__timing__hold__ws", "route__timing__hold__ws"),
    "hold_tns_ns": ("finish__timing__hold__tns", "route__timing__hold__tns"),
    "sequential_cells": ("finish__design__instance__count__class:sequential_cell",),
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


# ORFS writes each stage's real diagnostics to logs/<platform>/<design>/base/,
# NOT to make's stdout -- make only reports "Error 2". Without reading the log a
# failed run tells the user nothing they can act on.
_ORFS_ERROR_RE = re.compile(r"^\s*(?:\[ERROR|Error:|ERROR:).*$", re.MULTILINE)

# Failures common enough, and cryptic enough, to be worth translating into the
# port the user should actually change.
_ORFS_HINTS = (
    ("PDN-0185",
     "The die is too small for this PDK's power grid — usually because the design "
     "is tiny, so a handful of cells gives a core narrower than the power straps "
     "need. Lower 'core_utilization' (which makes the die BIGGER), or set an "
     "explicit 'die_area'/'core_area' such as '0 0 60 60'."),
    # An empty netlist is silent everywhere until routing, where the first tool
    # to actually object names a command that DID run. These three cover a run
    # started past the synth stage, where the gate below cannot help.
    ("EST-0005",
     "Global routing produced no result because the design has no nets — "
     "synthesis mapped it to an empty netlist. Check that every output of the "
     "top module is driven, and that 'top' names the module you meant to build."),
    ("GRT-0094",
     "There were no nets to route: synthesis produced an empty netlist. Check "
     "that every output of the top module is driven, and that 'top' names the "
     "module you meant to build."),
    ("CTS-0083",
     "No clock net was found: 'clock_port' matches no port of the design, or the "
     "design has no registers for a clock to reach."),
    ("no space to place",
     "Placement ran out of room. Lower 'core_utilization' to enlarge the die."),
    ("Unable to find a site",
     "The design has no placeable cells — check that synthesis produced a netlist "
     "and that 'top' names a real module."),
)


def explain_orfs_failure(log_text: str) -> str:
    """
    Turn an ORFS stage log into something a user can act on.

    Returns the tool's own ERROR lines plus, where the failure is recognised, a
    sentence naming the port to change. Best-effort: an unrecognised failure still
    surfaces the raw error lines, which beats make's bare exit code.
    """
    errors = _ORFS_ERROR_RE.findall(log_text or "")
    parts: List[str] = []
    if errors:
        parts.append("\n".join(e.strip() for e in errors[-6:]))
    # The tool's own ERROR lines are searched first, so the hint describes what
    # actually failed rather than something incidental further up. Only when none
    # of them is recognised does the whole log get a look: some causes are only
    # ever reported as a WARNING, by a stage that shrugged and carried on several
    # stages before the crash.
    for haystack in (" ".join(errors).lower(), (log_text or "").lower()):
        if not haystack:
            continue
        hint = next((text for marker, text in _ORFS_HINTS
                     if marker.lower() in haystack), "")
        if hint:
            parts.append(hint)
            break
    return "\n".join(parts).strip()


# ORFS reports the synthesized size twice in the `synth` stage's output. Either
# one reading zero means yosys mapped the design to nothing at all.
_EMPTY_AREA_RE = re.compile(r"Design area\s+0(?:\.0+)?\s*um\^2")
_EMPTY_INSTANCES_RE = re.compile(r"number instances in verilog is 0\b")

# The lines that usually explain WHY it came out empty. Quoted back verbatim so
# the user reads the tool's own words rather than our paraphrase of them.
#
# Deliberately absent: "Ignoring module ... because it contains processes (run
# 'proc' command first)". Yosys prints that during the normal
# 1_1_yosys_canonicalize step of every healthy run, and repeating it here as
# though it were a diagnosis would send people chasing a non-problem.
_EMPTY_EVIDENCE_RES = (
    re.compile(r"^.*\bis used but has no driver\b.*$", re.MULTILINE),
    re.compile(r"^.*\[WARNING STA-0366\].*$", re.MULTILINE),
    re.compile(r"^.*\bWire .* is unused\b.*$", re.MULTILINE),
)


def synth_produced_nothing(stage_log: str, top: str = "") -> str:
    """
    Explain an empty post-synthesis netlist; "" when synthesis produced cells.

    An empty netlist is a FAILED synthesis even though yosys and ``make synth``
    both exit 0. Every later stage then quietly no-ops on it — the floorplan
    holds nothing, GPL reports "no placeable instances" and skips placement, CTS
    finds no clock nets — and the first tool rude enough to object is
    ``estimate_parasitics -global_routing``, four stages later, with "[ERROR
    EST-0005] Run global_route before estimating parasitics". That names a command
    which DID run, points at the wrong tool, and says nothing whatsoever about the
    undriven output that caused it.

    Checking here costs one regex against a log we already have in hand, and turns
    a five-stage pod run into a twenty-second answer.
    """
    text = stage_log or ""
    if not (_EMPTY_AREA_RE.search(text) or _EMPTY_INSTANCES_RE.search(text)):
        return ""

    evidence: List[str] = []
    for pattern in _EMPTY_EVIDENCE_RES:
        for line in pattern.findall(text):
            line = line.strip()
            if line and line not in evidence:
                evidence.append(line)

    module = (top or "").strip() or "the top module"
    parts = [
        "Synthesis produced an EMPTY netlist — 0 cells, 0 nets — so there is "
        "nothing to floorplan, place or route. Later stages would no-op on it "
        "and the route stage would fail with [ERROR EST-0005], because global "
        "routing had no nets to route.",
    ]
    if evidence:
        parts.append("The tools reported:\n"
                     + "\n".join(f"  {line}" for line in evidence[:10]))
    parts.append(
        "Usual causes, most likely first:\n"
        f"  - An output of module '{module}' is never assigned, so all of its "
        "logic is dead and opt_clean deletes it. Drive every output port.\n"
        f"  - The 'top' port names the wrong module (it is '{module}' here).\n"
        f"  - Module '{module}' declares no output ports, so nothing it computes "
        "is observable from outside."
    )
    parts.append(
        "Wire this RTL into a verilator block (mode=lint) or a yosys block first "
        "— both catch this in seconds, without renting a pod."
    )
    return "\n\n".join(parts)


# ── Locating and explaining a failure ───────────────────────────────────────
#
# Everything from here to `summarize_failures` answers the two questions a red
# verilator block used to leave unanswered: WHY did this test fail, and WHERE is
# the error.  The evidence was always present and was simply being discarded —
# cocotb writes the assertion's traceback into the BODY of the <failure> element
# while the parser kept only the one-line `message` attribute, and Verilator
# prints `%Error-CODE: file:line:col:` which nothing ever read.
#
# The shape deliberately mirrors _ORFS_HINTS / explain_orfs_failure above: tuple
# rule tables, searched most-specific-field first, returning "" when nothing
# matches.  "" is never a failure of this code — an unrecognised failure still
# shows its assertion, its location and its quoted source line, which is already
# far more than the collapsed one-liner it replaces.


def quote_source_line(source: str, line: int, *, col: int = 0,
                      context: int = 1) -> str:
    """
    The offending source line, numbered, with the lines before it and a caret.

    The point of this function is that it costs NOTHING.  ``req.rtl``,
    ``req.testbench`` and ``req.sva`` are already in hand server-side — they were
    written into the pod moments earlier — so pointing at ``counter.v:12:5`` and
    showing the line needs no extra round trip to the machine.

    Returns "" for a line number outside the file rather than guessing.  A
    diagnostic that quotes the WRONG line is worse than one that quotes none: it
    sends the reader to a line that is fine and makes them doubt the tool.
    """
    if not source or line <= 0:
        return ""
    lines = source.splitlines()
    if line > len(lines):
        return ""
    first = max(1, line - max(0, context))
    width = len(str(line))
    out: List[str] = []
    for number in range(first, line + 1):
        out.append(f"  {str(number).rjust(width)} | {lines[number - 1].rstrip()}")
    if col > 0:
        # 1-based, counted in characters — Verilator's own convention.
        out.append("  " + " " * width + " | " + " " * (col - 1) + "^")
    return "\n".join(out)


# A Python traceback frame, the exception line that closes a traceback, and the
# assertion source cocotb echoes into the failure body.
_PY_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)\s*$',
    re.MULTILINE)
_PY_EXC_RE = re.compile(
    r'^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Failure))'
    r'(?::\s*(?P<msg>.*))?$', re.MULTILINE)
_ASSERT_SRC_RE = re.compile(r'^\s*(assert\b.*)$', re.MULTILINE)

# cocotb's reduced log format puts the simulation time first on every line.
# Anchored at line start so a stray "10ns" inside an assertion message is never
# mistaken for the time the test died at.
_COCOTB_LOG_TIME_RE = re.compile(
    r'^\s*(?P<t>\d+(?:\.\d+)?)\s*(?P<u>[fpnum]?s)\s+\w+', re.MULTILINE)

_ATTR_NAME_RE = re.compile(r"has no attribute '(?P<sig>[^']+)'")
_EXPECTED_GOT_RE = re.compile(
    r'expected[\s:=]+(?P<exp>.+?)[,;]?\s+(?:but\s+)?(?:got|was|actual)[\s:=]+(?P<act>\S+)',
    re.IGNORECASE)
_GOT_EXPECTED_RE = re.compile(
    r'got[\s:=]+(?P<act>\S+).{0,60}?expected[\s:=]+(?P<exp>\S+)',
    re.IGNORECASE | re.DOTALL)
_ASSERT_EQ_RE = re.compile(
    r'assert\s+(?P<act>.+?)\s*==\s*(?P<exp>[^,]+?)\s*(?:,|$)')

# Grafux's own generated runner.  A traceback that stops here is not the user's
# bug and must never be reported as though it were their testbench.
_RUNNER_FILENAME = "run_cocotb.py"


def locate_python_failure(detail: str, *, prefer: str = "") -> Dict[str, Any]:
    """
    The frame in a cocotb traceback that the USER can act on.

    ``prefer`` is the whole point of this function.  The DEEPEST frame in a
    cocotb traceback is almost always inside cocotb's own scheduler or somewhere
    under site-packages, and pointing a user there is worse than pointing
    nowhere — it reads as "the bug is in cocotb".  The actionable frame is the
    LAST one in their own test file, so callers pass ``prefer="test_fifo.py"``
    and the deepest frame is used only when the traceback never enters it.

    Returns {} when there is no traceback at all.  Never raises: this runs over
    text a user's testbench produced.
    """
    frames = [
        {"file": os.path.basename(m.group("file")),
         "path": m.group("file"),
         "line": int(m.group("line")),
         "func": m.group("func")}
        for m in _PY_FRAME_RE.finditer(detail or "")
    ]
    if not frames:
        return {}
    chosen = frames[-1]
    wanted = os.path.basename((prefer or "").strip())
    if wanted:
        owned = [f for f in frames if f["file"] == wanted]
        if owned:
            chosen = owned[-1]
    out = dict(chosen)
    out["frames"] = frames
    return out


def _exception_line(text: str) -> Tuple[str, str]:
    """The (type, message) of the LAST exception named in a traceback."""
    matches = list(_PY_EXC_RE.finditer(text or ""))
    if not matches:
        return "", ""
    last = matches[-1]
    return last.group("type") or "", (last.group("msg") or "").strip()


def _first_line(text: str, limit: int = 200) -> str:
    """One line, for a field a UI paints on a single row."""
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    line = " ".join(stripped.splitlines()[0].split())
    return line[:limit].rstrip() if len(line) > limit else line


def extract_failure_facts(test: Dict[str, Any], *, testbench: str = "",
                          test_file: str = "",
                          log_slice: str = "") -> Dict[str, Any]:
    """
    Turn one parsed testcase into the facts its failure block is printed from.

    Everything here is best-effort and ADDITIVE: a field that could not be
    extracted is simply absent and the caller prints what it has.  That is what
    lets the format degrade to "name + message" — exactly what the old one-liner
    gave — on a failure shape nothing here recognises, instead of degrading to an
    exception inside the run.
    """
    facts: Dict[str, Any] = {}
    detail = str(test.get("detail") or "")
    message = str(test.get("message") or "")
    ftype = str(test.get("type") or "")

    where = locate_python_failure(detail, prefer=test_file)
    if where:
        quoted = ""
        if test_file and where.get("file") == os.path.basename(test_file):
            quoted = quote_source_line(testbench, int(where.get("line") or 0))
        facts["where"] = {
            "file": where.get("file", ""),
            "line": int(where.get("line") or 0),
            "func": where.get("func", ""),
            "text": quoted,
        }

    # The simulation time the test died at, read off the LAST log line of its own
    # slice.  Wall-clock `time` is already on the testcase; sim time is the one a
    # hardware engineer reasons with.
    if log_slice:
        times = _COCOTB_LOG_TIME_RE.findall(log_slice)
        if times:
            facts["sim_time"] = f"{times[-1][0]} {times[-1][1]}"

    if not ftype:
        ftype = _exception_line(detail)[0]
    if ftype:
        facts["type"] = ftype

    # The assertion: the quoted source line when it is one, else the last
    # `assert ...` echoed into the traceback body.
    assertion = ""
    quoted_text = (facts.get("where") or {}).get("text") or ""
    for row in quoted_text.splitlines():
        body = row.split("|", 1)[-1].strip()
        if body.startswith("assert "):
            assertion = body
    if not assertion:
        found = _ASSERT_SRC_RE.findall(detail)
        if found:
            assertion = " ".join(found[-1].split())
    if assertion:
        facts["assertion"] = assertion

    blob = f"{message}\n{detail}"
    for pattern in (_EXPECTED_GOT_RE, _GOT_EXPECTED_RE):
        match = pattern.search(blob)
        if match:
            facts["expected"] = match.group("exp").strip().strip(".,")
            facts["actual"] = match.group("act").strip().strip(".,")
            break
    else:
        match = _ASSERT_EQ_RE.search(assertion)
        if match:
            # `assert dut.q.value == 4` states the EXPECTATION.  The actual value
            # is only known if the message carried it, so it is left ABSENT
            # rather than invented — a wrong "got" is worse than no "got".
            facts["expected"] = match.group("exp").strip()

    why = _first_line(message) or _first_line(detail)
    if ftype and not why.startswith(ftype):
        why = f"{ftype}: {why}" if why else ftype
    if facts.get("expected") and facts.get("actual"):
        why = f"{why} (expected {facts['expected']}, got {facts['actual']})"
    if why:
        facts["why"] = why[:240].rstrip()
    return facts


# ── Why a cocotb test failed ────────────────────────────────────────────────
#
# (id, markers, template).  The id is what the tests assert on, so the prose can
# be improved without touching a single test.  Markers are lowercase substrings;
# any one matching is enough.  ORDER IS SIGNIFICANT — most specific first, and
# `assert_mismatch` must stay last of the family or it swallows the rules above
# it, since cocotb reports nearly every failure as an AssertionError.
_COCOTB_HINTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("no_such_signal", ("has no attribute",),
     "The testbench referenced `dut.{signal}`, which the elaborated design does "
     "not have.{suggest} cocotb resolves names against the design hierarchy, so "
     "this is a NAME mismatch rather than a design bug: check the spelling "
     "against the module's port list{ports}, and remember that a signal declared "
     "inside a generate block or an unnamed always block is not reachable as "
     "`dut.<name>`."),
    ("xz_in_comparison",
     ("is not resolvable", "unresolvable", "contains 'x'", "contains x",
      "contains 'z'", "non-resolvable"),
     "A signal in the comparison still holds X or Z, so it could not be resolved "
     "to a number. Either it is never driven, or the test sampled it before "
     "reset finished. Drive it from reset, await a clock edge after deasserting "
     "reset, and where partially-unknown bits are legitimate at that point, test "
     "`.value.is_resolvable` or compare `.value.binstr` instead."),
    ("unresolvable_binary",
     ("invalid literal for int", "cannot convert", "non-numeric",
      "could not be converted"),
     "The test called `int(...)` or read `.integer` on a value that still "
     "contains X or Z, which raises instead of returning a number. Same root "
     "cause as an unresolved comparison: drive the signal, or read "
     "`.value.binstr` and assert on the string when unknown bits are expected."),
    ("sim_timeout",
     ("simtimeouterror", "timeouterror", "timed out", "timeout occurred"),
     "The test waited for something that never happened. Usual causes, most "
     "likely first:\n"
     "  - The clock was never started — `cocotb.start_soon(Clock(dut.{clk}, 10, "
     "units='ns').start())` must run before the first `await RisingEdge`.\n"
     "  - The design never asserts the signal the test awaits, so the handshake "
     "never completes. Check the logic that drives it.\n"
     "  - The timeout is simply shorter than the design's real latency."),
    ("assert_mismatch", ("assertionerror",),
     "{verdict}{at_time}. Check the logic driving the signal named in the "
     "assertion. If the value looks one cycle late or early, the test is "
     "probably sampling on the same delta as the write -- `await "
     "RisingEdge(clk)` then `await ReadOnly()` before reading, or sample on the "
     "opposite edge."),
)

# Run-level problems: not a property of any one test, so they are keyed by id and
# looked up directly rather than matched.  `libpython_missing` and `build_failed`
# were inline in run_cocotb before this table existed; their wording is preserved
# VERBATIM because existing tests match on it.
_RUN_HINTS: Dict[str, str] = {
    "libpython_missing":
        "The design built, but the simulator could not start: cocotb embeds "
        "CPython in the simulator and needs the shared libpython, which is "
        "missing from this pod's image. This is an image problem, not a "
        "problem with the design or the testbench. Fix it by pinning a "
        "verify image built after the libpython fix (EDA/models.py "
        "DEFAULT_VERIFY_IMAGE), or, on a pod that is already up, by running "
        "`apt-get install -y libpython3.10`.",
    "build_failed":
        "The design did not build, so no test ran. The build log is above.",
    "build_failed_sva":
        "  The `sva` port was wired in and is compiled alongside the "
        "design — unbind it to rule the assertions out as the cause.",
    "no_tests":
        "cocotb collected no tests from `test_{stem}.py`. Check that at least one "
        "function is decorated `@cocotb.test()` — WITH the parentheses, since the "
        "bare decorator collects nothing — and that the file imports cleanly: an "
        "ImportError at module scope makes the module contribute zero tests "
        "without failing the run.",
    "test_raised_exception":
        "{type} was raised inside {file}, not by an assertion. This is a bug in "
        "the TESTBENCH, not necessarily in the design: fix the test first and "
        "re-run before drawing any conclusion about the RTL.",
}


def _clock_name(ports: Sequence[str]) -> str:
    """The port most likely to be the clock, for use in an example line."""
    for candidate in ("clk", "clock", "i_clk", "clk_i"):
        if candidate in ports:
            return candidate
    for port in ports:
        if "clk" in port.lower() or "clock" in port.lower():
            return port
    return "clk"


def explain_cocotb_failure(test: Dict[str, Any], *, top: str = "",
                           ports: Sequence[str] = (), stem: str = "",
                           facts: Optional[Dict[str, Any]] = None
                           ) -> Tuple[str, str]:
    """
    (text, id): the plain-language cause of one failing test and what to do.

    Searched most-specific field first — the failure's `type` attribute, then its
    `message`, then the traceback body — for exactly the reason
    ``explain_orfs_failure`` searches ERROR lines before the whole log: the first
    match should describe what actually failed, not something incidental further
    down the traceback.

    ("", "") is a normal answer, not a miss to be worked around.  The caller
    still prints WHY, WHERE and the quoted source line for that test.
    """
    facts = facts or {}
    ftype = str(test.get("type") or facts.get("type") or "")
    message = str(test.get("message") or "")
    detail = str(test.get("detail") or "")

    hint_id = ""
    template = ""
    for haystack in (ftype.lower(), message.lower(), detail.lower()):
        if not haystack:
            continue
        for candidate_id, markers, text in _COCOTB_HINTS:
            if any(marker in haystack for marker in markers):
                hint_id, template = candidate_id, text
                break
        if hint_id:
            break

    # A non-assertion exception raised in the user's OWN test file is a testbench
    # bug, and saying so is more useful than any rule above.  Checked after the
    # table so a recognised cause (a timeout, an X in a comparison) still wins.
    if not hint_id and ftype and ftype != "AssertionError":
        where = facts.get("where") or {}
        if where.get("file") and where["file"] != _RUNNER_FILENAME:
            return (_RUN_HINTS["test_raised_exception"].format(
                type=ftype, file=where["file"]), "test_raised_exception")

    if not hint_id:
        return "", ""

    signal = ""
    match = _ATTR_NAME_RE.search(f"{message}\n{detail}")
    if match:
        signal = match.group("sig")
    suggest = ""
    if signal and ports:
        close = difflib.get_close_matches(signal, list(ports), n=1, cutoff=0.6)
        if close:
            suggest = f"  Did you mean `{close[0]}`?"
    # module_ports() returning [] means "could not tell", NEVER "portless" — so
    # the port-list clause is dropped entirely rather than claiming a design has
    # no ports.  See module_ports' own docstring.
    port_clause = ""
    if ports:
        shown = ", ".join(list(ports)[:12])
        port_clause = f" ({top or 'the top module'} declares: {shown})"

    at_time = f" at {facts['sim_time']}" if facts.get("sim_time") else ""
    expected, actual = facts.get("expected"), facts.get("actual")
    if expected and actual:
        verdict = f"The design produced {actual} where the test required {expected}"
    elif expected:
        # The actual value was never recoverable from the report. Saying "the
        # design produced something else" would imply this text knows a value it
        # is declining to print.
        verdict = f"The assertion required {expected} and the design did not meet it"
    else:
        verdict = "An assertion in the test did not hold"

    return template.format(
        signal=signal or "<signal>",
        suggest=suggest,
        ports=port_clause,
        clk=_clock_name(ports),
        expected=expected or "a different value",
        actual=actual or "something else",
        verdict=verdict,
        at_time=at_time,
    ), hint_id


# ── Where a Verilator error is ──────────────────────────────────────────────

# Anchored on `%Error`/`%Warning` rather than on line position, and MULTILINE.
# That is precisely what lets the same regex find Verilator's diagnostics when
# they arrive WRAPPED INSIDE a Python traceback out of cocotb's `runner.build()`,
# with no separate code path for that case.
_VERILATOR_DIAG_RE = re.compile(
    r"^%(?P<sev>Error|Warning)(?:-(?P<code>[A-Z0-9_]+))?:\s*"
    r"(?:(?P<file>[^\s:][^:]*):(?P<line>\d+):(?:(?P<col>\d+):)?\s*)?"
    r"(?P<msg>.*)$", re.MULTILINE)

# What each Verilator message code MEANS, in terms of this block's ports.
# Verilator's own text says what it found; these say what to change.
_VERILATOR_HINTS: Dict[str, str] = {
    "DECLFILENAME":
        "Grafux writes the RTL to `<top>.v`, so this almost always means the "
        "`top` port names a different module than the one the RTL declares. "
        "Set `top` to the module name in the source.",
    "PINMISSING":
        "The instantiation leaves out a port the module declares. Connect it "
        "explicitly, or give the port a default in its declaration.",
    "PINNOTFOUND":
        "The instantiation connects a port the module does not declare — a "
        "typo, or an instance of an older version of the module.",
    "WIDTH":
        "The two sides are different widths, so bits are silently zero-extended "
        "or dropped. Size the literal (`8'd0`) or fix the declaration.",
    "WIDTHEXPAND":
        "The right-hand side is narrower than the left and is being "
        "zero-extended. Harmless when intended; size the literal to say so.",
    "WIDTHTRUNC":
        "The right-hand side is WIDER than the left, so the top bits are "
        "discarded. This is the width bug that usually matters — widen the "
        "target or slice deliberately.",
    "MULTIDRIVEN":
        "The signal is assigned from more than one always block. In synthesis "
        "that is a short; drive it from exactly one block.",
    "LATCH":
        "An incomplete assignment inferred a latch. Give every branch of the "
        "if/case an assignment, or add a default before the branches.",
    "COMBDLY":
        "A non-blocking assignment (`<=`) in a combinational block. Use "
        "blocking (`=`) in `always_comb`, non-blocking in `always_ff`.",
    "BLKSEQ":
        "A blocking assignment (`=`) in a sequential block. Use non-blocking "
        "(`<=`) in `always_ff`, or simulation and synthesis will disagree.",
    "IMPLICIT":
        "A signal is used without being declared, so Verilator created a 1-bit "
        "wire for it. Usually a typo; declare it with its real width.",
    "MODDUP":
        "Two modules with the same name were compiled. When the `sva` port is "
        "wired in it is compiled alongside the design — check it does not "
        "redeclare the module it is meant to bind to.",
    "SYNTAX":
        "The location is where the parser gave up, which is often ONE LINE "
        "AFTER the real mistake — a missing `;`, `end` or `endmodule` on the "
        "line above.",
    "CASEINCOMPLETE":
        "The case statement does not cover every value. Add a `default:` arm.",
    "UNDRIVEN":
        "The signal is read but never assigned, so it stays X. Drive it, or "
        "delete it if it is dead.",
    "SELRANGE":
        "A bit select is outside the signal's declared range, which yields X. "
        "Check the index expression against the declared width.",
    "TIMESCALEMOD":
        "Some modules declare a timescale and others do not. Add a `timescale "
        "to every file, or pass none at all.",
    "MISSINGFILE":
        "A source file named on the command line does not exist in the pod. "
        "Check the `files` port: every entry must be one of the uploaded "
        "inputs, not a path from your own machine.",
    "STMTDLY":
        "A delay (`#`) in a statement. Delays are not synthesizable; drive "
        "timing from the clock instead.",
}

# Some diagnostics carry no message code at all -- a parse failure is reported as
# a bare `%Error:` -- so the code table alone would never explain the single most
# common first error a user hits. These match on the message text instead.
_VERILATOR_TEXT_HINTS: Tuple[Tuple[str, str], ...] = (
    ("syntax error", "SYNTAX"),
    ("cannot find file", "MISSINGFILE"),
    ("cannot open", "MISSINGFILE"),
)

_MAX_DIAGS = 12
_ERRORS_MAX_CHARS = 8000

# `%Error: Exiting due to 3 error(s)` carries no location and repeats a count the
# header already gives.  Noise, once a located diagnostic exists.
_VERILATOR_NOISE = ("exiting due to",)


def explain_verilator_diagnostics(text: str, *,
                                  sources: Optional[Dict[str, str]] = None,
                                  max_diags: int = _MAX_DIAGS,
                                  max_chars: int = _ERRORS_MAX_CHARS) -> str:
    """
    Verilator's diagnostics, located, quoted and explained; "" when there are none.

    This is the whole content of the `errors` port for every non-cocotb failure.
    Errors win over warnings when both are present, but in LINT mode the warnings
    ARE the product, so they are rendered whenever no error exists.
    """
    sources = sources or {}
    errors: List[Any] = []
    warnings: List[Any] = []
    for match in _VERILATOR_DIAG_RE.finditer(text or ""):
        (errors if match.group("sev") == "Error" else warnings).append(match)
    chosen = errors or warnings
    if not chosen:
        return ""

    # Drop the un-located summary lines, but only once something located exists —
    # on a failure whose ONLY output is "Exiting due to 1 error(s)", that line is
    # all the user has and dropping it would empty the port.
    if any(m.group("file") and m.group("line") for m in chosen):
        chosen = [m for m in chosen
                  if m.group("file")
                  or not any(n in (m.group("msg") or "").lower()
                             for n in _VERILATOR_NOISE)]

    seen = set()
    blocks: List[str] = []
    for match in chosen:
        key = (match.group("code"), match.group("file"),
               match.group("line"), (match.group("msg") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        if len(blocks) >= max_diags:
            continue

        code = match.group("code") or ""
        label = f"%{match.group('sev')}" + (f"-{code}" if code else "")
        name = match.group("file") or ""
        line = int(match.group("line") or 0)
        col = int(match.group("col") or 0)

        part: List[str] = []
        if name and col:
            part.append(f"{name}:{line}:{col}  {label}")
        elif name:
            part.append(f"{name}:{line}  {label}")
        else:
            part.append(label)
        quoted = quote_source_line(sources.get(name, ""), line, col=col)
        if quoted:
            part.append(quoted)
        message = (match.group("msg") or "").strip()
        if message:
            part.append(f"  {message}")
        hint = _VERILATOR_HINTS.get(code, "")
        if not hint:
            lowered = message.lower()
            for marker, mapped in _VERILATOR_TEXT_HINTS:
                if marker in lowered:
                    hint = _VERILATOR_HINTS.get(mapped, "")
                    break
        if hint:
            part.append(f"  -> {hint}")
        blocks.append("\n".join(part))

    if not blocks:
        return ""
    noun = "error" if errors else "warning"
    header = f"{len(seen)} Verilator {noun}{'' if len(seen) == 1 else 's'}."
    if len(seen) > len(blocks):
        header += f"  (showing the first {len(blocks)})"
    out = header + "\n\n" + "\n\n".join(blocks)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n... (truncated)"
    return out


def locate_python_error(traceback_text: str, *,
                        sources: Optional[Dict[str, str]] = None,
                        test_file: str = "") -> str:
    """
    Where a Python error in the testbench is, quoted; "" when there is no traceback.

    A traceback that never leaves ``run_cocotb.py`` is Grafux's OWN generated
    runner failing (see build_cocotb_runner_script), not the user's testbench.
    Saying so is the difference between a user debugging their design and a user
    debugging a file they have never seen and cannot edit.
    """
    sources = sources or {}
    where = locate_python_failure(traceback_text, prefer=test_file)
    if not where:
        return ""
    ftype, message = _exception_line(traceback_text)

    if where.get("file") == _RUNNER_FILENAME:
        head = (f"{_RUNNER_FILENAME}:{where['line']}  in {where['func']}\n"
                "  This is Grafux's generated cocotb runner, not your testbench — "
                "the run failed before it reached your tests.")
        return f"{head}\n  {ftype}: {message}".rstrip() if ftype else head

    part = [f"{where['file']}:{where['line']}  in {where['func']}"]
    quoted = quote_source_line(sources.get(where["file"], ""), int(where["line"]))
    if quoted:
        part.append(quoted)
    if ftype:
        part.append(f"  {ftype}: {message}".rstrip())
    return "\n".join(part)


def error_locations(*, verilator_stderr: str = "", python_traceback: str = "",
                    sources: Optional[Dict[str, str]] = None,
                    test_file: str = "", fallback: str = "") -> str:
    """
    The `errors` port: WHERE the run broke, and nothing else.

    One assembler with five call sites, so the "locations only" contract cannot
    drift between the lint path, the C++ sim path, the cocotb build path and the
    two failure paths.

    Returns "" when nothing was located AND there is no fallback — which is the
    decision that makes this port mean something: a run whose build was clean and
    whose tests merely failed has no error LOCATION, so it says nothing at all
    and `failures` carries the story.  The block's red/green comes from `status`
    and `passed`, never from this port, so an empty `errors` cannot make a failing
    run look clean.
    """
    sources = sources or {}
    located = explain_verilator_diagnostics(verilator_stderr, sources=sources)
    if not located and python_traceback:
        located = locate_python_error(python_traceback, sources=sources,
                                      test_file=test_file)
    parts = [p for p in ((fallback or "").strip(), located) if p]
    return "\n\n".join(parts).strip()


# ── Per-test slices of the cocotb log ───────────────────────────────────────

# cocotb's regression manager brackets every test in the log.  Both spellings are
# matched because 1.x and 2.x word the banner differently, and a version that
# words it a third way yields {} — the failure blocks then simply carry no log
# excerpt.  Degrade, never guess.
_COCOTB_TEST_START_RE = re.compile(
    r"^\s*[\d.]+\s*[fpnum]?s\s+INFO\s+cocotb\.regression\s+"
    r"(?:running\s+(?P<a>\w+)|Running\s+test\s+\d+/\d+:\s*(?P<b>\w+))",
    re.MULTILINE | re.IGNORECASE)

_LOG_TAIL_LINES = 25


def slice_cocotb_log(log_text: str, *,
                     tail_lines: int = _LOG_TAIL_LINES) -> Dict[str, str]:
    """
    {test name: the last lines of its own log}, for attaching to a failure block.

    Sliced start-of-test to start-of-NEXT-test rather than to the passed/failed
    banner, because anything a test printed after its assertion — a teardown
    dump, a scoreboard summary — belongs to that test and is often the line that
    explains it.
    """
    text = log_text or ""
    starts = list(_COCOTB_TEST_START_RE.finditer(text))
    if not starts:
        return {}
    out: Dict[str, str] = {}
    for index, match in enumerate(starts):
        name = match.group("a") or match.group("b") or ""
        if not name:
            continue
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[match.start():end].rstrip().splitlines()
        out[name] = "\n".join(body[-tail_lines:] if tail_lines > 0 else body)
    return out


# How much of a failure's evidence is kept per test.  All three are TAIL caps:
# a traceback's meaning is its last lines, and so is a captured stdout's.
_MAX_DETAIL_CHARS = 4000
_MAX_STREAM_CHARS = 2000
# Only this many FAILING tests carry their full evidence.  A regression with 300
# failures is one bug reported 300 times; the first twenty carry the diagnosis and
# the rest still carry their name, status and message.  Matches summarize_failures'
# default max_tests, so a test that gets a block always has evidence to fill it.
_MAX_DETAIL_TESTS = 20


def _tail_cap(text: str, limit: int) -> str:
    """Keep the LAST ``limit`` characters, and say so."""
    value = (text or "").strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return (f"...[truncated, {len(value) - limit} earlier characters]\n"
            + value[-limit:])


def parse_cocotb_results(xml_text: str) -> Dict[str, Any]:
    """
    Turn cocotb's JUnit ``results.xml`` into a structured summary.

    THE EXIT CODE IS NOT THE VERDICT.  Depending on the cocotb and simulator
    versions a run with failing tests can still exit 0, and a run where the
    testbench declared no tests at all exits 0 every time — which would read as a
    clean pass and is the worst possible lie for a verification block to tell.
    ``failed == 0 and total > 0`` is the condition callers must decide on.

    Never raises.  A missing or truncated report is itself a result the block has
    to show, so bad input comes back as an empty summary carrying ``error``.

    **``message`` is a compatibility anchor and must stay byte-identical.**  It
    is what saved ``results.txt`` files hold, what the app's
    ``VerificationResults::parse`` reads, and what four tests assert on.  The
    richer fields below are ADDITIVE, and are filled only for FAILING tests:
    ``detail`` carries the ``<failure>`` element's BODY (the traceback), which
    the attribute-wins rule for ``message`` throws away on every real cocotb
    report; ``extra`` carries any second failure child; ``stdout``/``stderr``
    carry ``<system-out>``/``<system-err>``, which nothing ever looked at.
    """
    summary: Dict[str, Any] = {
        "total": 0, "passed": 0, "failed": 0, "skipped": 0, "tests": [],
    }
    text = (xml_text or "").strip()
    if not text:
        summary["error"] = "no results.xml was produced"
        return summary
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        summary["error"] = f"results.xml could not be parsed: {exc}"
        return summary

    detailed = 0
    for case in root.iter("testcase"):
        status = "passed"
        message = ""
        # Verbatim from the original: the FIRST failure/error child wins and
        # stops the walk, and its `message` attribute beats its body.  Anything
        # "cleaner" here changes the compatibility anchor.
        for child in case:
            tag = (child.tag or "").lower()
            if tag in ("failure", "error"):
                status = "failed"
            elif tag == "skipped":
                status = "skipped"
            else:
                continue
            message = (child.get("message") or (child.text or "")).strip()
            if tag in ("failure", "error"):
                break
        try:
            elapsed = float(case.get("time") or 0.0)
        except ValueError:
            elapsed = 0.0
        entry: Dict[str, Any] = {
            "name": (case.get("name") or "").strip(),
            "classname": (case.get("classname") or "").strip(),
            "status": status,
            "time": elapsed,
            "message": message,
        }

        # Everything below is for failing tests only.  Passing and skipped tests
        # are the bulk of a large run and carry no evidence worth keeping.
        if status == "failed" and detailed < _MAX_DETAIL_TESTS:
            detailed += 1
            diagnostics = [c for c in case
                           if (c.tag or "").lower() in ("failure", "error")]
            if diagnostics:
                first = diagnostics[0]
                entry["detail_kind"] = (first.tag or "").lower()
                ftype = (first.get("type") or "").strip()
                if ftype:
                    entry["type"] = ftype
                body = _tail_cap(first.text or "", _MAX_DETAIL_CHARS)
                if body:
                    entry["detail"] = body
                extra = []
                for child in diagnostics[1:]:
                    extra.append({
                        "kind": (child.tag or "").lower(),
                        "type": (child.get("type") or "").strip(),
                        "message": (child.get("message") or "").strip(),
                        "detail": _tail_cap(child.text or "", _MAX_STREAM_CHARS),
                    })
                if extra:
                    entry["extra"] = extra
            source = (case.get("file") or "").strip()
            if source:
                entry["file"] = os.path.basename(source)
            for tag, key in (("system-out", "stdout"), ("system-err", "stderr")):
                node = case.find(tag)
                if node is not None:
                    captured = _tail_cap(node.text or "", _MAX_STREAM_CHARS)
                    if captured:
                        entry[key] = captured

        summary["tests"].append(entry)
        summary["total"] += 1
        summary[status] += 1

    if summary["total"] == 0:
        summary["error"] = (
            "the testbench declared no tests — check that it defines at least one "
            "@cocotb.test() and that its module name matches"
        )
    return summary


# lcov records, summed across every source file in the report:
#   LF/LH  lines found / hit      BRF/BRH  branches found / hit
_LCOV_RECORD_RE = {
    "lines": (re.compile(r"^LF:(\d+)", re.MULTILINE),
              re.compile(r"^LH:(\d+)", re.MULTILINE)),
    "branches": (re.compile(r"^BRF:(\d+)", re.MULTILINE),
                 re.compile(r"^BRH:(\d+)", re.MULTILINE)),
}


def parse_lcov_summary(info_text: str) -> Dict[str, Any]:
    """
    Summarize an lcov ``.info`` file into hit/total/percent per metric.

    Verilator's coverage is line and branch (toggle coverage is folded into lines
    by ``verilator_coverage --write-info``), and a report with no records at all —
    a design that never ran — yields zeros rather than a division error.
    """
    text = info_text or ""
    out: Dict[str, Any] = {}
    for metric, (found_re, hit_re) in _LCOV_RECORD_RE.items():
        total = sum(int(n) for n in found_re.findall(text))
        hit = sum(int(n) for n in hit_re.findall(text))
        pct = round(100.0 * hit / total, 1) if total else 0.0
        out[metric] = {"hit": hit, "total": total, "pct": pct}
    return out


def enrich_failures(results: Dict[str, Any], *, testbench: str = "", rtl: str = "",
                    top: str = "", stem: str = "",
                    log_slices: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Add the derived WHY/WHERE/cause fields to every failing test, in place.

    Idempotent, so both ``run_cocotb`` (which wants them inside the ``results``
    JSON, where the app's verdict panel reads them) and ``summarize_failures``
    (which may be handed a raw parse) can call it without paying twice or
    disagreeing about what a test's cause is.

    Never raises: it runs over text a user's testbench produced, and a run that
    already has its verdict must not be lost to a regex.
    """
    tests = results.get("tests") or []
    if not tests:
        return results
    slices = log_slices or {}
    test_file = f"test_{stem}.py" if stem else ""
    ports: Sequence[str] = ()
    if rtl and top:
        try:
            ports = module_ports(rtl, top)
        except Exception:  # pragma: no cover - module_ports is defensive already
            ports = ()

    for test in tests:
        if test.get("status") != "failed" or "hint_id" in test:
            continue
        try:
            facts = extract_failure_facts(
                test, testbench=testbench, test_file=test_file,
                log_slice=slices.get(test.get("name") or "", ""))
            hint, hint_id = explain_cocotb_failure(
                test, top=top, ports=ports, stem=stem, facts=facts)
        except Exception:
            logger.debug("failure enrichment failed for %s", test.get("name"),
                         exc_info=True)
            facts, hint, hint_id = {}, "", ""
        test.update(facts)
        test["hint"] = hint
        test["hint_id"] = hint_id
    return results


# One failing test's block is built from these sections, in this order.  The
# order is the reading order a person actually uses: what went wrong, where, why,
# and only then the log around it.
_FAILURE_RULE = "-" * 60
_WHY_MAX_CHARS = 1500


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line.strip() else ""
                     for line in (text or "").splitlines())


# A literal a reader can compare against, as opposed to the name of one.
_VALUE_RE = re.compile(r"^[-+]?(?:\d+'[bodhBODH][0-9a-fA-FxXzZ_?]+|0[xXbBoO][0-9a-fA-F_]+"
                       r"|\d[\d_]*(?:\.\d+)?|True|False|None)$")


def _looks_like_a_value(text: Any) -> bool:
    return bool(text) and bool(_VALUE_RE.match(str(text).strip()))


def _failure_header(test: Dict[str, Any]) -> str:
    """``FAILED <name>`` plus when it died.  The literal token is load-bearing."""
    name = test.get("name") or "(unnamed test)"
    when: List[str] = []
    if test.get("sim_time"):
        when.append(f"failed at {test['sim_time']}")
    try:
        elapsed = float(test.get("time") or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    if elapsed:
        when.append(f"{elapsed:g} s")
    suffix = f"        ({', '.join(when)})" if when else ""
    return f"FAILED {name}{suffix}"


def _render_failure(test: Dict[str, Any], *, level: int, hint_text: str,
                    log_excerpt: str = "") -> str:
    """
    One failing test, at a given level of detail.

    ``level`` 0 is everything; 1 drops the log excerpt; 2 also drops the source
    quote.  Degrading section by section keeps the ANSWER (what failed, and why)
    on the page when a run has too many failures to show in full — a blind
    ``text[:max_chars]`` would instead cut the last tests off mid-sentence and
    keep evidence for the first ones nobody needed twice.
    """
    parts: List[str] = [_FAILURE_RULE, _failure_header(test)]

    why: List[str] = []
    message = str(test.get("message") or "").strip()
    if not message:
        message = str(test.get("why") or "").strip()
    if len(message) > _WHY_MAX_CHARS:
        message = (message[:_WHY_MAX_CHARS].rstrip()
                   + f"\n...[truncated, {len(message) - _WHY_MAX_CHARS} more characters]")
    if message:
        why.append(message)
    if test.get("expected") and test.get("actual"):
        why.append(f"expected: {test['expected']}")
        why.append(f"got:      {test['actual']}")
    elif _looks_like_a_value(test.get("expected")):
        # Only when the expectation is a VALUE. `assert count == DEPTH` yields
        # the identifier "DEPTH", and echoing that back tells the reader nothing
        # they did not just read on the quoted source line above.
        why.append(f"the test required: {test['expected']}")
    if why:
        parts.append("")
        parts.append("WHY")
        parts.append(_indent("\n".join(why)))

    where = test.get("where") or {}
    if where.get("file") and where.get("line"):
        parts.append("")
        parts.append("WHERE")
        location = f"{where['file']}:{where['line']}"
        if where.get("func"):
            location += f"  in {where['func']}"
        rows = [location]
        if level < 2 and where.get("text"):
            rows.append(where["text"])
        parts.append(_indent("\n".join(rows)))

    if hint_text:
        parts.append("")
        parts.append("LIKELY CAUSE")
        parts.append(_indent(hint_text))

    if level < 1 and log_excerpt:
        parts.append("")
        parts.append("LAST LINES BEFORE THE FAILURE")
        parts.append(_indent(log_excerpt))

    return "\n".join(parts)


def _assemble_failures(failed: List[Dict[str, Any]], *, shown: int, total: int,
                       level: int, dedupe_hints: bool,
                       log_slices: Optional[Dict[str, str]] = None) -> str:
    """Header plus one block per shown test, plus the "and N more" line."""
    header = f"{len(failed)} of {total} cocotb tests failed."
    out: List[str] = [header]
    seen_hints: Dict[str, str] = {}
    for index, test in enumerate(failed[:shown]):
        hint = str(test.get("hint") or "")
        hint_id = str(test.get("hint_id") or "")
        if hint and dedupe_hints:
            if hint_id and hint_id in seen_hints:
                hint = f"(same cause as {seen_hints[hint_id]})"
            elif index >= 3:
                hint = ""
            elif hint_id:
                seen_hints[hint_id] = test.get("name") or "the test above"
        out.append("")
        out.append(_render_failure(
            test, level=level, hint_text=hint,
            log_excerpt=(log_slices or {}).get(test.get("name") or "", "")))
    if len(failed) > shown:
        out.append("")
        out.append(f"... and {len(failed) - shown} more failing tests.")
    return "\n".join(out)


# Fields that exist only so the enrichers can read them.  They are the bulk of a
# parsed report -- a traceback plus two captured streams is up to 8000 characters
# per failing test -- and every one of them has already been rendered into
# `failures` in the form a person reads, so serialising them onto the `results`
# port would put ~80 KB of duplicate text in a file whose remaining readers want
# name, status and one line of why.  It would also guarantee that the review's
# head cap cuts `results` mid-JSON.  The originals stay available: results.xml
# comes back as an artifact.
_RAW_EVIDENCE_FIELDS = ("detail", "stdout", "stderr", "extra", "log_excerpt")


def slim_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``results`` without the raw evidence, for the port."""
    out = dict(results)
    tests = []
    for test in results.get("tests") or []:
        slim = {k: v for k, v in test.items() if k not in _RAW_EVIDENCE_FIELDS}
        # The quoted source block is the same two lines `failures` already prints
        # under WHERE, and nothing reads it from here -- the app's verdict panel
        # wants only the file and the line.
        where = slim.get("where")
        if isinstance(where, dict) and "text" in where:
            slim["where"] = {k: v for k, v in where.items() if k != "text"}
        tests.append(slim)
    out["tests"] = tests
    return out


def summarize_failures(
    results: Dict[str, Any],
    *,
    max_tests: int = 20,
    max_chars: int = 20000,
    testbench: str = "",
    rtl: str = "",
    top: str = "",
    stem: str = "",
    log_slices: Optional[Dict[str, str]] = None,
) -> str:
    """
    The feedback payload: what failed, WHY, WHERE, and what to try.

    This text is what a human reads on the ``failures`` port AND what gets wired
    into the design block's ``feedback`` port to drive an RTL repair, so it names
    the test, quotes its assertion, points at the line that raised it and says
    what usually causes that — the fix prompt has to be able to tell which
    behaviour was wrong, and a user has to be able to act without opening a log.

    Returns "" when nothing failed, so an empty ``failures`` port still means
    "clean".  That is the verify loop's stop condition and must not change.

    When it does not fit, it degrades SECTION BY SECTION — log excerpt, then
    source quote, then repeated causes, then the number of tests — and only cuts
    mid-text as a last resort.  Losing the newest evidence uniformly beats losing
    the last test entirely.
    """
    if not results:
        return ""
    if results.get("error") and not results.get("tests"):
        return str(results["error"])

    failed = [t for t in results.get("tests", []) if t.get("status") == "failed"]
    if not failed:
        return str(results.get("error") or "")

    enrich_failures(results, testbench=testbench, rtl=rtl, top=top, stem=stem,
                    log_slices=log_slices)

    total = int(results.get("total", 0) or 0)
    shown = min(max_tests, len(failed))

    # Section-by-section, then test-by-test.
    for level, dedupe in ((0, False), (1, False), (2, False), (2, True)):
        text = _assemble_failures(failed, shown=shown, total=total,
                                  level=level, dedupe_hints=dedupe,
                                  log_slices=log_slices)
        if len(text) <= max_chars:
            return text
    while shown > 1:
        shown -= 1
        text = _assemble_failures(failed, shown=shown, total=total,
                                  level=2, dedupe_hints=True,
                                  log_slices=log_slices)
        if len(text) <= max_chars:
            return text

    return text[:max_chars].rstrip() + "\n... (truncated)"


def globs_for(kind: str, *, work_dir: str = WORK_DIR, platform: str = "",
              design: str = "") -> List[str]:
    """
    Artifact globs to pull back after a run.

    Collected even when a stage FAILED, so a route that blew up still returns the
    placement DEF and the reports that explain why.
    """
    if kind == "verilator":
        # The last five are the cocotb mode's: the JUnit report and the lcov
        # summary are read back inline as well, but they are collected as
        # artifacts too so a user can open the raw report, and depending on the
        # cocotb version the waveform and build log land either beside the design
        # or inside the runner's own build directory — so both are globbed.
        return [f"{work_dir}/*.vcd", f"{work_dir}/*.fst",
                f"{work_dir}/*.log", f"{work_dir}/obj_dir/*.log",
                f"{work_dir}/{COCOTB_RESULTS_XML}", f"{work_dir}/{COCOTB_COVERAGE_INFO}",
                f"{work_dir}/{COCOTB_BUILD_DIR}/*.vcd",
                f"{work_dir}/{COCOTB_BUILD_DIR}/*.fst",
                f"{work_dir}/{COCOTB_BUILD_DIR}/*.log"]
    if kind == "yosys":
        return [f"{work_dir}/*.v", f"{work_dir}/*.log", f"{work_dir}/*.json",
                f"{work_dir}/*.txt"]
    if kind in ("openram", "opengcram"):
        # Every generated view, plus the two files that explain the run: the
        # ``.py`` is the config OpenRAM ACTUALLY used with every default it
        # filled in -- the single most useful artifact when a macro comes out
        # wrong -- and the ``.log`` is its own record, which is why nothing here
        # pipes the compile through ``tee``.
        out = f"{work_dir}/{OPENRAM_OUT_DIR}"
        return [f"{out}/*.gds", f"{out}/*.lef", f"{out}/*.lib",
                f"{out}/*.v", f"{out}/*.sp", f"{out}/*.spice",
                f"{out}/*.html", f"{out}/*.log", f"{out}/*.py",
                f"{out}/*.lvs", f"{out}/*.json"]
    if kind == "analogue_simulator":
        # The deck as run, ngspice's own log, and the full rawfile -- the
        # waveforms port is decimated, this is not.  Named files rather than
        # *.raw globs so a model file staged through `files` is not echoed back.
        return [f"{work_dir}/{ngspice.DECK_FILE}", f"{work_dir}/{ngspice.LOG_FILE}",
                f"{work_dir}/{ngspice.RAW_FILE}"]
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


# Scoped to the cocotb runner, deliberately NOT folded into _EDA_ENV: prepending
# a venv to the PATH of every run would shadow the python3 that ORFS's own tooling
# uses, which is the classic way to break yosys and openroad from a distance.
#
# PYTHONUNBUFFERED is not cosmetic. Without it cocotb's output arrives in one lump
# when the process exits, so the live log tail shows nothing for the whole run and
# the stage markers all arrive at once, after the stages they announce.
#
# There is no SIM=verilator here on purpose: that variable belongs to cocotb's
# Makefile flow, and get_runner(...) does not read it.
_COCOTB_ENV = (
    'export PATH="/opt/cocotb-venv/bin:$PATH"; '
    'export PYTHONUNBUFFERED=1; '
    'export PYTHONDONTWRITEBYTECODE=1; '
    f'export PYTHONPATH="{WORK_DIR}:$PYTHONPATH"; '
)


def _sh_cocotb(command: str) -> str:
    """``_sh`` plus the cocotb virtualenv and unbuffered Python output."""
    return _sh(_COCOTB_ENV + command)


# Pods are created once and then REUSED for the life of their block, and the
# image tag is not part of any reuse key -- so fixing Dockerfile.verify does not
# retroactively fix a pod that is already warm, and a user keeps hitting the
# stale image until it idles out. This heals such a pod in place instead.
#
# `ldconfig -p` is the check because the library's exact name and directory are
# distro-specific (libpython3.10.so.1.0 under /usr/lib/x86_64-linux-gnu here) and
# the linker cache is the one place that knows all of them. It is also cheap: on
# a correct image this is a grep against an in-memory table, which is what makes
# it acceptable on every run. -dev goes in alongside the runtime package because
# it carries the unversioned libpython3.10.so symlink that cocotb's own
# find_libpython matches on -- the same pair the image installs at build time.
#
# Every apt line is redirected away and one deliberate marker is echoed instead:
# a hundred lines of dpkg output in the block's log would bury the simulation
# the user actually came to read.
_LIBPYTHON_PREFLIGHT = (
    "ldconfig -p 2>/dev/null | grep -q libpython3 "
    "&& echo GRAFUX_LIBPYTHON_PRESENT "
    "|| { apt-get update -qq >/dev/null 2>&1 "
    "&& DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
    "--no-install-recommends libpython3.10 libpython3.10-dev >/dev/null 2>&1 "
    "&& echo GRAFUX_LIBPYTHON_INSTALLED "
    "|| echo GRAFUX_LIBPYTHON_UNAVAILABLE; }"
)


def _ensure_libpython(client, notes: List[str],
                      should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """
    Give the pod the SHARED libpython cocotb needs, installing it if it is absent.

    cocotb embeds CPython inside the simulator binary, so without this library a
    design VERILATES AND COMPILES CLEANLY and only then dies in the runner with
    "Unable to find libpython" -- a failure that reads like a broken testbench.
    The verify image installs it; this covers the pod that was provisioned before
    it did, which is the pod the user is waiting on right now.

    Not gated on ``simulator``: the embedding lives in cocotb's own VPI library,
    which icarus loads exactly as verilator does.  Scoped to :func:`run_cocotb`
    alone, so lint, sim, yosys and openroad never touch apt.

    Appends to ``notes`` (the block's ``warnings`` port) only when it changed
    something or could not.  NEVER raises and never fails the run: a pod with no
    route to the archives must still get its run and its own diagnosis from the
    GRAFUX_LIBPYTHON_MISSING marker the runner script prints later.
    """
    if should_cancel and should_cancel():
        return
    try:
        _code, out, _err = exec_simple(
            client, _sh(_LIBPYTHON_PREFLIGHT), timeout=180)
    except Exception as exc:  # noqa: BLE001 -- a preflight must not fail a run
        logger.warning("libpython preflight did not complete: %s", exc)
        notes.append(
            "Could not check this pod for the shared libpython that cocotb "
            "needs; the run went ahead without it.")
        return
    if "GRAFUX_LIBPYTHON_INSTALLED" in out:
        notes.append(
            "This pod's image predates the libpython fix, so the shared "
            "libpython cocotb needs was installed before the run -- that is "
            "the extra start-up time. Recreate the pod on a current image to "
            "stop paying it."
        )
    elif "GRAFUX_LIBPYTHON_UNAVAILABLE" in out:
        notes.append(
            "This pod's image is missing the shared libpython that cocotb "
            "needs and it could not be installed. If the run below did not "
            "simulate, that is why."
        )
    # GRAFUX_LIBPYTHON_PRESENT is the steady state on a current image: nothing
    # happened, and saying so on the warnings port would be noise.


def _resolve_liberty(client, pdk: str, override: str = "") -> str:
    """
    Find the standard-cell liberty file for the platform inside the container.

    Globbed on-device rather than hardcoded because the filename encodes process
    corner and voltage (sky130_fd_sc_hd__tt_025C_1v80.lib) and differs per platform.
    ``ls -S`` sorts largest-first so ``pick_liberty`` can use size as the signal for
    "this is the standard-cell library"; see that function for why.
    Returns "" when nothing matches, which callers treat as "synthesize generically".
    """
    if override.strip():
        return override.strip()
    cmd = _sh(f"ls -S -1 {liberty_glob(pdk)} 2>/dev/null")
    _code, out, _err = exec_simple(client, cmd, timeout=60)
    return pick_liberty(out.splitlines())


def run_verilator(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Lint, simulate, or run cocotb tests against a design with Verilator.

    Returns the ``outputs`` map for the block's ports plus bookkeeping keys
    (``_globs``, ``_status``, ``_stage``).  Never raises for a tool failure — a
    design that does not compile is a normal, expected result here, and the whole
    point of the block is to report it clearly.

    ``mode="cocotb"`` hands off to :func:`run_cocotb`.  So does ``mode="sim"``
    when the testbench is plainly Python: every verilator block created before
    cocotb existed carries ``mode=sim``, and wiring a testbench block into one
    must simply work rather than feeding Python to a C++ compiler.  Lint never
    looks at the testbench, so it is left alone.
    """
    top = (req.top or "").strip() or infer_top_module(req.rtl)
    mode, mode_note = resolve_verilator_mode(req.mode, req.testbench or "")
    if mode == "cocotb":
        if mode_note:
            logger.info("verilator: %s", mode_note)
        return run_cocotb(client, req, on_stage=on_stage, on_line=on_line,
                          should_cancel=should_cancel, note=mode_note)
    trace = str(req.trace or "").strip().lower() in ("1", "true", "yes", "on")
    # The source file MUST be named after the top module. Verilator's -Wall (which
    # lint mode needs, since lint findings are the whole point there) promotes
    # DECLFILENAME — "filename does not match MODULE name" — to a fatal error, so
    # writing every design to a fixed design.v made lint fail on literally every
    # input. Naming the file after the module is also just what a user would do.
    source = f"{_safe_filename(top)}.v"
    sftp = client.open_sftp()
    try:
        _write_file(sftp, f"{WORK_DIR}/{source}", req.rtl or "")
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
        sources=[source],
        testbench_file="tb.cpp" if (mode != "lint" and tb.strip()) else "",
        defines=req.defines,
        include_dirs=req.include_dirs,
        trace=trace,
        extra_flags=req.verilator_flags,
    )
    # The sources as the pod sees them.  Held here so a diagnostic can quote the
    # offending line without a second trip to the machine -- see quote_source_line.
    sources_map = {source: req.rtl or "", "tb.cpp": tb}
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
        # `lint` above still carries the raw stderr verbatim; this port now
        # carries only WHERE the errors are.  The fallback covers a failure
        # Verilator did not put a %Error on -- a cancel, a timeout, a crash.
        outputs["errors"] = error_locations(
            verilator_stderr=err or out, sources=sources_map,
            fallback=(
                "Verilator was cancelled" if code == -1 else
                "Verilator exceeded its timeout" if code == -2 else
                "" if _VERILATOR_DIAG_RE.search(err or out or "")
                else f"Verilator exited with code {code}"
            ),
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
    # A $fatal prints `%Error: file:line: ...`, which the same parser locates.
    # When it does not, the exit code is all there is to say.
    outputs["errors"] = "" if passed else error_locations(
        verilator_stderr=err, sources=sources_map,
        fallback=("" if _VERILATOR_DIAG_RE.search(err or "")
                  else f"Simulation exited with code {code}"),
    )
    if err.strip() and passed:
        outputs["warnings"] = (outputs.get("warnings", "") + "\n" + err.strip()).strip()
    return {"outputs": outputs, "_status": "ok" if passed else "error", "_stage": "sim",
            "_globs": globs_for("verilator")}


def run_cocotb(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    note: str = "",
) -> Dict[str, Any]:
    """
    Run a cocotb testbench against the design and report per-test results.

    This is what makes a generated testbench executable rather than merely
    readable: the design plus the Python tests go into the pod, cocotb's runner
    builds and simulates them, and what comes back is a structured verdict
    (``results``), a feedback payload naming what broke (``failures``) and a
    coverage summary — the three things the fix loop needs.

    Same contract as :func:`run_verilator`: returns the block's ``outputs`` plus
    the ``_status``/``_stage``/``_globs`` bookkeeping keys, and never raises for a
    tool failure. A design that fails its tests is the expected case here.

    ``note`` is carried onto the ``warnings`` port — it is how a user who left the
    mode at "sim" finds out why the run took the cocotb path.
    """
    top = (req.top or "").strip() or infer_top_module(req.rtl)
    trace = str(req.trace or "").strip().lower() in ("1", "true", "yes", "on")
    coverage = str(getattr(req, "coverage", "1") or "").strip().lower() in (
        "1", "true", "yes", "on")
    simulator = normalize_simulator(getattr(req, "simulator", ""))
    notes: List[str] = [note] if note else []

    # Coverage is a Verilator feature; icarus is the escape hatch for designs
    # Verilator refuses, and silently reporting 0% there would look like a broken
    # testbench rather than an unsupported combination.
    if coverage and simulator != "verilator":
        coverage = False
        notes.append(f"Coverage is not collected with simulator={simulator}.")

    # An unbound assertion module compiles, checks nothing and reports success —
    # so SVA that cannot be bound is skipped loudly rather than compiled quietly.
    sva = (getattr(req, "sva", "") or "").strip()
    sva_problem = sva_binding_problem(sva)
    if sva_problem:
        notes.append(sva_problem)
        sva = ""

    # Same filename rule as sim/lint mode: Verilator's DECLFILENAME check is fatal
    # when the file does not match the module name.
    stem = _safe_filename(top)
    source = f"{stem}.v"
    test_module = f"test_{stem}"
    sources = [source] + (["sva.sv"] if sva else [])

    script = build_cocotb_runner_script(
        top=top,
        sources=sources,
        test_module=test_module,
        simulator=simulator,
        trace=trace,
        coverage=coverage,
        assertions=bool(sva),
        seed=getattr(req, "seed", "") or "",
        tests=getattr(req, "tests", "") or "",
        extra_flags=req.verilator_flags or "",
    )

    # Up front, before anything is staged: the point is that THIS run works, not
    # that it comes back with a good explanation of why it did not.
    _ensure_libpython(client, notes, should_cancel)

    sftp = client.open_sftp()
    try:
        _write_file(sftp, f"{WORK_DIR}/{source}", req.rtl or "")
        _write_file(sftp, f"{WORK_DIR}/{test_module}.py", req.testbench or "")
        if sva:
            _write_file(sftp, f"{WORK_DIR}/sva.sv", sva)
        _write_file(sftp, f"{WORK_DIR}/run_cocotb.py", script)
    finally:
        sftp.close()

    # The sources exactly as the pod sees them, so a %Error or a traceback frame
    # can be quoted at its line without a second trip to the machine.
    sources_map = {source: req.rtl or "",
                   f"{test_module}.py": req.testbench or ""}
    if sva:
        sources_map["sva.sv"] = sva

    outputs: Dict[str, str] = {"top": top, "rtl": req.rtl or "", "lint": ""}
    log_parts: List[str] = [f"$ cat run_cocotb.py\n{script}"]

    # The generated script prints GRAFUX_STAGE markers so one python invocation
    # still reports build and simulation as separate stages, instead of the UI
    # sitting on a single opaque "running" for the whole run.
    stage = ""

    def handle_line(text: str) -> None:
        nonlocal stage
        marker = text.strip()
        if marker.startswith("GRAFUX_STAGE "):
            nxt = marker.split(None, 1)[1].strip()
            if stage:
                on_stage(stage, "done")
            if nxt != "report":
                on_stage(nxt, "running")
            stage = nxt
            return
        if on_line:
            on_line(text)

    code, out, err = exec_stream(
        client, _sh_cocotb(f"cd {WORK_DIR} && python3 run_cocotb.py"),
        timeout=int(req.timeout or 900),
        on_line=handle_line, should_cancel=should_cancel,
    )
    log_parts.append(f"$ python3 run_cocotb.py\n{out}\n{err}")
    last_stage = stage or "build"
    if last_stage != "report":
        on_stage(last_stage, "done" if code == 0 else "failed")

    if code in (-1, -2):
        outputs.update({
            "status": "error", "passed": "false", "results": "", "failures": "",
            "coverage": "", "sim_output": out.strip(), "warnings": "\n".join(notes),
            "errors": ("The cocotb run was cancelled" if code == -1
                       else "The cocotb run exceeded its timeout"),
            "log": "\n".join(log_parts),
        })
        return {"outputs": outputs, "_status": "error", "_stage": last_stage,
                "_globs": globs_for("verilator")}

    build_failed = "GRAFUX_BUILD_FAILED" in out or "GRAFUX_BUILD_FAILED" in err

    # A pod whose image lacks the shared libpython builds the design perfectly
    # and then fails in cocotb's runner, so without this the user is handed a
    # clean Verilator log and a ValueError about their design's environment.
    # Checked in both streams and in both spellings: the generated script prints
    # the marker, and an older script in a long-lived pod only shows cocotb's own
    # message.
    joined = out + err
    libpython_missing = ("GRAFUX_LIBPYTHON_MISSING" in joined
                         or "Unable to find libpython" in joined)

    # results.xml is the verdict, NOT the exit code — see parse_cocotb_results.
    _rc, xml_text, _re = exec_simple(
        client, _sh(f"cat {WORK_DIR}/{COCOTB_RESULTS_XML} 2>/dev/null"), timeout=60)
    results = parse_cocotb_results(xml_text)
    passed = results["failed"] == 0 and results["total"] > 0 and not build_failed

    # The FULL log, fetched the way results.xml is, because exec_stream only ever
    # kept the last 400 lines of each stream -- the lines that explain an early
    # failure in a long run are gone before anything can parse them.
    #
    # A pod is created once per block and REUSED for its lifetime, and the image
    # tag is not part of the reuse key, so a warm pod is still running the
    # run_cocotb.py it was handed on its first run -- which may predate the tee
    # and write no log at all. Falling back to the 400-line tail keeps those pods
    # working with less detail rather than with none.
    _lc, log_text, _le = exec_simple(
        client,
        _sh(f"tail -c {COCOTB_LOG_MAX_BYTES} {WORK_DIR}/{COCOTB_LOG} 2>/dev/null"),
        timeout=120)
    log_slices = slice_cocotb_log(log_text) or slice_cocotb_log(out)

    coverage_summary: Dict[str, Any] = {}
    if coverage and results["total"] and not build_failed:
        on_stage("coverage", "running")
        cov_cmd = build_coverage_cmd()
        ccode, cout, cerr = exec_stream(
            client, _sh(f"cd {WORK_DIR} && {cov_cmd}"), timeout=300,
            on_line=on_line, should_cancel=should_cancel,
        )
        log_parts.append(f"$ {cov_cmd}\n{cout}\n{cerr}")
        if ccode == 0:
            _cc, info_text, _ce = exec_simple(
                client, _sh(f"cat {WORK_DIR}/{COCOTB_COVERAGE_INFO} 2>/dev/null"),
                timeout=60)
            coverage_summary = parse_lcov_summary(info_text)
        else:
            notes.append("Coverage could not be summarized; see the log.")
        # Coverage is a nice-to-have: a design whose tests all passed must not be
        # reported as failed because verilator_coverage had nothing to chew on.
        on_stage("coverage", "done" if ccode == 0 else "failed")

    failures = summarize_failures(
        results, testbench=req.testbench or "", rtl=req.rtl or "", top=top,
        stem=stem, log_slices=log_slices)

    # Run-level problems, prepended: they explain why the per-test story below is
    # thin or absent. The wording moved to _RUN_HINTS unchanged, so that one
    # table is the only place these sentences exist.
    run_problem = ""
    if libpython_missing:
        run_problem = _RUN_HINTS["libpython_missing"]
    elif build_failed:
        run_problem = _RUN_HINTS["build_failed"]
        if sva:
            run_problem += _RUN_HINTS["build_failed_sva"]
    elif "declared no tests" in str(results.get("error", "")):
        # summarize_failures returned the terse parser sentence; replace it with
        # the one that says what to actually check.
        run_problem = _RUN_HINTS["no_tests"].format(stem=stem)
        failures = ""
    if run_problem:
        failures = run_problem if not failures else f"{run_problem}\n\n{failures}"

    # What the `errors` port says when nothing could be LOCATED. It speaks only
    # when the run actually broke: a run that produced a verdict -- tests ran and
    # some failed -- did not break, so this stays empty and `failures` carries the
    # whole story. Without the total>0 guard every ordinary failing run would end
    # up with "the cocotb run exited with code 0" on a port meant for locations.
    cocotb_fallback = run_problem or str(results.get("error", ""))
    if not cocotb_fallback and not passed and int(results.get("total") or 0) == 0:
        cocotb_fallback = f"The cocotb run exited with code {code}"

    outputs.update({
        "results": json.dumps(slim_results(results), ensure_ascii=False),
        "failures": failures,
        "coverage": (json.dumps(coverage_summary, ensure_ascii=False)
                     if coverage_summary else ""),
        "sim_output": out.strip(),
        # Notes explain the run's shape (why cocotb, why no coverage, skipped
        # SVA); stderr is only worth surfacing when the run otherwise went well.
        "warnings": "\n".join(notes + ([err.strip()] if err.strip() and passed else [])),
        "passed": "true" if passed else "false",
        "status": "ok" if passed else "error",
        # WHERE the run broke, and nothing else. A clean build whose tests merely
        # failed has no error LOCATION, so this stays empty and `failures` carries
        # the story. The block's red/green comes from `status`/`passed`, never
        # from here, so an empty `errors` cannot make a failing run look clean.
        "errors": "" if passed else error_locations(
            verilator_stderr=joined, python_traceback=joined,
            sources=sources_map, test_file=f"{test_module}.py",
            fallback=cocotb_fallback,
        ),
        "log": "\n".join(log_parts),
    })
    return {"outputs": outputs, "_status": "ok" if passed else "error",
            "_stage": last_stage, "_globs": globs_for("verilator")}


# Scoped to run_openram, deliberately NOT folded into _EDA_ENV -- the same
# reason _COCOTB_ENV above is scoped: prepending a virtualenv to the PATH of
# every run is the classic way to break yosys and openroad from a distance.
#
# The defaults match Dockerfile.openram, but the image's own values win: a user
# pinning a different OpenRAM build through the `image` port gets that build's
# paths rather than this file's idea of them.  PYTHONUNBUFFERED is not cosmetic
# -- without it OpenRAM's phase banners arrive in one lump at exit, so a long
# compile shows a dead log tail for its whole duration.
_OPENRAM_ENV = (
    'export OPENRAM_HOME="${OPENRAM_HOME:-/opt/openram/compiler}"; '
    'export OPENRAM_TECH="${OPENRAM_TECH:-/opt/openram/technology}"; '
    'export PYTHONPATH="$OPENRAM_HOME:$PYTHONPATH"; '
    'export PYTHONUNBUFFERED=1; '
    'export PATH="/opt/openram-venv/bin:$PATH"; '
    # How the compiler is invoked.  OpenRAM has moved its entry point between
    # releases, so the IMAGE owns this: Dockerfile.openram writes it into
    # /etc/profile.d, which a `bash -lc` login shell sources -- the one channel
    # by which image-level configuration reaches an SSH exec, since Docker ENV
    # does not.  The ``:-`` keeps a pre-2026 image working.
    'export GRAFUX_OPENRAM_CMD='
    '"${GRAFUX_OPENRAM_CMD:-python3 $OPENRAM_HOME/../sram_compiler.py}"; '
)

# Unquoted on purpose: the value is several words and must split into a command
# plus its arguments.  It comes from the image or from this file, never from a
# port, so there is no untrusted text here.  ``EDA_OPENRAM_CMD`` on the devices
# server overrides it outright, which is the escape hatch when a pinned image
# turns out to be wrong and redeploying is faster than rebuilding.
_OPENRAM_CMD_DEFAULT = "$GRAFUX_OPENRAM_CMD"


def _sh_openram(command: str) -> str:
    """``_sh`` plus OpenRAM's paths and unbuffered Python output."""
    return _sh(_OPENRAM_ENV + command)


# The gain-cell compiler's environment: _OPENRAM_ENV's shape and reasoning, with
# OpenGCRAM's install paths, its own entry point (gain_cell_compiler.py, not
# sram_compiler.py), and the block's uploaded technology directory PREPENDED to
# OPENRAM_TECH -- OpenRAM accepts a colon list and imports the first match, so an
# uploaded tsmcN40 wins over anything the image carries.  The directory is
# created first because import_tech asserts every entry on the list exists.
_OPENGCRAM_ENV = (
    'mkdir -p ' + WORK_DIR + '/' + OPENGCRAM_TECH_DIR + '; '
    'export OPENRAM_HOME="${OPENRAM_HOME:-/opt/opengcram/compiler}"; '
    'export OPENRAM_TECH="' + WORK_DIR + '/' + OPENGCRAM_TECH_DIR
    + ':${OPENRAM_TECH:-/opt/opengcram/technology}"; '
    'export PYTHONPATH="$OPENRAM_HOME:$PYTHONPATH"; '
    'export PYTHONUNBUFFERED=1; '
    'export PATH="/opt/opengcram-venv/bin:$PATH"; '
    'export GRAFUX_OPENGCRAM_CMD='
    '"${GRAFUX_OPENGCRAM_CMD:-python3 $OPENRAM_HOME/../gain_cell_compiler.py}"; '
)

# Unquoted for the reason _OPENRAM_CMD_DEFAULT is.  ``EDA_OPENGCRAM_CMD`` on the
# devices server overrides it outright.
_OPENGCRAM_CMD_DEFAULT = "$GRAFUX_OPENGCRAM_CMD"


def _sh_opengcram(command: str) -> str:
    """``_sh`` plus OpenGCRAM's paths, the uploaded technology and unbuffered output."""
    return _sh(_OPENGCRAM_ENV + command)


def _read_pod_text(client, path: str, limit: Optional[int] = None) -> Tuple[str, bool]:
    """
    Read a generated file back for an inline port.

    Returns ``(text, truncated)``.  A file past the limit comes back empty and
    truncated, so the caller can point the port at the downloaded artifact
    instead -- the same read-or-attach rule ``run_yosys`` applies to a netlist.
    A missing file is reported, never raised: it is one port of many.

    ``limit`` resolves NETLIST_INLINE_MAX at CALL time rather than as a default
    argument: a default is bound at import, so the env-driven constant would
    freeze at whatever it was when this module first loaded.
    """
    if limit is None:
        limit = NETLIST_INLINE_MAX
    sftp = client.open_sftp()
    try:
        with sftp.open(path, "rb") as handle:
            data = handle.read(limit + 1)
    except Exception as exc:  # noqa: BLE001 -- a missing view is reported, not raised
        logger.warning("could not read back %s: %s", path, exc)
        return "", False
    finally:
        sftp.close()
    if len(data) > limit:
        return "", True
    return data.decode("utf-8", "replace"), False


def _truthy_port(req, field: str) -> bool:
    """A '1'/'true'/'yes'/'on' text port, read the way every EDA flag port is."""
    return (getattr(req, field, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _compile_and_collect_memory(
    client,
    req,
    *,
    kind: str,
    label: str,
    tech: str,
    name: str,
    config_text: str,
    notes: List[str],
    wrap: Callable[[str], str],
    command: str,
    fail: Callable[..., Dict[str, Any]],
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]],
    should_cancel: Optional[Callable[[], bool]],
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The compile and collect stages shared by openram and opengcram.

    OpenGCRAM is a fork of OpenRAM with the same output contract -- the same
    ``{output_path}{output_name}.{ext}`` files and the same config copy -- so the
    rules below are ONE set of rules, not two copies that drift: no ``tee``,
    ``ls -S`` order, the name-preferring classifier, and "a GDS or it failed".
    """
    out_dir = "{0}/{1}".format(WORK_DIR, OPENRAM_OUT_DIR)

    # ---- compile -----------------------------------------------------------
    on_stage("compile", "running")
    # No `| tee`: the compiler writes its own <output_name>.log into the output
    # directory, and a pipeline would hand the run tee's exit status instead of
    # the compiler's -- the exact trap the run_yosys comment below documents.
    code, out, err = exec_stream(
        client,
        wrap("mkdir -p {0} && cd {1} && {2} {3}".format(
            shlex.quote(out_dir), shlex.quote(WORK_DIR), command,
            shlex.quote(OPENRAM_CONFIG_FILE))),
        timeout=int(getattr(req, "timeout", 0) or 3600),
        on_line=on_line, should_cancel=should_cancel,
    )
    on_stage("compile", "done" if code == 0 else "failed")

    log_text = (out or "").strip()
    if code != 0:
        reason = (err or "").strip() or log_text or (
            "The {0} run was cancelled".format(label) if code == -1 else
            "The {0} run exceeded its timeout".format(label) if code == -2 else
            "{0} exited with code {1}".format(label, code)
        )
        result = fail("compile", reason, log_text)
        result["outputs"]["config"] = config_text
        result["outputs"]["warnings"] = "\n".join(notes)
        return result

    # ---- collect -----------------------------------------------------------
    on_stage("collect", "running")
    # `ls -S` (largest first) is load-bearing: classify_openram_outputs takes the
    # FIRST match per extension, which is how a multi-corner run's several .lib
    # files resolve to the substantive one.
    _code, listing, _err = exec_simple(
        client, _sh("ls -S -1 {0} 2>/dev/null".format(shlex.quote(out_dir))),
        timeout=60)
    found = classify_openram_outputs((listing or "").splitlines(), name)

    verilog, verilog_big = ("", False)
    if found.get("verilog_model"):
        verilog, verilog_big = _read_pod_text(
            client, "{0}/{1}".format(out_dir, found["verilog_model"]))
    resolved = ""
    if found.get("config"):
        resolved, _ = _read_pod_text(
            client, "{0}/{1}".format(out_dir, found["config"]))
    if verilog_big:
        notes.append(
            "The behavioural model exceeded the inline limit and was attached "
            "as the artifact {0} instead.".format(found["verilog_model"]))

    # A GDS is the deliverable, so a run that produced none is red even though
    # the compiler exited 0 -- "the tool said fine and built no macro" must not
    # read as success.  Unless the block asked for netlist_only, which is the
    # documented way to ask for exactly that.
    netlist_only = _truthy_port(req, "netlist_only")
    produced = bool(found.get("gds")) or (netlist_only and bool(found.get("spice")))

    stats = parse_openram_summary(req, tech=tech, output_name=name,
                                  log_text=log_text, found=found)
    stats.update(extra_stats or {})
    outputs: Dict[str, str] = {
        "top": name,
        "tech_name": tech,
        "verilog_model": verilog,
        # The config the compiler ACTUALLY ran, with every default it filled in
        # -- the only place the resolved defaults are visible, and the reason
        # this port is in both the input and the output list.
        "config": resolved or config_text,
        "stats": json.dumps(stats),
        "log": log_text,
        "reports": "",
        "status": "ok" if produced else "error",
        "errors": "" if produced else (
            "{0} finished without producing a macro. The log above is the "
            "whole story; the commonest causes are a tech_name the image does "
            "not carry and a parameter combination the technology cannot build."
            .format(label)
        ),
        "warnings": "\n".join(n for n in notes if n),
    }
    on_stage("collect", "done")
    return {"outputs": outputs, "_status": outputs["status"], "_stage": "collect",
            "_globs": globs_for(kind)}


def _memory_fail(kind: str, name: str, tech: str) -> Callable[..., Dict[str, Any]]:
    """The error result both memory runners return before or instead of a macro."""
    def _fail(stage: str, message: str, log_text: str = "") -> Dict[str, Any]:
        outputs = {
            "top": name, "tech_name": tech, "status": "error",
            "errors": message, "warnings": "", "log": log_text,
            "verilog_model": "", "config": "", "stats": "{}", "reports": "",
        }
        return {"outputs": outputs, "_status": "error", "_stage": stage,
                "_globs": globs_for(kind)}
    return _fail


def run_openram(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Compile an SRAM macro from memory parameters and collect its views."""
    out_dir = "{0}/{1}".format(WORK_DIR, OPENRAM_OUT_DIR)
    tech = (getattr(req, "tech_name", "") or "").strip() or DEFAULT_OPENRAM_TECH
    name = openram_output_name(req, tech)
    _fail = _memory_fail("openram", name, tech)

    # ---- stage 1: config ---------------------------------------------------
    on_stage("config", "running")
    notes, refusal = openram_size_warnings(req)
    if refusal:
        on_stage("config", "failed")
        return _fail("config", refusal)

    config_text = build_openram_config(req, tech=tech, output_name=name,
                                       out_dir=out_dir)
    if (getattr(req, "config", "") or "").strip():
        notes.append(
            "The `config` port was set, so every memory-parameter port was "
            "ignored and that config was used as written (output_path and "
            "output_name excepted -- the server reads the results from there)."
        )
    if _truthy_port(req, "check_lvsdrc"):
        notes.append(
            "check_lvsdrc is on. DRC and LVS need magic and netgen, which the "
            "default openram image does not carry -- pin an image that does on "
            "the `image` port, or the run fails at the checking step."
        )
    sftp = client.open_sftp()
    try:
        _write_file(sftp, "{0}/{1}".format(WORK_DIR, OPENRAM_CONFIG_FILE), config_text)
    finally:
        sftp.close()
    on_stage("config", "done")

    command = (os.environ.get("EDA_OPENRAM_CMD", "") or "").strip() or _OPENRAM_CMD_DEFAULT
    return _compile_and_collect_memory(
        client, req, kind="openram", label="OpenRAM", tech=tech, name=name,
        config_text=config_text, notes=notes, wrap=_sh_openram, command=command,
        fail=_fail, on_stage=on_stage, on_line=on_line, should_cancel=should_cancel,
    )


def run_opengcram(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Compile a gain-cell memory macro with OpenGCRAM and collect its views.

    One stage more than run_openram: ``tech``, which unpacks the block's
    technology archive and proves the gain cell is in it BEFORE the compiler
    starts.  Upstream ships no technology that has one, so without that check
    the commonest run would fail deep inside the compiler with a KeyError or a
    missing-GDS traceback that says nothing about the actual problem.
    """
    out_dir = "{0}/{1}".format(WORK_DIR, OPENRAM_OUT_DIR)
    tech_dir = "{0}/{1}".format(WORK_DIR, OPENGCRAM_TECH_DIR)
    override = bool((getattr(req, "config", "") or "").strip())
    tech = (getattr(req, "tech_name", "") or "").strip() or DEFAULT_OPENGCRAM_TECH
    name = opengcram_output_name(req, tech)
    _fail = _memory_fail("opengcram", name, tech)

    # Everything decidable from the request alone is refused before a single
    # command reaches the pod.
    notes, refusal = opengcram_preflight(req)
    if refusal:
        on_stage("tech", "failed")
        return _fail("tech", refusal)
    gc_type, _ = normalize_gc_type(getattr(req, "gc_type", ""))

    # ---- stage 1: tech -----------------------------------------------------
    on_stage("tech", "running")
    archive, archive_error = find_staged_archive(req)
    if archive_error:
        on_stage("tech", "failed")
        return _fail("tech", archive_error)

    if archive:
        scratch = "{0}/_extract".format(tech_dir)
        lower = archive.lower()
        unpack = ("python3 -m zipfile -e {0} {1}" if lower.endswith(".zip")
                  else "tar -xf {0} -C {1}").format(shlex.quote(archive), shlex.quote(scratch))
        code, listing, err = exec_simple(client, _sh(
            "rm -rf {0} && mkdir -p {0} && {1} && ls -1A {0}".format(
                shlex.quote(scratch), unpack)), timeout=600)
        if code != 0:
            on_stage("tech", "failed")
            return _fail("tech", "Could not unpack the technology archive {0}: {1}".format(
                archive.rsplit("/", 1)[-1], (err or listing or "").strip()))
        resolved_tech, subdir, layout_refusal = resolve_tech_layout(
            (listing or "").splitlines(),
            tech_port=(getattr(req, "tech_name", "") or "").strip(),
            stem=archive_stem(archive))
        if layout_refusal:
            on_stage("tech", "failed")
            return _fail("tech", layout_refusal)
        target = "{0}/{1}".format(tech_dir, resolved_tech)
        source = "{0}/{1}".format(scratch, subdir) if subdir else scratch
        code, _out, err = exec_simple(client, _sh(
            "rm -rf {0} && mv {1} {0} && rm -rf {2}".format(
                shlex.quote(target), shlex.quote(source), shlex.quote(scratch))),
            timeout=120)
        if code != 0:
            on_stage("tech", "failed")
            return _fail("tech", "Could not install the technology {0}: {1}".format(
                resolved_tech, (err or "").strip()))
        if resolved_tech != tech:
            tech = resolved_tech
            name = opengcram_output_name(req, tech)
            _fail = _memory_fail("opengcram", name, tech)

    if override:
        notes.append(
            "The `config` port was set, so every memory-parameter port was "
            "ignored and that config was used as written (output_path and "
            "output_name excepted -- the server reads the results from there). "
            "The gain-cell check was skipped: the technology and gc_type are "
            "whatever that config says."
        )
    else:
        _code, cells, _err = exec_simple(
            client, _sh_opengcram(gain_cell_probe_command(tech)), timeout=60)
        cell_refusal = check_gain_cell_listing(
            cells, tech=tech, gc_type=gc_type,
            netlist_only=_truthy_port(req, "netlist_only"))
        if cell_refusal:
            on_stage("tech", "failed")
            return _fail("tech", cell_refusal)
    on_stage("tech", "done")

    # ---- stage 2: config ---------------------------------------------------
    on_stage("config", "running")
    config_text = build_openram_config(req, tech=tech, output_name=name,
                                       out_dir=out_dir, gain_cell=True)
    if _truthy_port(req, "check_lvsdrc"):
        notes.append(
            "check_lvsdrc is on. OpenGCRAM's DRC and LVS decks are Calibre's, "
            "which no public image can carry -- pin an image that does on the "
            "`image` port, or the run fails at the checking step."
        )
    sftp = client.open_sftp()
    try:
        _write_file(sftp, "{0}/{1}".format(WORK_DIR, OPENRAM_CONFIG_FILE), config_text)
    finally:
        sftp.close()
    on_stage("config", "done")

    rw, r, w = gain_cell_ports(req)
    extra_stats: Dict[str, Any] = {"gc_type": gc_type, "num_rw_ports": rw,
                                   "num_r_ports": r, "num_w_ports": w}
    command = (os.environ.get("EDA_OPENGCRAM_CMD", "") or "").strip() or _OPENGCRAM_CMD_DEFAULT
    return _compile_and_collect_memory(
        client, req, kind="opengcram", label="OpenGCRAM", tech=tech, name=name,
        config_text=config_text, notes=notes, wrap=_sh_opengcram, command=command,
        fail=_fail, on_stage=on_stage, on_line=on_line, should_cancel=should_cancel,
        extra_stats=None if override else extra_stats,
    )


# ngspice's environment.  The defaults match Dockerfile.ngspice; the image's own
# profile.d values win, so a user pinning another build through the `image` port
# gets that build's paths.  OMP_NUM_THREADS feeds the OpenMP model evaluation
# the image is built with.
_NGSPICE_ENV = (
    'export PDK_ROOT="${PDK_ROOT:-' + ngspice.DEFAULT_PDK_ROOT + '}"; '
    'export GRAFUX_NGSPICE_CMD="${GRAFUX_NGSPICE_CMD:-ngspice}"; '
    'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"; '
)

# Unquoted for the reason _OPENRAM_CMD_DEFAULT is.  ``EDA_NGSPICE_CMD`` on the
# devices server overrides it outright.
_NGSPICE_CMD_DEFAULT = "$GRAFUX_NGSPICE_CMD"

# How much of ngspice's log is read back.  The log's tail is where the verdict,
# the measurements and the errors are; a runaway `print` in a user's .control
# block must not become a megabyte port.
NGSPICE_LOG_TAIL_BYTES = 256 * 1024


def _sh_ngspice(command: str) -> str:
    """``_sh`` plus ngspice's PDK root, entry point and thread count."""
    return _sh(_NGSPICE_ENV + command)


def _ngspice_probe(text: str) -> Tuple[str, int, str]:
    """``(pdk_root, nproc, version)`` from the deck stage's one-shot probe."""
    root, nproc = "", 0
    for line in (text or "").splitlines():
        if line.startswith("ROOT:"):
            root = line[5:].strip()
        elif line.startswith("NPROC:"):
            try:
                nproc = int(line[6:].strip())
            except ValueError:
                nproc = 0
    return root or ngspice.DEFAULT_PDK_ROOT, nproc, ngspice.parse_version(text)


def _analyses_summary(plots: Sequence[Dict[str, Any]], requested: Sequence[str]) -> str:
    """The `analyses` output port: what actually ran, with its point counts."""
    if not plots:
        return "\n".join("{0} -- no results".format(a) for a in requested)
    lines = []
    for plot in plots:
        vectors = plot.get("vectors") or []
        lines.append("{0}: {1} point{2}, {3} vector{4}".format(
            plot.get("name", "?"), plot.get("points", 0),
            "" if plot.get("points", 0) == 1 else "s",
            len(vectors), "" if len(vectors) == 1 else "s"))
    return "\n".join(lines)


def run_analogue_simulator(
    client,
    req,
    *,
    on_stage: Callable[[str, str], None],
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Simulate a SPICE deck with ngspice and turn its log and rawfile into ports.

    Three stages -- ``deck`` (preflight, resolve the PDK on the pod, write the
    deck), ``simulate`` and ``collect``.  The verdict is NOT ngspice's exit code:
    it exits 0 after dropping a malformed element line and 1 only for some load
    failures, so ``status`` needs a clean log AND results (see
    ngspice.classify_ngspice_log for which log lines are fatal and why).
    """
    pdk, _ = ngspice.normalize_pdk(getattr(req, "pdk", ""))
    corner, _ = ngspice.normalize_corner(pdk, getattr(req, "corner", ""))
    work = WORK_DIR

    def _fail(stage: str, message: str, *, log_text: str = "", deck: str = "",
              notes: Optional[List[str]] = None) -> Dict[str, Any]:
        outputs = {
            "status": "error", "errors": message,
            "warnings": "\n".join(n for n in (notes or []) if n),
            "log": log_text, "netlist": deck, "measurements": "{}",
            "waveforms": "", "operating_point": "{}", "analyses": "",
            "stats": json.dumps({"simulator": "ngspice", "pdk": pdk, "corner": corner}),
            "raw": "",
        }
        return {"outputs": outputs, "_status": "error", "_stage": stage,
                "_globs": globs_for("analogue_simulator")}

    # ---- stage 1: deck -----------------------------------------------------
    on_stage("deck", "running")
    refusal = ngspice.analogue_preflight(req)
    if refusal:
        on_stage("deck", "failed")
        return _fail("deck", refusal)

    _code, probe, _err = exec_simple(client, _sh_ngspice(
        'printf "ROOT:%s\\nNPROC:%s\\n" "$PDK_ROOT" "$(nproc)"; '
        '$GRAFUX_NGSPICE_CMD -v 2>&1 | head -5'), timeout=60)
    pdk_root, nproc, version = _ngspice_probe(probe)

    netlist = getattr(req, "netlist", "") or ""
    if pdk in ngspice.NGSPICE_PDKS and not ngspice.references_pdk_library(netlist, pdk):
        paths = ngspice.pdk_library_paths(pdk, pdk_root)
        _code, missing, _err = exec_simple(client, _sh(
            "for f in {0}; do test -f \"$f\" || echo \"MISSING:$f\"; done".format(
                " ".join(shlex.quote(p) for p in paths))), timeout=60)
        gone = [ln[8:] for ln in (missing or "").splitlines() if ln.startswith("MISSING:")]
        if gone:
            on_stage("deck", "failed")
            return _fail("deck", (
                "This image does not carry the {0} device models (missing: {1}). Leave "
                "the image port empty to use the default analogue_simulator image, or "
                "set pdk to none for a deck that loads its own models.").format(
                    pdk, ", ".join(gone)))

    deck, analyses_run, notes = ngspice.build_ngspice_deck(req, pdk_root=pdk_root)
    script_path = ngspice.__file__
    with open(script_path, "r", encoding="utf-8") as fh:
        script_text = fh.read()
    sftp = client.open_sftp()
    try:
        _write_file(sftp, "{0}/{1}".format(work, ngspice.DECK_FILE), deck)
        _write_file(sftp, "{0}/{1}".format(work, ngspice.SPICEINIT_FILE),
                    ngspice.spiceinit_for(pdk, threads=nproc))
        _write_file(sftp, "{0}/{1}".format(work, ngspice.SCRIPT_FILE), script_text)
    finally:
        sftp.close()
    on_stage("deck", "done")

    # ---- stage 2: simulate -------------------------------------------------
    on_stage("simulate", "running")
    command = (os.environ.get("EDA_NGSPICE_CMD", "") or "").strip() or _NGSPICE_CMD_DEFAULT
    # `tee` WITH pipefail.  The live log tail needs ngspice's output as it
    # happens, and the measurements need all of it (exec_stream keeps only a
    # tail), so the output goes both ways.  pipefail is what makes the pipeline
    # report ngspice's exit status rather than tee's -- the trap run_yosys
    # documents.  The rawfile is deleted first because the deck appends to it.
    code, _out, err = exec_stream(
        client,
        _sh_ngspice("cd {0} && rm -f {1} {2} && set -o pipefail && {3} -b {4} 2>&1 | tee {2}".format(
            shlex.quote(work), ngspice.RAW_FILE, ngspice.LOG_FILE, command, ngspice.DECK_FILE)),
        timeout=int(getattr(req, "timeout", 0) or 900),
        on_line=on_line, should_cancel=should_cancel,
    )
    if code in (-1, -2):
        on_stage("simulate", "failed")
        return _fail("simulate", "The simulation was cancelled." if code == -1 else (
            "The simulation exceeded its {0}s timeout. Shorten the analysis (a smaller "
            "stop time or a coarser step), or raise the timeout port.").format(
                int(getattr(req, "timeout", 0) or 900)),
            log_text=(_out or "").strip(), deck=deck, notes=notes)
    on_stage("simulate", "done" if code == 0 else "failed")

    # ---- stage 3: collect --------------------------------------------------
    on_stage("collect", "running")
    _c, log_text, _e = exec_simple(client, _sh("tail -c {0} {1}/{2} 2>/dev/null".format(
        NGSPICE_LOG_TAIL_BYTES, shlex.quote(work), ngspice.LOG_FILE)), timeout=60)
    log_text = (log_text or _out or err or "").strip()

    max_points = ngspice._int_or(getattr(req, "max_points", ""), ngspice.DEFAULT_MAX_POINTS)
    probes = ngspice.split_probes(getattr(req, "probes", ""))
    _c, post_out, post_err = exec_simple(client, _sh_ngspice(
        "cd {0} && python3 {1} postprocess {2} {3} {4}".format(
            shlex.quote(work), ngspice.SCRIPT_FILE, ngspice.RAW_FILE, max_points,
            shlex.quote(json.dumps(probes)))), timeout=600)
    try:
        post = json.loads(post_out or "{}")
    except ValueError:
        post = {}
        notes.append("Could not read the simulation results back: {0}".format(
            (post_err or post_out or "no output").strip()[:500]))

    measurements, meas_notes = ngspice.parse_measurements(
        log_text, ngspice.measure_names(deck))
    errors, ng_warnings, hints = ngspice.classify_ngspice_log(log_text, code)
    plots = post.get("plots") or []
    produced = any((p.get("points") or 0) > 0 for p in plots)
    own_control = ngspice.has_control_block(netlist)
    if not errors and not produced and not own_control:
        errors.append("ngspice finished without writing any results. The log is the whole "
                      "story; the commonest cause is a deck that failed to load.")
    if own_control and not produced:
        notes.append("The deck's own .control block wrote no sim.raw, so the waveforms "
                     "and operating_point ports are empty.")

    raw_bytes = int(post.get("raw_bytes") or 0)
    if raw_bytes > 32 * 1024 * 1024:
        notes.append("The rawfile is {0:.0f} MB; artifacts past the download cap arrive "
                     "truncated. The waveforms port is unaffected.".format(raw_bytes / 1e6))

    status = "error" if errors else "ok"
    error_text = "\n".join(errors)
    if hints:
        error_text += "\n\nLikely cause:\n" + "\n".join("- " + h for h in hints)

    stats = {
        "simulator": "ngspice", "version": version, "pdk": pdk, "corner": corner,
        "temperature": (getattr(req, "temperature", "") or "").strip() or "27",
        "analyses": analyses_run, "plots": plots,
        "waveform_plot": post.get("waveform_plot", ""),
        "measurements": len(measurements),
        "measurements_failed": sum(1 for v in measurements.values() if v is None),
        "raw_bytes": raw_bytes, "exit_code": code,
    }
    outputs: Dict[str, str] = {
        "status": status,
        "netlist": deck,
        "measurements": json.dumps(measurements),
        "waveforms": post.get("waveforms", "") or "",
        "operating_point": json.dumps(post.get("operating_point") or {}),
        "analyses": _analyses_summary(plots, analyses_run),
        "stats": json.dumps(stats),
        "errors": error_text,
        "warnings": "\n".join(n for n in (notes + meas_notes + list(post.get("notes") or [])
                                         + ng_warnings) if n),
        "log": log_text,
        "raw": ngspice.RAW_FILE if produced else "",
    }
    on_stage("collect", "done")
    return {"outputs": outputs, "_status": status, "_stage": "collect",
            "_globs": globs_for("analogue_simulator")}


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

    # `yosys -l` writes the log itself. The obvious `yosys ... | tee yosys.log`
    # is a trap: a shell pipeline exits with the status of its LAST command, so
    # tee's 0 masks a failing yosys entirely and the run reports the downstream
    # symptom ("produced no netlist") instead of the actual error.
    code, out, err = exec_stream(
        client, _sh(f"cd {WORK_DIR} && yosys -l yosys.log -s synth.ys"),
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
    # The design source ORFS will synthesize.
    #
    # OpenROAD-flow-scripts always runs its own yosys: ``1_synth.odb`` depends on
    # the yosys chain unconditionally and there is no variable to bypass it. So a
    # netlist wired in from an upstream yosys block cannot skip synthesis -- it is
    # simply used as the source instead (gate-level Verilog is still Verilog, and
    # the cells resolve against the platform's liberty).
    #
    # `rtl` therefore wins when both are wired: re-synthesizing from RTL is what
    # ORFS is built to do, whereas re-synthesizing an already-mapped netlist is
    # the less-travelled path. The upstream yosys block remains valuable for what
    # it reports -- the netlist to inspect, cell counts, area -- rather than as a
    # way to save the flow work.
    rtl = (req.rtl or "").strip()
    netlist = (req.netlist or "").strip()
    source_text = rtl or netlist
    stages = stages_between(req.from_stage, req.to_stage)

    # Nothing to build. Caught before the first make rather than out on the pod,
    # because ORFS's own failure for a zero-byte source is a yosys parse error
    # that says nothing about which port the user forgot to wire.
    if not source_text:
        return {
            "outputs": {
                "top": design, "pdk": platform, "stage": "", "status": "error",
                "metrics": "{}", "log": "", "warnings": "",
                "errors": (
                    "No design source: the 'rtl' and 'netlist' input ports are "
                    "both empty. Wire a code block's output into 'rtl', or a "
                    "yosys block's 'netlist' output into 'netlist'."
                ),
            },
            "_status": "error",
            "_stage": "",
            "_stages_done": [],
            "_globs": globs_for("openroad", platform=platform, design=design),
        }

    # config.mk is read by make, which does not expand $FLOW_HOME inside the file
    # the same way the shell does, so paths inside it use make's own variable.
    cfg_design_dir = f"./designs/{platform}/{design}"
    cfg_src_dir = f"./designs/src/{design}"

    # Reconcile clock_port against the ports the design actually declares before
    # writing the SDC. A create_clock aimed at a port that does not exist does not
    # fail — OpenSTA substitutes a virtual clock, CTS then builds no clock tree,
    # and nothing says so until someone reads the log line by line. A user-supplied
    # SDC is used verbatim; it is theirs, not ours to second-guess.
    preflight: List[str] = []
    if req.sdc.strip():
        sdc = req.sdc.strip()
    else:
        clock_port, clock_note = resolve_clock_port(
            source_text, design, req.clock_port)
        if clock_note:
            preflight.append(clock_note)
        sdc = (default_sdc(clock_port, req.clock_period) if clock_port
               else virtual_clock_sdc(req.clock_period))
    config = build_orfs_config(
        design=design,
        platform=platform,
        verilog_files=[f"{cfg_src_dir}/{design}.v"],
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
        # Named after the design so yosys' DECLFILENAME check stays quiet, matching
        # what the verilator runner does.
        _write_file(sftp, f"{abs_src_dir}/{design}.v", source_text)
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
            # `make synth` exits 0 even on a design that synthesized to nothing.
            # Stop here rather than spend four more stages to fail at route with
            # an error naming the wrong tool — see synth_produced_nothing.
            empty = ("" if stage != "synth" or _allow_empty_netlist()
                     else synth_produced_nothing(f"{out}\n{err}", design))
            if empty:
                on_stage(stage, "failed")
                failed = True
                outputs["errors"] = (
                    f"Stage 'synth' produced no netlist.\n\n{empty}")
                break
            stages_done.append(stage)
            on_stage(stage, "done")
            continue

        on_stage(stage, "failed")
        failed = True
        if code == -1:
            outputs["errors"] = f"Stage '{stage}' was cancelled."
        elif code == -2:
            outputs["errors"] = (
                f"Stage '{stage}' exceeded its {stage_timeout(stage)}s timeout.")
        else:
            # make prints only "Error 2"; the actual diagnostic is in the stage log.
            _c, log_text, _e = exec_simple(
                client,
                _sh(f"cat $(ls -t $FLOW_HOME/logs/{platform}/{design}/base/*.log "
                    f"2>/dev/null | head -1) 2>/dev/null || true"),
                timeout=60,
            )
            detail = explain_orfs_failure(log_text)
            outputs["errors"] = (
                f"Stage '{stage}' failed (exit {code}).\n"
                + (detail or err.strip() or out.strip()[-4000:])
            )
        break

    # Metrics and reports are collected whether or not the flow completed — a
    # failed route is exactly when the user most needs to see the numbers.
    # ORFS writes its metrics under logs/, not reports/, and names the file after
    # the step that produced it -- a completed flow leaves 6_report.json, while a
    # partial run (from_stage/to_stage) leaves only that stage's json. Prefer the
    # summary report, then fall back to the most recent stage json so a partial
    # run still reports numbers.
    log_dir = f"$FLOW_HOME/logs/{platform}/{design}/base"
    metrics: Dict[str, Any] = {}
    _c, meta, _e = exec_simple(
        client,
        _sh(
            f'f=$(ls -t {log_dir}/*report*.json 2>/dev/null | head -1); '
            f'[ -z "$f" ] && f=$(ls -t {log_dir}/*.json 2>/dev/null | head -1); '
            f'[ -n "$f" ] && cat "$f" || true'
        ),
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
    scraped = "\n".join(
        line for part in log_parts for line in part.splitlines()
        if "warning" in line.lower()
    )
    # Preflight notes lead: a clock_port that matches nothing explains a good
    # share of the warnings underneath it.
    outputs["warnings"] = "\n".join(preflight + [scraped]).strip()[:20000]
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
