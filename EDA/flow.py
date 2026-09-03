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
import json
import logging
import os
import re
import shlex
import xml.etree.ElementTree as ET
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

    for case in root.iter("testcase"):
        status = "passed"
        message = ""
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
        summary["tests"].append({
            "name": (case.get("name") or "").strip(),
            "classname": (case.get("classname") or "").strip(),
            "status": status,
            "time": elapsed,
            "message": message,
        })
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


def summarize_failures(
    results: Dict[str, Any],
    *,
    max_tests: int = 10,
    max_chars: int = 4000,
) -> str:
    """
    The feedback payload: what failed, named, with the assertion message.

    This text is what a human reads on the ``failures`` port AND what gets wired
    into the code block's ``feedback`` port to drive an RTL repair, so it names the
    test and quotes its assertion rather than dumping a log — the fix prompt has to
    be able to tell which behaviour was wrong.

    Returns "" when nothing failed, so an empty ``failures`` port means "clean".
    """
    if not results:
        return ""
    if results.get("error") and not results.get("tests"):
        return str(results["error"])

    failed = [t for t in results.get("tests", []) if t.get("status") == "failed"]
    if not failed:
        return str(results.get("error") or "")

    total = int(results.get("total", 0) or 0)
    header = f"{len(failed)} of {total} cocotb tests failed."
    parts = [header, ""]
    for test in failed[:max_tests]:
        name = test.get("name") or "(unnamed test)"
        parts.append(f"FAILED {name}")
        message = " ".join(str(test.get("message", "")).split())
        if message:
            parts.append(f"  {message}")
    if len(failed) > max_tests:
        parts.append(f"... and {len(failed) - max_tests} more failing tests.")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n... (truncated)"
    return text


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

    sftp = client.open_sftp()
    try:
        _write_file(sftp, f"{WORK_DIR}/{source}", req.rtl or "")
        _write_file(sftp, f"{WORK_DIR}/{test_module}.py", req.testbench or "")
        if sva:
            _write_file(sftp, f"{WORK_DIR}/sva.sv", sva)
        _write_file(sftp, f"{WORK_DIR}/run_cocotb.py", script)
    finally:
        sftp.close()

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

    # results.xml is the verdict, NOT the exit code — see parse_cocotb_results.
    _rc, xml_text, _re = exec_simple(
        client, _sh(f"cat {WORK_DIR}/{COCOTB_RESULTS_XML} 2>/dev/null"), timeout=60)
    results = parse_cocotb_results(xml_text)
    passed = results["failed"] == 0 and results["total"] > 0 and not build_failed

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

    failures = summarize_failures(results)
    if build_failed:
        hint = (
            "The design did not build, so no test ran. The build log is above."
        )
        if sva:
            hint += (
                "  The `sva` port was wired in and is compiled alongside the "
                "design — unbind it to rule the assertions out as the cause."
            )
        failures = hint if not failures else f"{hint}\n\n{failures}"

    outputs.update({
        "results": json.dumps(results, ensure_ascii=False),
        "failures": failures,
        "coverage": (json.dumps(coverage_summary, ensure_ascii=False)
                     if coverage_summary else ""),
        "sim_output": out.strip(),
        # Notes explain the run's shape (why cocotb, why no coverage, skipped
        # SVA); stderr is only worth surfacing when the run otherwise went well.
        "warnings": "\n".join(notes + ([err.strip()] if err.strip() and passed else [])),
        "passed": "true" if passed else "false",
        "status": "ok" if passed else "error",
        "errors": "" if passed else (
            failures or str(results.get("error", ""))
            or err.strip() or f"The cocotb run exited with code {code}"
        ),
        "log": "\n".join(log_parts),
    })
    return {"outputs": outputs, "_status": "ok" if passed else "error",
            "_stage": last_stage, "_globs": globs_for("verilator")}


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
