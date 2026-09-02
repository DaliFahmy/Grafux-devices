"""
test_eda_flow.py
Unit tests for the pure half of ``EDA.flow`` — script/config generation and log
parsing.

Everything here is string-in/string-out, so these tests need no cloud account, no
container and no SSH.  That is deliberate: the tool-specific domain knowledge
(what a yosys script must say to map onto a liberty file, how ORFS reports a cell
count) is the part most likely to be wrong and the part cheapest to pin down.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from EDA import flow  # noqa: E402
from EDA.models import ORFS_STAGES  # noqa: E402


# ---------------------------------------------------------------------------
# stages_between
# ---------------------------------------------------------------------------

def test_stages_between_full_flow():
    assert flow.stages_between("synth", "final") == ORFS_STAGES


def test_stages_between_partial():
    assert flow.stages_between("floorplan", "place") == ("floorplan", "place")


def test_stages_between_single():
    assert flow.stages_between("route", "route") == ("route",)


def test_stages_between_unknown_bounds_fall_back_to_full_flow():
    """A typo in a port must not fail the run before a tool has even started."""
    assert flow.stages_between("nonsense", "") == ORFS_STAGES


def test_stages_between_reversed_bounds_collapse():
    """to_stage before from_stage yields just the start, never an empty flow."""
    assert flow.stages_between("route", "floorplan") == ("route",)


# ---------------------------------------------------------------------------
# infer_top_module
# ---------------------------------------------------------------------------

def test_infer_top_module_picks_last_module():
    """Verilog convention puts submodules above the thing that instantiates them."""
    rtl = "module half_adder(); endmodule\nmodule full_adder(); endmodule\n"
    assert flow.infer_top_module(rtl) == "full_adder"


def test_infer_top_module_handles_parameters_and_ports():
    rtl = "module  counter #(parameter W=8) (input clk, output [W-1:0] q);\nendmodule"
    assert flow.infer_top_module(rtl) == "counter"


def test_infer_top_module_falls_back_when_no_module():
    assert flow.infer_top_module("") == "top"
    assert flow.infer_top_module("// just a comment", fallback="mydesign") == "mydesign"


# ---------------------------------------------------------------------------
# Verilator command construction
# ---------------------------------------------------------------------------

def test_build_verilator_cmd_lint_mode():
    cmd = flow.build_verilator_cmd(
        top="counter", mode="lint", sources=["design.v"], defines="W=8"
    )
    assert "--lint-only" in cmd
    assert "-Wall" in cmd
    assert "+define+W=8" in cmd
    assert "--top-module counter" in cmd
    # Lint must not try to build or link a binary.
    assert "--build" not in cmd


def test_build_verilator_cmd_sim_mode_with_trace():
    cmd = flow.build_verilator_cmd(
        top="counter", mode="sim", sources=["design.v"],
        testbench_file="tb.cpp", trace=True,
    )
    assert "--cc" in cmd and "--exe" in cmd and "--build" in cmd
    assert "--trace" in cmd
    assert cmd.endswith("design.v tb.cpp")


def test_build_verilator_cmd_sim_without_trace_omits_flag():
    cmd = flow.build_verilator_cmd(
        top="c", mode="sim", sources=["design.v"], testbench_file="tb.cpp", trace=False
    )
    assert "--trace" not in cmd


def test_build_verilator_cmd_lint_ignores_testbench():
    """A C++ harness is meaningless to a lint-only run and must not be passed."""
    cmd = flow.build_verilator_cmd(
        top="c", mode="lint", sources=["design.v"], testbench_file="tb.cpp"
    )
    assert "tb.cpp" not in cmd


def test_default_testbench_references_the_generated_class():
    tb = flow.default_testbench("counter", trace=True)
    assert '#include "Vcounter.h"' in tb
    assert "Vcounter* dut = new Vcounter" in tb
    assert "sim.vcd" in tb
    assert "int main(" in tb


def test_default_testbench_without_trace_has_no_vcd():
    tb = flow.default_testbench("counter", trace=False)
    assert "verilated_vcd_c.h" not in tb
    assert "sim.vcd" not in tb


# ---------------------------------------------------------------------------
# Yosys script construction
# ---------------------------------------------------------------------------

def test_build_yosys_script_maps_onto_liberty():
    script = flow.build_yosys_script(
        top="counter", sources=["design.v"], liberty="/pdk/sky130.lib",
        netlist_out="netlist.v", stat_out="stat.txt",
    )
    assert "read_verilog -sv design.v" in script
    assert "hierarchy -check -top counter" in script
    assert "synth -top counter" in script
    assert "dfflibmap -liberty /pdk/sky130.lib" in script
    assert "abc -liberty /pdk/sky130.lib" in script
    assert "write_verilog -noattr netlist.v" in script
    assert "tee -o stat.txt stat -liberty /pdk/sky130.lib" in script


def test_build_yosys_script_without_liberty_stays_generic():
    """No liberty file must degrade to generic synthesis, not fail the run."""
    script = flow.build_yosys_script(
        top="counter", sources=["design.v"], liberty="", netlist_out="netlist.v",
    )
    assert "abc -liberty" not in script
    assert "dfflibmap" not in script
    assert "synth -top counter" in script
    assert "write_verilog -noattr netlist.v" in script


def test_build_yosys_script_emits_stat_exactly_once():
    """A duplicated stat would double the parsed cell counts."""
    script = flow.build_yosys_script(
        top="c", sources=["design.v"], liberty="/l.lib",
        netlist_out="n.v", stat_out="stat.txt",
    )
    assert len([ln for ln in script.splitlines() if "stat" in ln]) == 1


def test_build_yosys_script_passes_defines_and_includes():
    script = flow.build_yosys_script(
        top="c", sources=["design.v"], liberty="", netlist_out="n.v",
        defines="W=8 DEBUG", include_dirs="/rtl /inc",
    )
    assert "-DW=8" in script and "-DDEBUG" in script
    assert "-I/rtl" in script and "-I/inc" in script


# ---------------------------------------------------------------------------
# Liberty selection
# ---------------------------------------------------------------------------

# `ls -S` order (largest first) for a real sky130hd platform directory.
_SKY130_LIBS = [
    "/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
    "/flow/platforms/sky130hd/lib/sky130_dummy_io.lib",
]


def test_pick_liberty_skips_the_io_stub():
    """
    The bug this exists to prevent.

    sky130hd's lib/ sorts sky130_dummy_io.lib ahead of the standard cells
    alphabetically. Mapping onto it makes dfflibmap fail with "dffs with async set
    or reset are not supported" -- an error that reads like the user's RTL is at
    fault when really we handed yosys a library with no flip-flops in it.
    """
    assert flow.pick_liberty(_SKY130_LIBS).endswith("sky130_fd_sc_hd__tt_025C_1v80.lib")


def test_pick_liberty_skips_macro_and_fill_libraries():
    libs = [
        "/p/lib/sky130_sram_1kbyte.lib",
        "/p/lib/foo_fill.lib",
        "/p/lib/foo_antenna.lib",
        "/p/lib/NangateOpenCellLibrary_typical.lib",
    ]
    assert flow.pick_liberty(libs).endswith("NangateOpenCellLibrary_typical.lib")


def test_pick_liberty_prefers_the_largest_remaining_file():
    """Callers pass `ls -S` output, so position encodes size."""
    libs = ["/p/lib/big_sc.lib", "/p/lib/small_sc.lib"]
    assert flow.pick_liberty(libs) == "/p/lib/big_sc.lib"


def test_pick_liberty_falls_back_rather_than_giving_up():
    """
    An unfamiliar platform whose every file looks like a support library should
    still synthesize onto something, not silently drop to generic cells.
    """
    libs = ["/p/lib/only_io.lib", "/p/lib/other_ram.lib"]
    assert flow.pick_liberty(libs) == "/p/lib/only_io.lib"


def test_pick_liberty_on_empty_listing():
    assert flow.pick_liberty([]) == ""
    assert flow.pick_liberty(["", "  "]) == ""


# ---------------------------------------------------------------------------
# Source filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top, expected", [
    ("counter", "counter"),
    ("my_module", "my_module"),
    ("gen$block", "gen_block"),     # $ is legal in Verilog, awkward in a path
    ("a-b c", "a_b_c"),
    ("", "top"),
])
def test_safe_filename(top, expected):
    """
    Verilator's -Wall makes DECLFILENAME fatal, so the source file must be named
    after the top module -- writing every design to a fixed design.v made lint
    fail on every single input.
    """
    assert flow._safe_filename(top) == expected


# ---------------------------------------------------------------------------
# ORFS config construction
# ---------------------------------------------------------------------------

def test_build_orfs_config_has_no_bypass_synthesis_option():
    """
    ORFS always runs its own yosys.

    ``1_synth.odb`` depends on the yosys chain unconditionally and there is no
    variable to skip it, so emitting something like SYNTH_NETLIST would be
    silently ignored -- the flow would synthesize anyway and the config would be
    lying about what it does. The source is always VERILOG_FILES.
    """
    cfg = flow.build_orfs_config(
        design="counter", platform="sky130hd",
        verilog_files=["./designs/src/counter/counter.v"],
        sdc_file="./designs/sky130hd/counter/constraint.sdc",
        clock_period="10", core_utilization="45", aspect_ratio="1",
    )
    assert "export DESIGN_NAME     = counter" in cfg
    assert "export PLATFORM        = sky130hd" in cfg
    assert "SYNTH_NETLIST" not in cfg
    assert "VERILOG_FILES" in cfg
    assert "export CORE_UTILIZATION = 45" in cfg


def test_build_orfs_config_from_rtl():
    cfg = flow.build_orfs_config(
        design="gcd", platform="sky130hd",
        verilog_files=["./designs/src/gcd/gcd.v"], sdc_file="c.sdc",
        core_utilization="45",
    )
    assert "VERILOG_FILES" in cfg


def test_build_orfs_config_explicit_die_area_overrides_utilization():
    """Setting an explicit floorplan must not leave a conflicting utilization."""
    cfg = flow.build_orfs_config(
        design="gcd", platform="sky130hd", verilog_files=["a.v"], sdc_file="c.sdc",
        die_area="0 0 100 100", core_area="5 5 95 95", core_utilization="45",
    )
    assert "export DIE_AREA        = 0 0 100 100" in cfg
    assert "export CORE_AREA       = 5 5 95 95" in cfg
    assert "CORE_UTILIZATION" not in cfg


def test_build_orfs_config_converts_clock_period_to_picoseconds():
    cfg = flow.build_orfs_config(
        design="d", platform="sky130hd", verilog_files=["a.v"], sdc_file="c.sdc",
        clock_period="2.5",
    )
    assert "export ABC_CLOCK_PERIOD_IN_PS = 2500" in cfg


def test_build_orfs_config_appends_extra_verbatim():
    cfg = flow.build_orfs_config(
        design="d", platform="sky130hd", verilog_files=["a.v"], sdc_file="c.sdc",
        extra="export PLACE_DENSITY = 0.7\nexport SETUP_SLACK_MARGIN = 0.1",
    )
    assert "export SETUP_SLACK_MARGIN = 0.1" in cfg


# ---------------------------------------------------------------------------
# SDC
# ---------------------------------------------------------------------------

def test_default_sdc_uses_the_clock_ports():
    sdc = flow.default_sdc("sysclk", "4")
    assert "create_clock" in sdc
    assert "-period 4" in sdc
    assert "get_ports {sysclk}" in sdc


def test_default_sdc_falls_back_on_blank_values():
    sdc = flow.default_sdc("", "")
    assert "get_ports {clk}" in sdc
    assert "-period 10" in sdc


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_YOSYS_STAT = r"""
=== counter ===

   Number of wires:                 12
   Number of wire bits:             48
   Number of cells:                 24
     sky130_fd_sc_hd__a21oi_1        5
     sky130_fd_sc_hd__dfxtp_1        8
     sky130_fd_sc_hd__inv_2         11

   Chip area for module '\counter': 245.678400
"""


# Real `stat` output from yosys 0.68 (the version openroad/orfs ships), captured
# from an actual sky130hd synthesis of a 4-bit counter. The modern format is a
# right-aligned table, NOT the legacy "Number of cells: N" list -- parsing only
# the legacy shape made a successful synthesis report zero cells.
_YOSYS_068_STAT = r"""
=== counter ===

   Number of modules:            1

       10        - wires
       16        - wire bits
        3        - public wires
        6        - public wire bits
        3        - ports
        6        - port bits
       14  152.646 cells
        1    5.005   sky130_fd_sc_hd__a21oi_1
        5   18.768   sky130_fd_sc_hd__clkinv_1
        4  100.096   sky130_fd_sc_hd__dfrtp_1
        1    6.256   sky130_fd_sc_hd__lpflow_isobufsrc_1
        1    5.005   sky130_fd_sc_hd__nand3_1
        1    8.758   sky130_fd_sc_hd__xnor2_1
        1    8.758   sky130_fd_sc_hd__xor2_1

   Chip area for module '\counter': 152.646400
     of which used for sequential elements: 100.096000 (65.57%)
"""


def test_parse_yosys_068_table_format():
    """The format the shipping yosys actually emits."""
    stats = flow.parse_yosys_stats(_YOSYS_068_STAT)
    assert stats["cell_count"] == 14
    assert stats["wire_count"] == 10
    assert stats["area_um2"] == pytest.approx(152.6464)
    assert stats["sequential_area_um2"] == pytest.approx(100.096)


def test_parse_yosys_068_counts_flip_flops():
    """sky130 spells its flip-flop __dfrtp_, not __dff."""
    stats = flow.parse_yosys_stats(_YOSYS_068_STAT)
    assert stats["sequential_cells"] == 4


def test_parse_yosys_068_cell_breakdown():
    stats = flow.parse_yosys_stats(_YOSYS_068_STAT)
    by_type = stats["by_cell_type"]
    assert by_type["sky130_fd_sc_hd__clkinv_1"] == 5
    assert by_type["sky130_fd_sc_hd__dfrtp_1"] == 4
    assert sum(by_type.values()) == 14
    # Summary rows must not be mistaken for cell types.
    for word in ("cells", "wires", "ports", "bits"):
        assert word not in by_type


def test_parse_yosys_stats_extracts_counts_and_area():
    stats = flow.parse_yosys_stats(_YOSYS_STAT)
    assert stats["cell_count"] == 24
    assert stats["wire_count"] == 12
    assert stats["area_um2"] == pytest.approx(245.6784)


def test_parse_yosys_stats_counts_flip_flops_as_sequential():
    """Sequential-cell count is the only portable proxy for design state."""
    stats = flow.parse_yosys_stats(_YOSYS_STAT)
    assert stats["sequential_cells"] == 8


def test_parse_yosys_stats_breaks_down_by_cell_type():
    stats = flow.parse_yosys_stats(_YOSYS_STAT)
    assert stats["by_cell_type"]["sky130_fd_sc_hd__inv_2"] == 11
    # The "Number of ..." summary lines must not be mistaken for cell types.
    assert not any(k.lower().startswith("number") for k in stats["by_cell_type"])


def test_parse_yosys_stats_on_empty_input_is_all_zeros():
    stats = flow.parse_yosys_stats("")
    assert stats["cell_count"] == 0
    assert stats["by_cell_type"] == {}


def test_parse_orfs_metrics_maps_known_keys():
    metrics = flow.parse_orfs_metrics({
        "finish__timing__setup__ws": -0.12,
        "finish__timing__setup__tns": -3.4,
        "finish__design__instance__area": 1234.5,
        "finish__design__instance__count": 240,
        "detailedroute__route__drc_errors": 0,
        "finish__power__total": 0.0021,
    })
    assert metrics["wns_ns"] == -0.12
    assert metrics["tns_ns"] == -3.4
    assert metrics["area_um2"] == 1234.5
    assert metrics["num_instances"] == 240
    assert metrics["drc_violations"] == 0
    assert metrics["power_mw"] == 0.0021


def test_parse_orfs_metrics_from_a_real_6_report_json():
    """
    Keys captured from a real sky130hd run of a 4-bit counter.

    ORFS writes these to logs/<platform>/<design>/base/6_report.json -- NOT to
    reports/.../metadata.json, which is where run_orfs first looked, so a
    perfectly good layout reported no metrics at all.
    """
    metrics = flow.parse_orfs_metrics({
        "finish__timing__setup__ws": 9.07498,
        "finish__timing__setup__tns": 0,
        "finish__timing__hold__ws": 0.373081,
        "finish__timing__hold__tns": 0,
        "finish__design__instance__count": 480,
        "finish__design__instance__area": 282.771,
        "finish__design__instance__count__class:sequential_cell": 4,
        "finish__design__instance__utilization": 0.0827839,
        "finish__power__total": 3.69695e-05,
    })
    assert metrics["wns_ns"] == pytest.approx(9.07498)
    assert metrics["hold_wns_ns"] == pytest.approx(0.373081)
    assert metrics["num_instances"] == 480
    assert metrics["area_um2"] == pytest.approx(282.771)
    assert metrics["sequential_cells"] == 4
    assert metrics["utilization"] == pytest.approx(0.0827839)


def test_explain_orfs_failure_translates_a_too_small_die():
    """
    The failure a first-time user hits: a tiny design gives a core narrower than
    sky130's power straps need. The raw error names a metal layer and two widths
    and says nothing about which port to change.
    """
    log = (
        "[INFO ORD-0030] Using 8 thread(s).\n"
        "[ERROR PDN-0185] Insufficient width (18.86 um) to add straps on layer "
        "met4 in grid 'grid' with total strap width 15.2 um and offset 13.6 um.\n"
        "Error: pdn.tcl, 6 PDN-0185\n"
    )
    out = flow.explain_orfs_failure(log)
    assert "PDN-0185" in out                 # the tool's own words are kept
    assert "core_utilization" in out         # and the lever is named
    assert "die_area" in out


def test_explain_orfs_failure_surfaces_unknown_errors_verbatim():
    """An unrecognised failure must still beat make's bare exit code."""
    out = flow.explain_orfs_failure("[ERROR XYZ-9] something new went wrong")
    assert "XYZ-9" in out


def test_explain_orfs_failure_on_a_clean_log():
    assert flow.explain_orfs_failure("") == ""
    assert flow.explain_orfs_failure("[INFO] all good") == ""


def test_parse_orfs_metrics_collects_per_stage_runtimes():
    metrics = flow.parse_orfs_metrics({
        "synth__runtime__total": 12.3, "route__runtime__total": 402.1,
    })
    assert metrics["runtime_s"] == {"synth": 12.3, "route": 402.1}


def test_parse_orfs_metrics_accepts_json_text():
    metrics = flow.parse_orfs_metrics(json.dumps({"finish__timing__setup__ws": -1.5}))
    assert metrics["wns_ns"] == -1.5


def test_parse_orfs_metrics_omits_absent_keys_rather_than_defaulting():
    """
    A missing DRC count must not read as zero violations.

    "The flow did not report this" and "the flow reported zero" mean completely
    different things to someone deciding whether a design is clean.
    """
    metrics = flow.parse_orfs_metrics({"finish__timing__setup__ws": -1.0})
    assert "drc_violations" not in metrics


def test_parse_orfs_metrics_tolerates_garbage():
    assert flow.parse_orfs_metrics("not json at all") == {}
    assert flow.parse_orfs_metrics(None) == {}
    assert flow.parse_orfs_metrics("[1,2,3]") == {}


# ---------------------------------------------------------------------------
# Artifact globs
# ---------------------------------------------------------------------------

def test_globs_for_openroad_covers_results_and_reports():
    globs = flow.globs_for("openroad", platform="sky130hd", design="gcd")
    joined = " ".join(globs)
    assert "results/sky130hd/gcd/base/*.gds" in joined
    assert "results/sky130hd/gcd/base/*.def" in joined
    assert "reports/sky130hd/gcd/base/*.json" in joined


def test_globs_for_verilator_collects_waveforms():
    assert any(g.endswith("*.vcd") for g in flow.globs_for("verilator"))


def test_globs_for_yosys_collects_the_netlist():
    assert any(g.endswith("*.v") for g in flow.globs_for("yosys"))


# ---------------------------------------------------------------------------
# Module ports and clock reconciliation
# ---------------------------------------------------------------------------

def test_module_ports_reads_an_ansi_header():
    rtl = "module calc(input wire clk, input [3:0] a, output reg [7:0] q);\nendmodule\n"
    assert flow.module_ports(rtl, "calc") == ["clk", "a", "q"]


def test_module_ports_reads_a_non_ansi_header():
    rtl = "module calc(a, b, y);\n  input a, b;\n  output y;\nendmodule\n"
    assert flow.module_ports(rtl, "calc") == ["a", "b", "y"]


def test_module_ports_skips_the_parameter_list():
    rtl = "module counter #(parameter W = 8) (input clk, output [W-1:0] q);\nendmodule\n"
    assert flow.module_ports(rtl, "counter") == ["clk", "q"]


def test_module_ports_ignores_comments():
    rtl = (
        "module calc(\n"
        "    input clk,          // the clock\n"
        "    /* the answer */ output [7:0] q\n"
        ");\nendmodule\n"
    )
    assert flow.module_ports(rtl, "calc") == ["clk", "q"]


def test_module_ports_on_an_unreadable_header_says_it_does_not_know():
    """An empty list must read as "could not tell", never "this module is portless"."""
    assert flow.module_ports("module calc(input clk", "calc") == []   # never closes
    assert flow.module_ports("module other(input clk);", "calc") == []  # wrong module
    assert flow.module_ports("", "calc") == []


def test_resolve_clock_port_accepts_a_real_port():
    rtl = "module calc(input clk, output q); endmodule"
    assert flow.resolve_clock_port(rtl, "calc", "clk") == ("clk", "")


def test_resolve_clock_port_swaps_in_the_designs_own_clock_name():
    rtl = "module calc(input sys_clk, input rst, output q); endmodule"
    port, note = flow.resolve_clock_port(rtl, "calc", "clk")
    assert port == "sys_clk"
    assert "sys_clk" in note
    assert "clock_port" in note


def test_resolve_clock_port_reports_a_design_with_no_clock():
    """
    The shape that produced EST-0005: no clock port at all, so ``create_clock
    [get_ports {clk}]`` had nothing to attach to and OpenSTA quietly made the
    clock virtual instead of complaining.
    """
    rtl = ("module calculator(input [7:0] a, input [7:0] b, "
           "output [7:0] result); endmodule")
    port, note = flow.resolve_clock_port(rtl, "calculator", "clk")
    assert port == ""
    assert "no clock port" in note
    assert "result" in note          # the real ports are named back to the user


def test_resolve_clock_port_does_not_guess_from_an_unreadable_header():
    assert flow.resolve_clock_port("!! not verilog !!", "calc", "clk") == ("clk", "")


def test_virtual_clock_sdc_attaches_to_nothing():
    sdc = flow.virtual_clock_sdc("4")
    assert "create_clock" in sdc
    assert "-period 4" in sdc
    assert "get_ports" not in sdc     # the whole point — there is no port to name
    assert "all_inputs" in sdc
    assert "all_outputs" in sdc


# ---------------------------------------------------------------------------
# The empty-netlist gate
# ---------------------------------------------------------------------------

# Trimmed from a real failing run: an openroad block on a `calculator` design
# whose RTL had no clock port and an undriven output.  `make synth` exited 0.
_EMPTY_SYNTH_LOG = r"""
3.3. Executing CHECK pass (checking for obvious problems).
Warning: Wire calculator.\valid_operation is used but has no driver.
link_design calculator
read_sdc ./results/sky130hd/calculator/base/1_2_yosys.sdc
[WARNING STA-0366] port 'clk' not found.
eliminate_dead_logic
Removed 0 unused instances and 0 unused nets.
Design area 0 um^2 100% utilization.
write_db ./results/sky130hd/calculator/base/1_synth.odb
"""

_HEALTHY_SYNTH_LOG = r"""
7. Executing OPT_CLEAN pass (remove unused cells and wires).
Warning: Ignoring module counter because it contains processes (run 'proc' command first).
eliminate_dead_logic
Removed 0 unused instances and 0 unused nets.
Design area 1284 um^2 14% utilization.
"""


def test_synth_produced_nothing_catches_the_empty_netlist():
    out = flow.synth_produced_nothing(_EMPTY_SYNTH_LOG, "calculator")
    assert "EMPTY netlist" in out
    assert "valid_operation" in out      # the tool's own words, quoted back
    assert "STA-0366" in out
    assert "calculator" in out           # and the module to go and fix


def test_synth_produced_nothing_stays_quiet_on_a_real_design():
    assert flow.synth_produced_nothing(_HEALTHY_SYNTH_LOG, "counter") == ""


def test_synth_produced_nothing_reads_the_instance_count_too():
    assert flow.synth_produced_nothing(
        "number instances in verilog is 0\n", "calc") != ""


def test_synth_produced_nothing_does_not_blame_the_benign_processes_warning():
    """
    "contains processes (run 'proc' command first)" is printed by the
    canonicalize step of EVERY healthy run.  Repeating it as though it were the
    diagnosis would send people chasing a non-problem.
    """
    log = (_EMPTY_SYNTH_LOG + "\nWarning: Ignoring module calculator because it "
           "contains processes (run 'proc' command first).\n")
    assert "contains processes" not in flow.synth_produced_nothing(log, "calculator")


def test_explain_orfs_failure_translates_the_route_stage_symptom():
    """
    The error the user actually saw.  It names ``estimate_parasitics`` — a command
    that ran perfectly — and says nothing about the empty netlist four stages
    upstream that is the real cause.
    """
    log = (
        "[INFO GRT-0018] Total wirelength: 0 um\n"
        "[ERROR EST-0005] Run global_route before estimating parasitics for "
        "global routing.\n"
        "Error: global_route.tcl, 175 EST-0005\n"
    )
    out = flow.explain_orfs_failure(log)
    assert "EST-0005" in out
    assert "empty netlist" in out


def test_explain_orfs_failure_finds_a_hint_that_was_only_ever_a_warning():
    """
    CTS reports a missing clock as a WARNING and carries on, so a hint keyed on it
    is only reachable if the whole log — not just the ERROR lines — is searched.
    """
    log = ("[WARNING CTS-0083] No clock nets have been found.\n"
           "[ERROR XYZ-1] something later blew up\n")
    out = flow.explain_orfs_failure(log)
    assert "XYZ-1" in out            # the unrecognised error is still surfaced
    assert "clock_port" in out       # and the earlier warning explains it
