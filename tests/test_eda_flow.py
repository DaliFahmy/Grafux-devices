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
# ORFS config construction
# ---------------------------------------------------------------------------

def test_build_orfs_config_from_netlist():
    cfg = flow.build_orfs_config(
        design="counter", platform="sky130hd", verilog_files=[],
        netlist_file="./designs/sky130hd/counter/netlist.v",
        sdc_file="./designs/sky130hd/counter/constraint.sdc",
        clock_period="10", core_utilization="45", aspect_ratio="1",
    )
    assert "export DESIGN_NAME     = counter" in cfg
    assert "export PLATFORM        = sky130hd" in cfg
    assert "SYNTH_NETLIST" in cfg
    # With a netlist wired in, ORFS must not be told to synthesize from RTL.
    assert "VERILOG_FILES" not in cfg
    assert "export CORE_UTILIZATION = 45" in cfg


def test_build_orfs_config_from_rtl():
    cfg = flow.build_orfs_config(
        design="gcd", platform="sky130hd",
        verilog_files=["./designs/src/gcd/design.v"], sdc_file="c.sdc",
        core_utilization="45",
    )
    assert "VERILOG_FILES" in cfg
    assert "SYNTH_NETLIST" not in cfg


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
