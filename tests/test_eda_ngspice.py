"""
test_eda_ngspice.py
The analogue_simulator block's runtime: the deck it builds, how it reads ngspice's
rawfile and log, and run_analogue_simulator with the pod faked out.

The fixtures under tests/fixtures/eda/ngspice_* are REAL ngspice-47 output from
the image Dockerfile.ngspice builds (open_pdks 1689ac3f), not hand-written
imitations -- see fakes-must-be-strict: a parser tested against the format we
imagine is a parser tested against nothing.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

_DEVICES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _DEVICES_DIR not in sys.path:
    sys.path.insert(0, _DEVICES_DIR)

from EDA import flow, ngspice  # noqa: E402
from EDA.models import EDA_KINDS, AnalogueSimRunRequest  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "eda")


def _fixture(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return fh.read()


def _req(**kw):
    kw.setdefault("netlist", "RC\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1u 1m\n.end\n")
    return AnalogueSimRunRequest(**kw)


def test_analogue_simulator_is_an_eda_kind():
    assert "analogue_simulator" in EDA_KINDS


def test_the_module_is_stdlib_only():
    """It is uploaded to and run on a pod whose python3 has no packages."""
    import ast
    tree = ast.parse(open(ngspice.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)


# ---------------------------------------------------------------------------
# PDK / corner normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, want", [("", "sky130A"), ("SKY130", "sky130A"),
                                       ("gf180", "gf180mcuD"), ("GF180MCUD", "gf180mcuD"),
                                       ("none", "none"), ("generic", "none")])
def test_pdk_aliases(raw, want):
    assert ngspice.normalize_pdk(raw) == (want, "")


def test_an_unknown_pdk_names_what_the_image_carries():
    _pdk, error = ngspice.normalize_pdk("tsmc28")
    assert "sky130A" in error and "gf180mcuD" in error


def test_corners_default_and_translate_between_pdk_spellings():
    assert ngspice.normalize_corner("sky130A", "") == ("tt", "")
    assert ngspice.normalize_corner("gf180mcuD", "") == ("typical", "")
    assert ngspice.normalize_corner("gf180mcuD", "TT") == ("typical", "")
    assert ngspice.normalize_corner("sky130A", "typical") == ("tt", "")
    assert ngspice.normalize_corner("sky130A", "SS") == ("ss", "")


def test_a_corner_the_library_lacks_is_refused_with_the_list():
    _c, error = ngspice.normalize_corner("gf180mcuD", "ll")
    assert "typical" in error and "ll" in error


def test_gf180_loads_design_first_and_matching_passive_corners():
    lines = ngspice.pdk_library_lines("gf180mcuD", "ff", "/opt/pdks")
    assert lines[0] == '.include "/opt/pdks/gf180mcuD/libs.tech/ngspice/design.ngspice"'
    assert '.lib "/opt/pdks/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" ff' in lines
    assert any(ln.endswith(" res_ff") for ln in lines)
    # A skewed MOS corner has no skewed passive section upstream.
    skewed = ngspice.pdk_library_lines("gf180mcuD", "fs", "/opt/pdks")
    assert any(ln.endswith(" res_typical") for ln in skewed)


def test_the_spiceinit_sets_what_must_precede_the_deck():
    text = ngspice.spiceinit_for("sky130A", threads=8)
    for line in ("set ngbehavior=hsa", "set filetype=ascii", "set skywaterpdk",
                 "option klu", "set num_threads=8"):
        assert line in text
    assert "skywaterpdk" not in ngspice.spiceinit_for("gf180mcuD")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def test_a_plain_deck_passes_preflight():
    assert ngspice.analogue_preflight(_req()) == ""


@pytest.mark.parametrize("kw, fragment", [
    ({"netlist": ""}, "block_description"),
    ({"pdk": "tsmc28"}, "Unknown pdk"),
    ({"corner": "zz"}, "does not exist"),
    ({"temperature": "hot"}, "temperature"),
    ({"supply_voltage": "high"}, "supply_voltage"),
    ({"max_points": "5"}, "max_points"),
    ({"max_points": "999999"}, "max_points"),
    ({"netlist": "no analysis\nR1 a 0 1k\n.end\n"}, "no analysis"),
])
def test_preflight_refusals(kw, fragment):
    assert fragment in ngspice.analogue_preflight(_req(**kw))


def test_an_analysis_on_the_port_satisfies_preflight():
    assert ngspice.analogue_preflight(_req(netlist="t\nR1 a 0 1k\n", analyses="op")) == ""


def test_spice_numbers_accept_scale_suffixes():
    assert ngspice.parse_spice_number("1.8") == pytest.approx(1.8)
    assert ngspice.parse_spice_number("10m") == pytest.approx(0.01)
    assert ngspice.parse_spice_number("2meg") == pytest.approx(2e6)
    assert ngspice.parse_spice_number("x") is None


# ---------------------------------------------------------------------------
# build_ngspice_deck
# ---------------------------------------------------------------------------

def test_dot_analyses_move_into_control_each_followed_by_write():
    deck, run, _notes = ngspice.build_ngspice_deck(
        _req(pdk="none", netlist="t\nV1 a 0 1\nR1 a 0 1k\n.op\n.tran 1u 1m\n.end\n"))
    assert run == ["op", "tran 1u 1m"]
    assert "* (run from .control by Grafux) .op" in deck
    control = deck[deck.index("\n.control\n"):]
    assert control.index("op\nwrite sim.raw") < control.index("tran 1u 1m\nwrite sim.raw")
    assert "set appendwrite" in control
    assert deck.rstrip().endswith(".end")
    assert deck.count(".end\n") == 1


def test_the_analyses_port_replaces_the_decks_cards_and_says_so():
    deck, run, notes = ngspice.build_ngspice_deck(_req(pdk="none", analyses=".ac dec 10 1 1meg"))
    assert run == ["ac dec 10 1 1meg"]
    assert "\ntran 1u 1m\n" not in deck
    assert any("replaced" in n for n in notes)


def test_the_pdk_library_is_injected_after_the_title():
    deck, _run, _notes = ngspice.build_ngspice_deck(_req(corner="ff"), pdk_root="/pdk")
    lines = deck.splitlines()
    assert lines[0] == "RC"
    assert '.lib "/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice" ff' in lines[1:4]


def test_a_deck_that_loads_the_library_itself_is_not_given_a_second_copy():
    netlist = 't\n.lib "/my/sky130.lib.spice" tt\nR1 a 0 1k\n.op\n'
    deck, _run, notes = ngspice.build_ngspice_deck(_req(netlist=netlist, corner="ss"))
    assert deck.count("sky130.lib.spice") == 1
    assert any("corner port was not applied" in n for n in notes)


def test_temperature_and_vdd_are_added_unless_the_deck_sets_them():
    deck, _r, _n = ngspice.build_ngspice_deck(_req(pdk="none", temperature="85",
                                                   supply_voltage="3.3"))
    assert ".temp 85" in deck and ".param vdd=3.3" in deck
    netlist = "t\n.temp 0\n.param vdd=1.2\nR1 a 0 1k\n.op\n"
    deck, _r, notes = ngspice.build_ngspice_deck(_req(pdk="none", netlist=netlist,
                                                      temperature="85", supply_voltage="3.3"))
    assert ".temp 85" not in deck and ".param vdd=3.3" not in deck
    assert len(notes) == 2


def test_meas_statements_get_their_dot_and_are_named():
    deck, _r, _n = ngspice.build_ngspice_deck(
        _req(pdk="none", meas_statements="meas tran t1 WHEN v(out)=0.5\n.measure tran T2 MAX v(out)"))
    assert ".meas tran t1 WHEN v(out)=0.5" in deck
    assert ngspice.measure_names(deck) == ["t1", "t2"]


def test_a_deck_starting_with_a_dot_card_gets_a_title_so_the_card_survives():
    deck, _r, _n = ngspice.build_ngspice_deck(
        _req(pdk="none", netlist=".subckt inv a y\nR1 a y 1k\n.ends\nX1 in out inv\n.op\n"))
    assert deck.splitlines()[0].startswith("*")
    assert ".subckt inv a y" in deck.splitlines()[1:]


def test_probes_are_saved_before_the_analyses():
    deck, _r, _n = ngspice.build_ngspice_deck(_req(pdk="none", probes="v(out), i(V1)"))
    control = deck[deck.index("\n.control\n"):]
    assert control.index("save v(out) i(V1)") < control.index("tran 1u 1m")


def test_a_deck_with_its_own_control_block_runs_as_written():
    netlist = "t\nV1 a 0 1\nR1 a 0 1k\n.control\nop\nprint v(a)\n.endc\n.end\n"
    deck, run, notes = ngspice.build_ngspice_deck(_req(pdk="none", netlist=netlist,
                                                       analyses="tran 1u 1m"))
    assert deck.count(".control") == 1
    assert "tran 1u 1m" not in deck
    assert run == []
    assert any("own .control" in n for n in notes)


def test_extra_control_runs_after_the_analyses_and_before_quit():
    deck, _r, _n = ngspice.build_ngspice_deck(_req(pdk="none", extra_control="print v(out)"))
    assert deck.index("write sim.raw") < deck.index("print v(out)") < deck.index("quit")


def test_the_fixture_deck_is_what_the_builder_emits():
    """tests/fixtures/eda/ngspice_tran_op.sp is the deck that produced the raw/log fixtures."""
    deck = _fixture("ngspice_tran_op.sp")
    assert "tran 200u 4m\nwrite sim.raw" in deck
    assert ngspice.measure_names(deck) == ["trise", "never"]


# ---------------------------------------------------------------------------
# rawfile parsing (real ngspice-47 ASCII rawfiles)
# ---------------------------------------------------------------------------

def test_a_tran_then_op_rawfile_has_both_plots_in_order():
    plots = ngspice.parse_ascii_raw(_fixture("ngspice_tran_op.raw"))
    assert [p["name"] for p in plots] == ["Transient Analysis", "Operating Point"]
    tran = plots[0]
    assert [v[0] for v in tran["variables"]] == ["time", "v(in)", "v(out)", "i(v1)"]
    assert tran["npoints_read"] == tran["npoints"] == len(tran["points"])
    assert tran["points"][0][0] == 0.0


def test_decimation_keeps_the_first_and_last_point():
    full = ngspice.parse_ascii_raw(_fixture("ngspice_tran_op.raw"))[0]
    small = ngspice.parse_ascii_raw(_fixture("ngspice_tran_op.raw"), keep_points=10)[0]
    assert small["stride"] > 1
    assert len(small["points"]) <= 11
    assert small["points"][0] == full["points"][0]
    assert small["points"][-1] == full["points"][-1]


def test_an_ac_rawfile_is_complex():
    plot = ngspice.parse_ascii_raw(_fixture("ngspice_ac.raw"))[0]
    assert "complex" in plot["flags"]
    assert isinstance(plot["points"][0][1], complex)


def test_a_truncated_rawfile_yields_what_was_read():
    text = _fixture("ngspice_tran_op.raw")
    cut = text[: len(text) // 2]
    plots = ngspice.parse_ascii_raw(cut)
    assert plots and 0 < plots[0]["npoints_read"] < plots[0]["npoints"]


def test_the_operating_point_comes_from_the_op_plot():
    plots = ngspice.parse_ascii_raw(_fixture("ngspice_gf180_op_dc.raw"))
    op = ngspice.operating_point(plots)
    assert op["v(vdd)"] == pytest.approx(3.3)
    assert op["v(out)"] == pytest.approx(3.3, abs=1e-3)


# ---------------------------------------------------------------------------
# waveforms: the plotter's CSV shape
# ---------------------------------------------------------------------------

def test_tran_waveforms_are_a_header_plus_numeric_rows():
    plot = ngspice.choose_waveform_plot(ngspice.parse_ascii_raw(_fixture("ngspice_tran_op.raw")))
    csv, _notes = ngspice.waveforms_csv(plot, [])
    rows = csv.strip().splitlines()
    assert rows[0] == "time,v(in),v(out),i(v1)"
    for row in rows[1:]:
        assert all(cell != "" for cell in row.split(","))
        [float(cell) for cell in row.split(",")]


def test_ac_waveforms_are_bode_magnitude_and_phase():
    plot = ngspice.parse_ascii_raw(_fixture("ngspice_ac.raw"))[0]
    csv, _notes = ngspice.waveforms_csv(plot, ["out"])
    rows = csv.strip().splitlines()
    assert rows[0] == "frequency,vdb(out),vp(out)"
    first = [float(c) for c in rows[1].split(",")]
    last = [float(c) for c in rows[-1].split(",")]
    assert first[1] == pytest.approx(0.0, abs=0.01)     # passband
    assert last[1] < -40                                 # 1 MHz on a 159 Hz pole
    assert -91 < last[2] < -89


def test_the_dc_sweep_is_chosen_over_the_op_plot():
    plots = ngspice.parse_ascii_raw(_fixture("ngspice_gf180_op_dc.raw"))
    assert ngspice.choose_waveform_plot(plots)["name"] == "DC transfer characteristic"


def test_probes_select_columns_and_missing_ones_are_named():
    plot = ngspice.parse_ascii_raw(_fixture("ngspice_tran_op.raw"))[0]
    csv, notes = ngspice.waveforms_csv(plot, ["out", "v(nowhere)"])
    assert csv.splitlines()[0] == "time,v(out)"
    assert any("v(nowhere)" in n for n in notes)


def test_series_past_the_plotters_limit_are_dropped_with_a_note():
    variables = [("time", "time")] + [("v(n{0})".format(i), "voltage") for i in range(12)]
    plot = {"name": "Transient Analysis", "flags": "real", "variables": variables,
            "npoints": 2, "stride": 1, "points": [[0.0] + [1.0] * 12, [1.0] + [2.0] * 12]}
    csv, notes = ngspice.waveforms_csv(plot, [])
    assert len(csv.splitlines()[0].split(",")) == 1 + ngspice.MAX_SERIES
    assert any("v(n11)" in n for n in notes)


def test_voltages_are_ranked_before_currents():
    variables = [("time", "time"), ("i(v1)", "current"), ("v(a)", "voltage")]
    plot = {"name": "Transient Analysis", "flags": "real", "variables": variables,
            "npoints": 2, "stride": 1, "points": [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]}
    assert ngspice.waveforms_csv(plot, [])[0].splitlines()[0] == "time,v(a),i(v1)"


def test_postprocess_cli_on_the_real_fixture(tmp_path, capsys):
    raw = tmp_path / "sim.raw"
    raw.write_text(_fixture("ngspice_tran_op.raw"), encoding="utf-8")
    assert ngspice._main(["x", "postprocess", str(raw), "20", "[]"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [p["kind"] for p in result["plots"]] == ["tran", "op"]
    assert result["waveform_plot"] == "Transient Analysis"
    assert result["raw_bytes"] > 0
    assert len(result["waveforms"].splitlines()) <= 22


def test_postprocess_cli_without_a_rawfile_answers_instead_of_crashing(tmp_path, capsys):
    assert ngspice._main(["x", "postprocess", str(tmp_path / "missing.raw"), "20", "[]"]) == 0
    assert json.loads(capsys.readouterr().out)["plots"] == []


# ---------------------------------------------------------------------------
# log parsing (real ngspice-47 logs)
# ---------------------------------------------------------------------------

def test_measurements_parse_and_a_failed_one_is_null_with_ngspices_reason():
    values, notes = ngspice.parse_measurements(_fixture("ngspice_tran_op_log.txt"),
                                               ["trise", "never"])
    assert values["trise"] == pytest.approx(2.19562e-3)
    assert values["never"] is None
    assert any("out of interval" in n for n in notes)


def test_ac_measurement():
    values, notes = ngspice.parse_measurements(_fixture("ngspice_ac_log.txt"), ["f3db"])
    assert values["f3db"] == pytest.approx(158.78)
    assert notes == []


def test_a_failed_measure_is_a_warning_not_a_failed_run():
    errors, warnings, _hints = ngspice.classify_ngspice_log(_fixture("ngspice_tran_op_log.txt"), 0)
    assert errors == []
    assert any("measure" in w for w in warnings)


def test_an_unknown_device_is_an_error_with_the_device_hint():
    errors, _w, hints = ngspice.classify_ngspice_log(_fixture("ngspice_unknown_subckt_log.txt"), 1)
    assert any("unknown subckt" in e for e in errors)
    assert any("sky130_fd_pr__nfet_01v8" in h for h in hints)


def test_a_dropped_element_line_is_an_error_although_ngspice_exits_zero():
    errors, _w, hints = ngspice.classify_ngspice_log(_fixture("ngspice_dropped_line_log.txt"), 0)
    assert any("not a valid resistor instance line" in e for e in errors)
    assert any("DROPPED" in h for h in hints)


def test_a_klu_convergence_failure_is_an_error_naming_the_sparse_workaround():
    errors, warnings, hints = ngspice.classify_ngspice_log(_fixture("ngspice_floating_klu_log.txt"), 0)
    assert any("timestep too small" in e.lower() for e in errors)
    assert any("singular matrix" in w for w in warnings)
    assert any(".option sparse" in h for h in hints)


def test_recovered_convergence_warnings_on_a_green_run_give_no_hints():
    log = ("Warning: singular matrix:  check node x\nWarning: source stepping failed\n"
           "Note: Transient op finished successfully\nngspice-47 done\n")
    errors, warnings, hints = ngspice.classify_ngspice_log(log, 0)
    assert errors == [] and hints == []
    assert len(warnings) == 2


def test_the_hsa_mode_noise_warning_is_dropped():
    _e, warnings, _h = ngspice.classify_ngspice_log(_fixture("ngspice_sky130_ss_85c_log.txt"), 0)
    assert not any("m=xx" in w for w in warnings)


def test_the_sky130_log_carries_the_corner_temperature_and_measure():
    log = _fixture("ngspice_sky130_ss_85c_log.txt")
    assert "TEMP = 85" in log
    assert ngspice.parse_measurements(log, ["tpd"])[0]["tpd"] == pytest.approx(4.46548e-11)


def test_a_nonzero_exit_with_a_quiet_log_is_still_an_error():
    errors, _w, _h = ngspice.classify_ngspice_log("", 139)
    assert errors == ["ngspice exited with code 139."]


# ---------------------------------------------------------------------------
# run_analogue_simulator, with the pod faked out
# ---------------------------------------------------------------------------

class _FakeSftp:
    def close(self):
        pass


class _FakeClient:
    def open_sftp(self):
        return _FakeSftp()


def _post(**kw):
    raw = _fixture("ngspice_tran_op.raw").splitlines(keepends=True)
    result = ngspice.postprocess(raw, probes=kw.pop("probes", []), max_points=50)
    result["raw_bytes"] = 9205
    result.update(kw)
    return json.dumps(result)


@pytest.fixture
def pod(monkeypatch):
    state = {
        "written": {},
        "commands": [],
        "stages": [],
        "probe": "ROOT:/opt/pdks\nNPROC:8\n** ngspice-47 : Circuit level simulation program\n",
        "missing": "",
        "sim": (0, "ngspice-47 done", ""),
        "log": _fixture("ngspice_tran_op_log.txt"),
        "post": _post(),
    }

    def write_file(_sftp, path, content):
        state["written"][path.rsplit("/", 1)[-1]] = content

    def exec_stream(_client, command, **_kw):
        state["commands"].append(command)
        return state["sim"]

    def exec_simple(_client, command, **_kw):
        state["commands"].append(command)
        # Most specific first: every ngspice command carries _NGSPICE_ENV, whose
        # "${PDK_ROOT:-...}" would match a bare "ROOT:" test.
        if "postprocess" in command:
            return 0, state["post"], ""
        if "NPROC:" in command:
            return 0, state["probe"], ""
        if "MISSING:" in command:
            return 0, state["missing"], ""
        if "tail -c" in command:
            return 0, state["log"], ""
        return 0, "", ""

    monkeypatch.setattr(flow, "_write_file", write_file)
    monkeypatch.setattr(flow, "exec_stream", exec_stream)
    monkeypatch.setattr(flow, "exec_simple", exec_simple)
    return state


def _run(state, req=None):
    return flow.run_analogue_simulator(
        _FakeClient(), req or _req(pdk="none", meas_statements="meas tran trise TRIG v(out) "
                                   "VAL=0.1 RISE=1 TARG v(out) VAL=0.9 RISE=1"),
        on_stage=lambda s, d: state["stages"].append((s, d)),
    )


def test_a_successful_run_fills_every_port_it_owns(pod):
    result = _run(pod)
    out = result["outputs"]
    assert out["status"] == "ok", out["errors"]
    assert json.loads(out["measurements"])["trise"] == pytest.approx(2.19562e-3)
    assert out["waveforms"].startswith("time,v(in),v(out),i(v1)\n")
    assert "Transient Analysis:" in out["analyses"]
    assert out["netlist"] == pod["written"][ngspice.DECK_FILE]
    assert out["raw"] == ngspice.RAW_FILE
    stats = json.loads(out["stats"])
    assert stats["version"] == "47" and stats["simulator"] == "ngspice"
    assert result["_globs"] == flow.globs_for("analogue_simulator")
    assert set(out) >= {"status", "netlist", "measurements", "waveforms", "operating_point",
                        "analyses", "stats", "errors", "warnings", "log", "raw"}


def test_the_stages_are_deck_simulate_collect(pod):
    _run(pod)
    assert [s for s, _ in pod["stages"]] == [
        "deck", "deck", "simulate", "simulate", "collect", "collect"]


def test_the_deck_spiceinit_and_script_are_written_before_ngspice_runs(pod):
    _run(pod)
    assert set(pod["written"]) == {ngspice.DECK_FILE, ngspice.SPICEINIT_FILE, ngspice.SCRIPT_FILE}
    assert "set num_threads=8" in pod["written"][ngspice.SPICEINIT_FILE]
    assert pod["written"][ngspice.SCRIPT_FILE] == open(ngspice.__file__, encoding="utf-8").read()
    sim = next(c for c in pod["commands"] if "-b deck.sp" in c)
    assert "pipefail" in sim and "rm -f sim.raw sim.log" in sim


def test_the_pdk_root_comes_from_the_pod(pod):
    pod["probe"] = "ROOT:/custom/pdks\nNPROC:2\n"
    _run(pod, _req(pdk="sky130A"))
    assert '"/custom/pdks/sky130A/libs.tech/ngspice/sky130.lib.spice"' in pod["written"][ngspice.DECK_FILE]


def test_an_image_without_the_models_is_refused_before_simulating(pod):
    pod["missing"] = "MISSING:/opt/pdks/gf180mcuD/libs.tech/ngspice/design.ngspice\n"
    out = _run(pod, _req(pdk="gf180mcuD"))["outputs"]
    assert out["status"] == "error"
    assert "does not carry the gf180mcuD" in out["errors"]
    assert not any("-b deck.sp" in c for c in pod["commands"])


def test_a_bad_request_touches_nothing_on_the_pod(pod):
    out = _run(pod, _req(netlist=""))["outputs"]
    assert out["status"] == "error"
    assert pod["commands"] == []


def test_an_error_in_the_log_is_red_even_with_exit_zero(pod):
    pod["log"] = _fixture("ngspice_dropped_line_log.txt")
    out = _run(pod)["outputs"]
    assert out["status"] == "error"
    assert "Likely cause" in out["errors"]


def test_no_results_is_red(pod):
    pod["log"] = "ngspice-47 done\n"
    pod["post"] = json.dumps({"plots": [], "waveforms": "", "operating_point": {},
                              "notes": [], "missing_raw": True})
    out = _run(pod)["outputs"]
    assert out["status"] == "error"
    assert "without writing any results" in out["errors"]


def test_a_timeout_says_how_to_fix_it(pod):
    pod["sim"] = (-2, "partial", "")
    out = _run(pod)["outputs"]
    assert out["status"] == "error" and "timeout" in out["errors"]
    assert out["netlist"]


def test_failed_measurements_and_decimation_land_on_warnings(pod):
    req = _req(pdk="none", meas_statements="meas tran never WHEN v(out)=5")
    out = _run(pod, req)["outputs"]
    assert out["status"] == "ok"
    assert json.loads(out["measurements"]) == {"never": None}
    assert "Measurement never produced no value" in out["warnings"]
    assert "decimated" in out["warnings"]


def test_the_server_command_override_is_honoured(pod, monkeypatch):
    monkeypatch.setenv("EDA_NGSPICE_CMD", "/custom/ngspice")
    _run(pod)
    assert any("/custom/ngspice -b deck.sp" in c for c in pod["commands"])


def test_the_globs_name_the_run_files_not_staged_models():
    globs = flow.globs_for("analogue_simulator", work_dir="/w")
    assert globs == ["/w/deck.sp", "/w/sim.log", "/w/sim.raw"]


def test_a_duck_typed_request_works_for_the_ci_smoke_test():
    req = SimpleNamespace(netlist="t\nR1 a 0 1k\n", analyses="op", pdk="none")
    assert ngspice.analogue_preflight(req) == ""
    assert "op\nwrite sim.raw" in ngspice.build_ngspice_deck(req)[0]
