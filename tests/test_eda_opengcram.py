"""
test_eda_opengcram.py
The opengcram block's runtime: the gain-cell config it writes, the refusals it
makes before a compile, and the technology-archive stage run_opengcram adds.

Nothing here touches a pod.  The facts the refusals encode were read from
OpenGCRAM @ cbdc35b (see the comment block above OPENGCRAM_TECH_DIR in flow.py);
each test names the upstream failure it stands in front of.
"""

import os
import sys

import pytest

_DEVICES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _DEVICES_DIR not in sys.path:
    sys.path.insert(0, _DEVICES_DIR)

from EDA import flow  # noqa: E402
from EDA.models import EDA_KINDS, OpenGcRamRunRequest, OpenRamRunRequest  # noqa: E402

ARCHIVE_CLIENT_PATH = "C:/Users/me/pdks/tsmcN40.tar.gz"
ARCHIVE_POD_PATH = "/workspace/grafux/tsmcN40.tar.gz"


def _req(**kw):
    return OpenGcRamRunRequest(**kw)


def _staged(**kw):
    kw.setdefault("tech_archive", ARCHIVE_CLIENT_PATH)
    kw.setdefault("input_files", [{"path": ARCHIVE_POD_PATH, "content": "", "b64": True}])
    return _req(**kw)


def _config(req, **kw):
    kw.setdefault("tech", "tsmcN40")
    kw.setdefault("output_name", "gcram_8x32")
    kw.setdefault("out_dir", "/workspace/grafux/openram_out")
    return flow.build_openram_config(req, gain_cell=True, **kw)


def test_opengcram_is_an_eda_kind():
    assert "opengcram" in EDA_KINDS


# ---------------------------------------------------------------------------
# build_openram_config(gain_cell=True)
# ---------------------------------------------------------------------------

def test_the_gain_cell_variables_are_written():
    text = _config(_req(gc_type="si", vddio="1.5", word_size="8", num_words="32"))
    assert 'gc_type = "Si"' in text
    assert "vddio = 1.5" in text
    assert 'tech_name = "tsmcN40"' in text


def test_blank_ports_become_one_read_one_write_not_upstreams_single_rw():
    """Upstream's 1 rw + 0 + 0 adds up to one port, and only gain_cell_2port exists."""
    text = _config(_req())
    for line in ("num_rw_ports = 0", "num_r_ports = 1", "num_w_ports = 1"):
        assert line in text


def test_the_two_defaults_that_would_sink_a_run_are_overridden():
    """analytical_delay is False upstream (HSpice); use_pex is True (Calibre)."""
    text = _config(_req())
    assert "analytical_delay = True" in text
    assert "use_pex = False" in text
    assert "check_lvsdrc = False" in text


def test_extra_config_comes_last_so_it_can_turn_them_back_on():
    text = _config(_req(extra_config="use_pex = True"))
    assert text.rindex("use_pex = True") > text.index("use_pex = False")


def test_the_openram_config_is_unchanged_by_the_gain_cell_branch():
    text = flow.build_openram_config(OpenRamRunRequest(), tech="scn4m_subm",
                                     output_name="m", out_dir="/o")
    assert "gc_type" not in text
    assert "use_pex" not in text
    assert "num_r_ports" not in text


def test_the_config_port_still_wins_outright():
    text = _config(_req(gc_type="Si", config="word_size = 4\ngc_type = 'hybrid'\n"))
    assert 'gc_type = "Si"' not in text
    assert 'output_name = "gcram_8x32"' in text


def test_the_gain_cell_config_is_valid_python():
    compile(_config(_req(word_size="8", num_words="32", vddio="1.2",
                         process_corners="TT")), "<opengcram config>", "exec")


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, want", [("", "OS"), ("os", "OS"), ("SI", "Si"),
                                       ("Hybrid", "hybrid")])
def test_gc_type_is_normalised_to_the_exact_strings_upstream_compares(raw, want):
    assert flow.normalize_gc_type(raw) == (want, "")


def test_an_unknown_gc_type_is_an_error_not_a_silent_keyerror():
    value, error = flow.normalize_gc_type("edram")
    assert value == ""
    assert "OS" in error and "hybrid" in error


def test_output_name_carries_geometry_flavour_and_tech():
    assert flow.opengcram_output_name(_req(word_size="16", num_words="64", gc_type="Si"),
                                      "tsmcN40") == "gcram_16x64_si_tsmcN40"
    assert flow.opengcram_output_name(_req(output_name="my gc"), "t") == "my_gc"


@pytest.mark.parametrize("path, stem", [("/w/tsmcN40.tar.gz", "tsmcN40"),
                                        ("C:\\x\\tech.ZIP", "tech"),
                                        ("a.tgz", "a")])
def test_archive_stem(path, stem):
    assert flow.archive_stem(path) == stem


# ---------------------------------------------------------------------------
# opengcram_preflight
# ---------------------------------------------------------------------------

def test_a_default_request_passes_preflight():
    assert flow.opengcram_preflight(_req(word_size="8", num_words="32")) == ([], "")


@pytest.mark.parametrize("ports", [
    {"num_rw_ports": "1"},                                   # 1 rw + 1 r + 1 w = 3
    {"num_r_ports": "0"},                                    # write-only
    {"num_w_ports": "0", "num_rw_ports": "1"},               # no write port
])
def test_anything_but_one_read_one_write_is_refused(ports):
    _notes, refusal = flow.opengcram_preflight(_req(**ports))
    assert "two-port gain cell" in refusal


def test_a_bad_gc_type_or_vddio_is_refused():
    assert flow.opengcram_preflight(_req(gc_type="dram"))[1]
    assert "voltage" in flow.opengcram_preflight(_req(vddio="high"))[1]


def test_a_user_config_skips_the_gain_cell_rules_but_not_the_size_ceiling():
    assert flow.opengcram_preflight(_req(config="x = 1", num_rw_ports="3")) == ([], "")
    assert "Refusing" in flow.opengcram_preflight(
        _req(config="x = 1", word_size="64", num_words="1000000"))[1]


# ---------------------------------------------------------------------------
# find_staged_archive / resolve_tech_layout / check_gain_cell_listing
# ---------------------------------------------------------------------------

def test_the_archive_is_matched_by_base_name_across_client_and_pod_paths():
    assert flow.find_staged_archive(_staged()) == (ARCHIVE_POD_PATH, "")


def test_no_archive_port_means_no_archive_and_no_error():
    assert flow.find_staged_archive(_req()) == ("", "")


def test_a_named_archive_that_never_reached_the_pod_is_an_error():
    path, error = flow.find_staged_archive(_staged(input_files=[]))
    assert path == ""
    assert "no such file reached the pod" in error


def test_a_non_archive_is_an_error():
    assert "can unpack" in flow.find_staged_archive(_staged(tech_archive="t.rar"))[1]


def test_an_archive_of_the_tech_directory_uses_that_directory():
    assert flow.resolve_tech_layout(["tsmcN40/"], tech_port="", stem="x") == (
        "tsmcN40", "tsmcN40", "")


def test_an_archive_of_the_tech_contents_is_named_by_the_port_or_the_file():
    entries = ["__init__.py", "gds_lib", "sp_lib", "tech", "layers.map"]
    assert flow.resolve_tech_layout(entries, tech_port="", stem="tsmcN40") == (
        "tsmcN40", "", "")
    assert flow.resolve_tech_layout(entries, tech_port="n40", stem="x")[0] == "n40"


def test_several_top_level_dirs_need_tech_name():
    assert "several" in flow.resolve_tech_layout(["a", "b"], tech_port="", stem="x")[2]
    assert flow.resolve_tech_layout(["a", "b"], tech_port="b", stem="x") == ("b", "b", "")


def test_the_tech_name_must_be_importable():
    assert "identifier" in flow.resolve_tech_layout(["tsmc-n40"], tech_port="", stem="x")[2]


def _cells(*files, directory="/workspace/grafux/gcram_tech/tsmcN40"):
    return "\n".join(["DIR:" + directory] + list(files))


def test_a_technology_with_the_cell_passes():
    listing = _cells("GDS:os_gc.gds", "SP:os_gc.sp")
    assert flow.check_gain_cell_listing(listing, tech="tsmcN40", gc_type="OS",
                                        netlist_only=False) == ""


def test_a_missing_technology_explains_the_nda_situation():
    refusal = flow.check_gain_cell_listing("", tech="tsmcN40", gc_type="OS",
                                           netlist_only=False)
    assert "tech_archive" in refusal and "NDA" in refusal


def test_freepdk45_is_refused_as_sram_only():
    listing = _cells("GDS:cell_1rw.gds", "SP:cell_1rw.sp",
                     directory="/opt/opengcram/technology/freepdk45")
    refusal = flow.check_gain_cell_listing(listing, tech="freepdk45", gc_type="OS",
                                           netlist_only=False)
    assert "os_gc" in refusal and "SRAM-only" in refusal


def test_the_wrong_flavour_names_the_one_edit_fix():
    listing = _cells("GDS:si_gc.gds", "SP:si_gc.sp")
    refusal = flow.check_gain_cell_listing(listing, tech="t", gc_type="OS",
                                           netlist_only=False)
    assert "set gc_type to Si" in refusal


def test_netlist_only_needs_the_spice_cell_but_not_the_gds():
    listing = _cells("SP:hybrid_gc.sp")
    assert flow.check_gain_cell_listing(listing, tech="t", gc_type="hybrid",
                                        netlist_only=True) == ""
    assert "gds_lib/hybrid_gc.gds" in flow.check_gain_cell_listing(
        listing, tech="t", gc_type="hybrid", netlist_only=False)


def test_the_probe_searches_openram_tech_for_the_quoted_name():
    cmd = flow.gain_cell_probe_command("tsmcN40")
    assert "$OPENRAM_TECH" in cmd and "gds_lib" in cmd and "sp_lib" in cmd


def test_the_uploaded_tech_dir_is_first_on_the_search_path():
    wrapped = flow._sh_opengcram("true")
    assert "gcram_tech:${OPENRAM_TECH" in wrapped
    assert "gain_cell_compiler.py" in wrapped


def test_the_globs_match_openrams():
    assert flow.globs_for("opengcram", work_dir="/w") == flow.globs_for("openram", work_dir="/w")


# ---------------------------------------------------------------------------
# run_opengcram, with the pod faked out
# ---------------------------------------------------------------------------

class _FakeFile:
    def __init__(self, data):
        self._data = data

    def read(self, size=-1):
        return self._data if size is None or size < 0 else self._data[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSftp:
    def __init__(self, state):
        self._state = state

    def open(self, path, _mode="rb"):
        name = path.rsplit("/", 1)[-1]
        if name not in self._state["pod_files"]:
            raise FileNotFoundError(path)
        return _FakeFile(self._state["pod_files"][name].encode("utf-8"))

    def close(self):
        pass


class _FakeClient:
    def __init__(self, state):
        self._state = state

    def open_sftp(self):
        return _FakeSftp(self._state)


NAME = "gcram_8x32_os_tsmcN40"


@pytest.fixture
def pod(monkeypatch):
    state = {
        "written": {},
        "commands": [],
        "stages": [],
        "extract": (0, "tsmcN40\n", ""),
        "cells": _cells("GDS:os_gc.gds", "SP:os_gc.sp"),
        "run": (0, "Total area: 99.0 um^2", ""),
        "listing": "\n".join(NAME + ext for ext in (
            ".gds", ".sp", "_TT_1p0V_25C.lib", ".lef", ".v", ".html", ".py", ".log")),
        "pod_files": {NAME + ".v": "module gc(); endmodule\n",
                      NAME + ".py": "gc_type = 'OS'\n"},
    }

    def write_file(_sftp, path, content):
        state["written"][path] = content

    def exec_stream(_client, command, **_kw):
        state["commands"].append(command)
        return state["run"]

    def exec_simple(_client, command, **_kw):
        state["commands"].append(command)
        if "ls -1A" in command:
            return state["extract"]
        if "DIR:" in command:
            return 0, state["cells"], ""
        if "ls -S" in command:
            return 0, state["listing"], ""
        return 0, "", ""

    monkeypatch.setattr(flow, "_write_file", write_file)
    monkeypatch.setattr(flow, "exec_stream", exec_stream)
    monkeypatch.setattr(flow, "exec_simple", exec_simple)
    return state


def _run(state, req=None):
    return flow.run_opengcram(
        _FakeClient(state), req or _staged(word_size="8", num_words="32"),
        on_stage=lambda s, d: state["stages"].append((s, d)),
    )


def test_a_successful_run_fills_every_port_it_owns(pod):
    result = _run(pod)
    out = result["outputs"]
    assert out["status"] == "ok", out["errors"]
    assert out["top"] == NAME
    assert out["tech_name"] == "tsmcN40"
    assert out["verilog_model"] == "module gc(); endmodule\n"
    assert '"gc_type": "OS"' in out["stats"]
    assert result["_globs"] == flow.globs_for("opengcram")


def test_the_stages_are_tech_config_compile_collect(pod):
    _run(pod)
    assert [s for s, _ in pod["stages"]] == [
        "tech", "tech", "config", "config", "compile", "compile", "collect", "collect"]


def test_the_archive_is_unpacked_moved_and_probed_before_the_compile(pod):
    _run(pod)
    cmds = pod["commands"]
    extract = next(i for i, c in enumerate(cmds) if "tar -xf" in c)
    move = next(i for i, c in enumerate(cmds) if " mv " in c)
    probe = next(i for i, c in enumerate(cmds) if "DIR:" in c)
    compile_ = next(i for i, c in enumerate(cmds) if "GRAFUX_OPENGCRAM_CMD" in c
                    and flow.OPENRAM_CONFIG_FILE in c)
    assert extract < move < probe < compile_
    assert ARCHIVE_POD_PATH in cmds[extract]
    assert "gcram_tech/tsmcN40" in cmds[move]


def test_a_zip_is_unpacked_with_python_not_unzip(pod):
    _run(pod, _staged(tech_archive="C:/t/tsmcN40.zip",
                      input_files=[{"path": "/workspace/grafux/tsmcN40.zip"}]))
    assert any("python3 -m zipfile -e" in c for c in pod["commands"])


def test_the_config_written_is_the_gain_cell_one(pod):
    _run(pod)
    text = pod["written"][f"{flow.WORK_DIR}/{flow.OPENRAM_CONFIG_FILE}"]
    assert 'gc_type = "OS"' in text and "use_pex = False" in text


def test_a_technology_without_the_cell_never_reaches_the_compiler(pod):
    pod["cells"] = _cells("GDS:cell_1rw.gds", "SP:cell_1rw.sp")
    out = _run(pod)["outputs"]
    assert out["status"] == "error"
    assert "os_gc" in out["errors"]
    assert not any(flow.OPENRAM_CONFIG_FILE in c for c in pod["commands"])


def test_no_archive_and_no_installed_tech_is_refused_with_the_reason(pod):
    pod["cells"] = ""
    out = _run(pod, _req(word_size="8", num_words="32"))["outputs"]
    assert out["status"] == "error"
    assert "tech_archive" in out["errors"]
    assert not any("tar -xf" in c for c in pod["commands"])


def test_a_bad_request_touches_nothing_on_the_pod(pod):
    out = _run(pod, _staged(num_rw_ports="1"))["outputs"]
    assert "two-port" in out["errors"]
    assert pod["commands"] == []


def test_an_unpack_failure_is_reported(pod):
    pod["extract"] = (2, "", "tar: This does not look like a tar archive")
    out = _run(pod)["outputs"]
    assert "Could not unpack" in out["errors"]
    assert "does not look like a tar archive" in out["errors"]


def test_the_archive_can_rename_the_technology(pod):
    pod["extract"] = (0, "n40ulp\n", "")
    out = _run(pod)["outputs"]
    assert out["tech_name"] == "n40ulp"
    assert out["top"] == "gcram_8x32_os_n40ulp"


def test_a_user_config_skips_the_cell_probe_and_says_so(pod):
    pod["cells"] = ""
    out = _run(pod, _staged(config="word_size = 4\n"))["outputs"]
    assert out["status"] == "ok"
    assert "gain-cell check was skipped" in out["warnings"]
    assert not any("DIR:" in c for c in pod["commands"])


def test_a_nonzero_exit_names_opengcram(pod):
    pod["run"] = (-2, "", "")
    out = _run(pod)["outputs"]
    assert out["status"] == "error"
    assert "OpenGCRAM run exceeded its timeout" in out["errors"]


def test_asking_for_drc_warns_about_calibre(pod):
    out = _run(pod, _staged(check_lvsdrc="1"))["outputs"]
    assert "Calibre" in out["warnings"]


def test_the_server_command_override_is_honoured(pod, monkeypatch):
    monkeypatch.setenv("EDA_OPENGCRAM_CMD", "python3 /custom/gain_cell_compiler.py")
    _run(pod)
    assert any("/custom/gain_cell_compiler.py" in c for c in pod["commands"])
