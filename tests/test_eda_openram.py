"""
test_eda_openram.py
The openram block's runtime: the config it writes, what it makes of the files
that come back, and the decisions run_openram takes around the compile.

Nothing here touches a pod.  Everything that decides what OpenRAM is asked to
build is a pure function precisely so it can be pinned offline -- and because
the image's CI smoke test compiles the output of `build_openram_config`, an
option OpenRAM stops accepting fails the image build rather than a user's run.
"""

import json
import os
import sys

import pytest

_DEVICES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _DEVICES_DIR not in sys.path:
    sys.path.insert(0, _DEVICES_DIR)

from EDA import flow  # noqa: E402
from EDA.models import OpenRamRunRequest  # noqa: E402


def _req(**kw):
    return OpenRamRunRequest(**kw)


def _config(req, **kw):
    kw.setdefault("tech", "scn4m_subm")
    kw.setdefault("output_name", "sram_8x64")
    kw.setdefault("out_dir", "/workspace/grafux/openram_out")
    return flow.build_openram_config(req, **kw)


# ---------------------------------------------------------------------------
# openram_output_name
# ---------------------------------------------------------------------------

def test_output_name_is_derived_from_the_geometry_when_the_port_is_empty():
    """Two differently-sized macros in one project must not collide on disk."""
    name = flow.openram_output_name(_req(word_size="8", num_words="64"), "scn4m_subm")
    assert name == "sram_8x64_scn4m_subm"


def test_an_explicit_output_name_wins_and_is_made_path_safe():
    name = flow.openram_output_name(_req(output_name="my ram$1"), "scn4m_subm")
    assert name == "my_ram_1"


# ---------------------------------------------------------------------------
# build_openram_config
# ---------------------------------------------------------------------------

def test_the_parameter_ports_become_openram_config_variables():
    text = _config(_req(word_size="8", num_words="64", num_banks="2",
                        num_rw_ports="1", num_r_ports="1", num_w_ports="0",
                        write_size="8"))
    for line in ("word_size = 8", "num_words = 64", "num_banks = 2",
                 "num_rw_ports = 1", "num_r_ports = 1", "num_w_ports = 0",
                 "write_size = 8", 'tech_name = "scn4m_subm"'):
        assert line in text


def test_an_empty_port_writes_no_assignment_at_all():
    """
    An omitted assignment leaves OpenRAM's own default standing.  Writing a
    guess instead would silently build something the user never asked for.
    """
    text = _config(_req(word_size="8"))
    assert "word_size = 8" in text
    assert "num_banks" not in text
    assert "write_size" not in text


def test_drc_and_lvs_are_off_unless_asked_for():
    """
    They need magic and netgen, which the image does not carry, and they can run
    longer than the compile itself.  Written explicitly on every run rather than
    left to the default, because failing late is expensive: the layout work is
    already paid for by the time a checker runs.
    """
    assert "check_lvsdrc = False" in _config(_req())
    assert "check_lvsdrc = True" in _config(_req(check_lvsdrc="1"))


def test_characterization_is_analytical_because_the_image_has_no_simulator():
    """With this off, the run dies at the very end looking for ngspice."""
    assert "analytical_delay = True" in _config(_req())


def test_one_corner_by_default_because_the_lib_port_holds_one_filename():
    assert "nominal_corner_only = True" in _config(_req())


def test_asking_for_corners_replaces_the_single_corner_default():
    text = _config(_req(process_corners="TT,SS", supply_voltages="1.8",
                        temperatures="25, 85"))
    assert 'process_corners = ["TT", "SS"]' in text
    assert "supply_voltages = [1.8]" in text
    assert "temperatures = [25, 85]" in text
    assert "nominal_corner_only" not in text


def test_netlist_only_is_written_only_when_asked():
    assert "netlist_only" not in _config(_req())
    assert "netlist_only = True" in _config(_req(netlist_only="1"))


def test_extra_config_is_appended_verbatim():
    text = _config(_req(extra_config="route_supplies = False"))
    assert text.rstrip().endswith("route_supplies = False")
    assert "# --- extra_config port ---" in text


def test_the_config_port_replaces_the_generated_one_entirely():
    """
    A half-merge is the worst of both: the block face shows word_size 8 while the
    macro comes out 32 wide, and nothing on screen says which won.
    """
    text = _config(_req(word_size="8", config="word_size = 32\nnum_words = 1024\n"))
    assert "word_size = 32" in text
    assert "word_size = 8" not in text


def test_the_output_location_is_forced_even_over_a_user_config():
    """
    The server reads the results back from there.  A user config pointing
    somewhere else produces a run whose outputs are invisible.
    """
    text = _config(_req(config="output_path = '/tmp/elsewhere/'\n"),
                   out_dir="/workspace/grafux/openram_out")
    assert text.rindex('output_path = "/workspace/grafux/openram_out/"') > text.index("/tmp/elsewhere")
    assert 'output_name = "sram_8x64"' in text


def test_the_output_path_always_ends_in_a_slash():
    """OpenRAM concatenates output_path and output_name with NO separator."""
    text = _config(_req(), out_dir="/no/trailing/slash")
    assert 'output_path = "/no/trailing/slash/"' in text


def test_the_generated_config_is_valid_python():
    """It is exec'd by the compiler; a syntax error is a wasted pod."""
    compile(_config(_req(word_size="8", num_words="64",
                         process_corners="TT", supply_voltages="1.8")),
            "<openram config>", "exec")


# ---------------------------------------------------------------------------
# openram_size_warnings
# ---------------------------------------------------------------------------

def test_an_ordinary_size_warns_about_nothing():
    notes, refusal = flow.openram_size_warnings(_req(word_size="8", num_words="64"))
    assert notes == []
    assert refusal == ""


def test_a_large_macro_warns_and_points_at_banking():
    notes, refusal = flow.openram_size_warnings(_req(word_size="32", num_words="16384"))
    assert refusal == ""
    assert any("num_banks" in n for n in notes)


def test_a_non_power_of_two_word_count_is_called_out():
    notes, _ = flow.openram_size_warnings(_req(word_size="8", num_words="1000"))
    assert any("power of two" in n for n in notes)


def test_an_absurd_size_is_refused_before_the_compiler_is_started():
    """
    Nothing else bounds the cost: the block face shows two numbers, and the
    difference between them is the difference between a thirty-second run and a
    multi-hour bill on a rented machine.
    """
    _notes, refusal = flow.openram_size_warnings(_req(word_size="64", num_words="1000000"))
    assert "Refusing to start" in refusal
    assert "num_words" in refusal


# ---------------------------------------------------------------------------
# classify_openram_outputs
# ---------------------------------------------------------------------------

def test_outputs_are_matched_by_extension_not_by_predicted_filename():
    """
    OpenRAM encodes the process corner into the Liberty name and its naming has
    moved between releases, so a filename spelled out in the runner would be a
    guess that fails on a rented machine.
    """
    found = flow.classify_openram_outputs([
        "sram_8x64_scn4m_subm.gds",
        "sram_8x64_scn4m_subm_TT_3p3V_25C.lib",
        "sram_8x64_scn4m_subm.lef",
        "sram_8x64_scn4m_subm.v",
        "sram_8x64_scn4m_subm.sp",
        "sram_8x64_scn4m_subm.html",
        "sram_8x64_scn4m_subm.py",
        "sram_8x64_scn4m_subm.log",
    ])
    assert found["gds"].endswith(".gds")
    assert found["lib"].endswith("_TT_3p3V_25C.lib")
    assert found["lef"].endswith(".lef")
    assert found["verilog_model"].endswith(".v")
    assert found["spice"].endswith(".sp")
    assert found["datasheet"].endswith(".html")
    assert found["config"].endswith(".py")


def test_a_multi_corner_run_takes_the_first_listed_lib():
    """
    The caller passes `ls -S` output (largest first) and must not re-sort it:
    first-match-wins is what resolves several .lib files to the substantive one.
    """
    found = flow.classify_openram_outputs(["m_TT_3p3V_25C.lib", "m_SS_3p0V_85C.lib"])
    assert found["lib"] == "m_TT_3p3V_25C.lib"


def test_a_missing_view_is_simply_absent_rather_than_empty_or_raising():
    found = flow.classify_openram_outputs(["m.sp", "m.v"])
    assert "gds" not in found
    assert found["spice"] == "m.sp"


def test_unknown_extensions_and_blank_lines_are_ignored():
    found = flow.classify_openram_outputs(["", "subdir/", "m.unknown", "m.gds"])
    assert found == {"gds": "m.gds"}


# ---------------------------------------------------------------------------
# globs_for
# ---------------------------------------------------------------------------

def test_the_globs_collect_every_view_plus_the_two_files_that_explain_the_run():
    globs = flow.globs_for("openram", work_dir="/w")
    out = "/w/" + flow.OPENRAM_OUT_DIR
    for ext in ("gds", "lef", "lib", "v", "sp", "html", "log", "py"):
        assert f"{out}/*.{ext}" in globs, ext


def test_the_other_kinds_globs_are_untouched():
    """globs_for is shared; a new branch must not shadow an existing one."""
    assert "/w/*.vcd" in flow.globs_for("verilator", work_dir="/w")
    assert "/w/*.v" in flow.globs_for("yosys", work_dir="/w")


# ---------------------------------------------------------------------------
# run_openram — the decisions around the compile, with the pod faked out
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


@pytest.fixture
def pod(monkeypatch):
    """A pod whose written files, commands and canned outputs are inspectable."""
    state = {
        "written": {},
        "commands": [],
        "stages": [],
        "run": (0, "Total area: 1234.5 um^2", ""),
        # What `ls -S -1` reports, and what those files contain.
        "listing": ("m.gds\nm_TT_3p3V_25C.lib\nm.lef\nm.v\nm.sp\nm.html\n"
                    "m.py\nm.log\n"),
        "pod_files": {
            "m.v": "module m(); endmodule\n",
            "m.py": "word_size = 8\nnum_words = 64\nnum_banks = 1\n",
        },
    }

    def write_file(_sftp, path, content):
        state["written"][path] = content

    def exec_stream(_client, command, **_kw):
        state["commands"].append(command)
        return state["run"]

    def exec_simple(_client, command, **_kw):
        state["commands"].append(command)
        if "ls -S" in command:
            return 0, state["listing"], ""
        return 0, "", ""

    monkeypatch.setattr(flow, "_write_file", write_file)
    monkeypatch.setattr(flow, "exec_stream", exec_stream)
    monkeypatch.setattr(flow, "exec_simple", exec_simple)
    return state


def _run(state, req=None):
    return flow.run_openram(
        _FakeClient(state), req or _req(word_size="8", num_words="64"),
        on_stage=lambda s, d: state["stages"].append((s, d)),
    )


def test_a_successful_run_fills_every_port_it_owns(pod):
    out = _run(pod)["outputs"]
    assert out["status"] == "ok"
    assert out["top"] == "sram_8x64_scn4m_subm"
    assert out["tech_name"] == "scn4m_subm"
    assert out["verilog_model"] == "module m(); endmodule\n"
    assert out["errors"] == ""


def test_the_config_port_reports_what_openram_actually_ran(pod):
    """
    The copy OpenRAM writes beside its outputs carries every default it filled
    in -- the only place the resolved settings are visible, and the reason this
    port is in both the input and the output list.
    """
    out = _run(pod)["outputs"]
    assert out["config"] == pod["pod_files"]["m.py"]


def test_the_generated_config_falls_back_when_openram_wrote_no_copy(pod):
    pod["listing"] = "m.gds\nm.v\n"
    out = _run(pod)["outputs"]
    assert "tech_name" in out["config"]


def test_the_config_is_written_into_the_pod_before_the_compile(pod):
    _run(pod)
    path = f"{flow.WORK_DIR}/{flow.OPENRAM_CONFIG_FILE}"
    assert path in pod["written"]
    assert "word_size = 8" in pod["written"][path]


def test_the_compile_is_not_piped_through_tee(pod):
    """
    A shell pipeline exits with the status of its LAST command, so tee's 0 would
    mask a failing compile entirely -- the same trap run_yosys documents.
    OpenRAM writes its own .log, so there is nothing to pipe.
    """
    _run(pod)
    compile_cmd = next(c for c in pod["commands"] if flow.OPENRAM_CONFIG_FILE in c)
    assert "tee" not in compile_cmd


def test_the_stats_port_is_json_describing_what_was_built(pod):
    out = _run(pod)["outputs"]
    stats = json.loads(out["stats"])
    assert stats["word_size"] == 8
    assert stats["num_words"] == 64
    assert stats["total_bits"] == 512
    assert "gds" in stats["views"]


def test_a_nonzero_exit_reports_the_error_and_still_collects_artifacts(pod):
    pod["run"] = (1, "some output", "ERROR: no such technology")
    result = _run(pod)
    assert result["_status"] == "error"
    assert "no such technology" in result["outputs"]["errors"]
    assert result["_globs"] == flow.globs_for("openram")


def test_a_clean_exit_that_produced_no_macro_is_still_an_error(pod):
    """
    "The tool said fine and built nothing" must not read as success -- OpenRAM
    can exit 0 on a partial run.
    """
    pod["listing"] = "m.log\n"
    out = _run(pod)["outputs"]
    assert out["status"] == "error"
    assert "without producing a macro" in out["errors"]


def test_netlist_only_is_green_without_a_gds(pod):
    """It is the documented way to ask for exactly that."""
    pod["listing"] = "m.sp\nm.v\nm.log\n"
    out = _run(pod, _req(word_size="8", num_words="64", netlist_only="1"))["outputs"]
    assert out["status"] == "ok"


def test_an_absurd_size_never_reaches_the_compiler(pod):
    out = _run(pod, _req(word_size="64", num_words="1000000"))["outputs"]
    assert out["status"] == "error"
    assert "Refusing to start" in out["errors"]
    assert pod["commands"] == []


def test_using_the_config_port_is_reported_on_warnings(pod):
    """
    Otherwise "I changed word_size and nothing happened" is unanswerable from
    the block.
    """
    out = _run(pod, _req(word_size="8", config="word_size = 32\n"))["outputs"]
    assert "`config` port was set" in out["warnings"]


def test_asking_for_drc_warns_that_the_default_image_cannot_do_it(pod):
    out = _run(pod, _req(word_size="8", num_words="64", check_lvsdrc="1"))["outputs"]
    assert "magic and netgen" in out["warnings"]


def test_an_oversized_model_points_the_port_at_the_artifact_instead(pod, monkeypatch):
    monkeypatch.setattr(flow, "NETLIST_INLINE_MAX", 4)
    out = _run(pod)["outputs"]
    assert out["verilog_model"] == ""
    assert "exceeded the inline limit" in out["warnings"]


def test_the_stages_bracket_the_compile(pod):
    _run(pod)
    assert [s for s, _ in pod["stages"]] == [
        "config", "config", "compile", "compile", "collect", "collect"]
