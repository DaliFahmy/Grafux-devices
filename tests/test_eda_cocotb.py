"""
test_eda_cocotb.py
Unit tests for the cocotb half of ``EDA.flow`` — testbench classification, the
generated runner script, and the result/coverage parsers.

Pure string-in/string-out, like ``test_eda_flow.py``: no container, no cloud, no
simulator.  That matters more here than anywhere else in this package, because the
alternative way to find out that a build flag is wrong or that ``results.xml``
landed in the wrong directory is to rent a machine and wait for it to boot.

The fixtures under ``tests/fixtures/eda`` are shared with the end-to-end script.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from EDA import flow  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "eda")


def fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# testbench_kind / resolve_verilator_mode
# ---------------------------------------------------------------------------

def test_testbench_kind_recognises_cocotb():
    assert flow.testbench_kind(fixture("test_sync_fifo.py")) == "python"


def test_testbench_kind_recognises_the_generated_cpp_harness():
    assert flow.testbench_kind(flow.default_testbench("sync_fifo")) == "cpp"


def test_testbench_kind_falls_back_to_parsing_it_as_python():
    """
    A Python testbench that imports cocotb indirectly still has to be recognised:
    handing it to a C++ compiler produces a wall of syntax errors that reads as
    the user's fault rather than the tool's misclassification.
    """
    assert flow.testbench_kind("from helpers import *\n\nasync def go(dut):\n    pass\n") == "python"


def test_testbench_kind_is_empty_when_it_cannot_tell():
    assert flow.testbench_kind("") == ""
    assert flow.testbench_kind("   \n  ") == ""
    assert flow.testbench_kind("module tb; endmodule") == ""


def test_resolve_mode_promotes_sim_to_cocotb_for_a_python_testbench():
    """
    The default everywhere is mode="sim", so wiring a testbench block into an
    existing verilator block has to just work.
    """
    mode, note = flow.resolve_verilator_mode("sim", fixture("test_sync_fifo.py"))
    assert mode == "cocotb"
    assert "cocotb" in note


def test_resolve_mode_leaves_a_cpp_harness_alone():
    assert flow.resolve_verilator_mode("sim", flow.default_testbench("f")) == ("sim", "")


def test_resolve_mode_never_promotes_lint():
    """Lint is a deliberate "do not simulate"; promoting it would rent a pod."""
    assert flow.resolve_verilator_mode("lint", fixture("test_sync_fifo.py")) == ("lint", "")


def test_resolve_mode_notes_an_explicit_cocotb_request_with_no_testbench():
    mode, note = flow.resolve_verilator_mode("cocotb", "")
    assert mode == "cocotb"
    assert note


def test_resolve_mode_of_an_explicit_clean_request_has_nothing_to_say():
    assert flow.resolve_verilator_mode("cocotb", fixture("test_sync_fifo.py")) == ("cocotb", "")


# ---------------------------------------------------------------------------
# build flags
# ---------------------------------------------------------------------------

def test_cocotb_build_args_always_pass_timing_and_public_flat_rw():
    """
    Without --timing cocotb 2.x hangs on the first clock edge instead of failing,
    and without --public-flat-rw every dut.<signal> lookup fails with "not found"
    — which reads as a broken testbench. These two are the flags whose absence is
    hardest to diagnose from inside a pod.
    """
    args = flow.cocotb_build_args("verilator")
    assert "--timing" in args
    assert "--public-flat-rw" in args


def test_cocotb_build_args_follow_the_trace_and_coverage_switches():
    on = flow.cocotb_build_args("verilator", trace=True, coverage=True)
    assert "--trace" in on and "--coverage" in on
    off = flow.cocotb_build_args("verilator", trace=False, coverage=False)
    assert "--trace" not in off and "--coverage" not in off
    assert "--timing" in off


def test_cocotb_build_args_only_assert_when_sva_is_wired():
    assert "--assert" not in flow.cocotb_build_args("verilator")
    assert "--assert" in flow.cocotb_build_args("verilator", assertions=True)


def test_cocotb_build_args_for_icarus_use_the_2012_dialect():
    assert flow.cocotb_build_args("icarus") == ["-g2012"]


def test_cocotb_build_args_append_the_users_own_flags_last():
    args = flow.cocotb_build_args("verilator", extra_flags="-Wno-WIDTH --x-assign 0")
    assert args[-3:] == ["-Wno-WIDTH", "--x-assign", "0"]


@pytest.mark.parametrize("given,expected", [
    ("iverilog", "icarus"), ("Icarus", "icarus"), ("ICARUS", "icarus"),
    ("verilator", "verilator"), ("", "verilator"), ("nonsense", "verilator"),
])
def test_normalize_simulator(given, expected):
    assert flow.normalize_simulator(given) == expected


# ---------------------------------------------------------------------------
# the generated runner script
# ---------------------------------------------------------------------------

def _script(**kwargs) -> str:
    base = dict(top="sync_fifo", sources=["sync_fifo.v"], test_module="test_sync_fifo")
    base.update(kwargs)
    return flow.build_cocotb_runner_script(**base)


def test_runner_script_is_valid_python():
    """It is written into the pod and run there; a quoting slip surfaces late."""
    ast.parse(_script())


def test_runner_script_prefers_the_cocotb_2x_import():
    """``cocotb.runner`` is the 1.x path; both are tried, 2.x first."""
    script = _script()
    assert "from cocotb_tools.runner import get_runner" in script
    assert "from cocotb.runner import get_runner" in script


def test_runner_script_writes_results_to_an_absolute_path():
    """A relative results_xml lands in the build dir, where nothing looks for it."""
    assert 'os.path.abspath("results.xml")' in _script()


def test_runner_script_pins_the_test_dir():
    """
    cocotb's runner defaults test_dir to the build directory, which would both
    hide results.xml and break the import of the test module written beside the
    design.
    """
    script = _script()
    assert "TEST_DIR = os.getcwd()" in script
    assert "test_dir=TEST_DIR" in script


def test_runner_script_sets_both_seed_env_spellings():
    """cocotb 2.0 renamed these; setting only one silently does nothing."""
    script = _script(seed="4242")
    assert 'COCOTB_RANDOM_SEED' in script
    assert 'RANDOM_SEED' in script
    assert 'SEED = "4242"' in script


def test_runner_script_disables_ansi_colour():
    """Escape codes in the log port are noise the user cannot turn off."""
    assert 'COCOTB_ANSI_OUTPUT="0"' in _script()


def test_runner_script_threads_the_testcase_filter():
    assert 'TESTCASES = ["test_a", "test_b"]' in _script(tests="test_a, test_b")


def test_runner_script_drops_a_non_numeric_seed():
    """A stray character in a port must not crash the run inside the pod."""
    assert 'SEED = ""' in _script(seed="not-a-number")


def test_runner_script_carries_every_source():
    script = _script(sources=["f.v", "sva.sv"], assertions=True)
    assert 'SOURCES = ["f.v", "sva.sv"]' in script


def test_runner_script_survives_a_failing_test_without_exiting_nonzero():
    """
    cocotb's runner raises SystemExit when a test fails. That is an expected
    outcome for this block, and letting it escape would make a failing design
    indistinguishable from a crashed runner.
    """
    script = _script()
    assert "except SystemExit as exc:" in script
    assert "GRAFUX_TESTS_FAILED" in script


def test_runner_script_resolves_libpython_before_the_simulation():
    """
    cocotb EMBEDS CPython in the simulator, so it needs the SHARED libpython and
    dies with "Unable to find libpython" without it -- AFTER a clean build, which
    is what makes it read like a design problem. The image installs the library;
    the script resolves it anyway, because a pod pinned to an older image is
    otherwise unusable for reasons the log does not explain.
    """
    script = _script()
    assert "def resolve_libpython():" in script
    # os.environ, not extra_env alone: the runner copies os.environ over its own
    # env dict, so the extra_env half is the version-dependent one.
    assert 'os.environ["LIBPYTHON_LOC"] = LIBPYTHON' in script
    assert 'ENV["LIBPYTHON_LOC"] = LIBPYTHON' in script
    assert script.index("resolve_libpython()") < script.index("runner.test(")


def test_runner_script_names_a_missing_libpython_as_a_marker():
    """run_cocotb keys off this to say "image", not "your testbench"."""
    assert "GRAFUX_LIBPYTHON_MISSING" in _script()


def test_runner_script_announces_its_stages():
    script = _script()
    assert 'print("GRAFUX_STAGE build", flush=True)' in script
    assert 'print("GRAFUX_STAGE sim", flush=True)' in script


# ---------------------------------------------------------------------------
# sva_binding_problem
# ---------------------------------------------------------------------------

def test_sva_with_a_bind_statement_is_accepted():
    assert flow.sva_binding_problem("bind sync_fifo fifo_chk chk(.*);") == ""


def test_sva_without_a_bind_is_refused():
    """
    An unbound checker compiles cleanly, checks nothing and reports success — a
    false green, the one outcome a verification block must never produce.
    """
    problem = flow.sva_binding_problem("module chk; assert property (1); endmodule")
    assert "bind" in problem


def test_sva_bind_match_is_a_whole_word():
    assert flow.sva_binding_problem("// rebinding the checker later") != ""


def test_no_sva_is_not_a_problem():
    assert flow.sva_binding_problem("") == ""
    assert flow.sva_binding_problem("   ") == ""


# ---------------------------------------------------------------------------
# parse_cocotb_results
# ---------------------------------------------------------------------------

def test_parse_results_counts_a_clean_run():
    results = flow.parse_cocotb_results(fixture("results_pass.xml"))
    assert (results["total"], results["passed"], results["failed"]) == (3, 3, 0)
    assert "error" not in results


def test_parse_results_separates_failures_from_skips():
    results = flow.parse_cocotb_results(fixture("results_fail.xml"))
    assert (results["total"], results["passed"], results["failed"],
            results["skipped"]) == (5, 2, 2, 1)
    failed = [t["name"] for t in results["tests"] if t["status"] == "failed"]
    assert failed == ["test_full_asserts_at_depth", "test_count_must_not_exceed_depth"]


def test_parse_results_keeps_the_assertion_message():
    results = flow.parse_cocotb_results(fixture("results_fail.xml"))
    failing = next(t for t in results["tests"] if t["name"] == "test_full_asserts_at_depth")
    assert "full must assert after 8 writes" in failing["message"]


def test_parse_results_treats_a_run_with_no_tests_as_an_error():
    """
    The worst lie a verification block can tell is "passed" for a testbench that
    declared nothing — which is exactly what an exit code of 0 would imply.
    """
    results = flow.parse_cocotb_results("<testsuites></testsuites>")
    assert results["total"] == 0
    assert "no tests" in results["error"]


def test_parse_results_survives_a_truncated_report():
    results = flow.parse_cocotb_results("<testsuites><testsuite><testca")
    assert results["total"] == 0
    assert "could not be parsed" in results["error"]


def test_parse_results_survives_a_missing_report():
    assert flow.parse_cocotb_results("")["error"] == "no results.xml was produced"


def test_parse_results_counts_elements_not_the_suite_attribute():
    """Some cocotb versions write a tests= attribute that disagrees with reality."""
    xml = ('<testsuites><testsuite tests="99">'
           '<testcase name="a"/><testcase name="b"/></testsuite></testsuites>')
    assert flow.parse_cocotb_results(xml)["total"] == 2


def test_parse_results_treats_an_error_element_as_a_failure():
    xml = ('<testsuites><testsuite><testcase name="a">'
           '<error message="the simulator crashed"/></testcase></testsuite></testsuites>')
    results = flow.parse_cocotb_results(xml)
    assert results["failed"] == 1
    assert results["tests"][0]["message"] == "the simulator crashed"


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_parse_lcov_sums_every_source_file():
    coverage = flow.parse_lcov_summary(fixture("coverage.info"))
    assert coverage["lines"] == {"hit": 11, "total": 14, "pct": 78.6}
    assert coverage["branches"] == {"hit": 4, "total": 6, "pct": 66.7}


def test_parse_lcov_of_an_empty_report_is_zero_not_a_crash():
    assert flow.parse_lcov_summary("")["lines"] == {"hit": 0, "total": 0, "pct": 0.0}


def test_coverage_cmd_locates_the_dat_file_at_run_time():
    """
    Verilator writes coverage.dat wherever the simulation ran, and which directory
    that is depends on the cocotb version — so it is found, not assumed.
    """
    cmd = flow.build_coverage_cmd()
    assert "coverage.dat" in cmd
    assert "sim_build/coverage.dat" in cmd
    assert "verilator_coverage --write-info coverage.info" in cmd


# ---------------------------------------------------------------------------
# summarize_failures — this text becomes the RTL fix prompt's feedback
# ---------------------------------------------------------------------------

def test_summarize_failures_names_the_test_and_quotes_the_assertion():
    results = flow.parse_cocotb_results(fixture("results_fail.xml"))
    text = flow.summarize_failures(results)
    assert text.startswith("2 of 5 cocotb tests failed.")
    assert "FAILED test_full_asserts_at_depth" in text
    assert "full must assert after 8 writes" in text


def test_summarize_failures_leaves_passing_tests_out_of_the_report():
    results = flow.parse_cocotb_results(fixture("results_fail.xml"))
    assert "test_reset_values" not in flow.summarize_failures(results)


def test_summarize_failures_is_empty_for_a_clean_run():
    """An empty ``failures`` port is what tells the loop it can stop."""
    assert flow.summarize_failures(flow.parse_cocotb_results(fixture("results_pass.xml"))) == ""


def test_summarize_failures_reports_a_missing_report_as_the_failure():
    assert flow.summarize_failures(flow.parse_cocotb_results("")) == (
        "no results.xml was produced")


def test_summarize_failures_caps_the_number_of_tests_listed():
    results = {"total": 30, "failed": 30, "passed": 0, "skipped": 0, "tests": [
        {"name": f"test_{i}", "status": "failed", "message": "boom"} for i in range(30)
    ]}
    text = flow.summarize_failures(results, max_tests=3)
    assert text.count("FAILED ") == 3
    assert "and 27 more failing tests" in text


def test_summarize_failures_caps_total_length():
    """It goes into a port file and an LLM prompt; neither wants a megabyte."""
    results = {"total": 2, "failed": 2, "passed": 0, "skipped": 0, "tests": [
        {"name": "test_a", "status": "failed", "message": "x" * 5000},
        {"name": "test_b", "status": "failed", "message": "y" * 5000},
    ]}
    text = flow.summarize_failures(results, max_chars=500)
    assert len(text) < 600
    assert text.endswith("(truncated)")


def test_summarize_failures_collapses_multiline_assertion_messages():
    results = {"total": 1, "failed": 1, "passed": 0, "skipped": 0, "tests": [
        {"name": "test_a", "status": "failed", "message": "line one\n  line two\n"},
    ]}
    assert "line one line two" in flow.summarize_failures(results)


# ---------------------------------------------------------------------------
# artifact globs
# ---------------------------------------------------------------------------

def test_globs_for_verilator_collect_the_cocotb_artifacts():
    globs = flow.globs_for("verilator", work_dir="/w")
    for expected in ("/w/results.xml", "/w/coverage.info", "/w/sim_build/*.fst",
                     "/w/sim_build/*.vcd"):
        assert expected in globs


def test_globs_for_verilator_still_collect_the_cpp_sim_waveform():
    """The sim/lint mode did not go away; its artifacts must still come back."""
    globs = flow.globs_for("verilator", work_dir="/w")
    assert "/w/*.vcd" in globs
    assert "/w/obj_dir/*.log" in globs


# ---------------------------------------------------------------------------
# run_cocotb — the decision logic, with the pod faked out
# ---------------------------------------------------------------------------

class _FakeSftp:
    def close(self):
        pass


class _FakeClient:
    def open_sftp(self):
        return _FakeSftp()


class _Req:
    """The subset of VerilatorRunRequest that run_cocotb reads."""

    def __init__(self, **kw):
        self.rtl = kw.get("rtl", "module sync_fifo(input clk); endmodule")
        self.testbench = kw.get("testbench", "import cocotb")
        self.top = kw.get("top", "sync_fifo")
        self.mode = kw.get("mode", "sim")
        self.trace = kw.get("trace", "1")
        self.timeout = kw.get("timeout", 900)
        self.verilator_flags = kw.get("verilator_flags", "")
        self.sva = kw.get("sva", "")
        self.simulator = kw.get("simulator", "verilator")
        self.tests = kw.get("tests", "")
        self.seed = kw.get("seed", "")
        self.coverage = kw.get("coverage", "1")


@pytest.fixture
def pod(monkeypatch):
    """A pod whose files, commands and canned outputs are all inspectable."""
    state = {
        "files": {}, "commands": [], "stages": [],
        "run": (0, "", ""),          # exit code, stdout, stderr of run_cocotb.py
        "results_xml": "", "coverage_info": "", "coverage_rc": 0,
        # The steady state on a current image: the library is already there.
        "libpython": "GRAFUX_LIBPYTHON_PRESENT",
    }

    def write_file(_sftp, path, content):
        state["files"][path] = content

    def exec_stream(_client, command, **kwargs):
        state["commands"].append(command)
        on_line = kwargs.get("on_line")
        if "run_cocotb.py" in command:
            code, out, err = state["run"]
            if on_line:
                for line in out.splitlines():
                    on_line(line)
            return code, out, err
        return state["coverage_rc"], "", ""

    def exec_simple(_client, command, **_kw):
        state["commands"].append(command)
        if "results.xml" in command:
            return 0, state["results_xml"], ""
        if "coverage.info" in command:
            return 0, state["coverage_info"], ""
        if "GRAFUX_LIBPYTHON" in command:
            return 0, state["libpython"], ""
        return 0, "", ""

    monkeypatch.setattr(flow, "_write_file", write_file)
    monkeypatch.setattr(flow, "exec_stream", exec_stream)
    monkeypatch.setattr(flow, "exec_simple", exec_simple)
    return state


def _run(pod, req=None):
    return flow.run_cocotb(
        _FakeClient(), req or _Req(),
        on_stage=lambda name, status: pod["stages"].append((name, status)),
    )


def test_run_cocotb_reports_a_clean_run_as_passed(pod):
    pod["results_xml"] = fixture("results_pass.xml")
    pod["coverage_info"] = fixture("coverage.info")
    outcome = _run(pod)
    outputs = outcome["outputs"]
    assert outputs["passed"] == "true"
    assert outputs["status"] == "ok"
    assert outputs["failures"] == ""
    assert outcome["_status"] == "ok"
    assert "78.6" in outputs["coverage"]


def test_run_cocotb_fails_a_run_whose_tests_failed_despite_exit_code_zero(pod):
    """
    THE test for this feature. cocotb's runner can exit 0 with failing tests, and
    trusting the exit code would paint a broken design green.
    """
    pod["run"] = (0, "GRAFUX_STAGE build\nGRAFUX_STAGE sim\n", "")
    pod["results_xml"] = fixture("results_fail.xml")
    outcome = _run(pod)
    assert outcome["outputs"]["passed"] == "false"
    assert outcome["_status"] == "error"
    assert "test_full_asserts_at_depth" in outcome["outputs"]["failures"]
    assert "test_full_asserts_at_depth" in outcome["outputs"]["errors"]


def test_run_cocotb_fails_a_run_that_declared_no_tests(pod):
    """A testbench that failed to import collects nothing and exits 0."""
    pod["results_xml"] = "<testsuites></testsuites>"
    outputs = _run(pod)["outputs"]
    assert outputs["passed"] == "false"
    assert "no tests" in outputs["errors"]


def test_run_cocotb_fails_when_results_xml_is_missing(pod):
    pod["run"] = (0, "", "")
    pod["results_xml"] = ""
    outputs = _run(pod)["outputs"]
    assert outputs["passed"] == "false"
    assert "results.xml" in outputs["errors"]


def test_run_cocotb_reports_a_build_failure_before_any_test_ran(pod):
    pod["run"] = (2, "GRAFUX_STAGE build\nGRAFUX_BUILD_FAILED\n", "syntax error")
    outputs = _run(pod)["outputs"]
    assert outputs["passed"] == "false"
    assert "did not build" in outputs["failures"]


def test_run_cocotb_blames_the_image_when_libpython_is_missing(pod):
    """
    The design built and Verilator was happy; only the simulator failed to start.
    Without this the user is handed cocotb's ValueError under a clean build log
    and reasonably concludes their RTL or testbench is at fault.
    """
    pod["run"] = (3, "GRAFUX_STAGE build\nGRAFUX_LIBPYTHON_MISSING\nGRAFUX_STAGE sim\n",
                  "ValueError: Unable to find libpython")
    outputs = _run(pod)["outputs"]
    assert outputs["passed"] == "false"
    assert "libpython" in outputs["failures"]
    assert "image problem" in outputs["failures"]
    assert "libpython" in outputs["errors"]


def test_run_cocotb_recognises_cocotbs_own_libpython_message(pod):
    """A pod running an older run_cocotb.py prints no marker of ours."""
    pod["run"] = (3, "GRAFUX_STAGE build\nGRAFUX_STAGE sim\n",
                  "ValueError: Unable to find libpython, please make sure ...")
    assert "image problem" in _run(pod)["outputs"]["failures"]


def test_run_cocotb_does_not_cry_libpython_on_an_ordinary_failure(pod):
    pod["results_xml"] = fixture("results_fail.xml")
    assert "libpython" not in _run(pod)["outputs"]["failures"]


def test_run_cocotb_heals_the_pod_before_it_runs_anything(pod):
    """
    A pod is created once and reused for the life of its block, and the image tag
    is not part of any reuse key -- so fixing the image does nothing for a pod
    that is already warm.  The check therefore has to happen on the way IN, while
    it can still make this run succeed.
    """
    _run(pod)
    preflight = [i for i, c in enumerate(pod["commands"]) if "ldconfig" in c]
    runner = [i for i, c in enumerate(pod["commands"]) if "run_cocotb.py" in c]
    assert preflight and runner
    assert preflight[0] < runner[0]


def test_run_cocotb_says_nothing_when_the_pod_already_has_libpython(pod):
    """
    This runs on every cocotb invocation, so on a correct image it must be
    invisible; a warning that fires every time is one nobody reads.
    """
    pod["results_xml"] = fixture("results_pass.xml")
    pod["coverage_info"] = fixture("coverage.info")
    outputs = _run(pod)["outputs"]
    assert "libpython" not in outputs["warnings"]
    assert outputs["passed"] == "true"


def test_run_cocotb_says_so_when_it_installed_libpython_itself(pod):
    """
    The install costs the user half a minute they did not ask for; the run should
    still pass, and the warnings port should explain where the time went.
    """
    pod["libpython"] = "GRAFUX_LIBPYTHON_INSTALLED"
    pod["results_xml"] = fixture("results_pass.xml")
    pod["coverage_info"] = fixture("coverage.info")
    outcome = _run(pod)
    assert "libpython" in outcome["outputs"]["warnings"]
    assert outcome["_status"] == "ok"


def test_run_cocotb_runs_anyway_when_libpython_cannot_be_installed(pod):
    """
    An offline pod must still get its run and its own diagnosis from the runner
    script's marker, rather than being failed here on a guess.
    """
    pod["libpython"] = "GRAFUX_LIBPYTHON_UNAVAILABLE"
    pod["results_xml"] = fixture("results_pass.xml")
    pod["coverage_info"] = fixture("coverage.info")
    outcome = _run(pod)
    assert "libpython" in outcome["outputs"]["warnings"]
    assert any("run_cocotb.py" in c for c in pod["commands"])


def test_run_cocotb_survives_a_pod_that_does_not_answer_the_preflight(pod, monkeypatch):
    """A preflight that raises must not take the run down with it."""
    real = flow.exec_simple

    def flaky(client, command, **kw):
        if "GRAFUX_LIBPYTHON" in command:
            raise OSError("socket closed")
        return real(client, command, **kw)

    monkeypatch.setattr(flow, "exec_simple", flaky)
    outputs = _run(pod)["outputs"]
    assert "libpython" in outputs["warnings"]
    assert any("run_cocotb.py" in c for c in pod["commands"])


def test_lint_never_touches_apt(pod):
    """
    The preflight is scoped to the cocotb path on purpose: a lint run has no
    business installing packages on someone's pod.
    """
    class _LintReq(_Req):
        """Lint reads two fields the cocotb path never looks at."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.defines = ""
            self.include_dirs = ""

    flow.run_verilator(_FakeClient(), _LintReq(mode="lint", testbench=""),
                       on_stage=lambda *a: None)
    assert not any("ldconfig" in c for c in pod["commands"])


def test_run_cocotb_writes_the_design_testbench_and_runner_into_the_pod(pod):
    _run(pod)
    names = sorted(path.rsplit("/", 1)[-1] for path in pod["files"])
    assert names == ["run_cocotb.py", "sync_fifo.v", "test_sync_fifo.py"]


def test_run_cocotb_names_the_source_after_the_top_module(pod):
    """Verilator's DECLFILENAME check is fatal when they disagree."""
    _run(pod, _Req(top="my$fifo"))
    assert any(path.endswith("/my_fifo.v") for path in pod["files"])


def test_run_cocotb_emits_build_and_sim_stages_from_the_markers(pod):
    pod["run"] = (0, "GRAFUX_STAGE build\nbuilding...\nGRAFUX_STAGE sim\nrunning...\n"
                     "GRAFUX_STAGE report\n", "")
    pod["results_xml"] = fixture("results_pass.xml")
    _run(pod)
    assert ("build", "running") in pod["stages"]
    assert ("sim", "running") in pod["stages"]
    assert ("build", "done") in pod["stages"]


def test_run_cocotb_does_not_forward_the_stage_markers_as_log_lines(pod):
    """They are protocol, not output; showing them would be noise in the log tail."""
    seen = []
    pod["run"] = (0, "GRAFUX_STAGE build\nreal output\n", "")
    pod["results_xml"] = fixture("results_pass.xml")
    flow.run_cocotb(_FakeClient(), _Req(),
                    on_stage=lambda *a: None, on_line=seen.append)
    assert seen == ["real output"]


def test_run_cocotb_explains_why_it_took_the_cocotb_path(pod):
    pod["results_xml"] = fixture("results_pass.xml")
    outputs = flow.run_cocotb(_FakeClient(), _Req(), on_stage=lambda *a: None,
                              note="promoted from sim")["outputs"]
    assert "promoted from sim" in outputs["warnings"]


def test_run_cocotb_skips_unbindable_sva_rather_than_compiling_it(pod):
    """An unbound checker compiles, checks nothing, and reports success."""
    pod["results_xml"] = fixture("results_pass.xml")
    outputs = _run(pod, _Req(sva="module chk; assert property (1); endmodule"))["outputs"]
    assert not any(path.endswith("/sva.sv") for path in pod["files"])
    assert "bind" in outputs["warnings"]


def test_run_cocotb_compiles_sva_that_binds_itself(pod):
    pod["results_xml"] = fixture("results_pass.xml")
    _run(pod, _Req(sva="bind sync_fifo chk c(.*);"))
    assert any(path.endswith("/sva.sv") for path in pod["files"])


def test_run_cocotb_turns_coverage_off_for_icarus(pod):
    """Coverage is a Verilator feature; 0% would look like a broken testbench."""
    pod["results_xml"] = fixture("results_pass.xml")
    outputs = _run(pod, _Req(simulator="icarus"))["outputs"]
    assert outputs["coverage"] == ""
    assert "icarus" in outputs["warnings"]


def test_run_cocotb_reports_a_timeout_without_claiming_a_verdict(pod):
    pod["run"] = (-2, "", "")
    outcome = _run(pod)
    assert outcome["outputs"]["passed"] == "false"
    assert "timeout" in outcome["outputs"]["errors"]
    assert outcome["outputs"]["results"] == ""


def test_run_cocotb_reports_cancellation_distinctly(pod):
    pod["run"] = (-1, "", "")
    assert "cancelled" in _run(pod)["outputs"]["errors"]


def test_run_cocotb_runs_python_with_the_cocotb_venv_and_unbuffered(pod):
    """
    Docker ENV is not inherited by a non-login SSH exec, and buffered output
    would make the live log and the stage markers arrive only at the very end.
    """
    _run(pod)
    run_cmd = next(c for c in pod["commands"] if "run_cocotb.py" in c)
    assert "/opt/cocotb-venv/bin" in run_cmd
    assert "PYTHONUNBUFFERED=1" in run_cmd


def test_run_cocotb_echoes_the_design_through_for_the_next_block(pod):
    """verilator.rtl -> yosys.rtl keeps the canvas pipeline linear."""
    pod["results_xml"] = fixture("results_pass.xml")
    outputs = _run(pod)["outputs"]
    assert outputs["rtl"] == _Req().rtl
    assert outputs["top"] == "sync_fifo"


def test_run_verilator_hands_a_python_testbench_to_the_cocotb_runner(pod):
    """
    The wiring users are already told to make - testbench.testbench into
    verilator.testbench - has to work on a block whose mode is still "sim".
    """
    pod["results_xml"] = fixture("results_pass.xml")
    outputs = flow.run_verilator(
        _FakeClient(), _Req(testbench=fixture("test_sync_fifo.py")),
        on_stage=lambda *a: None)["outputs"]
    assert outputs["passed"] == "true"
    assert "results" in outputs          # only the cocotb path produces this
