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
import json
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


def test_summarize_failures_preserves_the_shape_of_a_multiline_message():
    """
    The message's own line breaks are the diagnosis, not noise.

    This used to be flattened by ``" ".join(msg.split())``, which is exactly what
    turned an expected-vs-got block, or a small traceback, into an unreadable
    ribbon.  Both lines must survive, each on its own line.
    """
    results = {"total": 1, "failed": 1, "passed": 0, "skipped": 0, "tests": [
        {"name": "test_a", "status": "failed", "message": "line one\n  line two\n"},
    ]}
    text = flow.summarize_failures(results)
    assert "line one" in text
    assert "line two" in text
    assert "line one line two" not in text


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
        # The full log the generated runner tees to disk. Empty is the STALE POD
        # case -- a warm pod still running a run_cocotb.py that predates the tee
        # -- and the runner must fall back to the 400-line stdout tail there.
        "cocotb_log": "",
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
        if "cocotb.log" in command:
            return 0, state["cocotb_log"], ""
        if "GRAFUX_LIBPYTHON" in command:
            return 0, state["libpython"], ""
        return 0, "", ""

    monkeypatch.setattr(flow, "_write_file", write_file)
    monkeypatch.setattr(flow, "exec_stream", exec_stream)
    monkeypatch.setattr(flow, "exec_simple", exec_simple)
    return state



def _by_name(results, name):
    """One testcase out of a parsed report, by its test name."""
    return next(t for t in results["tests"] if t["name"] == name)

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
    # `errors` is WHERE the run broke. This run did not break -- it built clean
    # and produced a verdict -- so it has no error location at all, and naming a
    # failing test here would just duplicate `failures`.
    assert outcome["outputs"]["errors"] == ""


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


# ---------------------------------------------------------------------------
# Failure detail: WHY a test failed
#
# The old parser kept one whitespace-collapsed line per failure, and that line
# was simultaneously the `failures` port, the `errors` port, the block tooltip
# and the whole input to the RTL fix prompt. These tests pin the evidence that
# used to be thrown away on the floor of parse_cocotb_results.
# ---------------------------------------------------------------------------

def test_parse_keeps_the_failure_body_the_message_attribute_used_to_hide():
    """
    `child.get("message") or child.text` let the ATTRIBUTE win, and cocotb always
    sets it -- so the traceback in the element body was discarded on every real
    report.
    """
    results = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    failed = _by_name(results, "test_full_asserts_at_depth")
    assert failed["type"] == "AssertionError"
    assert "Traceback (most recent call last):" in failed["detail"]
    assert "test_sync_fifo.py" in failed["detail"]


def test_parse_keeps_message_byte_identical_to_the_old_contract():
    """
    `message` is the compatibility anchor: saved results.txt files hold it, the
    app's VerificationResults::parse reads it, and four tests assert on it. The
    attribute still wins and the body must not leak into it.
    """
    results = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    failed = _by_name(results, "test_full_asserts_at_depth")
    assert failed["message"] == (
        "full must assert after 8 writes (spec 5: full iff count == 8), "
        "got 0 with count=8")
    assert "Traceback" not in failed["message"]


def test_parse_keeps_a_second_diagnostic_child_and_the_captured_streams():
    """The old loop `break`ed after the first, and never looked at system-out."""
    results = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    second = _by_name(results, "test_count_must_not_exceed_depth")
    assert second["extra"][0]["type"] == "RuntimeError"
    first = _by_name(results, "test_full_asserts_at_depth")
    assert "write 8 accepted" in first["stdout"]
    assert "test_full_asserts_at_depth failed" in first["stderr"]


def test_parse_does_not_pay_for_detail_on_passing_tests():
    """They are the bulk of a large run and carry nothing worth keeping."""
    results = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    passing = _by_name(results, "test_reset_values")
    assert passing["status"] == "passed"
    for key in ("detail", "stdout", "stderr", "extra"):
        assert key not in passing


def test_locate_python_failure_prefers_the_users_file_over_cocotbs_own():
    """
    The DEEPEST frame is inside cocotb. Reporting it reads as "the bug is in
    cocotb" and sends the user to a file they cannot edit; the actionable frame
    is the last one in their own testbench.
    """
    results = flow.parse_cocotb_results(fixture("results_xz.xml"))
    detail = results["tests"][0]["detail"]
    deepest = flow.locate_python_failure(detail)
    assert deepest["file"] == "handle.py"
    chosen = flow.locate_python_failure(detail, prefer="test_sync_fifo.py")
    assert chosen["file"] == "test_sync_fifo.py"
    assert chosen["line"] == 73


def test_quote_source_line_never_guesses():
    source = "one\ntwo\nthree\n"
    assert "1 | one" in flow.quote_source_line(source, 1)
    assert "3 | three" in flow.quote_source_line(source, 3)
    assert flow.quote_source_line(source, 4) == ""
    assert flow.quote_source_line(source, 0) == ""
    assert flow.quote_source_line("", 1) == ""


def test_quote_source_line_puts_the_caret_under_the_column():
    quoted = flow.quote_source_line("abcdef", 1, col=3)
    caret = quoted.splitlines()[-1]
    body = quoted.splitlines()[0]
    assert caret.index("^") == body.index("abcdef") + 2


@pytest.mark.parametrize("name,hint_id", [
    ("results_xz.xml", "xz_in_comparison"),
    ("results_timeout.xml", "sim_timeout"),
    ("results_attr.xml", "no_such_signal"),
    ("results_fail_detail.xml", "assert_mismatch"),
])
def test_each_cocotb_rule_fires_on_its_own_fixture(name, hint_id):
    """
    Asserted on the ID, not the prose, so the wording can be improved without
    rewriting a test -- the same reason _ORFS_HINTS entries are matched by marker.
    """
    results = flow.parse_cocotb_results(fixture(name))
    flow.enrich_failures(results, testbench=fixture("test_sync_fifo.py"),
                         rtl=fixture("sync_fifo_good.v"), top="sync_fifo",
                         stem="sync_fifo")
    failing = [t for t in results["tests"] if t["status"] == "failed"]
    assert failing[0]["hint_id"] == hint_id
    assert failing[0]["hint"]


def test_the_missing_signal_rule_names_the_closest_real_port():
    results = flow.parse_cocotb_results(fixture("results_attr.xml"))
    flow.enrich_failures(results, testbench=fixture("test_sync_fifo.py"),
                         rtl=fixture("sync_fifo_good.v"), top="sync_fifo",
                         stem="sync_fifo")
    hint = results["tests"][0]["hint"]
    assert "dut.fulll" in hint
    assert "Did you mean `full`?" in hint


def test_the_missing_signal_rule_stays_silent_when_the_ports_are_unknown():
    """
    module_ports() returning [] means "could not tell", never "portless". Listing
    an empty port list would be a confident lie about the user's design.
    """
    results = flow.parse_cocotb_results(fixture("results_attr.xml"))
    flow.enrich_failures(results, testbench=fixture("test_sync_fifo.py"),
                         rtl="", top="", stem="sync_fifo")
    hint = results["tests"][0]["hint"]
    assert "declares:" not in hint
    assert "Did you mean" not in hint


def test_an_unrecognised_failure_still_gets_where_and_why():
    """("", "") from the rule table is a normal answer, not a hole in the report."""
    results = {"total": 1, "failed": 1, "passed": 0, "skipped": 0, "tests": [{
        "name": "test_a", "status": "failed", "type": "ZeroDivisionError",
        "message": "division by zero",
        "detail": 'Traceback (most recent call last):\n'
                  '  File "/w/test_sync_fifo.py", line 73, in test_a\n'
                  '    x = 1 / 0\n'
                  'ZeroDivisionError: division by zero\n',
    }]}
    text = flow.summarize_failures(results, testbench=fixture("test_sync_fifo.py"),
                                   stem="sync_fifo")
    assert "FAILED test_a" in text
    assert "test_sync_fifo.py:73" in text
    # Not an assertion, and raised in the user's own file: a testbench bug.
    assert results["tests"][0]["hint_id"] == "test_raised_exception"


def test_summarize_failures_shows_why_where_and_the_cause_for_each_test():
    results = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    text = flow.summarize_failures(
        results, testbench=fixture("test_sync_fifo.py"),
        rtl=fixture("sync_fifo_good.v"), top="sync_fifo", stem="sync_fifo",
        log_slices=flow.slice_cocotb_log(fixture("cocotb_run.log")))
    assert text.startswith("2 of 3 cocotb tests failed.")
    assert text.count("FAILED ") == 2
    for section in ("WHY", "WHERE", "LIKELY CAUSE", "LAST LINES BEFORE THE FAILURE"):
        assert section in text
    # The location, and the line the RTL fix prompt needs to see.
    assert "test_sync_fifo.py:76" in text
    assert "assert int(dut.full.value) == 1" in text
    assert "failed at 145.00 ns" in text


def test_summarize_failures_degrades_section_by_section_before_dropping_tests():
    """
    A blind text[:max_chars] would keep full evidence for the first tests and cut
    the last ones off mid-sentence. The log excerpt is the first thing to go.
    """
    results = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    kwargs = dict(testbench=fixture("test_sync_fifo.py"),
                  rtl=fixture("sync_fifo_good.v"), top="sync_fifo",
                  stem="sync_fifo",
                  log_slices=flow.slice_cocotb_log(fixture("cocotb_run.log")))
    full = flow.summarize_failures(results, **kwargs)
    squeezed = flow.summarize_failures(results, max_chars=len(full) - 400, **kwargs)
    assert "LAST LINES BEFORE THE FAILURE" not in squeezed
    # Both tests survive: what was dropped is evidence, not the answer.
    assert squeezed.count("FAILED ") == 2


def test_slice_cocotb_log_keys_each_test_by_name():
    slices = flow.slice_cocotb_log(fixture("cocotb_run.log"))
    assert set(slices) == {"test_reset_values", "test_full_asserts_at_depth",
                           "test_count_must_not_exceed_depth"}
    assert "write 8 accepted" in slices["test_full_asserts_at_depth"]
    # Sliced to the NEXT test's banner, so a test owns what it printed last.
    assert "write 9 accepted" not in slices["test_full_asserts_at_depth"]


def test_slice_cocotb_log_degrades_to_nothing_on_an_unfamiliar_format():
    """A cocotb that words its banner a third way must yield {}, never a guess."""
    assert flow.slice_cocotb_log("some log with no regression banner at all") == {}
    assert flow.slice_cocotb_log("") == {}


# ---------------------------------------------------------------------------
# Error locations: WHERE the error is
# ---------------------------------------------------------------------------

def test_verilator_diagnostics_are_located_quoted_and_explained():
    text = flow.explain_verilator_diagnostics(
        fixture("verilator_errors.txt"),
        sources={"broken_fifo.v": fixture("verilator_broken.v")})
    assert text.startswith("3 Verilator errors.")
    assert "broken_fifo.v:21:30" in text
    assert "sub_block u0 (.clk(clk), .rstn(rst_n));" in text   # the quoted line
    assert "Pin not found: 'rstn'" in text                     # Verilator's own words
    assert "does not declare" in text                          # what to change
    # A parse failure has no message code at all, so it is matched on its text.
    assert "ONE LINE AFTER" in text


def test_verilator_warnings_are_shown_only_when_nothing_errored():
    """In lint mode the warnings ARE the product; in a build failure they are noise."""
    src = {"broken_fifo.v": fixture("verilator_broken.v")}
    both = flow.explain_verilator_diagnostics(fixture("verilator_errors.txt"), sources=src)
    assert "WIDTHEXPAND" not in both
    warning_only = "\n".join(
        line for line in fixture("verilator_errors.txt").splitlines()
        if not line.startswith("%Error"))
    assert "WIDTHEXPAND" in flow.explain_verilator_diagnostics(warning_only, sources=src)


def test_verilator_errors_are_found_inside_a_python_build_traceback():
    """
    cocotb's runner.build() wraps Verilator's stderr in a SystemExit traceback.
    Anchoring the regex on %Error rather than on line position is what makes that
    work with no separate code path.
    """
    text = flow.explain_verilator_diagnostics(
        fixture("verilator_build_traceback.txt"),
        sources={"sync_fifo.v": fixture("sync_fifo_good.v")})
    assert "sync_fifo.v:12:5" in text
    assert "PINMISSING" in text


def test_the_summary_line_survives_when_it_is_all_there_is():
    """Dropping "Exiting due to N error(s)" unconditionally would empty the port."""
    text = flow.explain_verilator_diagnostics("%Error: Exiting due to 1 error(s)")
    assert "Exiting due to 1 error" in text


def test_quoting_a_source_line_costs_no_pod_round_trip(monkeypatch):
    """
    The whole reason `errors` is free: req.rtl is already in hand server-side.
    """
    def explode(*_a, **_kw):  # pragma: no cover - the point is that it never runs
        raise AssertionError("explain_verilator_diagnostics must not touch the pod")
    monkeypatch.setattr(flow, "exec_simple", explode)
    monkeypatch.setattr(flow, "exec_stream", explode)
    text = flow.explain_verilator_diagnostics(
        fixture("verilator_errors.txt"),
        sources={"broken_fifo.v": fixture("verilator_broken.v")})
    assert "assign level = count;" in text


def test_a_traceback_in_grafuxs_own_runner_is_not_blamed_on_the_testbench():
    """run_cocotb.py is generated by Grafux; the user has never seen it."""
    trace = ('Traceback (most recent call last):\n'
             '  File "/workspace/grafux/run_cocotb.py", line 88, in <module>\n'
             '    runner.build()\n'
             'RuntimeError: boom\n')
    text = flow.locate_python_error(trace, test_file="test_sync_fifo.py")
    assert "generated cocotb runner, not your testbench" in text


def test_error_locations_is_empty_when_nothing_broke():
    """
    The decision that makes the port mean something: a clean build whose tests
    merely failed has no error LOCATION, so it says nothing at all.
    """
    assert flow.error_locations() == ""
    assert flow.error_locations(verilator_stderr="all fine here") == ""


def test_the_results_port_keeps_the_derived_fields_and_drops_the_raw_bulk(pod):
    """
    A traceback plus two captured streams is up to 8000 characters per failing
    test, and every one of them is already rendered into `failures` in the form a
    person reads. Serialising them here too would put tens of kilobytes of
    duplicate text in a port whose readers want a name, a status and one line of
    why -- and would guarantee the review's head cap cuts this JSON mid-document.
    The originals are still reachable: results.xml comes back as an artifact.
    """
    pod["run"] = (0, "GRAFUX_STAGE build\nGRAFUX_STAGE sim\n", "")
    pod["results_xml"] = fixture("results_fail_detail.xml")
    pod["cocotb_log"] = fixture("cocotb_run.log")
    outcome = _run(pod, _Req(testbench=fixture("test_sync_fifo.py")))

    results = json.loads(outcome["outputs"]["results"])
    failing = _by_name(results, "test_full_asserts_at_depth")
    # What the app's verdict panel reads.
    assert failing["why"]
    assert failing["where"]["file"] == "test_sync_fifo.py"
    assert failing["where"]["line"] == 76
    assert failing["message"]                       # the compatibility anchor
    # What only the enrichers needed.
    for gone in ("detail", "stdout", "stderr", "extra", "log_excerpt"):
        assert gone not in failing, gone
    assert "text" not in failing["where"]
    # But the parser itself still keeps them -- this is a serialisation rule, not
    # a parsing one, or the enrichers would have nothing to read.
    parsed = flow.parse_cocotb_results(fixture("results_fail_detail.xml"))
    assert "detail" in _by_name(parsed, "test_full_asserts_at_depth")


def test_run_cocotb_puts_locations_on_errors_and_the_story_on_failures(pod):
    pod["run"] = (0, "GRAFUX_STAGE build\nGRAFUX_STAGE sim\n", "")
    pod["results_xml"] = fixture("results_fail_detail.xml")
    pod["cocotb_log"] = fixture("cocotb_run.log")
    outcome = _run(pod, _Req(testbench=fixture("test_sync_fifo.py")))
    failures = outcome["outputs"]["failures"]
    assert "WHY" in failures and "WHERE" in failures
    assert "test_sync_fifo.py:76" in failures
    assert outcome["outputs"]["errors"] == ""


def test_run_cocotb_locates_a_build_failure_in_the_rtl(pod):
    pod["run"] = (2, fixture("verilator_build_traceback.txt"), "")
    pod["results_xml"] = ""
    outcome = _run(pod, _Req(rtl=fixture("sync_fifo_good.v")))
    errors = outcome["outputs"]["errors"]
    assert "did not build" in errors            # why there is no test verdict
    assert "sync_fifo.v:12:5" in errors         # and exactly where
    assert "PINMISSING" in errors


def test_run_cocotb_falls_back_to_the_stdout_tail_on_a_stale_pod(pod):
    """
    A pod is reused for the life of its block and the image tag is not part of
    the reuse key, so a warm pod is still running a run_cocotb.py that predates
    the tee and writes no cocotb.log at all.
    """
    pod["run"] = (0, fixture("cocotb_run.log"), "")
    pod["results_xml"] = fixture("results_fail_detail.xml")
    pod["cocotb_log"] = ""                      # the stale pod
    outcome = _run(pod, _Req(testbench=fixture("test_sync_fifo.py")))
    assert "LAST LINES BEFORE THE FAILURE" in outcome["outputs"]["failures"]


def test_the_generated_runner_tees_both_streams_without_merging_them():
    """
    Merging stderr into stdout would empty `err` in the parent, and `err` decides
    build_failed, libpython_missing and what lands on `warnings`.
    """
    script = flow.build_cocotb_runner_script(
        top="sync_fifo", sources=["sync_fifo.v"], test_module="test_sync_fifo")
    assert "GRAFUX_TEE" in script
    assert script.count("subprocess.PIPE") == 2
    assert "subprocess.STDOUT" not in script
    ast.parse(script)
