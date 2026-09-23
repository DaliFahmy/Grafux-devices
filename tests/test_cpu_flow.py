"""
test_cpu_flow.py
Tests for the post-silicon verification flow.

The pure half (``normalize_language``, ``build_compile_cmd``, ``run_target_for``,
``build_bench_script``, the parsers, ``benchmark_stats``) is tested directly —
no cloud, no container, no SSH.  ``run_cpu`` is driven through an in-memory fake
transport, because the bugs worth catching there are about ORDER and VERDICT:
that flags land after the source, that a failing check beats a zero exit code,
and that a stopped run never reports a measurement.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from CPU import flow  # noqa: E402
from CPU.models import CpuRunRequest  # noqa: E402


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("c", "c"), ("C", "c"), ("c11", "c"),
    ("cpp", "cpp"), ("c++", "cpp"), ("C++20", "cpp"), ("cxx", "cpp"),
    ("python", "python"), ("py", "python"), ("python3", "python"),
    ("  CPP  ", "cpp"),
])
def test_known_languages_resolve(raw, expected):
    language, note = flow.normalize_language(raw)
    assert language == expected
    assert note == ""


def test_an_empty_language_port_falls_back_without_complaining():
    """An unwired port is not a mistake, so it must not produce a warning."""
    assert flow.normalize_language("") == ("cpp", "")


def test_an_unknown_language_warns_but_still_runs():
    """
    Refusing here would mean a user who typed "golang" stares at an error instead
    of a result. The cost of guessing is one wasted compile; the cost of refusing
    is the whole run.
    """
    language, note = flow.normalize_language("golang")
    assert language == "cpp"
    assert "golang" in note and "language port" in note


# ---------------------------------------------------------------------------
# The compile command
# ---------------------------------------------------------------------------

def test_c_and_cpp_pick_different_compilers():
    c_cmd = flow.build_compile_cmd("c", source="case.c", binary="case")
    cpp_cmd = flow.build_compile_cmd("cpp", source="case.cpp", binary="case")
    assert c_cmd.startswith("gcc ")
    assert cpp_cmd.startswith("g++ ")


def test_python_does_not_compile():
    assert flow.build_compile_cmd("python", source="case.py", binary="case") is None


def test_build_flags_come_after_the_source_and_output():
    """
    Regression test for a link failure with no visible cause.

    ``-l`` libraries must follow the object that references them or the linker
    reports undefined symbols, so a user adding '-lm' to the build_flags port
    would get an error pointing at their code rather than at flag order.
    """
    cmd = flow.build_compile_cmd(
        "c", source="case.c", binary="case", build_flags="-O2 -lm",
    )
    parts = cmd.split()
    assert parts.index("case.c") < parts.index("-o") < parts.index("-lm")


def test_defines_and_includes_are_prefixed_only_when_needed():
    """Users write both 'WIDTH=8' and '-DWIDTH=8'; neither should double up."""
    cmd = flow.build_compile_cmd(
        "cpp", source="case.cpp", binary="case",
        defines="WIDTH=8 -DDEBUG", include_dirs="/inc\n-I/other",
    )
    assert "-DWIDTH=8" in cmd and "-DDEBUG" in cmd and "-D-DDEBUG" not in cmd
    assert "-I/inc" in cmd and "-I/other" in cmd and "-I-I/other" not in cmd


# ---------------------------------------------------------------------------
# The run target
# ---------------------------------------------------------------------------

def test_python_runs_through_the_interpreter_and_c_runs_the_binary():
    assert flow.run_target_for("python", binary="case", source="case.py") == "python3 case.py"
    assert flow.run_target_for("cpp", binary="case", source="case.cpp") == "./case"


def test_args_are_appended_to_the_run_target():
    target = flow.run_target_for("c", binary="case", source="case.c", args="--iters 10")
    assert target == "./case --iters 10"


# ---------------------------------------------------------------------------
# Counts off text ports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "   ", "empty", "unconnected", "0", "-3", "abc"])
def test_a_missing_or_nonsense_count_uses_the_default(raw):
    """
    An unwired port arrives as "" or as the frontend's sentinel text. Reading
    either as 0 would run the case zero times and report a measurement of
    nothing, which is worse than ignoring the port.
    """
    assert flow._positive_int(raw, 5, maximum=200) == 5


def test_a_count_is_capped():
    """A pasted 100000 must not hold a pod open until the watchdog fires."""
    assert flow._positive_int("100000", 5, maximum=200) == 200
    assert flow._positive_int("7", 5, maximum=200) == 7


# ---------------------------------------------------------------------------
# The on-pod benchmark script
# ---------------------------------------------------------------------------

def test_the_script_times_each_repetition_individually():
    """
    A single total divided by N cannot produce a spread, and a mean with no
    spread beside it invites a comparison the data does not support.
    """
    script = flow.build_bench_script("./case", warmup=2, repetitions=5)
    assert "date +%s%N" in script
    assert script.count(flow.MS_MARKER) == 1  # inside the loop, emitted per run
    assert "seq 1 5" in script and "seq 1 2" in script


def test_warmup_output_is_discarded_and_measured_output_is_kept():
    script = flow.build_bench_script("./case", warmup=1, repetitions=3)
    warm_line = [ln for ln in script.splitlines() if ">/dev/null 2>&1" in ln][0]
    assert "./case" in warm_line
    assert f"> {flow.STDOUT_FILE}" in script


def test_the_script_does_not_abort_on_a_failing_repetition():
    """
    ``set -e`` here would abandon the remaining repetitions the moment a case
    failed -- losing the measurement of exactly the run a user most wants to see.
    """
    script = flow.build_bench_script("./case", warmup=0, repetitions=3)
    assert "set -u" in script
    assert "set -e" not in script


def test_peak_rss_is_measured_on_a_separate_unmeasured_run():
    """
    /usr/bin/time's own overhead must not land inside a number reported as the
    case's duration, so its run sits after the loop and is never timed.
    """
    script = flow.build_bench_script("./case", warmup=0, repetitions=2)
    body, _, tail = script.partition("/usr/bin/time -v")
    assert flow.MS_MARKER in body          # every timing marker precedes it
    assert flow.MS_MARKER not in tail


# ---------------------------------------------------------------------------
# Parsing what the script printed
# ---------------------------------------------------------------------------

def test_parse_bench_output_reads_samples_codes_and_verdict():
    text = "\n".join([
        "noise from the program",
        f"{flow.MS_MARKER}12", f"{flow.RC_MARKER}0",
        f"{flow.MS_MARKER}15", f"{flow.RC_MARKER}3",
        f"{flow.EXIT_MARKER}3",
    ])
    samples, codes, worst = flow.parse_bench_output(text)
    assert samples == [12, 15]
    assert codes == [0, 3]
    assert worst == 3


def test_parse_bench_output_ignores_malformed_markers():
    """The case's own stdout is interleaved with these; it must not break parsing."""
    samples, codes, worst = flow.parse_bench_output(
        f"{flow.MS_MARKER}not-a-number\n{flow.MS_MARKER}8\n")
    assert samples == [8] and codes == [] and worst is None


def test_benchmark_stats_reports_spread():
    stats = flow.benchmark_stats([10, 20, 30])
    assert stats["exec_ms_mean"] == 20.0
    assert stats["exec_ms_min"] == 10 and stats["exec_ms_max"] == 30
    assert stats["exec_ms_stddev"] > 0
    assert stats["repetitions"] == 3


def test_benchmark_stats_with_one_sample_reports_zero_spread():
    """
    Reported as 0.0 rather than omitted so a consumer never branches on a missing
    key; min == max is the honest signal that there is no spread to speak of.
    """
    stats = flow.benchmark_stats([42])
    assert stats["exec_ms_stddev"] == 0.0
    assert stats["exec_ms_min"] == stats["exec_ms_max"] == 42


def test_benchmark_stats_with_no_samples_does_not_divide_by_zero():
    assert flow.benchmark_stats([])["repetitions"] == 0


@pytest.mark.parametrize("line,name,verdict", [
    ("PASS: cache writeback", "cache writeback", "pass"),
    ("FAIL: dcache line fill", "dcache line fill", "fail"),
    ("[ OK ] tlb refill", "tlb refill", "pass"),
    ("[FAIL] branch predictor", "branch predictor", "fail"),
    ("l1 coherency: PASS", "l1 coherency", "pass"),
    ("atomics -- FAIL", "atomics", "fail"),
    ("ERROR: memory ordering", "memory ordering", "fail"),
])
def test_parse_check_lines_reads_the_shapes_cases_actually_print(line, name, verdict):
    assert flow.parse_check_lines(line) == {name: verdict}


def test_parse_check_lines_ignores_prose_and_huge_lines():
    text = "starting the run\n" + ("x" * 500) + "\nall done\n"
    assert flow.parse_check_lines(text) == {}


def test_parse_max_rss_kb():
    assert flow.parse_max_rss_kb(
        "\tMaximum resident set size (kbytes): 20484\n") == 20484
    assert flow.parse_max_rss_kb("nothing useful") == 0


def test_parse_machine_picks_the_fields_worth_showing():
    machine = flow.parse_machine(
        "Architecture:            x86_64\n"
        "CPU(s):                  8\n"
        "Model name:              AMD EPYC 7763\n"
    )
    assert machine == {"arch": "x86_64", "vcpus": "8", "cpu_model": "AMD EPYC 7763"}


def test_globs_do_not_drag_back_the_compiled_binary():
    """It is large, opaque, and reproducible from the `code` port."""
    globs = flow.globs_for()
    assert not any(g.rstrip("/").endswith(f"/{flow.BINARY_NAME}") for g in globs)
    assert any(g.endswith("*.txt") for g in globs)


# ---------------------------------------------------------------------------
# run_cpu, over a fake transport
# ---------------------------------------------------------------------------

class _FakeFile:
    def __init__(self, sink, path):
        self._sink, self._path = sink, path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, data):
        self._sink[self._path] = data.decode("utf-8")


class _FakeSftp:
    def __init__(self, sink):
        self.sink = sink

    def open(self, path, _mode):
        return _FakeFile(self.sink, path)

    def close(self):
        pass


class _FakeClient:
    def __init__(self):
        self.written = {}

    def open_sftp(self):
        return _FakeSftp(self.written)


@pytest.fixture
def transport(monkeypatch):
    """
    An in-memory stand-in for the pod.

    ``flow`` imports exec_simple/exec_stream by name, so patching them on the
    module is what a real SSH session would otherwise provide.
    """
    state = {
        "compile_code": 0, "compile_out": "",
        "bench_code": 0,
        "bench_out": "\n".join([
            f"{flow.MS_MARKER}10", f"{flow.RC_MARKER}0",
            f"{flow.MS_MARKER}12", f"{flow.RC_MARKER}0",
            f"{flow.EXIT_MARKER}0",
        ]),
        "stdout": "PASS: cache writeback\n",
        "stderr": "",
        "time_v": "\tMaximum resident set size (kbytes): 4096\n",
        "lscpu": "Model name:   AMD EPYC 7763\nCPU(s):    8\n",
        "commands": [],
    }

    def fake_exec_stream(_client, command, *, timeout, on_line=None, should_cancel=None):
        state["commands"].append(command)
        if "bench.sh" in command:
            return state["bench_code"], state["bench_out"], ""
        return state["compile_code"], state["compile_out"], ""

    def fake_exec_simple(_client, command, timeout):
        state["commands"].append(command)
        if flow.STDOUT_FILE in command:
            return 0, state["stdout"], ""
        if flow.STDERR_FILE in command:
            return 0, state["stderr"], ""
        if flow.TIME_FILE in command:
            return 0, state["time_v"], ""
        if "lscpu" in command:
            return 0, state["lscpu"], ""
        return 0, "", ""

    monkeypatch.setattr(flow, "exec_stream", fake_exec_stream)
    monkeypatch.setattr(flow, "exec_simple", fake_exec_simple)
    return state


def _run(state, **fields):
    stages = []
    req = CpuRunRequest(**fields)
    outcome = flow.run_cpu(
        _FakeClient(), req, on_stage=lambda s, d: stages.append((s, d)),
    )
    outcome["_stages"] = stages
    return outcome


def test_a_clean_run_passes_and_reports_a_measurement(transport):
    outcome = _run(transport, code="int main(){return 0;}", language="c",
                   repetitions="2", warmup="1")
    outputs = outcome["outputs"]
    assert outcome["_status"] == "ok"
    assert outputs["passed"] == "true"
    assert outputs["response"] == "PASS: cache writeback\n"
    assert json.loads(outputs["results"]) == {"cache writeback": "pass"}
    benchmark = json.loads(outputs["benchmark"])
    assert benchmark["exec_ms_min"] == 10 and benchmark["exec_ms_max"] == 12
    assert benchmark["max_rss_kb"] == 4096
    assert benchmark["cpu_model"] == "AMD EPYC 7763"
    assert outputs["duration"] == "11.000"
    assert outputs["errors"] == ""


def test_an_empty_code_port_is_refused_before_provisioning_work(transport):
    """The message must name the port and the usual wiring, not just say 'empty'."""
    outcome = _run(transport, code="")
    assert outcome["_status"] == "error"
    assert "`code` port is empty" in outcome["outputs"]["errors"]
    assert "post_silicon_verification" in outcome["outputs"]["errors"]
    assert transport["commands"] == []   # nothing was run on the pod


def test_a_failing_check_beats_a_zero_exit_code(transport):
    """
    The case returned 0 but printed a failure. Trusting the exit code alone would
    report a green block for a chip that did not do what the case asked.
    """
    transport["stdout"] = "PASS: tlb refill\nFAIL: cache writeback\n"
    outcome = _run(transport, code="int main(){return 0;}")
    assert outcome["outputs"]["passed"] == "false"
    assert outcome["_status"] == "failed"
    assert "cache writeback" in outcome["outputs"]["errors"]


def test_a_nonzero_exit_with_no_check_lines_is_judged_on_the_exit_code(transport):
    transport["stdout"] = "nothing parseable here\n"
    transport["stderr"] = "segfault details"
    transport["bench_out"] = "\n".join([
        f"{flow.MS_MARKER}9", f"{flow.RC_MARKER}139", f"{flow.EXIT_MARKER}139",
    ])
    outcome = _run(transport, code="int main(){return 139;}")
    assert outcome["outputs"]["passed"] == "false"
    assert "exited with code 139" in outcome["outputs"]["errors"]
    assert "segfault details" in outcome["outputs"]["errors"]


def test_a_compile_failure_reports_the_diagnostics_and_never_benchmarks(transport):
    transport["compile_code"] = 1
    transport["compile_out"] = "case.c:3:5: error: 'x' undeclared"
    outcome = _run(transport, code="int main(){x;}", language="c")
    assert outcome["_status"] == "error"
    assert outcome["_stage"] == "build"
    assert "undeclared" in outcome["outputs"]["errors"]
    assert not any("bench.sh" in c for c in transport["commands"])


def test_a_missing_compiler_says_which_port_to_fix(transport):
    """
    Exit 127 is 'command not found'. Pointing at the image port is the only
    actionable advice; the raw shell error is not.
    """
    transport["compile_code"] = 127
    outcome = _run(transport, code="int main(){}", language="c")
    assert "`image` port" in outcome["outputs"]["errors"]


def test_compiler_warnings_on_a_successful_build_are_surfaced(transport):
    """In a verification case a sign-compare warning is often the bug it hunts."""
    transport["compile_out"] = "case.c:4:9: warning: comparison of integer expressions"
    outcome = _run(transport, code="int main(){}", language="c")
    assert outcome["outputs"]["passed"] == "true"
    assert "comparison of integer" in outcome["outputs"]["warnings"]


def test_python_skips_the_compile_step_entirely(transport):
    outcome = _run(transport, code="print('PASS: x')", language="python")
    assert outcome["outputs"]["passed"] == "true"
    assert json.loads(outcome["outputs"]["benchmark"])["compile_ms"] == 0
    assert not any("gcc" in c or "g++" in c for c in transport["commands"])


def test_a_stopped_run_reports_no_measurement(transport):
    """-1 from exec_stream means the user pressed Stop."""
    transport["bench_code"] = -1
    outcome = _run(transport, code="int main(){}")
    assert outcome["_status"] == "error"
    assert "stopped" in outcome["outputs"]["errors"].lower()
    assert outcome["outputs"]["duration"] == ""


def test_a_timeout_names_the_ports_that_fix_it(transport):
    """-2 from exec_stream means the run exceeded its wall-clock limit."""
    transport["bench_code"] = -2
    outcome = _run(transport, code="int main(){}", timeout=120, repetitions="50")
    assert "`timeout`" in outcome["outputs"]["errors"]
    assert "repetitions" in outcome["outputs"]["errors"]


def test_no_timing_samples_is_an_error_not_a_zero_measurement(transport):
    """
    A case that never started must not be reported as having run in 0 ms -- that
    is the most misleading possible output from a benchmark.
    """
    transport["bench_out"] = "bash: ./case: No such file or directory"
    outcome = _run(transport, code="int main(){}")
    assert outcome["_status"] == "error"
    assert outcome["outputs"]["duration"] == ""
    assert "no timing samples" in outcome["outputs"]["errors"].lower()


def test_the_unknown_language_warning_reaches_the_block(transport):
    outcome = _run(transport, code="int main(){}", language="golang")
    assert "golang" in outcome["outputs"]["warnings"]


def test_stages_are_reported_in_order(transport):
    """The block face must say something true while a long run is in flight."""
    outcome = _run(transport, code="int main(){}", warmup="1")
    names = [s for s, _ in outcome["_stages"]]
    assert names.index("build") < names.index("bench") <= names.index("collect")
