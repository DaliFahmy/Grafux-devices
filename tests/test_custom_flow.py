"""
test_custom_flow.py
The generic runner, driven through an in-memory fake transport.

The bugs worth catching are about the CONTRACT: every input lands as its own
file (defaults filled, placeholders treated as empty), bad typed inputs are
refused before the pod does anything, the entry runs under a login shell, and
the verdict is "exit 0 AND the required outputs exist" -- never exit 0 alone.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from CUSTOM import flow  # noqa: E402
from CUSTOM.manifest import parse_manifest  # noqa: E402
from CUSTOM.models import CustomRunRequest  # noqa: E402
from EDA.pod_client import WORK_DIR  # noqa: E402

OUT = f"{WORK_DIR}/{flow.OUT_SUBDIR}"
IN = f"{WORK_DIR}/{flow.IN_SUBDIR}"

MANIFEST = {
    "schema": 1,
    "slug": "demo",
    "runtime": {"image": "ghcr.io/dalifahmy/grafux-gen:demo-1", "timeout_s": 77},
    "inputs": [
        {"name": "words", "type": "int", "default": "16"},
        {"name": "mode", "type": "enum", "choices": ["fast", "slow"], "default": "fast"},
        {"name": "flag", "type": "bool"},
        {"name": "code", "type": "text"},
    ],
    "outputs": [
        {"name": "summary", "kind": "text"},
        {"name": "gds", "kind": "artifact", "glob": "files/*.gds"},
    ],
    "status": {"require_outputs": ["gds"]},
}


class _File:
    def __init__(self, store, path, mode):
        self.store, self.path, self.mode = store, path, mode
        self.buf = b""

    def write(self, data):
        self.buf += data

    def read(self, n=-1):
        if self.path not in self.store:
            raise IOError(self.path)
        data = self.store[self.path]
        return data if n < 0 else data[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if "w" in self.mode:
            self.store[self.path] = self.buf


class _Sftp:
    def __init__(self, store):
        self.store = store
        self.dirs = []

    def mkdir(self, d):
        self.dirs.append(d)

    def open(self, path, mode="rb"):
        return _File(self.store, path, mode)

    def close(self):
        pass


class _FakeClient:
    def __init__(self):
        self.files = {}

    def open_sftp(self):
        return _Sftp(self.files)


@pytest.fixture
def pod(monkeypatch):
    """A pod whose run writes whatever ``state['produce']`` says."""
    client = _FakeClient()
    state = {"code": 0, "out": "hello\n", "err": "", "commands": [], "timeouts": [],
             "produce": {f"{OUT}/summary": b"done"}, "matches": {"files/*.gds": "files/sram.gds\n"}}

    def fake_exec_stream(_c, command, *, timeout, on_line=None, should_cancel=None):
        state["commands"].append(command)
        state["timeouts"].append(timeout)
        client.files.update(state["produce"])
        return state["code"], state["out"], state["err"]

    def fake_exec_simple(_c, command, timeout):
        state["commands"].append(command)
        for glob, listing in state["matches"].items():
            if glob in command:
                return 0, listing, ""
        return 0, "", ""

    monkeypatch.setattr(flow, "exec_stream", fake_exec_stream)
    monkeypatch.setattr(flow, "exec_simple", fake_exec_simple)
    state["client"] = client
    return state


def _run(pod, manifest=MANIFEST, **fields):
    import json
    stages = []
    req = CustomRunRequest(manifest=json.dumps(manifest), **fields)
    out = flow.run_custom(pod["client"], req, on_stage=lambda s, d: stages.append((s, d)))
    out["_stages"] = stages
    return out


def test_a_clean_run_stages_inputs_runs_the_entry_and_reports_ok(pod):
    outcome = _run(pod, inputs={"words": " 32 ", "code": "  int x;\n", "mode": "unconnected"})
    files = pod["client"].files
    assert files[f"{IN}/words"] == b"32"
    assert files[f"{IN}/code"] == b"  int x;\n"      # text is byte-for-byte
    assert files[f"{IN}/mode"] == b"fast"            # placeholder -> default
    assert files[f"{IN}/flag"] == b"false"
    assert outcome["_status"] == "ok"
    o = outcome["outputs"]
    assert o["status"] == "ok" and o["errors"] == ""
    assert o["summary"] == "done" and o["gds"] == "sram.gds"
    assert "hello" in o["log"]
    assert outcome["_globs"] == [f"{OUT}/files/*.gds"]
    run_cmd = next(c for c in pod["commands"] if "/opt/grafux/run.sh" in c)
    assert run_cmd.startswith("bash -lc ") and "GRAFUX_IN=" in run_cmd
    assert pod["timeouts"] == [77]                   # manifest timeout when the port is 0
    assert [s for s, _ in outcome["_stages"]] == ["stage", "stage", "run", "run", "collect", "collect"]


def test_the_timeout_port_overrides_the_manifest(pod):
    _run(pod, timeout=5)
    assert pod["timeouts"] == [5]


def test_exit_zero_without_a_required_output_is_a_failure(pod):
    pod["matches"] = {}
    outcome = _run(pod)
    assert outcome["_status"] == "error"
    assert "produced no `gds`" in outcome["outputs"]["errors"]


def test_a_nonzero_exit_fails_and_explains_127(pod):
    pod["code"], pod["err"] = 127, "run.sh: not found"
    o = _run(pod)["outputs"]
    assert o["status"] == "error"
    assert "exited with code 127" in o["errors"] and "not found" in o["errors"]


@pytest.mark.parametrize("inputs,needle", [
    ({"words": "abc"}, "must be an integer"),
    ({"mode": "medium"}, "must be one of fast, slow"),
    ({"flag": "maybe"}, "true or false"),
])
def test_bad_typed_inputs_are_refused_before_the_pod_runs_anything(pod, inputs, needle):
    outcome = _run(pod, inputs=inputs)
    assert outcome["_status"] == "error"
    assert needle in outcome["outputs"]["errors"]
    assert pod["commands"] == [] and pod["client"].files == {}


def test_a_required_input_without_value_or_default_is_refused(pod):
    m = dict(MANIFEST, inputs=[{"name": "code", "type": "text", "required": True}])
    outcome = _run(pod, manifest=m)
    assert "is required" in outcome["outputs"]["errors"]


def test_a_broken_manifest_is_reported_not_raised(pod):
    import json
    req = CustomRunRequest(manifest="{", inputs={})
    outcome = flow.run_custom(pod["client"], req, on_stage=lambda s, d: None)
    assert outcome["_status"] == "error" and "not valid JSON" in outcome["outputs"]["errors"]
    assert json.dumps(outcome)


def test_adapter_errors_and_warnings_files_reach_the_ports(pod):
    pod["produce"] = {f"{OUT}/summary": b"x", f"{OUT}/warnings": b"slow corner skipped",
                      f"{OUT}/errors": b"partial result"}
    o = _run(pod)["outputs"]
    assert "slow corner skipped" in o["warnings"]
    assert "partial result" in o["errors"]


def test_an_oversized_text_output_becomes_an_artifact(pod):
    m = dict(MANIFEST, outputs=[{"name": "summary", "kind": "text", "inline_max": 3},
                                {"name": "gds", "kind": "artifact", "glob": "files/*.gds"}])
    outcome = _run(pod, manifest=m)
    assert outcome["outputs"]["summary"] == ""
    assert f"{OUT}/summary" in outcome["_globs"]
    assert "attached as the artifact" in outcome["outputs"]["warnings"]


def test_undeclared_inputs_are_ignored_but_noted(pod):
    o = _run(pod, inputs={"surprise": "1"})["outputs"]
    assert "surprise" in o["warnings"]


def test_coerce_input_keeps_empty_numbers_empty():
    m = parse_manifest(MANIFEST)
    assert flow.coerce_input(m.inputs[0], "  ") == ("", "")
