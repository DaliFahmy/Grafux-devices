"""
test_generator_session.py
The Generator's loop, end to end, with a fake sandbox (whose "agent" is a Python
function that edits the workspace), a fake builder and a fake smoke run.

What must hold: plan turns cannot leave files behind; a verified block needs a
passing build AND a passing smoke run; every failure goes back to the agent as
text; a repair that changes nothing stops the loop; infrastructure failures are
never sent to the agent; keys never appear in events.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from GENERATOR import contract, session as gsession  # noqa: E402
from GENERATOR.builder import BuildResult  # noqa: E402
from GENERATOR.models import CreateSessionRequest  # noqa: E402

KEY = "sk-ant-SECRET-1234567890"

GOOD = {
    "grafux-block.json": json.dumps({
        "schema": 1, "slug": "demo",
        "inputs": [{"name": "n", "type": "int", "default": "1"}],
        "outputs": [{"name": "out", "kind": "text"}],
        "status": {"require_outputs": ["out"]},
        "selftest": {"inputs": {"n": "2"}, "expect_outputs": ["out"]}}),
    "Dockerfile": "FROM ubuntu:22.04\n",
    "run.sh": "#!/bin/bash\n",
}


class FakeSandbox:
    """A pod whose agent is ``script(prompt, mode_flag, gen)``, mutating ``gen``."""

    def __init__(self, script, ensure_error=""):
        self.script = script
        self.fresh = True
        self.gen = {}
        self.checkpoint = {}
        self.written = {}
        self.commands = []
        self.ensure_error = ensure_error
        self.terminated = False

    def ensure(self):
        return self.ensure_error

    def exec(self, cmd, timeout=120):
        self.commands.append(cmd)
        if "git status --porcelain" in cmd:
            return 0, ("M x\n" if self.gen != self.checkpoint else ""), ""
        if "git reset -q --hard" in cmd:
            self.gen = dict(self.checkpoint)
        elif "commit -qm" in cmd:
            self.checkpoint = dict(self.gen)
        elif cmd.startswith("git clone"):
            return 0, "", ""
        return 0, "", ""

    def stream(self, cmd, *, timeout, on_line, should_cancel):
        self.commands.append(cmd)
        prompt = self.written[gsession.PROMPT_FILE]
        mode = "plan" if "--permission-mode plan" in cmd else "edit"
        answer = self.script(prompt, mode, self.gen) or "ok"
        for obj in ({"type": "system", "subtype": "init", "session_id": "S1"},
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": f"using {KEY}"}]}},
                    {"type": "result", "subtype": "success", "is_error": False, "result": answer,
                     "total_cost_usd": 0.1, "session_id": "S1"}):
            on_line(json.dumps(obj))
        return 0, "", ""

    def write_file(self, path, text, mode=0o644):
        self.written[path] = text
        if path.startswith(contract.GEN_DIR + "/"):
            self.gen[path[len(contract.GEN_DIR) + 1:]] = text

    def read_tree(self, directory):
        return dict(self.gen), []

    def keepalive(self):
        pass

    def terminate(self):
        self.terminated = True


class FakeBuilder:
    def __init__(self, results):
        self.results = list(results)
        self.built = []

    def build(self, tag, files, on_status=None, should_cancel=None):
        self.built.append((tag, files))
        return self.results.pop(0)


def _req(**kw):
    base = dict(agent="claude_code", mode="edit", prompt="a demo block",
                repo_url="https://github.com/VLSIDA/OpenRAM/tree/stable",
                owner="ahmed", anthropic_api_key=KEY)
    base.update(kw)
    return CreateSessionRequest(**base)


def _run(script, *, builder=None, smoke=None, mode="edit", **req_kw):
    sb = FakeSandbox(script)
    s = gsession.GeneratorSession(_req(mode=mode, **req_kw), sandbox=sb, builder=builder,
                                  smoke=smoke, max_repairs=2)
    s.start(mode)
    s.join(10)
    return s, sb


def _kinds(s):
    return [e["kind"] for e in s.events]


def write_good(prompt, mode, gen):
    gen.update(GOOD)
    return "wrote the block"


def ok_build():
    return BuildResult(True, "img")


def test_repo_urls():
    assert gsession.parse_repo_url("https://github.com/VLSIDA/OpenRAM/tree/stable") == \
        ("https://github.com/VLSIDA/OpenRAM.git", "stable")
    assert gsession.parse_repo_url("https://gitlab.com/a/b.git") == ("https://gitlab.com/a/b.git", "")
    assert gsession.parse_repo_url("") == ("", "")
    with pytest.raises(ValueError):
        gsession.parse_repo_url("file:///etc; rm -rf /")


def test_edit_happy_path_builds_smokes_and_returns_the_block():
    smoke_calls = []

    def smoke(manifest_text, **kw):
        smoke_calls.append(json.loads(manifest_text))
        return True, "", False

    s, sb = _run(write_good, builder=FakeBuilder([ok_build()]), smoke=smoke)
    assert s.state == "done", s.events
    assert s.result["verified"] is True
    assert s.result["image"].startswith(f"{contract.IMAGE_REPO}:ahmed-demo-")
    assert smoke_calls[0]["runtime"]["image"] == s.result["image"]
    assert any("git clone --depth 1 --branch stable" in c for c in sb.commands)
    assert s.agent_session_id == "S1"
    assert pytest.approx(s.cost_usd) == 0.1


def test_keys_never_reach_events_or_the_command_line():
    s, sb = _run(write_good, builder=FakeBuilder([ok_build()]), smoke=lambda m, **k: (True, "", False))
    blob = json.dumps(s.events)
    assert KEY not in blob and "***" in blob
    assert all(KEY not in c for c in sb.commands)
    assert KEY in sb.written[gsession.KEYS_FILE]            # only in the 0600 file


def test_plan_mode_reverts_stray_edits_and_returns_the_plan():
    def planner(prompt, mode, gen):
        assert mode == "plan" and "Do NOT write any files" in prompt
        gen["Dockerfile"] = "FROM sneaky\n"
        return "PLAN: ports n -> out"

    s, sb = _run(planner, mode="plan")
    assert s.state == "awaiting_user"
    assert sb.gen == {}
    assert "warning" in _kinds(s)
    assert next(e for e in s.events if e["kind"] == "plan")["text"] == "PLAN: ports n -> out"


def test_invalid_files_are_repaired_then_built():
    calls = []

    def agent(prompt, mode, gen):
        calls.append(prompt)
        if len(calls) == 1:
            gen["Dockerfile"] = "FROM x\n"          # no manifest, no run.sh
        else:
            assert "rejected before building" in prompt and "run.sh is missing" in prompt
            gen.update(GOOD)

    s, _ = _run(agent, builder=FakeBuilder([ok_build()]), smoke=lambda m, **k: (True, "", False))
    assert s.state == "done" and len(calls) == 2
    assert "repair" in _kinds(s)


def test_build_failure_log_is_the_repair_order():
    prompts = []

    def agent(prompt, mode, gen):
        prompts.append(prompt)
        gen.update(GOOD)
        gen["run.sh"] = f"#!/bin/bash\n# attempt {len(prompts)}\n"

    b = FakeBuilder([BuildResult(False, "img", "GRAFUX SELFTEST FAILED: out"), ok_build()])
    s, _ = _run(agent, builder=b, smoke=lambda m, **k: (True, "", False))
    assert s.state == "done"
    assert "GRAFUX SELFTEST FAILED: out" in prompts[1]


def test_a_repair_that_changes_nothing_stops_the_loop():
    b = FakeBuilder([BuildResult(False, "img", "boom")] * 5)
    s, _ = _run(write_good, builder=b, smoke=None)
    assert s.state == "awaiting_user"
    assert len(b.built) == 1          # the identical repair is caught before a second build
    assert any("did not change any file" in e["text"] for e in s.events)


def test_gives_up_after_max_repairs():
    n = [0]

    def agent(prompt, mode, gen):
        n[0] += 1
        gen.update(GOOD)
        gen["run.sh"] = f"#!/bin/bash\n# {n[0]}\n"

    b = FakeBuilder([BuildResult(False, "img", "boom")] * 5)
    s, _ = _run(agent, builder=b, smoke=None)
    assert s.state == "awaiting_user" and len(b.built) == 3
    assert any("Still failing after 2 repairs" in e["text"] for e in s.events)


def test_smoke_failure_is_repaired_but_infra_failure_is_not():
    n = [0]

    def agent(prompt, mode, gen):
        n[0] += 1
        gen.update(GOOD)
        gen["run.sh"] = f"#!/bin/bash\n# {n[0]}\n"

    smokes = iter([(False, "empty outputs: out", False), (True, "", False)])
    s, _ = _run(agent, builder=FakeBuilder([ok_build(), ok_build()]), smoke=lambda m, **k: next(smokes))
    assert s.state == "done" and n[0] == 2

    s2, _ = _run(write_good, builder=FakeBuilder([ok_build()]),
                 smoke=lambda m, **k: (False, "no RunPod capacity", True))
    assert s2.state == "error" and "repair" not in _kinds(s2)

    s3, _ = _run(write_good, builder=FakeBuilder([BuildResult(False, "i", "401", infra=True)]))
    assert s3.state == "error" and "repair" not in _kinds(s3)


def test_missing_builder_or_key_or_sandbox_are_explained():
    s, _ = _run(write_good, builder=None)
    assert s.state == "error" and "builder is not configured" in s.events[-2]["text"]

    s2, _ = _run(write_good, anthropic_api_key="")
    assert any("needs your Anthropic API key" in e["text"] for e in s2.events)

    sb = FakeSandbox(write_good, ensure_error="image not published")
    s3 = gsession.GeneratorSession(_req(), sandbox=sb)
    s3.start("edit")
    s3.join(10)
    assert s3.state == "error" and any("image not published" in e["text"] for e in s3.events)


def test_follow_up_messages_resume_the_agent_session():
    prompts = []

    def agent(prompt, mode, gen):
        prompts.append((prompt, mode))
        return "plan v%d" % len(prompts)

    sb = FakeSandbox(agent)
    s = gsession.GeneratorSession(_req(mode="plan"), sandbox=sb)
    s.start("plan")
    s.join(10)
    s.message("use sky130 too", "plan")
    s.join(10)
    assert prompts[1] == ("use sky130 too", "plan")
    assert any("--resume S1" in c for c in sb.commands)
    assert s.events_after(0)[0]["kind"] == "user"
    s.stop()
    assert sb.terminated and s.state == "stopped"
    with pytest.raises(RuntimeError):
        s.message("again", "plan")


def test_files_survive_a_reaped_sandbox():
    sb = FakeSandbox(write_good)
    s = gsession.GeneratorSession(_req(mode="plan"), sandbox=sb)
    s.files = dict(GOOD)
    s.start("plan")
    s.join(10)
    assert f"{contract.GEN_DIR}/run.sh" in sb.written


def test_registry_sweeps_idle_sessions():
    reg = gsession.SessionRegistry(idle_timeout_s=0)
    sb = FakeSandbox(write_good)
    s = gsession.GeneratorSession(_req(), sandbox=sb)
    s.updated = 0
    reg._sessions[s.id] = s
    reg.sweep()
    assert reg.get(s.id) is None and sb.terminated


def test_summary_shape():
    s = gsession.GeneratorSession(_req(), sandbox=FakeSandbox(write_good))
    assert set(s.summary()) >= {"session_id", "state", "busy", "last_seq", "verified"}
