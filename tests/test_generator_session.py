"""
test_generator_session.py
The Generator's loop, end to end, with a fake runner (whose "agent" is a Python
function that edits the block's files, as the Actions job would) and a fake
builder.  No pod, no GitHub, no LLM.

What must hold: plan turns cannot change the block; a finished block needs a
passing build (with its self-test) and NOTHING else -- no pod; every failure
goes back to the agent as text; a repair that changes nothing stops the loop;
infrastructure failures are never sent to the agent; live lines are not
duplicated by the end-of-turn replay; keys never appear in events.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from GENERATOR import contract, session as gsession  # noqa: E402
from GENERATOR.actions import TurnOutcome  # noqa: E402
from GENERATOR.builder import BuildResult  # noqa: E402
from GENERATOR.models import CreateSessionRequest, unseal  # noqa: E402

KEY = "sk-ant-SECRET-1234567890"
WRAP = Fernet.generate_key().decode()

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

SERVER_KEY_VARS = ("GENERATOR_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY",
                   "GENERATOR_OPENAI_API_KEY", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Each test decides which server keys exist; the developer's env must not leak in."""
    for name in SERVER_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GENERATOR_WRAP_KEY", WRAP)
    monkeypatch.setenv("GENERATOR_PUBLIC_URL", "https://devices.example")


def _lines(answer):
    return [json.dumps(o) for o in (
        {"type": "system", "subtype": "init", "session_id": "S1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"using {KEY}"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": answer,
         "total_cost_usd": 0.1, "session_id": "S1"})]


class FakeRunner:
    """
    Stands in for the Actions job.  ``script(prompt, mode, files)`` edits a copy
    of the files (as the agent would, in either mode -- the SESSION must discard
    a plan turn's edits) and returns the answer.  ``live`` lines are pushed
    through ``session.ingest`` mid-turn, like the job's forwarder.
    """

    def __init__(self, script, *, live=0, outcome=None):
        self.script = script
        self.live = live
        self.outcome = outcome
        self.specs = []
        self.session = None

    def run_turn(self, spec, *, on_status=lambda *a: None, should_cancel=lambda: False):
        self.specs.append(spec)
        if self.outcome is not None:
            return self.outcome
        files = dict(spec.files)
        answer = self.script(spec.prompt, spec.mode, files) or "ok"
        lines = _lines(answer)
        if self.live and self.session is not None:
            self.session.ingest(self.session.callback_token, spec.turn,
                                [{"n": i, "line": ln} for i, ln in enumerate(lines[: self.live])])
        on_status("agent run: https://gh/run/1", "https://gh/run/1")
        return TurnOutcome(ok=True, exit_code=0, files=files, lines=lines,
                           state={"claude/-workspace/S1.jsonl": b"{}"}, run_url="https://gh/run/1")


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


def _run(script, *, builder=None, mode="edit", runner=None, **req_kw):
    runner = runner or FakeRunner(script)
    s = gsession.GeneratorSession(_req(mode=mode, **req_kw), runner=runner, builder=builder,
                                  max_repairs=2)
    runner.session = s
    s.start(mode)
    s.join(10)
    return s, runner


def _kinds(s):
    return [e["kind"] for e in s.events]


def write_good(prompt, mode, files):
    files.update(GOOD)
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


def test_the_generator_rents_no_pods():
    """The only place pods are created is a block's Regenerate."""
    root = os.path.join(os.path.dirname(__file__), "..", "GENERATOR")
    for name in os.listdir(root):
        if name.endswith(".py"):
            text = open(os.path.join(root, name), encoding="utf-8").read()
            for forbidden in ("provision_eda", "EDA.runtime", "EDA import runtime", "pod_client",
                              "start_custom_job", "runpod"):
                assert forbidden not in text, f"{name} mentions {forbidden}"


def test_edit_happy_path_builds_and_returns_the_block():
    s, runner = _run(write_good, builder=FakeBuilder([ok_build()]))
    assert s.state == "done", s.events
    assert s.result["verified"] is True
    assert s.result["image"].startswith(f"{contract.IMAGE_REPO}:ahmed-demo-")
    assert s.agent_session_id == "S1"
    assert pytest.approx(s.cost_usd) == 0.1
    spec = runner.specs[0]
    assert spec.repo_url == "https://github.com/VLSIDA/OpenRAM.git" and spec.ref == "stable"
    assert "--permission-mode acceptEdits" in spec.command and "--resume" not in spec.command
    assert spec.cli_versions["claude_code"]
    assert s.state_files == {"claude/-workspace/S1.jsonl": b"{}"}
    assert any(e.get("data", {}).get("url") == "https://gh/run/1" for e in s.events)


def test_the_job_gets_the_key_only_sealed():
    s, runner = _run(write_good, builder=FakeBuilder([ok_build()]))
    spec = runner.specs[0]
    assert KEY not in spec.command and KEY not in spec.prompt and KEY not in json.dumps(spec.files)
    payload = unseal(spec.sealed)
    assert payload["ANTHROPIC_API_KEY"] == KEY and "OPENAI_API_KEY" not in payload
    assert payload["callback_url"] == f"https://devices.example/generator/sessions/{s.id}/ingest"
    assert payload["callback_token"] == s.callback_token


def test_keys_never_reach_events():
    s, _ = _run(write_good, builder=FakeBuilder([ok_build()]))
    blob = json.dumps(s.events)
    assert KEY not in blob and s.callback_token not in blob and "***" in blob


def test_plan_mode_discards_edits_and_returns_the_plan():
    def planner(prompt, mode, files):
        assert mode == "plan" and "Do NOT write any files" in prompt
        files["Dockerfile"] = "FROM sneaky\n"
        return "PLAN: ports n -> out"

    s, runner = _run(planner, mode="plan")
    assert s.state == "awaiting_user"
    assert s.files == {}
    assert "--permission-mode plan" in runner.specs[0].command
    assert "warning" in _kinds(s)
    assert next(e for e in s.events if e["kind"] == "plan")["text"] == "PLAN: ports n -> out"


def test_live_lines_are_not_duplicated_by_the_replay():
    runner = FakeRunner(write_good, live=2)
    s, _ = _run(None, runner=runner, builder=FakeBuilder([ok_build()]))
    sessions_seen = [e for e in s.events if e["kind"] == "session"]
    thoughts = [e for e in s.events if e["kind"] == "thought"]
    answers = [e for e in s.events if e["kind"] == "answer"]
    assert len(sessions_seen) == 1 and len(thoughts) == 1 and len(answers) == 1


def test_ingest_checks_the_token_and_the_turn():
    runner = FakeRunner(write_good)
    s = gsession.GeneratorSession(_req(), runner=runner)
    with pytest.raises(PermissionError):
        s.ingest("wrong", 0, [{"n": 0, "line": "x"}])
    assert s.ingest(s.callback_token, 99, [{"n": 0, "line": "x"}]) == 0   # stale turn
    s._turn_no = 1
    assert s.ingest(s.callback_token, 1, [{"n": 1, "line": "a"}, {"n": 0, "line": "b"}]) == 2
    assert s.ingest(s.callback_token, 1, [{"n": 1, "line": "a"}]) == 0     # duplicate


def test_invalid_files_are_repaired_then_built():
    calls = []

    def agent(prompt, mode, files):
        calls.append(prompt)
        if len(calls) == 1:
            files["Dockerfile"] = "FROM x\n"          # no manifest, no run.sh
        else:
            assert "rejected before building" in prompt and "run.sh is missing" in prompt
            files.update(GOOD)

    s, runner = _run(agent, builder=FakeBuilder([ok_build()]))
    assert s.state == "done" and len(calls) == 2
    assert "repair" in _kinds(s)
    assert "--resume S1" in runner.specs[1].command
    assert runner.specs[1].state == {"claude/-workspace/S1.jsonl": b"{}"}


def test_build_failure_log_is_the_repair_order():
    prompts = []

    def agent(prompt, mode, files):
        prompts.append(prompt)
        files.update(GOOD)
        files["run.sh"] = f"#!/bin/bash\n# attempt {len(prompts)}\n"

    b = FakeBuilder([BuildResult(False, "img", "GRAFUX SELFTEST FAILED: out"), ok_build()])
    s, _ = _run(agent, builder=b)
    assert s.state == "done"
    assert "GRAFUX SELFTEST FAILED: out" in prompts[1]


def test_a_repair_that_changes_nothing_stops_the_loop():
    b = FakeBuilder([BuildResult(False, "img", "boom")] * 5)
    s, _ = _run(write_good, builder=b)
    assert s.state == "awaiting_user"
    assert len(b.built) == 1          # the identical repair is caught before a second build
    assert any("did not change any file" in e["text"] for e in s.events)


def test_gives_up_after_max_repairs():
    n = [0]

    def agent(prompt, mode, files):
        n[0] += 1
        files.update(GOOD)
        files["run.sh"] = f"#!/bin/bash\n# {n[0]}\n"

    b = FakeBuilder([BuildResult(False, "img", "boom")] * 5)
    s, _ = _run(agent, builder=b)
    assert s.state == "awaiting_user" and len(b.built) == 3
    assert any("Still failing after 2 repairs" in e["text"] for e in s.events)


def test_infra_failures_are_not_repair_orders():
    s, _ = _run(write_good, builder=FakeBuilder([BuildResult(False, "i", "401", infra=True)]))
    assert s.state == "error" and "repair" not in _kinds(s)

    runner = FakeRunner(None, outcome=TurnOutcome(False, infra=True, detail="job never appeared"))
    s2, _ = _run(None, runner=runner, builder=FakeBuilder([]))
    assert s2.state == "error" and any("job never appeared" in e["text"] for e in s2.events)


def test_a_failed_agent_turn_waits_for_the_user():
    runner = FakeRunner(None, outcome=TurnOutcome(False, exit_code=1, lines=[
        json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "gave up"})]))
    s, _ = _run(None, runner=runner, builder=FakeBuilder([]))
    assert s.state == "awaiting_user"
    assert any("The agent's turn failed: gave up" in e["text"] for e in s.events)


def test_missing_builder_or_key_are_explained():
    s, _ = _run(write_good, builder=None)
    assert s.state == "error" and any("builder is not configured" in e["text"] for e in s.events)

    s2, runner = _run(write_good, anthropic_api_key="")
    assert any("No Anthropic key" in e["text"] and "GENERATOR_ANTHROPIC_API_KEY" in e["text"]
               for e in s2.events)
    assert runner.specs == []                     # nothing dispatched without a key


def test_follow_up_messages_resume_the_agent_session():
    prompts = []

    def agent(prompt, mode, files):
        prompts.append((prompt, mode))
        return "plan v%d" % len(prompts)

    runner = FakeRunner(agent)
    s = gsession.GeneratorSession(_req(mode="plan"), runner=runner)
    runner.session = s
    s.start("plan")
    s.join(10)
    s.message("use sky130 too", "plan")
    s.join(10)
    assert prompts[1] == ("use sky130 too", "plan")
    assert "--resume S1" in runner.specs[1].command
    assert [sp.turn for sp in runner.specs] == [1, 2]
    s.stop()
    assert s.state == "stopped" and s.cancel.is_set()
    with pytest.raises(RuntimeError):
        s.message("again", "plan")


def test_registry_sweeps_idle_sessions():
    reg = gsession.SessionRegistry(idle_timeout_s=0)
    s = gsession.GeneratorSession(_req(), runner=FakeRunner(write_good))
    s.updated = 0
    reg._sessions[s.id] = s
    reg.sweep()
    assert reg.get(s.id) is None


SERVER_KEY = "sk-ant-GRAFUX-SERVER-KEY-0987654321"


def test_without_a_user_key_the_session_runs_on_grafuxs_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SERVER_KEY)
    s, runner = _run(write_good, builder=FakeBuilder([ok_build()]), anthropic_api_key="")
    assert s.state == "done", s.events
    assert unseal(runner.specs[0].sealed)["ANTHROPIC_API_KEY"] == SERVER_KEY
    info = next(e for e in s.events if e["kind"] == "info")
    assert info["text"] == "Using Grafux's Anthropic key." and info["data"]["key_source"] == "grafux"
    assert SERVER_KEY not in json.dumps(s.events)


@pytest.mark.parametrize("env,given,expected,source", [
    ({"ANTHROPIC_API_KEY": "general"}, "", "general", "grafux"),
    ({"ANTHROPIC_API_KEY": "general", "GENERATOR_ANTHROPIC_API_KEY": "dedicated"}, "",
     "dedicated", "grafux"),
    ({"GENERATOR_ANTHROPIC_API_KEY": "dedicated"}, "  mine  ", "mine", "user"),
    ({"ANTHROPIC_API_KEY": "   "}, "", "", ""),
])
def test_key_precedence(monkeypatch, env, given, expected, source):
    from GENERATOR.models import resolve_agent_key
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert resolve_agent_key("anthropic", given) == (expected, source)


def test_a_user_key_is_announced_as_theirs():
    s, _ = _run(write_good, mode="plan")
    info = next(e for e in s.events if e["kind"] == "info")
    assert info["text"] == "Using your Anthropic key."
