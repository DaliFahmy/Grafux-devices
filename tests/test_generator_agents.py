"""
test_generator_agents.py
The two agent drivers: the command lines they build and the events they parse.

The parser fixtures are the documented stream shapes (Claude Code stream-json,
Codex exec --json).  When a CLI is bumped in the sandbox image, capture a real
turn and add its lines here -- a looser fake would hide a schema break.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from GENERATOR import agents  # noqa: E402

KW = dict(prompt_file="/w/.grafux/prompt.md", system_file="/w/.grafux/system.md",
          cwd="/w", env_file="/w/.grafux/keys.env")


def test_claude_plan_command():
    cmd = agents.build_command("claude_code", mode="plan", model="claude-opus-5-5", **KW)
    assert cmd.startswith("set -a; . /w/.grafux/keys.env; set +a; cd /w && claude -p ")
    assert '"$(cat /w/.grafux/prompt.md)"' in cmd
    assert "--output-format stream-json --verbose" in cmd
    assert "--permission-mode plan" in cmd and "acceptEdits" not in cmd
    assert "--model claude-opus-5-5" in cmd and "--resume" not in cmd
    assert cmd.endswith("< /dev/null")


def test_claude_edit_command_resumes_and_preapproves_tools():
    cmd = agents.build_command("claude_code", mode="edit", resume_id="abc-123", **KW)
    assert "--permission-mode acceptEdits" in cmd
    assert "--allowedTools 'Bash Read Edit Write" in cmd
    assert "--resume abc-123" in cmd
    assert f"--model {agents.default_model('claude_code')}" in cmd


def test_codex_command_uses_the_pod_as_the_sandbox():
    cmd = agents.build_command("codex", mode="edit", model="gpt-5-codex", **KW)
    assert "codex exec --json --skip-git-repo-check --sandbox danger-full-access -m gpt-5-codex" in cmd
    resumed = agents.build_command("codex", mode="edit", resume_id="th_1", **KW)
    assert "codex exec resume th_1 --json" in resumed


def test_keys_are_sourced_never_inlined():
    cmd = agents.build_command("claude_code", mode="edit", **KW)
    assert "ANTHROPIC_API_KEY" not in cmd


@pytest.mark.parametrize("agent,mode", [("nope", "plan"), ("claude_code", "yolo")])
def test_bad_agent_or_mode(agent, mode):
    with pytest.raises(ValueError):
        agents.build_command(agent, mode=mode, **KW)


CLAUDE_LINES = [
    {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-opus-5-5"},
    {"type": "assistant", "session_id": "s-1", "message": {"content": [
        {"type": "text", "text": "Reading the README."},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls /workspace/src"}},
        {"type": "tool_use", "id": "t2", "name": "Write",
         "input": {"file_path": "/workspace/gen/run.sh", "content": "#!/bin/bash"}},
        {"type": "tool_use", "id": "t3", "name": "Grep", "input": {"pattern": "sram_compiler"}},
    ]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}},
    {"type": "result", "subtype": "success", "is_error": False, "result": "Done: 16 inputs.",
     "total_cost_usd": 0.42, "num_turns": 7, "session_id": "s-1"},
]


def _parse_all(agent, objs):
    out = []
    for o in objs:
        out += agents.parse_line(agent, json.dumps(o))
    return out


def test_claude_stream_is_normalized():
    evs = _parse_all("claude_code", CLAUDE_LINES)
    kinds = [e["kind"] for e in evs]
    assert kinds == ["session", "thought", "tool", "file", "tool", "result"]
    assert evs[2]["data"]["command"] == "ls /workspace/src"
    assert evs[3]["data"]["path"] == "/workspace/gen/run.sh"
    assert evs[-1]["data"]["cost_usd"] == 0.42 and not evs[-1]["data"]["is_error"]
    assert agents.session_id_from(evs) == "s-1"
    assert agents.final_text("claude_code", evs) == "Done: 16 inputs."


def test_claude_error_result():
    evs = _parse_all("claude_code", [{"type": "result", "subtype": "error_max_turns",
                                      "is_error": True, "result": ""}])
    assert evs[0]["data"]["is_error"] is True


CODEX_LINES = [
    {"type": "thread.started", "thread_id": "th_9"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "Look at setup.py"}},
    {"type": "item.completed", "item": {"type": "command_execution", "command": "cat README.md",
                                        "exit_code": 0, "aggregated_output": "..."}},
    {"type": "item.completed", "item": {"type": "file_change", "changes": [
        {"path": "/workspace/gen/Dockerfile", "kind": "add"},
        {"path": "/workspace/gen/run.sh", "kind": "add"}]}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "Block written."}},
    {"type": "turn.completed", "usage": {"input_tokens": 10}},
    {"type": "some.future.event"},
]


def test_codex_stream_is_normalized():
    evs = _parse_all("codex", CODEX_LINES)
    assert [e["kind"] for e in evs] == ["session", "thought", "tool", "file", "file", "thought", "result"]
    assert evs[1]["data"]["reasoning"] is True
    assert agents.session_id_from(evs) == "th_9"
    assert agents.final_text("codex", evs) == "Block written."


def test_codex_failures_become_errors():
    evs = _parse_all("codex", [{"type": "turn.failed", "error": {"message": "rate limited"}},
                               {"type": "error", "message": "boom"}])
    assert [(e["kind"], e["text"]) for e in evs] == [("error", "rate limited"), ("error", "boom")]


def test_non_json_lines_are_surfaced_not_swallowed():
    assert agents.parse_line("claude_code", "npm WARN something")[0]["kind"] == "log"
    assert agents.parse_line("codex", "") == []
    assert agents.parse_line("codex", "[1,2]") == []


def test_long_text_is_cut():
    evs = agents.parse_line("codex", json.dumps(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "x" * 20000}}))
    assert len(evs[0]["text"]) <= agents.EVENT_TEXT_MAX + 2
