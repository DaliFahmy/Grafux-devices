"""
test_generator_actions.py
The Actions agent runner against a strict fake of the GitHub REST API: every
call must be one the fake knows, so a wrong path fails here, not on GitHub.
"""

from __future__ import annotations

import base64
import json
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from GENERATOR.actions import ActionsAgentRunner, TurnSpec, build_turn_tree, turn_title  # noqa: E402
from GENERATOR.builder import GitHubBuilder  # noqa: E402

R = "/repos/o/builds"


class _Resp:
    def __init__(self, status, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})
        self.content = self.text.encode()

    def json(self):
        return self._body if self._body is not None else {}


class FakeGitHub:
    """A branch store plus one dispatched run whose 'job' commits back ``result_tree``."""

    def __init__(self, result_tree=None, conclusion="success", job_commits=True):
        self.calls = []
        self.blobs = {}
        self.trees = {}
        self.commits = {}
        self.refs = {}
        self.dispatched = None
        self.polls = 0
        self.result_tree = result_tree or {}
        self.conclusion = conclusion
        self.job_commits = job_commits

    def _store_tree(self, files):
        entries = []
        for path, data in files.items():
            sha = f"b{len(self.blobs)}"
            self.blobs[sha] = data
            entries.append({"path": path, "type": "blob", "sha": sha})
        tsha = f"t{len(self.trees)}"
        self.trees[tsha] = entries
        csha = f"c{len(self.commits)}"
        self.commits[csha] = tsha
        return csha

    def request(self, method, path, json=None, params=None):
        self.calls.append((method, path, json))
        if method == "POST" and path == f"{R}/git/blobs":
            sha = f"b{len(self.blobs)}"
            self.blobs[sha] = base64.b64decode(json["content"])
            return _Resp(201, {"sha": sha})
        if method == "POST" and path == f"{R}/git/trees":
            tsha = f"t{len(self.trees)}"
            self.trees[tsha] = [{"path": e["path"], "type": "blob", "sha": e["sha"], "mode": e["mode"]}
                                for e in json["tree"]]
            return _Resp(201, {"sha": tsha})
        if method == "POST" and path == f"{R}/git/commits":
            csha = f"c{len(self.commits)}"
            self.commits[csha] = json["tree"]
            return _Resp(201, {"sha": csha})
        if method == "POST" and path == f"{R}/git/refs":
            self.refs[json["ref"].removeprefix("refs/heads/")] = json["sha"]
            return _Resp(201, {})
        if method == "GET" and path.startswith(f"{R}/git/ref/heads/"):
            return _Resp(200, {"object": {"sha": self.refs[path.removeprefix(f'{R}/git/ref/heads/')]}})
        if method == "GET" and path.startswith(f"{R}/git/commits/"):
            return _Resp(200, {"tree": {"sha": self.commits[path.rsplit('/', 1)[1]]}})
        if method == "GET" and path.startswith(f"{R}/git/trees/"):
            assert params == {"recursive": "1"}
            return _Resp(200, {"tree": self.trees[path.rsplit('/', 1)[1]]})
        if method == "GET" and path.startswith(f"{R}/git/blobs/"):
            data = self.blobs[path.rsplit("/", 1)[1]]
            return _Resp(200, {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        if method == "POST" and path == f"{R}/actions/workflows/agent.yml/dispatches":
            self.dispatched = json
            return _Resp(204, text="")
        if method == "GET" and path == f"{R}/actions/workflows/agent.yml/runs":
            if not self.dispatched:
                return _Resp(200, {"workflow_runs": []})
            i = self.dispatched["inputs"]
            return _Resp(200, {"workflow_runs": [
                {"id": 7, "display_title": f"agent {i['session_id']}-{i['turn']}",
                 "status": "queued", "html_url": "https://gh/run/7"}]})
        if method == "GET" and path == f"{R}/actions/runs/7":
            self.polls += 1
            done = self.polls >= 2
            if done and self.job_commits:
                branch = "session/" + self.dispatched["inputs"]["session_id"]
                self.refs[branch] = self._store_tree(self.result_tree)
            return _Resp(200, {"id": 7, "status": "completed" if done else "in_progress",
                               "conclusion": self.conclusion if done else None})
        if method == "GET" and path == f"{R}/actions/runs/7/jobs":
            return _Resp(200, {"jobs": [{"id": 70, "name": "turn", "conclusion": "failure",
                                         "steps": [{"name": "Run the agent", "status": "in_progress"}]}]})
        if method == "GET" and path == f"{R}/actions/jobs/70/logs":
            return _Resp(200, text="setup exploded")
        if method == "POST" and path == f"{R}/actions/runs/7/cancel":
            return _Resp(202, {})
        raise AssertionError(f"unexpected GitHub call {method} {path}")


def _runner(gh):
    b = GitHubBuilder("tok", "o/builds", http=gh, poll_s=0, sleep=lambda s: None)
    return ActionsAgentRunner(b, poll_s=0)


def _spec(**over):
    base = dict(session_id="abc123", turn=2, command="claude -p x", prompt="build it",
                system="SYSTEM", mode="edit", agent="claude_code",
                repo_url="https://github.com/VLSIDA/OpenRAM.git", ref="stable",
                cli_versions={"claude_code": "2.1.283"}, files={"run.sh": "#!/bin/bash\n"},
                state={"claude/-workspace/S1.jsonl": b"{}"}, sealed="SEALED-BLOB")
    base.update(over)
    return TurnSpec(**base)


def test_the_turn_tree_holds_everything_the_job_needs():
    tree = build_turn_tree(_spec())
    assert tree["gen/run.sh"] == b"#!/bin/bash\n"
    assert tree[".grafux/state/claude/-workspace/S1.jsonl"] == b"{}"
    assert tree[".grafux/run.sh"].startswith(b"#!/bin/bash\nclaude -p x")
    turn = json.loads(tree[".grafux/turn.json"])
    assert turn["mode"] == "edit" and turn["ref"] == "stable" and turn["cli_versions"]["claude_code"]
    assert b"SEALED" not in b"".join(tree.values())          # the secret is an input, not a file
    assert turn_title("abc123", 2) == "agent abc123-2"


def test_a_turn_round_trip():
    result = {"gen/run.sh": b"#!/bin/bash\necho hi\n", "gen/logo.png": b"\x89PNG\x00\xff",
              ".grafux/state/claude/-workspace/S1.jsonl": b"{\"t\":1}",
              ".grafux/out/agent.jsonl": b'{"type":"system","subtype":"init","session_id":"S1"}\nplain\n',
              ".grafux/out/exit_code": b"0\n", ".grafux/prompt.md": b"ignored"}
    gh = FakeGitHub(result_tree=result)
    statuses = []
    out = _runner(gh).run_turn(_spec(), on_status=lambda s, u="": statuses.append((s, u)))
    assert out.ok and not out.infra and out.exit_code == 0
    assert out.files == {"run.sh": "#!/bin/bash\necho hi\n"} and out.binaries == ["logo.png"]
    assert out.state == {"claude/-workspace/S1.jsonl": b"{\"t\":1}"}
    assert out.lines == ['{"type":"system","subtype":"init","session_id":"S1"}', "plain"]
    assert gh.dispatched == {"ref": "main", "inputs": {"session_id": "abc123", "turn": "2",
                                                        "sealed": "SEALED-BLOB"}}
    modes = {e["path"]: e["mode"] for tree in gh.trees.values() for e in tree if "mode" in e}
    assert modes[".grafux/run.sh"] == "100755"
    assert ("agent run: https://gh/run/7", "https://gh/run/7") in statuses
    assert ("Run the agent", "") in statuses


def test_a_nonzero_agent_exit_is_a_failed_turn_not_infra():
    gh = FakeGitHub(result_tree={".grafux/out/exit_code": b"1"})
    out = _runner(gh).run_turn(_spec())
    assert not out.ok and not out.infra and out.exit_code == 1


def test_a_job_that_never_committed_is_infra_with_its_log():
    gh = FakeGitHub(job_commits=False, conclusion="failure")
    out = _runner(gh).run_turn(_spec())
    assert not out.ok and out.infra and "setup exploded" in out.detail


def test_stop_cancels_the_job():
    gh = FakeGitHub()
    out = _runner(gh).run_turn(_spec(), should_cancel=lambda: True)
    assert out.infra and out.detail == "stopped"
    assert ("POST", f"{R}/actions/runs/7/cancel", None) in gh.calls


def test_oversized_state_is_dropped_with_a_note(monkeypatch):
    from GENERATOR import actions
    monkeypatch.setattr(actions, "MAX_STATE_BYTES", 3)
    gh = FakeGitHub(result_tree={".grafux/state/claude/x.jsonl": b"12345"})
    out = _runner(gh).run_turn(_spec())
    assert out.state == {} and "too large" in out.detail
