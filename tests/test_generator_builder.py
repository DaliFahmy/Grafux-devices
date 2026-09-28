"""
test_generator_builder.py
The GitHub Actions builder against a strict fake of the REST API: every call it
makes must be one the fake knows, so a typo in a path fails here, not on GitHub.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from GENERATOR.builder import BuilderNotConfigured, GitHubBuilder  # noqa: E402


class _Resp:
    def __init__(self, status, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})
        self.content = self.text.encode()

    def json(self):
        return self._body if self._body is not None else {}


class FakeGitHub:
    def __init__(self, conclusion="success", ref_exists=False, log="step 1\nGRAFUX SELFTEST FAILED: x\nend",
                 failed_steps=()):
        self.calls = []
        self.failed_steps = failed_steps
        self.conclusion = conclusion
        self.ref_exists = ref_exists
        self.log = log
        self.polls = 0
        self.dispatched = None

    def request(self, method, path, json=None, params=None):
        self.calls.append((method, path, json))
        r = "/repos/o/builds"
        if method == "POST" and path == f"{r}/git/blobs":
            return _Resp(201, {"sha": f"blob{len(self.calls)}"})
        if method == "POST" and path == f"{r}/git/trees":
            return _Resp(201, {"sha": "tree1"})
        if method == "POST" and path == f"{r}/git/commits":
            assert json["parents"] == []
            return _Resp(201, {"sha": "commit1"})
        if method == "POST" and path == f"{r}/git/refs":
            return _Resp(422, text="exists") if self.ref_exists else _Resp(201, {"ref": json["ref"]})
        if method == "PATCH" and path.startswith(f"{r}/git/refs/heads/"):
            return _Resp(200, {})
        if method == "POST" and path == f"{r}/actions/workflows/build.yml/dispatches":
            self.dispatched = json
            return _Resp(204, text="")
        if method == "GET" and path == f"{r}/actions/workflows/build.yml/runs":
            if not self.dispatched:
                return _Resp(200, {"workflow_runs": []})
            return _Resp(200, {"workflow_runs": [
                {"id": 1, "display_title": "build other"},
                {"id": 7, "display_title": f"build {self.dispatched['inputs']['tag']}",
                 "status": "queued", "html_url": "https://gh/run/7"}]})
        if method == "GET" and path == f"{r}/actions/runs/7":
            self.polls += 1
            done = self.polls >= 2
            return _Resp(200, {"id": 7, "status": "completed" if done else "in_progress",
                               "conclusion": self.conclusion if done else None})
        if method == "GET" and path == f"{r}/actions/runs/7/jobs":
            return _Resp(200, {"jobs": [{"id": 70, "name": "build", "conclusion": "failure", "steps": [
                {"name": n, "conclusion": "failure"} for n in self.failed_steps]}]})
        if method == "GET" and path == f"{r}/actions/jobs/70/logs":
            return _Resp(200, text=self.log)
        if method == "POST" and path == f"{r}/actions/runs/7/cancel":
            return _Resp(202, {})
        raise AssertionError(f"unexpected GitHub call {method} {path}")


def _builder(gh):
    return GitHubBuilder("tok", "o/builds", http=gh, poll_s=0, sleep=lambda s: None)


def test_not_configured_without_token_or_repo(monkeypatch):
    monkeypatch.delenv("GENERATOR_GITHUB_TOKEN", raising=False)
    with pytest.raises(BuilderNotConfigured):
        GitHubBuilder.from_env()


def test_a_successful_build():
    gh = FakeGitHub()
    res = _builder(gh).build("u-demo-abc", {"Dockerfile": b"FROM x", "run.sh": b"#!/bin/sh"})
    assert res.ok and res.image.endswith(":u-demo-abc") and res.run_url == "https://gh/run/7"
    assert gh.dispatched == {"ref": "main", "inputs": {"branch": "build/u-demo-abc", "tag": "u-demo-abc"}}
    tree = next(c[2] for c in gh.calls if c[1].endswith("/git/trees"))["tree"]
    modes = {t["path"]: t["mode"] for t in tree}
    assert modes == {"Dockerfile": "100644", "run.sh": "100755"}


def test_a_rebuild_force_updates_the_branch():
    gh = FakeGitHub(ref_exists=True)
    assert _builder(gh).build("t", {"Dockerfile": b"FROM x"}).ok
    assert any(c[0] == "PATCH" for c in gh.calls)


def test_a_failed_build_returns_the_log_tail_as_a_repair_order():
    gh = FakeGitHub(conclusion="failure")
    res = _builder(gh).build("t", {"Dockerfile": b"FROM x"})
    assert not res.ok and not res.infra
    assert "GRAFUX SELFTEST FAILED: x" in res.detail


def test_selftest_lines_survive_a_long_log():
    log = "GRAFUX SELFTEST FAILED: early\n" + "\n".join(f"noise {i}" for i in range(500))
    res = _builder(FakeGitHub(conclusion="failure", log=log)).build("t", {"Dockerfile": b"FROM x"})
    assert res.detail.startswith("GRAFUX SELFTEST FAILED: early")


def test_a_refused_push_is_infra_not_a_repair_order():
    log = ("GRAFUX SELFTEST OK\nThe push refers to repository [ghcr.io/dalifahmy/grafux-gen]\n"
           "denied: permission_denied: write_package")
    gh = FakeGitHub(conclusion="failure", log=log, failed_steps=["Push"])
    res = _builder(gh).build("t", {"Dockerfile": b"FROM x"})
    assert not res.ok and res.infra
    assert "passed its self-test" in res.detail and "o/builds" in res.detail
    assert "write_package" in res.detail


def test_a_denial_without_named_steps_is_still_infra():
    log = "denied: permission_denied: write_package"
    res = _builder(FakeGitHub(conclusion="failure", log=log)).build("t", {"Dockerfile": b"FROM x"})
    assert res.infra


def test_a_failed_build_step_stays_a_repair_order_even_if_it_says_denied():
    log = "E: Could not open lock file - open (13: Permission denied)\ndenied: permission_denied"
    gh = FakeGitHub(conclusion="failure", log=log,
                    failed_steps=["Build (includes the Grafux self-test)"])
    res = _builder(gh).build("t", {"Dockerfile": b"FROM x"})
    assert not res.ok and not res.infra


def test_api_failures_are_infra_not_repair_orders():
    class Broken(FakeGitHub):
        def request(self, method, path, json=None, params=None):
            return _Resp(401, text="Bad credentials")
    res = _builder(Broken()).build("t", {"Dockerfile": b"FROM x"})
    assert not res.ok and res.infra and "401" in res.detail


def test_stop_cancels_the_run():
    gh = FakeGitHub()
    res = _builder(gh).build("t", {"Dockerfile": b"FROM x"}, should_cancel=lambda: True)
    assert not res.ok and res.infra
    assert ("POST", "/repos/o/builds/actions/runs/7/cancel", None) in gh.calls
