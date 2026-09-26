"""
test_generator_router.py
The /generator REST surface through TestClient, with the runner and builder
seams replaced so no GitHub job, LLM or pod is touched.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from GENERATOR import router as grouter  # noqa: E402
from GENERATOR.builder import BuildResult  # noqa: E402
from GENERATOR.session import sessions  # noqa: E402
from tests.test_generator_session import GOOD, WRAP, FakeBuilder, FakeRunner  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    def agent(prompt, mode, files):
        if mode == "edit":
            files.update(GOOD)
        return "the plan" if mode == "plan" else "built"

    monkeypatch.setenv("GENERATOR_WRAP_KEY", WRAP)
    monkeypatch.setattr(grouter, "configuration_error", lambda: "")
    monkeypatch.setattr(grouter, "make_runner", lambda builder: FakeRunner(agent))
    monkeypatch.setattr(grouter, "make_builder", lambda: FakeBuilder([BuildResult(True, "i")] * 3))
    from device.app import app
    return TestClient(app)


def _wait(client, sid, until=lambda b: not b["busy"]):
    for _ in range(200):
        body = client.get(f"/generator/sessions/{sid}/events").json()
        if until(body):
            return body
        time.sleep(0.02)
    raise AssertionError("session never settled")


def _create(client, **over):
    body = {"agent": "claude_code", "mode": "plan", "prompt": "OpenRAM block",
            "repo_url": "https://github.com/VLSIDA/OpenRAM/tree/stable",
            "anthropic_api_key": "sk-ant-x-1234567890", "owner": "ahmed"}
    body.update(over)
    return client.post("/generator/sessions", json=body)


def test_agents_catalogue(client, monkeypatch):
    for name in ("GENERATOR_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY",
                 "GENERATOR_OPENAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server-secret-123")
    body = client.get("/generator/agents").json()
    assert [a["id"] for a in body["agents"]] == ["claude_code", "codex"]
    assert body["modes"] == ["plan", "edit"]
    assert body["configured"] is True and body["setup"] == ""
    # Whether Grafux holds a key -- never the key itself.
    assert [a["server_key"] for a in body["agents"]] == [True, False]
    assert "sk-ant-server-secret-123" not in json.dumps(body)


def test_an_unconfigured_server_refuses_with_the_missing_settings(monkeypatch):
    for name in ("GENERATOR_GITHUB_TOKEN", "GENERATOR_BUILDS_REPO", "GENERATOR_WRAP_KEY"):
        monkeypatch.delenv(name, raising=False)
    from device.app import app
    c = TestClient(app)
    r = _create(c)
    assert r.status_code == 503
    assert "GENERATOR_GITHUB_TOKEN" in r.json()["detail"] and "GENERATOR_WRAP_KEY" in r.json()["detail"]
    agents_body = c.get("/generator/agents").json()
    assert agents_body["configured"] is False and "GENERATOR_BUILDS_REPO" in agents_body["setup"]


def test_plan_then_edit_then_result(client):
    r = _create(client)
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    body = _wait(client, sid)
    assert body["state"] == "awaiting_user"
    assert any(e["kind"] == "plan" and e["text"] == "the plan" for e in body["events"])
    assert client.get(f"/generator/sessions/{sid}/result").status_code == 409

    last = body["next"]
    assert client.post(f"/generator/sessions/{sid}/message",
                       json={"text": "build it", "mode": "edit"}).status_code == 200
    body = _wait(client, sid)
    assert body["state"] == "done" and body["verified"]
    newer = client.get(f"/generator/sessions/{sid}/events", params={"after": last}).json()["events"]
    assert newer and all(e["seq"] > last for e in newer)

    result = client.get(f"/generator/sessions/{sid}/result").json()
    assert result["image"].startswith("ghcr.io/dalifahmy/grafux-gen:ahmed-demo-")
    assert "run.sh" in result["files"]

    assert client.post(f"/generator/sessions/{sid}/stop").json()["state"] == "stopped"
    assert client.delete(f"/generator/sessions/{sid}").status_code == 200
    assert client.get(f"/generator/sessions/{sid}").status_code == 404


def test_ingest_endpoint(client):
    sid = _create(client).json()["session_id"]
    _wait(client, sid)
    s = sessions.get(sid)
    turn = s._turn_no
    bad = client.post(f"/generator/sessions/{sid}/ingest",
                      json={"token": "nope", "turn": turn, "lines": []})
    assert bad.status_code == 403
    line = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "late"}]}})
    ok = client.post(f"/generator/sessions/{sid}/ingest",
                     json={"token": s.callback_token, "turn": turn, "lines": [{"n": 100, "line": line}]})
    assert ok.json() == {"accepted": 1}
    again = client.post(f"/generator/sessions/{sid}/ingest",
                        json={"token": s.callback_token, "turn": turn, "lines": [{"n": 100, "line": line}]})
    assert again.json() == {"accepted": 0}
    stale = client.post(f"/generator/sessions/{sid}/ingest",
                        json={"token": s.callback_token, "turn": turn + 5, "lines": [{"n": 101, "line": line}]})
    assert stale.json() == {"accepted": 0}
    assert client.post("/generator/sessions/nope/ingest",
                       json={"token": "x", "turn": 1, "lines": []}).status_code == 404


@pytest.mark.parametrize("body,status", [
    ({"agent": "gpt", "prompt": "x"}, 422),
    ({"mode": "yolo", "prompt": "x"}, 422),
    ({"prompt": "   "}, 422),
    ({"prompt": "x", "repo_url": "ftp://nope"}, 422),
])
def test_bad_requests(client, body, status):
    assert client.post("/generator/sessions", json=body).status_code == status


def test_unknown_session(client):
    assert client.get("/generator/sessions/nope/events").status_code == 404
    assert client.post("/generator/sessions/nope/message", json={"text": "x"}).status_code == 404
