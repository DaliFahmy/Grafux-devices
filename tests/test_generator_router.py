"""
test_generator_router.py
The /generator REST surface through TestClient, with the sandbox, builder and
smoke seams replaced so no pod, GitHub or LLM is touched.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from GENERATOR import router as grouter  # noqa: E402
from GENERATOR.builder import BuildResult  # noqa: E402
from tests.test_generator_session import GOOD, FakeBuilder, FakeSandbox  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    def agent(prompt, mode, gen):
        if mode == "edit":
            gen.update(GOOD)
        return "the plan" if mode == "plan" else "built"

    monkeypatch.setattr(grouter, "make_sandbox", lambda req: FakeSandbox(agent))
    monkeypatch.setattr(grouter, "make_builder", lambda: FakeBuilder([BuildResult(True, "i")] * 3))
    monkeypatch.setattr(grouter, "make_smoke", lambda: (lambda m, **k: (True, "", False)))
    from device.app import app
    return TestClient(app)


def _wait(client, sid, until=lambda b: not b["busy"]):
    for _ in range(200):
        body = client.get(f"/generator/sessions/{sid}/events").json()
        if until(body):
            return body
        time.sleep(0.02)
    raise AssertionError("session never settled")


def test_agents_catalogue(client):
    body = client.get("/generator/agents").json()
    assert [a["id"] for a in body["agents"]] == ["claude_code", "codex"]
    assert body["modes"] == ["plan", "edit"]


def test_plan_then_edit_then_result(client):
    r = client.post("/generator/sessions", json={
        "agent": "claude_code", "mode": "plan", "prompt": "OpenRAM block",
        "repo_url": "https://github.com/VLSIDA/OpenRAM/tree/stable",
        "anthropic_api_key": "sk-ant-x-1234567890", "owner": "ahmed"})
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
