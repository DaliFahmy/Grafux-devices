"""
test_claw_runtime.py
Unit tests for the OpenClaw (claw block) runtime and router.

No real Anthropic account or network is used: a fake ``anthropic`` module is
injected into ``sys.modules`` so the lazy ``from anthropic import AsyncAnthropic``
inside ``claw_runtime`` resolves to an in-memory recorder.  Tests exercise the
request-building logic (sampling-param gating, usage/cost), the block-level
session memory threading, in-place config patching, and the models endpoint.
"""

import sys
import types

import pytest

from openclaw import claw_runtime, connections
from openclaw.models import ClawSpec, ConfigPatchRequest, RunRequest
from openclaw.registry import registry
from openclaw.sessions import sessions
from openclaw import router as claw_router


# ---------------------------------------------------------------------------
# Fake Anthropic SDK
# ---------------------------------------------------------------------------

class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, **kw) -> None:
        self.input_tokens = kw.get("input_tokens", 0)
        self.output_tokens = kw.get("output_tokens", 0)
        self.cache_read_input_tokens = kw.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens = kw.get("cache_creation_input_tokens", 0)


class _FakeMessage:
    def __init__(self, text: str, usage: _FakeUsage) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = usage


class _Recorder:
    """Captures every messages.create() call so tests can assert on the request."""

    def __init__(self) -> None:
        self.calls = []
        self.reply_text = "hello from the claw"
        self.usage = _FakeUsage(input_tokens=1000, output_tokens=500)
        self.stream_chunks = ["Hello ", "from ", "the ", "claw"]


class _FakeStream:
    """Async context manager mimicking client.messages.stream()."""

    def __init__(self, rec: "_Recorder") -> None:
        self._rec = rec

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        chunks = list(self._rec.stream_chunks)

        async def gen():
            for c in chunks:
                yield c

        return gen()

    async def get_final_message(self):
        return _FakeMessage("".join(self._rec.stream_chunks), self._rec.usage)


def _install_fake_anthropic(monkeypatch) -> _Recorder:
    rec = _Recorder()

    class _Messages:
        async def create(self, **kwargs):
            rec.calls.append(kwargs)
            return _FakeMessage(rec.reply_text, rec.usage)

        def stream(self, **kwargs):
            rec.calls.append(kwargs)
            return _FakeStream(rec)

    class _Beta:
        def __init__(self):
            self.messages = _Messages()

    class _FakeAsyncAnthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = _Messages()
            self.beta = _Beta()

    mod = types.ModuleType("anthropic")
    mod.AsyncAnthropic = _FakeAsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return rec


@pytest.fixture
def fake_anthropic(monkeypatch):
    return _install_fake_anthropic(monkeypatch)


@pytest.fixture(autouse=True)
def _clean_registry():
    for s in list(registry.list()):
        registry.delete(s.claw_id)
    yield
    for s in list(registry.list()):
        registry.delete(s.claw_id)


def _spec(**kw) -> ClawSpec:
    base = dict(api_keys="sk-test-key", soul="You are a test claw.")
    base.update(kw)
    return ClawSpec(**base)


# ---------------------------------------------------------------------------
# Pure helpers — sampling support + cost
# ---------------------------------------------------------------------------

def test_model_accepts_sampling_table():
    assert claw_runtime._model_accepts_sampling("claude-sonnet-4-6") is True
    assert claw_runtime._model_accepts_sampling("claude-opus-4-6") is True
    assert claw_runtime._model_accepts_sampling("claude-haiku-4-5") is True
    # Opus 4.7/4.8 + Fable reject sampling → must be False
    assert claw_runtime._model_accepts_sampling("claude-opus-4-8") is False
    assert claw_runtime._model_accepts_sampling("claude-opus-4-7") is False
    assert claw_runtime._model_accepts_sampling("claude-fable-5") is False
    # Unknown models default to False (safe — the frontier default rejects them)
    assert claw_runtime._model_accepts_sampling("some-future-model") is False


def test_estimate_cost_usd():
    usage = _FakeUsage(input_tokens=1000, output_tokens=500)
    # opus-4-8: $5/1M in, $25/1M out → 1000*5e-6 + 500*25e-6 = 0.005 + 0.0125
    assert claw_runtime._estimate_cost_usd("claude-opus-4-8", usage) == pytest.approx(0.0175)
    # unknown model → 0.0 (badge falls back to token counts)
    assert claw_runtime._estimate_cost_usd("mystery", usage) == 0.0


# ---------------------------------------------------------------------------
# run_claw — sampling-param gating (the temperature-400 fix)
# ---------------------------------------------------------------------------

async def test_temperature_omitted_for_default_model(fake_anthropic):
    # Default agent ⇒ claude-opus-4-8, which 400s on temperature: must be absent.
    result = await claw_runtime.run_claw(_spec(), task="hi")
    assert result["status"] == "ok"
    assert "temperature" not in fake_anthropic.calls[0]
    assert "top_p" not in fake_anthropic.calls[0]


async def test_temperature_dropped_even_when_user_sets_it_on_nonsampling_model(fake_anthropic):
    spec = _spec(agent='{"model": "claude-opus-4-8", "temperature": 0.5}')
    await claw_runtime.run_claw(spec, task="hi")
    assert "temperature" not in fake_anthropic.calls[0]


async def test_temperature_sent_for_sampling_model_when_user_sets_it(fake_anthropic):
    spec = _spec(agent='{"model": "claude-sonnet-4-6", "temperature": 0.3}')
    await claw_runtime.run_claw(spec, task="hi")
    assert fake_anthropic.calls[0]["temperature"] == 0.3
    assert fake_anthropic.calls[0]["model"] == "claude-sonnet-4-6"


async def test_temperature_absent_when_user_does_not_set_it_on_sampling_model(fake_anthropic):
    spec = _spec(agent="claude-sonnet-4-6")  # bare id, no temperature
    await claw_runtime.run_claw(spec, task="hi")
    assert "temperature" not in fake_anthropic.calls[0]


# ---------------------------------------------------------------------------
# run_claw — usage + cost in the result
# ---------------------------------------------------------------------------

async def test_usage_and_cost_in_result(fake_anthropic):
    fake_anthropic.usage = _FakeUsage(input_tokens=800, output_tokens=500, cache_read_input_tokens=200)
    result = await claw_runtime.run_claw(_spec(), task="hi")
    assert result["output_tokens"] == 500
    # input_tokens (for the badge) sums uncached + cache read + cache write
    assert result["input_tokens"] == 800 + 200
    assert result["cache_read_input_tokens"] == 200
    assert result["cost_usd"] > 0.0


async def test_error_path_has_zero_usage_defaults(monkeypatch):
    # No anthropic installed ⇒ friendly error, and RunResponse usage fields default to 0.
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))
    result = await claw_runtime.run_claw(_spec(), task="hi")
    assert result["status"] == "error"
    assert "errors" in result
    assert result.get("cost_usd", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Router — block-level session memory (A1)
# ---------------------------------------------------------------------------

async def test_session_memory_threads_and_appends(fake_anthropic):
    claw_id = registry.create(_spec())
    sessions.clear(claw_id, "block", "s1")
    fake_anthropic.reply_text = "first answer"

    await claw_router.run_claw(claw_id, RunRequest(text_message="remember 42", session_id="s1"))
    transcript = sessions.get(claw_id, "block", "s1")
    assert "User: remember 42" in transcript
    assert "Assistant: first answer" in transcript

    # Second turn: prior transcript must be folded into the request the model sees.
    fake_anthropic.reply_text = "second answer"
    await claw_router.run_claw(claw_id, RunRequest(text_message="what was it?", session_id="s1"))
    user_content = fake_anthropic.calls[-1]["messages"][0]["content"]
    assert "remember 42" in user_content
    assert "first answer" in user_content


async def test_empty_session_id_is_stateless(fake_anthropic):
    claw_id = registry.create(_spec())
    await claw_router.run_claw(claw_id, RunRequest(text_message="no memory", session_id=""))
    # Nothing recorded under the block provider for any chat id.
    assert sessions.get(claw_id, "block", "") == ""
    # And the request carries no "prior context" preamble.
    user_content = fake_anthropic.calls[-1]["messages"][0]["content"]
    assert "Relevant prior context" not in user_content


async def test_reset_session_clears_transcript(fake_anthropic):
    claw_id = registry.create(_spec())
    await claw_router.run_claw(claw_id, RunRequest(text_message="hi", session_id="s2"))
    assert sessions.get(claw_id, "block", "s2") != ""
    out = await claw_router.clear_claw_session(claw_id, "s2")
    assert out["status"] == "cleared"
    assert sessions.get(claw_id, "block", "s2") == ""


# ---------------------------------------------------------------------------
# Router — in-place config patch (B1) + summary enrichment (D1)
# ---------------------------------------------------------------------------

async def test_config_patch_applies_without_recreate(fake_anthropic):
    claw_id = registry.create(_spec(soul="old soul"))
    summary = await claw_router.patch_claw_config(
        claw_id, ConfigPatchRequest(soul="new soul", agent="claude-haiku-4-5")
    )
    # Same claw id (no re-create), spec mutated in place.
    assert summary.claw_id == claw_id
    assert summary.model == "claude-haiku-4-5"
    assert registry.get(claw_id).soul == "new soul"
    # Secrets are never touched by a config patch.
    assert registry.get(claw_id).api_keys == "sk-test-key"


async def test_config_patch_connections_reflected_in_summary(fake_anthropic):
    claw_id = registry.create(_spec())
    conns = '[{"app": "telegram", "mcp_url": "https://x/mcp", "enabled": true}]'
    summary = await claw_router.patch_claw_config(claw_id, ConfigPatchRequest(connections=conns))
    assert "telegram" in summary.apps
    assert summary.tool_count == 1


async def test_get_claw_summary_enriched(fake_anthropic):
    claw_id = registry.create(_spec(agent="claude-sonnet-4-6"))
    summary = await claw_router.get_claw(claw_id)
    assert summary.model == "claude-sonnet-4-6"
    assert summary.apps == []


# ---------------------------------------------------------------------------
# Router — models endpoint (C1)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Streaming — stream_claw generator (C4)
# ---------------------------------------------------------------------------

async def test_stream_claw_yields_deltas_then_done(fake_anthropic):
    fake_anthropic.stream_chunks = ["Hel", "lo!"]
    frames = []
    async for kind, payload in claw_runtime.stream_claw(_spec(), task="hi"):
        frames.append((kind, payload))
    kinds = [k for k, _ in frames]
    assert kinds[-1] == "done"
    deltas = "".join(p for k, p in frames if k == "delta")
    assert deltas == "Hello!"
    done = frames[-1][1]
    assert done["status"] == "ok"
    assert done["output_tokens"] == 500
    assert done["cost_usd"] > 0.0


async def test_stream_claw_errors_without_key(monkeypatch):
    _install_fake_anthropic(monkeypatch)
    spec = ClawSpec(soul="x")  # no api_keys, no env key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frames = [f async for f in claw_runtime.stream_claw(spec, task="hi")]
    assert frames and frames[0][0] == "error"


def test_stream_websocket_endpoint(fake_anthropic):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    fake_anthropic.stream_chunks = ["abc", "def"]
    claw_id = registry.create(_spec())

    app = FastAPI()
    app.include_router(claw_router.router)
    client = TestClient(app)

    with client.websocket_connect(f"/claw/{claw_id}/run/stream") as ws:
        ws.send_json({"text_message": "hello", "session_id": "ws1"})
        deltas, done = [], None
        while True:
            frame = ws.receive_json()
            if frame["type"] == "delta":
                deltas.append(frame["text"])
            elif frame["type"] == "done":
                done = frame
                break
            elif frame["type"] == "error":
                raise AssertionError(frame["error"])

    assert "".join(deltas) == "abcdef"
    assert done["response"] == "abcdef"
    assert done["status"] == "ok"
    # The streamed turn was threaded into the shared block conversation.
    assert "abcdef" in sessions.get(claw_id, "block", "ws1")


async def test_models_endpoint_lists_priced_models():
    resp = await claw_router.list_claw_models()
    by_id = {m.id: m for m in resp.models}
    assert "claude-opus-4-8" in by_id
    assert by_id["claude-opus-4-8"].input_per_mtok == 5.0
    assert by_id["claude-opus-4-8"].output_per_mtok == 25.0
    assert by_id["claude-haiku-4-5"].input_per_mtok == 1.0
