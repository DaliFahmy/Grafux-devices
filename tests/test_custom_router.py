"""
test_custom_router.py
The /custom REST surface, through FastAPI's TestClient, plus the one piece of
the OpenRAM example that can be checked without a container: that its adapter
writes the same config the native openram block does.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from EDA import runtime as eda_runtime  # noqa: E402
from EDA.models import DEFAULT_IMAGE  # noqa: E402
from EDA.registry import registry  # noqa: E402

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "CUSTOM", "examples", "openram")
EXAMPLE = open(os.path.join(EXAMPLE_DIR, "grafux-block.json"), encoding="utf-8").read()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    from device.app import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_registry():
    for s in list(registry.list()):
        registry.delete(s.eda_id)
    yield
    for s in list(registry.list()):
        registry.delete(s.eda_id)


def test_run_accepts_a_manifest_and_an_inputs_map(client):
    resp = client.post("/custom/does-not-exist/run",
                       json={"manifest": EXAMPLE, "inputs": {"word_size": "4"}, "timeout": 0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "custom" and body["status"] == "error"
    assert "does-not-exist" in body["errors"]


def test_create_without_an_image_is_refused_before_renting_a_pod(client, monkeypatch):
    called = []
    monkeypatch.setattr(eda_runtime, "provision_eda_async", lambda spec: called.append(spec))
    for body in ({}, {"image": DEFAULT_IMAGE}, {"image": "  "}):
        resp = client.post("/custom/create_async", json=body)
        assert resp.status_code == 422, resp.text
        assert "own image" in resp.json()["detail"]
    assert called == []


def test_create_with_an_image_keeps_it_and_the_custom_kind(client, monkeypatch):
    seen = []

    def fake(spec):
        seen.append(spec)
        return {"eda_id": "x", "kind": spec.kind, "status": "creating", "phase": "creating"}

    monkeypatch.setattr(eda_runtime, "provision_eda_async", fake)
    resp = client.post("/custom/create_async",
                       json={"image": "ghcr.io/dalifahmy/grafux-gen:demo-1", "compute_type": "CPU"})
    assert resp.status_code == 200, resp.text
    assert seen[0].kind == "custom"
    assert seen[0].image == "ghcr.io/dalifahmy/grafux-gen:demo-1"
    assert seen[0].container_disk_gb == 20


def test_other_kinds_are_untouched_by_the_check_spec_hook(client, monkeypatch):
    seen = []
    monkeypatch.setattr(eda_runtime, "provision_eda_async",
                        lambda spec: seen.append(spec) or {"eda_id": "y", "kind": spec.kind,
                                                            "status": "creating"})
    resp = client.post("/cpu/create_async", json={})
    assert resp.status_code == 200, resp.text
    assert seen[0].kind == "cpu" and seen[0].image != DEFAULT_IMAGE


def test_manifest_endpoints(client):
    schema = client.get("/custom/manifest/schema")
    assert schema.status_code == 200 and "runtime" in schema.json()["properties"]

    ok = client.post("/custom/manifest/validate", json={"manifest": json.loads(EXAMPLE)}).json()
    assert ok["ok"] is True and ok["summary"]["slug"] == "openram"
    assert ok["summary"]["inputs"][-1] == "api_keys"

    ok_text = client.post("/custom/manifest/validate", json={"manifest": EXAMPLE}).json()
    assert ok_text["ok"] is True

    bad = client.post("/custom/manifest/validate", json={"manifest": {"slug": "x"}}).json()
    assert bad["ok"] is False and "runtime" in bad["errors"]


# ---------------------------------------------------------------------------
# The OpenRAM example's adapter vs the native block
# ---------------------------------------------------------------------------

def _load_adapter(tmp_path, ports):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    for k, v in ports.items():
        (tmp_path / "in" / k).write_text(v, encoding="utf-8")
    os.environ["GRAFUX_IN"] = str(tmp_path / "in")
    os.environ["GRAFUX_OUT"] = str(tmp_path / "out")
    os.environ["GRAFUX_WORK"] = str(tmp_path)
    spec = importlib.util.spec_from_file_location("openram_adapter",
                                                  os.path.join(EXAMPLE_DIR, "adapter.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _body(cfg: str):
    """The assignments, without comments or the forced output location."""
    lines = []
    for ln in cfg.splitlines():
        code = ln.split("#", 1)[0].strip()
        if code and not code.startswith(("output_path", "output_name")):
            lines.append(code)
    return lines


@pytest.mark.parametrize("ports", [
    {"word_size": "2", "num_words": "16"},
    {"word_size": "8", "num_words": "64", "num_banks": "2", "write_size": "4",
     "process_corners": "TT, SS", "supply_voltages": "1.8", "temperatures": "25,100",
     "check_lvsdrc": "true", "netlist_only": "yes", "extra_config": "route_supplies = True"},
    {"config": "word_size = 4\nnum_words = 8\ntech_name = 'scn4m_subm'"},
])
def test_the_example_adapter_writes_the_native_config(tmp_path, ports, monkeypatch):
    from EDA.flow import build_openram_config
    adapter = _load_adapter(tmp_path, ports)
    ours = adapter.build_config("scn4m_subm", "sram")
    native = build_openram_config(SimpleNamespace(**ports), tech="scn4m_subm",
                                  output_name="sram", out_dir="/x/")
    assert _body(ours) == _body(native)
    for k in ("GRAFUX_IN", "GRAFUX_OUT", "GRAFUX_WORK"):
        monkeypatch.delenv(k, raising=False)
