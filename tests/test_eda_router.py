"""
test_eda_router.py
Contract tests for the three EDA REST surfaces, driven through FastAPI's
TestClient so the routing and request/response wiring is exercised for real.

These complement ``test_eda_runtime.py`` (which tests the orchestration logic
directly): the bugs this file exists to catch are the ones that live in the layer
between HTTP and the runtime — a body parsed as a query parameter, a literal path
swallowed by a path parameter, a kind reachable through the wrong prefix.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from EDA.models import DEFAULT_IMAGE, DEFAULT_VERIFY_IMAGE, EDA_KINDS  # noqa: E402
from EDA.registry import EdaRecord, registry  # noqa: E402
from EDA.router_base import _coerce_kind  # noqa: E402
from EDA.models import EdaSpec  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """A TestClient over the whole devices app, with no RunPod key configured."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    from device.app import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_registry():
    for summary in list(registry.list()):
        registry.delete(summary.eda_id)
    yield
    for summary in list(registry.list()):
        registry.delete(summary.eda_id)


# ---------------------------------------------------------------------------
# The run endpoint's request body
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", EDA_KINDS)
def test_run_accepts_a_json_body(client, kind):
    """
    Regression test for a silent, total breakage of /run.

    ``make_router`` is handed a different Pydantic model per kind, so the run
    endpoint's body annotation is a runtime value. Under
    ``from __future__ import annotations`` that annotation stays the *string*
    "run_request_model", FastAPI cannot resolve it to a model, and it quietly
    demotes the request body to a required QUERY parameter — so every run request
    failed with 422 "Field required: query.body" while every other endpoint kept
    working. The run endpoint is what the entire client polling flow depends on,
    so this asserts a JSON body is accepted and reaches the runtime.
    """
    resp = client.post(f"/{kind}/does-not-exist/run", json={"top": "counter"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # It reached the runtime, which reports an unknown id as a result rather than
    # an HTTP error — the 422 above would never have got this far.
    assert body["status"] == "error"
    assert "does-not-exist" in body["errors"]


@pytest.mark.parametrize("kind", EDA_KINDS)
def test_run_body_fields_are_parsed_not_ignored(client, kind):
    """A populated body must validate against that kind's own model."""
    payloads = {
        "verilator": {"rtl": "module m(); endmodule", "mode": "lint", "trace": "0"},
        "yosys": {"rtl": "module m(); endmodule", "top": "m", "pdk": "sky130hd"},
        "openroad": {"netlist": "module m(); endmodule", "top": "m",
                     "clock_period": "5", "from_stage": "floorplan"},
    }
    resp = client.post(f"/{kind}/does-not-exist/run", json=payloads[kind])
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == kind


def test_run_accepts_the_cocotb_body_fields(client):
    """
    The cocotb inputs go through the same runtime-annotation path as everything
    else on /run, so a new field that FastAPI cannot resolve would demote the
    whole body to a query parameter and 422 every run.
    """
    resp = client.post("/verilator/does-not-exist/run", json={
        "rtl": "module m(); endmodule", "testbench": "import cocotb",
        "mode": "cocotb", "simulator": "icarus", "tests": "test_a",
        "seed": "42", "coverage": "1", "sva": "bind m chk c(.*);",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "verilator"


# ---------------------------------------------------------------------------
# Per-kind image and disk
# ---------------------------------------------------------------------------

def test_a_verilator_block_gets_the_light_verify_image():
    """
    A simulation takes seconds; pulling the multi-gigabyte ORFS image to run it is
    most of the wall clock and most of the cost, and the fix loop pays that on
    every iteration.
    """
    spec = _coerce_kind(EdaSpec(), "verilator")
    assert spec.kind == "verilator"
    assert spec.image == DEFAULT_VERIFY_IMAGE
    assert spec.container_disk_gb == 20


@pytest.mark.parametrize("kind", ["yosys", "openroad"])
def test_synthesis_kinds_keep_the_pdk_image(kind):
    spec = _coerce_kind(EdaSpec(), kind)
    assert spec.image == DEFAULT_IMAGE
    assert spec.container_disk_gb == 60


def test_an_explicitly_pinned_image_is_never_replaced():
    """An image on the block's port is the user pinning a toolchain."""
    spec = _coerce_kind(EdaSpec(image="ghcr.io/me/my-eda:tag",
                                container_disk_gb=100), "verilator")
    assert spec.image == "ghcr.io/me/my-eda:tag"
    assert spec.container_disk_gb == 100


# ---------------------------------------------------------------------------
# Route ordering: literal paths must not be swallowed by /{eda_id}
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", EDA_KINDS)
def test_pdks_endpoint_is_not_captured_by_the_id_route(client, kind):
    """"/pdks" must resolve as a literal, not as an eda_id called "pdks"."""
    resp = client.get(f"/{kind}/pdks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default"] == "sky130hd"
    assert "sky130hd" in body["pdks"]


@pytest.mark.parametrize("kind", EDA_KINDS)
def test_instances_endpoint_is_not_captured_by_the_id_route(client, kind):
    resp = client.get(f"/{kind}/instances")
    assert resp.status_code == 200
    instances = resp.json()["instances"]
    assert instances and all(i["id"] for i in instances)


# ---------------------------------------------------------------------------
# Kind isolation
# ---------------------------------------------------------------------------

def test_a_block_is_not_reachable_through_another_kinds_prefix(client):
    """
    A yosys pod must 404 on /openroad/{id}.

    The three kinds share one registry, so without the kind check a mistyped
    prefix would happily operate on another block type's pod and fail much later
    inside the tool.
    """
    eda_id = registry.create(EdaRecord(
        spec=EdaSpec(kind="yosys"), pod_id="p", public_ip="1.2.3.4",
        ssh_port=22, phase="ready",
    ))
    assert client.get(f"/yosys/{eda_id}").status_code == 200
    assert client.get(f"/openroad/{eda_id}").status_code == 404


def test_list_only_returns_this_kind(client):
    registry.create(EdaRecord(spec=EdaSpec(kind="yosys"), pod_id="p",
                              public_ip="1.2.3.4", ssh_port=22))
    registry.create(EdaRecord(spec=EdaSpec(kind="verilator"), pod_id="p2",
                              public_ip="1.2.3.5", ssh_port=22))
    assert len(client.get("/yosys").json()) == 1
    assert len(client.get("/verilator").json()) == 1
    assert client.get("/openroad").json() == []


def test_create_forces_the_kind_from_the_url(client):
    """
    A block sends its whole config blob and may set the wrong kind (or none).

    Taking the kind from the URL means a yosys block can never provision itself
    as an openroad one and then fail confusingly at run time.
    """
    resp = client.post("/yosys/create", json={"kind": "openroad", "name": "x"})
    assert resp.status_code == 200
    assert resp.json()["kind"] == "yosys"


# ---------------------------------------------------------------------------
# Error shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", EDA_KINDS)
def test_missing_key_is_a_result_not_a_500(client, kind):
    """An operational failure must reach the block's errors port, not blow up."""
    resp = client.post(f"/{kind}/create", json={"name": "x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert "RunPod API key" in body["errors"]


@pytest.mark.parametrize("kind", EDA_KINDS)
def test_status_and_result_404_for_an_unknown_id(client, kind):
    assert client.get(f"/{kind}/nope/status").status_code == 404
    assert client.get(f"/{kind}/nope/result").status_code == 404


@pytest.mark.parametrize("kind", EDA_KINDS)
def test_delete_404s_for_an_unknown_id(client, kind):
    assert client.delete(f"/{kind}/nope").status_code == 404


def test_delete_cancels_and_removes(client):
    eda_id = registry.create(EdaRecord(
        spec=EdaSpec(kind="verilator"), pod_id="p", public_ip="1.2.3.4", ssh_port=22))
    assert client.delete(f"/verilator/{eda_id}").status_code == 200
    assert registry.get(eda_id) is None


# ---------------------------------------------------------------------------
# Coexistence with the other runtimes on the same server
# ---------------------------------------------------------------------------

def test_mounting_eda_does_not_disturb_the_gpu_router(client):
    """The EDA routers are mounted alongside gpu/claw; none may shadow another."""
    assert client.get("/gpu/models").status_code == 200
    assert client.get("/health").status_code == 200
