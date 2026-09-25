"""
test_cpu_router.py
Contract tests for the /cpu REST surface, driven through FastAPI's TestClient so
the routing and request/response wiring is exercised for real.

These complement ``test_cpu_flow.py`` (which tests the run logic directly): the
bugs this file exists to catch live in the layer between HTTP and the runtime —
a body parsed as a query parameter, a literal path swallowed by a path
parameter, one kind reachable through another's prefix — plus the one risk
specific to this package, which is that borrowing EDA's plumbing disturbs the
six block types already using it.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from CPU.models import (  # noqa: E402
    CPU_DISK_GB,
    DEFAULT_CPU_IMAGE,
    LANGUAGE_CHOICES,
)
from EDA.models import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_VERIFY_IMAGE,
    EDA_KINDS,
    EdaSpec,
    disk_for_kind,
    image_for_kind,
)
from EDA.registry import EdaRecord, registry  # noqa: E402
from EDA.router_base import _coerce_kind  # noqa: E402


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


def _register(kind: str) -> str:
    """Put a fake provisioned pod of some kind in the registry."""
    return registry.create(EdaRecord(spec=EdaSpec(kind=kind), pod_id="pod-1",
                                     public_ip="1.2.3.4", ssh_port=22))


# ---------------------------------------------------------------------------
# The run endpoint's request body
# ---------------------------------------------------------------------------

def test_run_accepts_a_json_body(client):
    """
    Regression test for a silent, total breakage of /run.

    ``make_router`` is handed a different Pydantic model per kind, so the run
    endpoint's body annotation is a runtime value. Under
    ``from __future__ import annotations`` in router_base that annotation would
    stay the *string* "run_request_model", FastAPI could not resolve it to a
    model, and it would quietly demote the request body to a required QUERY
    parameter — every run request failing with 422 "Field required: query.body"
    while every other endpoint kept working.

    CPU/router.py DOES carry the future import, which is safe because nothing in
    it has that shape; this test is what proves that distinction holds.
    """
    resp = client.post("/cpu/does-not-exist/run", json={"code": "int main(){}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # It reached the runtime, which reports an unknown id as a result rather than
    # an HTTP error — a 422 would never have got this far.
    assert body["status"] == "error"
    assert "does-not-exist" in body["errors"]


def test_run_body_fields_are_parsed_not_ignored(client):
    """Every field a cpu block actually sends must validate against the model."""
    resp = client.post("/cpu/does-not-exist/run", json={
        "code": "int main(){return 0;}", "language": "c",
        "spec": "1. the L1 writes back on eviction",
        "args": "--iters 10", "build_flags": "-O2 -lm",
        "defines": "WIDTH=8", "include_dirs": "/inc",
        "repetitions": "10", "warmup": "2", "timeout": 600,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "cpu"


@pytest.mark.parametrize("language", LANGUAGE_CHOICES)
def test_every_supported_language_is_accepted_by_the_model(client, language):
    resp = client.post("/cpu/does-not-exist/run",
                       json={"code": "x", "language": language})
    assert resp.status_code == 200, resp.text


def test_an_unwired_block_still_validates(client):
    """Every field is optional so a partially-wired block reaches the runtime."""
    resp = client.post("/cpu/does-not-exist/run", json={})
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Route ordering and kind scoping
# ---------------------------------------------------------------------------

def test_instances_is_not_swallowed_by_the_id_path(client):
    """
    FastAPI matches in declaration order, so a literal path declared after
    ``/{eda_id}`` is unreachable — it would 404 as "no cpu with id 'instances'".
    """
    resp = client.get("/cpu/instances")
    assert resp.status_code == 200, resp.text
    assert "instances" in resp.json()


def test_cpu_instances_offer_every_runpod_cpu_flavour(client):
    """
    For a benchmark the machine is the experiment, so the cpu dropdown spans every
    flavour family RunPod accepts -- not just EDA's compute-first shortlist -- and
    still defaults to the size cpu blocks have always run on.
    """
    from CPU.models import CPU_INSTANCES, DEFAULT_CPU_INSTANCE
    from EDA import pod_client

    ids = [i["id"] for i in client.get("/cpu/instances").json()["instances"]]
    assert ids == [i["id"] for i in CPU_INSTANCES]
    assert DEFAULT_CPU_INSTANCE in ids
    families = {pod_client.split_instance_type(i)[0] for i in ids}
    assert families == set(pod_client.CPU_FLAVOR_FAMILIES)
    for inst_id in ids:
        # Only an id whose family is real: split_instance_type would otherwise
        # quietly rent a different machine than the dropdown said.
        assert inst_id.split("-")[0] in pod_client.CPU_FLAVOR_FAMILIES, inst_id
        assert pod_client.price_for(inst_id) > 0, inst_id


@pytest.mark.parametrize("kind", ["verilator", "yosys", "openroad", "openram"])
def test_the_cpu_machine_list_did_not_leak_into_eda_kinds(client, kind):
    from EDA import pod_client

    ids = [i["id"] for i in client.get(f"/{kind}/instances").json()["instances"]]
    assert ids == [i["id"] for i in pod_client.EDA_INSTANCES]


def test_starting_a_run_notes_a_substituted_machine(monkeypatch):
    from CPU import runtime as cpu_runtime
    from CPU.models import CpuRunRequest

    eda_id = registry.create(EdaRecord(
        spec=EdaSpec(kind="cpu", instance_type="xeon-8"), pod_id="pod-1",
        public_ip="1.2.3.4", ssh_port=22))
    seen = {}
    monkeypatch.setattr(cpu_runtime.eda_runtime, "start_job",
                        lambda i, req, kind: seen.setdefault("note", req._machine_note))
    cpu_runtime.start_cpu_job(eda_id, CpuRunRequest(code="x"))
    assert "xeon-8" in seen["note"] and "cpu3c-8" in seen["note"]


def test_pdks_answers_empty_rather_than_offering_an_orfs_platform(client):
    """
    No PDK applies to running a program on a CPU. The endpoint exists because it
    is part of the shared surface, so it must answer honestly rather than offer
    "sky130hd" to a block whose toolchain has never heard of it.
    """
    resp = client.get("/cpu/pdks")
    assert resp.status_code == 200, resp.text
    assert resp.json()["pdks"] == []


def test_the_list_endpoint_is_scoped_to_cpu(client):
    """A verilator pod must not appear in a cpu block's list, and vice versa."""
    cpu_id = _register("cpu")
    _register("verilator")
    listed = {row["eda_id"] for row in client.get("/cpu").json()}
    assert listed == {cpu_id}


def test_fetching_another_kinds_id_through_cpu_is_a_404(client):
    """
    Separate prefixes exist so a wrong-kind request fails at the router rather
    than deep inside a tool that was handed something it cannot run.
    """
    verilator_id = _register("verilator")
    assert client.get(f"/cpu/{verilator_id}").status_code == 404


def test_deleting_an_unknown_id_is_a_404(client):
    assert client.delete("/cpu/does-not-exist").status_code == 404


def test_status_for_an_unknown_id_is_a_404(client):
    assert client.get("/cpu/does-not-exist/status").status_code == 404


# ---------------------------------------------------------------------------
# The kind's image and disk
# ---------------------------------------------------------------------------

def test_a_cpu_block_gets_the_post_silicon_image():
    """
    Registered from CPU/models.py rather than hard-coded in EDA's maps, so this
    also proves the self-registration actually ran.
    """
    spec = _coerce_kind(EdaSpec(), "cpu")
    assert spec.kind == "cpu"
    assert spec.image == DEFAULT_CPU_IMAGE
    assert spec.container_disk_gb == CPU_DISK_GB


def test_a_pinned_image_is_never_silently_replaced():
    """
    An explicit value on the `image` port is the user pinning a toolchain.
    Overwriting it would be the most confusing bug this router could have.
    """
    spec = _coerce_kind(EdaSpec(image="ghcr.io/me/my-own:v1"), "cpu")
    assert spec.image == "ghcr.io/me/my-own:v1"


def test_registering_cpu_did_not_change_any_eda_kinds_image():
    """
    CPU writes into EDA's own image/disk maps. If that write were not confined to
    its own key, another kind would start provisioning the wrong container — a
    failure that would surface far from this package.
    """
    assert image_for_kind("verilator") == DEFAULT_VERIFY_IMAGE
    assert image_for_kind("openroad") == DEFAULT_IMAGE
    assert disk_for_kind("verilator") == 20
    assert disk_for_kind("openroad") == 60


def test_cpu_is_not_an_eda_kind():
    """
    It borrows EDA's plumbing but is a separate block type. Anything iterating
    EDA_KINDS (per-kind test parametrisation, dropdowns) must not pick it up.
    """
    assert "cpu" not in EDA_KINDS


# ---------------------------------------------------------------------------
# Mounting
# ---------------------------------------------------------------------------

def test_mounting_cpu_does_not_disturb_the_gpu_router(client):
    """The gpu block's own surface must be untouched by this package existing."""
    assert client.get("/gpu/models").status_code == 200


@pytest.mark.parametrize("kind", EDA_KINDS)
def test_mounting_cpu_does_not_disturb_the_eda_routers(client, kind):
    resp = client.post(f"/{kind}/does-not-exist/run", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == kind


def test_the_cpu_runner_is_registered_with_eda():
    """
    Without this registration a cpu run would fall through EDA's dispatch chain to
    the ORFS runner and try to place-and-route a C program — the single worst
    failure mode available to this package, and a silent one.
    """
    from CPU import runtime as cpu_runtime  # noqa: F401 — import performs it
    from CPU.flow import run_cpu
    from EDA.runtime import _RUNNERS

    assert _RUNNERS.get("cpu") is run_cpu
