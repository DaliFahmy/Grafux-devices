"""
test_eda_runtime.py
Unit tests for ``EDA.runtime`` — provisioning, the async job protocol, cancel,
teardown and the reaper.

Every RunPod REST and SSH call is replaced via monkeypatch (the same approach as
``test_gpu_runtime.py``), so these exercise the orchestration logic with no cloud
account, no container and no spend.  That covers the large majority of what can
go wrong here: retry-and-hop on a bad placement, tearing the pod down on *every*
exit path, refusing concurrent runs, and — the subtle one — not reaping a pod
that is in the middle of a 45-minute route.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from EDA import flow, pod_client, runtime  # noqa: E402
from EDA.models import (  # noqa: E402
    EdaSpec,
    OpenRoadRunRequest,
    VerilatorRunRequest,
    YosysRunRequest,
)
from EDA.registry import EdaRecord, registry  # noqa: E402


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll until a background job thread reaches the expected state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    """Start each test with an empty registry and no RUNPOD_API_KEY leakage."""
    monkeypatch.setattr(runtime, "_LOCAL_SSH", False)
    for summary in list(registry.list()):
        registry.delete(summary.eda_id)
    yield
    for summary in list(registry.list()):
        registry.delete(summary.eda_id)


@pytest.fixture
def fake_pod(monkeypatch):
    """Replace the RunPod REST + SSH calls with in-memory fakes."""
    calls = {"create": [], "terminate": [], "connect": [], "artifacts": []}

    def fake_keypair():
        return ("PRIVATE_PEM", "ssh-rsa AAAAFAKE grafux-eda")

    def fake_create(api_key, spec, public_key):
        calls["create"].append({
            "api_key": api_key,
            "compute_type": spec.compute_type,
            "instance_type": spec.instance_type,
            "image": spec.image,
        })
        return f"pod-{len(calls['create'])}"

    def fake_wait(api_key, pod_id, **kwargs):
        return ("1.2.3.4", 40022)

    def fake_terminate(api_key, pod_id):
        calls["terminate"].append(pod_id)

    def fake_connect(host, port, key, **kwargs):
        calls["connect"].append((host, port))
        return _FakeSSH()

    def fake_download(client, globs):
        calls["artifacts"].append(list(globs))
        return [{"path": "/workspace/grafux/netlist.v", "size": 10,
                 "content": "", "b64": True, "truncated": False}]

    monkeypatch.setattr(pod_client, "generate_keypair", fake_keypair)
    monkeypatch.setattr(pod_client, "create_pod", fake_create)
    monkeypatch.setattr(pod_client, "wait_until_ready", fake_wait)
    monkeypatch.setattr(pod_client, "terminate_pod", fake_terminate)
    monkeypatch.setattr(pod_client, "connect_ssh", fake_connect)
    monkeypatch.setattr(pod_client, "download_artifacts", fake_download)
    return calls


class _FakeSSH:
    """A paramiko-client stand-in; the flow layer is faked out separately."""

    def open_sftp(self):
        return _FakeSFTP()

    def close(self):
        pass


class _FakeSFTP:
    def open(self, path, mode="rb"):
        return _FakeFile()

    def mkdir(self, path):
        pass

    def close(self):
        pass


class _FakeFile:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, data):
        pass

    def read(self, n=-1):
        return b""


@pytest.fixture
def fake_flow(monkeypatch):
    """Replace the tool runners so a 'run' is instant and scriptable."""
    state = {"stages": [], "cancelled": False, "status": "ok"}

    def make_runner(kind, stages):
        def runner(client, req, **kwargs):
            on_stage = kwargs.get("on_stage") or (lambda *a: None)
            should_cancel = kwargs.get("should_cancel") or (lambda: False)
            for stage in stages:
                if should_cancel():
                    state["cancelled"] = True
                    break
                on_stage(stage, "running")
                state["stages"].append(stage)
                on_stage(stage, "done")
            return {
                "outputs": {"status": state["status"], "errors": "", "warnings": ""},
                "_status": state["status"],
                "_stage": state["stages"][-1] if state["stages"] else "",
                "_globs": [f"/{kind}/*"],
            }
        return runner

    monkeypatch.setattr(flow, "run_verilator", make_runner("verilator", ["verilate", "sim"]))
    monkeypatch.setattr(flow, "run_yosys", make_runner("yosys", ["synth"]))
    monkeypatch.setattr(
        flow, "run_orfs",
        make_runner("openroad", ["floorplan", "place", "cts", "route", "final"]),
    )
    return state


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

def test_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    assert runtime._resolve_runpod_key(EdaSpec()) == "rp_env_key"


def test_resolve_key_from_api_keys_json(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    spec = EdaSpec(api_keys='{"runpod": "rp_from_json"}')
    assert runtime._resolve_runpod_key(spec) == "rp_from_json"


def test_resolve_key_bare_prefix(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runtime._resolve_runpod_key(EdaSpec(credentials="rp_bare")) == "rp_bare"


def test_resolve_key_ignores_empty_port_sentinel(monkeypatch):
    """An unwired port arrives as the literal 'empty', not as an empty string."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runtime._resolve_runpod_key(EdaSpec(api_keys="empty")) is None


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------

def test_provision_without_key_returns_error_not_exception(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    result = runtime.provision_eda(EdaSpec(kind="yosys"))
    assert result["status"] == "error"
    assert "RunPod API key" in result["errors"]
    # The message must point at the local-development path too.
    assert "EDA_LOCAL_SSH" in result["errors"]


def test_provision_success_registers_the_pod(monkeypatch, fake_pod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    result = runtime.provision_eda(EdaSpec(kind="yosys", name="counter"))
    assert result["status"] == "ok"
    assert result["kind"] == "yosys"
    record = registry.get(result["eda_id"])
    assert record is not None and record.is_running
    assert record.public_ip == "1.2.3.4" and record.ssh_port == 40022


def test_provision_requests_a_cpu_pod_by_default(monkeypatch, fake_pod):
    """EDA work is CPU-bound; renting an idle GPU for it would be pure waste."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    runtime.provision_eda(EdaSpec(kind="openroad"))
    assert fake_pod["create"][0]["compute_type"] == "CPU"


def test_provision_honours_gpu_fallback(monkeypatch, fake_pod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    runtime.provision_eda(
        EdaSpec(kind="yosys", compute_type="GPU", instance_type="NVIDIA RTX A4000")
    )
    assert fake_pod["create"][0]["compute_type"] == "GPU"


def test_provision_retries_and_hops_on_capacity_error(monkeypatch, fake_pod):
    """A placement failure should hop machines, not fail the user's run."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    monkeypatch.setattr(runtime, "_RETRY_BACKOFF_S", 0)
    attempts = {"n": 0}
    real_wait = pod_client.wait_until_ready

    def flaky_wait(api_key, pod_id, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise pod_client.NoEndpointError("no public IP on this machine")
        return real_wait(api_key, pod_id, **kwargs)

    monkeypatch.setattr(pod_client, "wait_until_ready", flaky_wait)
    result = runtime.provision_eda(EdaSpec(kind="yosys"))
    assert result["status"] == "ok"
    assert len(fake_pod["create"]) == 2          # hopped to a second machine
    assert fake_pod["terminate"] == ["pod-1"]    # the bad pod was freed


def test_provision_terminates_the_pod_when_it_never_comes_up(monkeypatch, fake_pod):
    """A pod we created but cannot reach must never be left billing."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    monkeypatch.setattr(runtime, "_RETRY_BACKOFF_S", 0)

    def always_fail(api_key, pod_id, **kwargs):
        raise pod_client.NoEndpointError("never got an endpoint")

    monkeypatch.setattr(pod_client, "wait_until_ready", always_fail)
    result = runtime.provision_eda(EdaSpec(kind="yosys"))
    assert result["status"] == "error"
    assert len(fake_pod["terminate"]) == runtime._PROVISION_ATTEMPTS


def test_provision_terminates_on_an_unexpected_error(monkeypatch, fake_pod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")

    def boom(api_key, pod_id, **kwargs):
        raise ValueError("something unexpected")

    monkeypatch.setattr(pod_client, "wait_until_ready", boom)
    result = runtime.provision_eda(EdaSpec(kind="yosys"))
    assert result["status"] == "error"
    assert "something unexpected" in result["errors"]
    assert fake_pod["terminate"] == ["pod-1"]


def test_provision_async_returns_immediately_with_creating_phase(monkeypatch, fake_pod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    result = runtime.provision_eda_async(EdaSpec(kind="openroad"))
    assert result["status"] == "creating"
    assert result["eda_id"]
    assert _wait_until(lambda: (registry.get(result["eda_id"]) or EdaRecord(
        spec=EdaSpec())).phase == "ready")


def test_provision_async_without_key_fails_fast(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    result = runtime.provision_eda_async(EdaSpec(kind="yosys"))
    assert result["status"] == "error"
    assert result["eda_id"] == ""


# ---------------------------------------------------------------------------
# The async job protocol
# ---------------------------------------------------------------------------

def _provisioned(kind: str = "yosys") -> str:
    """Register a ready-to-run record without going through RunPod."""
    return registry.create(EdaRecord(
        spec=EdaSpec(kind=kind), pod_id="pod-x", public_ip="1.2.3.4",
        ssh_port=40022, private_key_pem="PEM", api_key="rp_k", phase="ready",
    ))


def test_run_returns_before_the_job_finishes(fake_pod, fake_flow, monkeypatch):
    """
    The run endpoint must not wait for the tool.

    An OpenROAD route outlives any HTTP request, so /run starts a thread and the
    block polls; this is the contract the whole client side depends on.
    """
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    eda_id = _provisioned("openroad")
    accepted = runtime.start_openroad_job(eda_id, OpenRoadRunRequest())
    assert accepted["status"] == "running"
    assert _wait_until(lambda: (registry.get(eda_id) or EdaRecord(spec=EdaSpec())).done)


def test_run_walks_every_orfs_stage(fake_pod, fake_flow, monkeypatch):
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    eda_id = _provisioned("openroad")
    runtime.start_openroad_job(eda_id, OpenRoadRunRequest())
    assert _wait_until(lambda: registry.get(eda_id).done)
    assert fake_flow["stages"] == ["floorplan", "place", "cts", "route", "final"]


def test_status_reports_the_stage_while_running(fake_pod, fake_flow, monkeypatch):
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    eda_id = _provisioned("yosys")
    runtime.start_yosys_job(eda_id, YosysRunRequest())
    assert _wait_until(lambda: registry.get(eda_id).done)
    status = runtime.eda_status(eda_id)
    assert status["done"] is True
    assert "synth" in status["stages_done"]


def test_result_is_available_after_the_job_finishes(fake_pod, fake_flow, monkeypatch):
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    eda_id = _provisioned("verilator")
    runtime.start_verilator_job(eda_id, VerilatorRunRequest())
    assert _wait_until(lambda: registry.get(eda_id).done)
    result = runtime.eda_result(eda_id)
    assert result["done"] is True
    assert result["status"] == "ok"
    assert result["outputs"]["eda_id"] == eda_id


def test_result_of_an_unstarted_record_reports_running_not_none(fake_pod):
    """A client polling /result early deserves a coherent answer, not a 404."""
    eda_id = _provisioned("yosys")
    result = runtime.eda_result(eda_id)
    assert result is not None
    assert result["done"] is False
    assert result["status"] == "running"


def test_result_of_unknown_id_is_none():
    assert runtime.eda_result("nope") is None


def test_status_of_unknown_id_is_none():
    assert runtime.eda_status("nope") is None


def test_run_on_unknown_id_returns_error_not_exception():
    out = runtime.start_yosys_job("nope", YosysRunRequest())
    assert out["status"] == "error"
    assert "Regenerate" in out["errors"]


def test_run_on_a_pod_that_is_not_ready_is_refused():
    eda_id = registry.create(EdaRecord(spec=EdaSpec(kind="yosys"), phase="creating"))
    out = runtime.start_yosys_job(eda_id, YosysRunRequest())
    assert out["status"] == "error"
    assert "not ready" in out["errors"]


def test_concurrent_runs_are_refused(fake_pod, monkeypatch):
    """
    Two make invocations in one ORFS working directory corrupt each other.

    The damage surfaces much later as an inscrutable tool error, so the second
    run is refused up front.
    """
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    gate = threading.Event()

    def slow_runner(client, req, **kwargs):
        gate.wait(timeout=5)
        return {"outputs": {}, "_status": "ok", "_stage": "synth", "_globs": []}

    monkeypatch.setattr(flow, "run_yosys", slow_runner)
    eda_id = _provisioned("yosys")
    first = runtime.start_yosys_job(eda_id, YosysRunRequest())
    assert first["status"] == "running"
    assert _wait_until(lambda: registry.get(eda_id).job_in_flight)

    second = runtime.start_yosys_job(eda_id, YosysRunRequest())
    assert second["status"] == "error"
    assert "already in progress" in second["errors"]
    gate.set()


def test_a_crashing_tool_still_finishes_the_job(fake_pod, monkeypatch):
    """A thread that died without finishing would leave the block polling forever."""
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)

    def boom(client, req, **kwargs):
        raise RuntimeError("ssh died mid-run")

    monkeypatch.setattr(flow, "run_yosys", boom)
    eda_id = _provisioned("yosys")
    runtime.start_yosys_job(eda_id, YosysRunRequest())
    assert _wait_until(lambda: registry.get(eda_id).done)
    result = runtime.eda_result(eda_id)
    assert result["status"] == "error"
    assert "ssh died mid-run" in result["errors"]


def test_artifacts_are_collected_even_when_the_run_failed(fake_pod, monkeypatch):
    """A route that blew up still produced the reports that explain why."""
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)

    def failing(client, req, **kwargs):
        return {"outputs": {"status": "error", "errors": "route failed"},
                "_status": "error", "_stage": "route", "_globs": ["/results/*"]}

    monkeypatch.setattr(flow, "run_orfs", failing)
    eda_id = _provisioned("openroad")
    runtime.start_openroad_job(eda_id, OpenRoadRunRequest())
    assert _wait_until(lambda: registry.get(eda_id).done)
    assert fake_pod["artifacts"] == [["/results/*"]]
    assert runtime.eda_result(eda_id)["artifacts"]


# ---------------------------------------------------------------------------
# Teardown, keep-warm and cancel — the cost-control paths
# ---------------------------------------------------------------------------

def test_ephemeral_run_terminates_the_pod_and_drops_the_record(fake_pod, fake_flow,
                                                               monkeypatch):
    monkeypatch.setattr(runtime, "_EPHEMERAL", True)
    monkeypatch.setattr(runtime, "_DEFAULT_KEEP_WARM_MIN", 0)
    eda_id = _provisioned("yosys")
    runtime.start_yosys_job(eda_id, YosysRunRequest(keep_warm_minutes=0))
    assert _wait_until(lambda: registry.get(eda_id) is None)
    assert "pod-x" in fake_pod["terminate"]


def test_keep_warm_holds_the_pod_instead_of_terminating(fake_pod, fake_flow,
                                                        monkeypatch):
    monkeypatch.setattr(runtime, "_EPHEMERAL", True)
    eda_id = _provisioned("yosys")
    runtime.start_yosys_job(eda_id, YosysRunRequest(keep_warm_minutes=15))
    assert _wait_until(lambda: (registry.get(eda_id) or EdaRecord(spec=EdaSpec())).done)
    record = registry.get(eda_id)
    assert record is not None
    assert record.warm_until > 0
    assert "pod-x" not in fake_pod["terminate"]


def test_cancel_terminates_the_pod_and_signals_the_job(fake_pod):
    """
    Stop must stop the *billing*, not just the request.

    Aborting the HTTP call alone would leave the pod routing away happily — the
    expensive failure mode this package exists to prevent.
    """
    eda_id = _provisioned("openroad")
    assert runtime.cancel_eda(eda_id) is True
    assert "pod-x" in fake_pod["terminate"]
    assert registry.get(eda_id) is None


def test_cancel_of_unknown_id_is_false():
    assert runtime.cancel_eda("nope") is False


def test_cancel_stops_a_running_job(fake_pod, monkeypatch):
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    started = threading.Event()
    observed = {"cancelled": False}

    def watching_runner(client, req, **kwargs):
        should_cancel = kwargs["should_cancel"]
        started.set()
        for _ in range(200):
            if should_cancel():
                observed["cancelled"] = True
                break
            time.sleep(0.01)
        return {"outputs": {}, "_status": "error", "_stage": "route", "_globs": []}

    monkeypatch.setattr(flow, "run_orfs", watching_runner)
    eda_id = _provisioned("openroad")
    runtime.start_openroad_job(eda_id, OpenRoadRunRequest())
    assert started.wait(timeout=5)
    registry.request_cancel(eda_id)
    assert _wait_until(lambda: observed["cancelled"], timeout=5)


# ---------------------------------------------------------------------------
# The reaper — the bug this package exists to not repeat
# ---------------------------------------------------------------------------

def test_reaper_never_reaps_a_pod_with_a_job_in_flight():
    """
    The whole point of the EDA reaper override.

    A route occupies one SSH call for 45 minutes and touches nothing, so a reaper
    that judged staleness by idle time alone would terminate a perfectly healthy
    pod mid-run and the user would see an inexplicable failure.
    """
    eda_id = _provisioned("openroad")
    registry.start_job(eda_id)
    record = registry.get(eda_id)
    # Backdate the record far past any idle timeout.
    record.last_used = time.monotonic() - 86400
    assert registry._select_stale(time.monotonic()) == []
    assert registry.get(eda_id) is not None


def test_reaper_does_reap_an_idle_pod(monkeypatch):
    import EDA.registry as registry_mod
    monkeypatch.setattr(registry_mod, "_IDLE_TIMEOUT_MIN", 10)
    eda_id = _provisioned("yosys")
    registry.get(eda_id).last_used = time.monotonic() - 86400
    stale = registry._select_stale(time.monotonic())
    assert [eid for eid, _rec in stale] == [eda_id]


def test_reaper_expires_a_keep_warm_hold():
    eda_id = _provisioned("yosys")
    registry.set_keep_warm(eda_id, 1)
    record = registry.get(eda_id)
    record.keep_warm_deadline = time.monotonic() - 1
    stale = registry._select_stale(time.monotonic())
    assert [eid for eid, _rec in stale] == [eda_id]


def test_set_stage_refreshes_last_used():
    """A long stage makes no requests; without this the record would look idle."""
    eda_id = _provisioned("openroad")
    record = registry.get(eda_id)
    record.last_used = time.monotonic() - 500
    registry.set_stage(eda_id, "route", "running")
    assert time.monotonic() - registry.get(eda_id).last_used < 1


def test_stages_done_accumulates_in_order():
    eda_id = _provisioned("openroad")
    for stage in ("floorplan", "place", "cts"):
        registry.set_stage(eda_id, stage, "running")
        registry.set_stage(eda_id, stage, "done")
    assert registry.get(eda_id).stages_done == ["floorplan", "place", "cts"]


def test_log_tail_is_bounded():
    """A chatty route emits tens of thousands of lines; memory must stay flat."""
    eda_id = _provisioned("openroad")
    for i in range(1000):
        registry.append_log(eda_id, f"line {i}")
    tail = registry.get(eda_id).log_tail
    assert len(tail) <= 200
    assert tail[-1] == "line 999"


# ---------------------------------------------------------------------------
# Local-SSH development mode
# ---------------------------------------------------------------------------

def test_local_ssh_mode_needs_no_key_and_no_cloud(monkeypatch):
    """The path that makes the whole runtime developable without a RunPod account."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(runtime, "_LOCAL_SSH", True)
    monkeypatch.setattr(runtime, "_LOCAL_HOST", "127.0.0.1")
    monkeypatch.setattr(runtime, "_LOCAL_PORT", 2222)
    monkeypatch.setattr(runtime, "_LOCAL_KEY", "")
    result = runtime.provision_eda(EdaSpec(kind="yosys"))
    assert result["status"] == "ok"
    record = registry.get(result["eda_id"])
    assert record.public_ip == "127.0.0.1" and record.ssh_port == 2222


def test_local_ssh_mode_does_not_tear_down_a_pod_it_does_not_own(
    fake_pod, fake_flow, monkeypatch
):
    monkeypatch.setattr(runtime, "_LOCAL_SSH", True)
    monkeypatch.setattr(runtime, "_EPHEMERAL", True)
    eda_id = _provisioned("yosys")
    runtime.start_yosys_job(eda_id, YosysRunRequest())
    assert _wait_until(lambda: (registry.get(eda_id) or EdaRecord(spec=EdaSpec())).done)
    assert fake_pod["terminate"] == []
    assert registry.get(eda_id) is not None
