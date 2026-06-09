"""
test_gpu_runtime.py
Unit tests for the GPU (cloud-GPU) runtime.

No real RunPod account, network, or SSH is used: every ``runpod_client`` call is
replaced via monkeypatch, so the tests exercise the orchestration logic in
``GPU.runtime`` (key resolution, provision/run/teardown, result mapping, error
paths) in isolation.
"""

import json

import pytest

from GPU import runtime, runpod_client
from GPU.models import GpuRunRequest, GpuSpec
from GPU.registry import registry
from GPU.runpod_client import _ssh_endpoint


@pytest.fixture(autouse=True)
def _clean_registry():
    """Start each test with an empty registry and no RUNPOD_API_KEY leakage."""
    for summary in list(registry.list()):
        registry.delete(summary.gpu_id)
    yield
    for summary in list(registry.list()):
        registry.delete(summary.gpu_id)


@pytest.fixture
def fake_runpod(monkeypatch):
    """Replace the RunPod REST + SSH calls with in-memory fakes."""
    calls = {"create": [], "terminate": [], "run": []}

    def fake_keypair():
        return ("PRIVATE_PEM", "ssh-rsa AAAAFAKE grafux-gpu")

    def fake_create(api_key, spec, public_key):
        calls["create"].append((api_key, spec.gpu_model, public_key))
        return "pod-123"

    def fake_wait(api_key, pod_id, **kwargs):
        return ("1.2.3.4", 40022)

    def fake_terminate(api_key, pod_id):
        calls["terminate"].append((api_key, pod_id))

    def fake_run_remote(host, port, key, **kwargs):
        calls["run"].append({"host": host, "port": port, **kwargs})
        return {
            "status": "ok",
            "response": "hello from gpu\n",
            "errors": "",
            "warnings": "ptxas info: used 12 registers",
            "benchmark": {
                "compile_ms": 800,
                "exec_ms": 12,
                "exit_code": 0,
                "gpu_model": kwargs.get("gpu_model", ""),
                "gpu_info": "NVIDIA GeForce RTX 4090, 512 MiB, 24564 MiB",
            },
        }

    monkeypatch.setattr(runpod_client, "generate_keypair", fake_keypair)
    monkeypatch.setattr(runpod_client, "create_pod", fake_create)
    monkeypatch.setattr(runpod_client, "wait_until_ready", fake_wait)
    monkeypatch.setattr(runpod_client, "terminate_pod", fake_terminate)
    monkeypatch.setattr(runpod_client, "run_remote", fake_run_remote)
    return calls


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

def test_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    assert runtime._resolve_runpod_key(GpuSpec()) == "rp_env_key"


def test_resolve_key_from_api_keys_json(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    spec = GpuSpec(api_keys=json.dumps({"runpod": "rp_json_key"}))
    assert runtime._resolve_runpod_key(spec) == "rp_json_key"


def test_resolve_key_bare_prefix(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runtime._resolve_runpod_key(GpuSpec(credentials="rp_bare_key")) == "rp_bare_key"


def test_resolve_key_placeholder_ignored(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    # "unconnected"/"empty" sentinels must never be treated as a real key.
    assert runtime._resolve_runpod_key(GpuSpec(api_keys="unconnected")) is None


# ---------------------------------------------------------------------------
# Provision (Regenerate)
# ---------------------------------------------------------------------------

def test_provision_no_key_errors(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    result = runtime.provision_gpu(GpuSpec())
    assert result["status"] == "error"
    assert "RunPod API key" in result["errors"]
    assert result["gpu_id"] == ""


def test_provision_success_caches_pod(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    spec = GpuSpec(gpu_model="NVIDIA GeForce RTX 4090", name="bench")
    result = runtime.provision_gpu(spec)

    assert result["status"] == "ok"
    assert result["pod_id"] == "pod-123"
    gpu_id = result["gpu_id"]
    assert gpu_id

    record = registry.get(gpu_id)
    assert record is not None
    assert record.is_running
    assert record.public_ip == "1.2.3.4"
    assert record.ssh_port == 40022
    assert record.private_key_pem == "PRIVATE_PEM"
    # The injected public key flows through to create_pod.
    assert fake_runpod["create"][0][2].startswith("ssh-rsa")


def test_provision_failure_terminates_pod(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")

    def boom(api_key, pod_id, **kwargs):
        raise RuntimeError("pod never became ready")

    monkeypatch.setattr(runpod_client, "wait_until_ready", boom)

    result = runtime.provision_gpu(GpuSpec())
    assert result["status"] == "error"
    assert "never became ready" in result["errors"]
    # The half-provisioned pod must be terminated so it does not bill.
    assert fake_runpod["terminate"] == [("rp_env_key", "pod-123")]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def test_run_maps_result(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec(gpu_model="NVIDIA GeForce RTX 4090"))["gpu_id"]

    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}", language="cuda"))
    assert out["status"] == "ok"
    assert out["response"] == "hello from gpu\n"
    assert "ptxas" in out["warnings"]

    bench = json.loads(out["benchmark"])
    assert bench["exec_ms"] == 12
    assert bench["exit_code"] == 0
    assert bench["gpu_model"] == "NVIDIA GeForce RTX 4090"

    # The source + language reached run_remote.
    assert fake_runpod["run"][0]["source"] == "int main(){}"
    assert fake_runpod["run"][0]["language"] == "cuda"


def test_run_unknown_id_errors():
    out = runtime.run_gpu("does-not-exist", GpuRunRequest(code="x"))
    assert out["status"] == "error"
    assert "No GPU with id" in out["errors"]


def test_run_empty_code_errors(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="   "))
    assert out["status"] == "error"
    assert "code" in out["errors"].lower()


def test_run_remote_failure_is_reported(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    def boom(*a, **k):
        raise OSError("ssh connection refused")

    monkeypatch.setattr(runpod_client, "run_remote", boom)
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}"))
    assert out["status"] == "error"
    assert "connection refused" in out["errors"]


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def test_terminate_removes_and_terminates(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    assert runtime.terminate_gpu(gpu_id) is True
    assert registry.get(gpu_id) is None
    assert ("rp_env_key", "pod-123") in fake_runpod["terminate"]


def test_terminate_unknown_id_returns_false():
    assert runtime.terminate_gpu("nope") is False


# ---------------------------------------------------------------------------
# SSH endpoint extraction — must handle every RunPod payload shape.
# ---------------------------------------------------------------------------

def test_ssh_endpoint_port_mappings_dict():
    pod = {"publicIp": "1.2.3.4", "portMappings": {"22": 40022}}
    assert _ssh_endpoint(pod) == ("1.2.3.4", 40022)


def test_ssh_endpoint_port_mappings_tcp_key():
    pod = {"publicIp": "1.2.3.4", "portMappings": {"22/tcp": 40022}}
    assert _ssh_endpoint(pod) == ("1.2.3.4", 40022)


def test_ssh_endpoint_runtime_ports_list():
    # publicIp null at top level, but the port entry carries the public IP.
    pod = {
        "publicIp": None,
        "runtime": {"ports": [
            {"privatePort": 22, "publicPort": 40022, "ip": "5.6.7.8",
             "isIpPublic": True, "type": "tcp"},
        ]},
    }
    assert _ssh_endpoint(pod) == ("5.6.7.8", 40022)


def test_ssh_endpoint_top_level_ports_list_picks_22():
    pod = {"ports": [
        {"privatePort": 8888, "publicPort": 1, "ip": "9.9.9.9"},
        {"privatePort": 22, "publicPort": 50022, "ip": "9.9.9.9", "isIpPublic": True},
    ]}
    assert _ssh_endpoint(pod) == ("9.9.9.9", 50022)


def test_ssh_endpoint_none_when_not_public():
    pod = {"publicIp": None, "runtime": {"ports": [
        {"privatePort": 22, "publicPort": 40022, "ip": "10.0.0.1", "isIpPublic": False},
    ]}}
    assert _ssh_endpoint(pod) is None


def test_ssh_endpoint_none_when_empty():
    assert _ssh_endpoint({"publicIp": None, "portMappings": {}}) is None
