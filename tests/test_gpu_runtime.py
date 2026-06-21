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


def test_provision_retries_no_public_ip_then_succeeds(monkeypatch, fake_runpod):
    """A no-public-IP placement is freed and a fresh pod is created (machine hop)."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    monkeypatch.setattr(runtime, "_RETRY_BACKOFF_S", 0)

    created = []

    def fake_create(api_key, spec, public_key):
        created.append(f"pod-{len(created)}")
        return created[-1]

    waits = {"n": 0}

    def fake_wait(api_key, pod_id, **kwargs):
        waits["n"] += 1
        if waits["n"] == 1:
            raise runpod_client.NoEndpointError("machine has no public IP")
        return ("1.2.3.4", 40022)

    terminated = []
    monkeypatch.setattr(runpod_client, "create_pod", fake_create)
    monkeypatch.setattr(runpod_client, "wait_until_ready", fake_wait)
    monkeypatch.setattr(runpod_client, "terminate_pod",
                        lambda api_key, pod_id: terminated.append(pod_id))

    result = runtime.provision_gpu(GpuSpec())
    assert result["status"] == "ok"
    assert result["pod_id"] == "pod-1"   # the second placement
    assert terminated == ["pod-0"]       # the no-public-IP pod was freed


def test_provision_retries_on_capacity_then_succeeds(monkeypatch, fake_runpod):
    """A transient 'no instances available' is retried until capacity frees up."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    monkeypatch.setattr(runtime, "_RETRY_BACKOFF_S", 0)

    n = {"create": 0}

    def fake_create(api_key, spec, public_key):
        n["create"] += 1
        if n["create"] == 1:
            raise runpod_client.CapacityError("no instances available")
        return "pod-ok"

    monkeypatch.setattr(runpod_client, "create_pod", fake_create)
    monkeypatch.setattr(runpod_client, "wait_until_ready",
                        lambda api_key, pod_id, **k: ("1.2.3.4", 40022))

    result = runtime.provision_gpu(GpuSpec())
    assert result["status"] == "ok"
    assert result["pod_id"] == "pod-ok"
    assert n["create"] == 2
    # CapacityError means no pod was created on the first try → nothing to terminate.
    assert fake_runpod["terminate"] == []


def test_provision_gives_up_after_attempts(monkeypatch, fake_runpod):
    """Exhausting retries returns a helpful error and frees every pod created."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    monkeypatch.setattr(runtime, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(runtime, "_PROVISION_ATTEMPTS", 3)

    created = []
    terminated = []
    monkeypatch.setattr(runpod_client, "create_pod",
                        lambda api_key, spec, pk: created.append(f"pod-{len(created)+1}") or created[-1])
    monkeypatch.setattr(runpod_client, "wait_until_ready",
                        lambda api_key, pod_id, **k: (_ for _ in ()).throw(
                            runpod_client.NoEndpointError("no public IP on this machine")))
    monkeypatch.setattr(runpod_client, "terminate_pod",
                        lambda api_key, pod_id: terminated.append(pod_id))

    result = runtime.provision_gpu(GpuSpec())
    assert result["status"] == "error"
    assert "after 3 attempts" in result["errors"]
    assert len(created) == 3
    assert terminated == ["pod-1", "pod-2", "pod-3"]  # every placement freed


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


def test_run_ephemeral_terminates_pod_after_success(monkeypatch, fake_runpod):
    """The cost fix: a successful run frees the pod immediately and clears gpu_id."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}"))
    assert out["status"] == "ok"
    # Pod terminated and dropped from the registry the instant the run finished.
    assert ("rp_env_key", "pod-123") in fake_runpod["terminate"]
    assert registry.get(gpu_id) is None
    # Empty gpu_id tells the block to re-provision on the next Run.
    assert out["gpu_id"] == ""


def test_run_ephemeral_terminates_pod_after_failure(monkeypatch, fake_runpod):
    """A failed run must still free the pod — no orphan billing on errors."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    def boom(*a, **k):
        raise OSError("ssh connection refused")

    monkeypatch.setattr(runpod_client, "run_remote", boom)
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}"))
    assert out["status"] == "error"
    assert ("rp_env_key", "pod-123") in fake_runpod["terminate"]
    assert registry.get(gpu_id) is None
    assert out["gpu_id"] == ""


def test_run_keep_alive_does_not_terminate(monkeypatch, fake_runpod):
    """GPU_EPHEMERAL=0 restores warm reuse: the pod survives the run."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    monkeypatch.setattr(runtime, "_EPHEMERAL", False)
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}"))
    assert out["status"] == "ok"
    # Pod kept warm: not terminated, still in the registry, live id reported back.
    assert fake_runpod["terminate"] == []
    assert registry.get(gpu_id) is not None
    assert out["gpu_id"] == gpu_id


def test_run_unknown_id_errors():
    out = runtime.run_gpu("does-not-exist", GpuRunRequest(code="x"))
    assert out["status"] == "error"
    assert "No GPU with id" in out["errors"]
    # Empty gpu_id so the next Run provisions a fresh pod instead of re-erroring.
    assert out["gpu_id"] == ""


def test_run_empty_code_terminates_and_errors(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="   "))
    assert out["status"] == "error"
    assert "code" in out["errors"].lower()
    # Even a no-op run must not leave a billing pod behind.
    assert ("rp_env_key", "pod-123") in fake_runpod["terminate"]
    assert registry.get(gpu_id) is None


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


# ---------------------------------------------------------------------------
# Provisioning failure classification — capacity + no-public-IP must be retryable.
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def json(self):
        return {}


class _FakeHttpxClient:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, *a, **k):
        return self._resp


def test_create_pod_capacity_error_is_retryable(monkeypatch):
    """The RunPod '500 no instances available' must surface as a retryable CapacityError."""
    resp = _FakeResp(
        500,
        '{"error":"create pod: There are no instances currently available","status":500}',
    )

    class FakeHttpx:
        @staticmethod
        def Client(*a, **k):
            return _FakeHttpxClient(resp)

    monkeypatch.setattr(runpod_client, "_httpx", lambda: FakeHttpx)
    with pytest.raises(runpod_client.CapacityError):
        runpod_client.create_pod("rp_x", GpuSpec(), "ssh-rsa AAA grafux")
    # CapacityError is a ProvisionError so provision_gpu's retry path catches it.
    assert issubclass(runpod_client.CapacityError, runpod_client.ProvisionError)


def test_create_pod_other_error_is_not_retryable(monkeypatch):
    resp = _FakeResp(400, '{"error":"bad image"}')

    class FakeHttpx:
        @staticmethod
        def Client(*a, **k):
            return _FakeHttpxClient(resp)

    monkeypatch.setattr(runpod_client, "_httpx", lambda: FakeHttpx)
    with pytest.raises(RuntimeError) as ei:
        runpod_client.create_pod("rp_x", GpuSpec(), "ssh-rsa AAA grafux")
    # A plain RuntimeError (NOT a ProvisionError) → provision_gpu fails immediately.
    assert not isinstance(ei.value, runpod_client.ProvisionError)


def test_wait_until_ready_fast_fails_when_no_public_ip(monkeypatch):
    """A pod placed on a machine with no public IP fails fast, well before timeout."""
    pod = {"desiredStatus": "RUNNING", "machineId": "m1",
           "publicIp": "", "portMappings": None}
    monkeypatch.setattr(runpod_client, "get_pod", lambda api_key, pod_id: pod)
    monkeypatch.setattr(runpod_client, "_PUBLIC_IP_GRACE", 10)

    clock = {"t": 0.0}
    monkeypatch.setattr(runpod_client.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(runpod_client.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))

    with pytest.raises(runpod_client.NoEndpointError) as ei:
        runpod_client.wait_until_ready("rp_x", "pod-1", timeout_s=600, poll_s=5)
    assert "public-IP networking" in str(ei.value)
    # Bailed at the grace (~10s of fake clock), not the 600s timeout.
    assert clock["t"] < 600


# ---------------------------------------------------------------------------
# run_remote — language dispatch (compiled cuda/cpp vs interpreted python)
#
# These drive run_remote directly against an in-memory SSH/SFTP fake so we can
# assert which commands it issues (compile vs python3) without a real pod.
# ---------------------------------------------------------------------------


class _FakeSftpFile:
    def __init__(self, store, path):
        self._store, self._path = store, path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, data):
        self._store[self._path] = data


class _FakeSftp:
    def __init__(self, store):
        self._store = store

    def open(self, path, _mode):
        return _FakeSftpFile(self._store, path)

    def close(self):
        pass


class _FakeSshClient:
    def __init__(self, store):
        self._store = store

    def open_sftp(self):
        return _FakeSftp(self._store)

    def close(self):
        pass


@pytest.fixture
def fake_ssh(monkeypatch):
    """Drive run_remote against an in-memory SSH/SFTP fake.

    Records uploaded source + every command, and lets a test set the result of
    the main run command (the on-device exit code via /tmp/exit + stdout/stderr).
    """
    state = {
        "uploaded": {},   # path -> source written via sftp
        "commands": [],   # every command passed to _exec, in order
        "run_out": "ran ok\n",
        "run_err": "",
        "exit_txt": "0",  # what `cat /tmp/exit` reports → becomes exit_code
        "ms_txt": "7",
    }

    def fake_connect(host, port, key, *, timeout=30.0):
        return _FakeSshClient(state["uploaded"])

    def fake_exec(client, command, timeout):
        state["commands"].append(command)
        if command.startswith("cat /tmp/exit"):
            return 0, state["exit_txt"], ""
        if command.startswith("cat /tmp/ms"):
            return 0, state["ms_txt"], ""
        if command.startswith("nvidia-smi"):
            return 0, "NVIDIA GeForce RTX 4090, 100 MiB, 24564 MiB", ""
        if "date +%s%N" in command:  # the main run command
            return 0, state["run_out"], state["run_err"]
        return 0, "", ""  # compile command: success, no warnings

    monkeypatch.setattr(runpod_client, "_connect_ssh", fake_connect)
    monkeypatch.setattr(runpod_client, "_exec", fake_exec)
    return state


def test_run_remote_python_skips_compile(fake_ssh):
    result = runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source="import torch\nprint('hi')\n",
        language="python",
        gpu_model="NVIDIA GeForce RTX 4090",
        compile_flags="-O3",
        args="",
        timeout=60,
    )
    assert result["status"] == "ok"
    # Source landed in a .py file.
    assert fake_ssh["uploaded"]["/tmp/job.py"] == "import torch\nprint('hi')\n"
    joined = "\n".join(fake_ssh["commands"])
    # No compiler was invoked …
    assert "nvcc" not in joined
    assert "g++" not in joined
    # … and the source ran directly under python3.
    assert "python3 /tmp/job.py" in joined
    # Interpreted → no compile time.
    assert result["benchmark"]["compile_ms"] == 0


def test_run_remote_python_missing_interpreter(fake_ssh):
    fake_ssh["exit_txt"] = "127"
    fake_ssh["run_err"] = "bash: python3: command not found"
    result = runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source="print(1)\n",
        language="python",
        gpu_model="",
        compile_flags="",
        args="",
        timeout=60,
    )
    assert result["status"] == "error"
    assert "python3 not found" in result["errors"]
    assert result["benchmark"]["exit_code"] == 127


def test_run_remote_cuda_still_compiles(fake_ssh):
    result = runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source="int main(){return 0;}",
        language="cuda",
        gpu_model="NVIDIA GeForce RTX 4090",
        compile_flags="-O3",
        args="",
        timeout=60,
    )
    assert result["status"] == "ok"
    assert fake_ssh["uploaded"]["/tmp/job.cu"] == "int main(){return 0;}"
    joined = "\n".join(fake_ssh["commands"])
    assert "nvcc" in joined
    assert "python3 /tmp/job" not in joined
