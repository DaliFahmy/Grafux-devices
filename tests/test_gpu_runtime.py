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


def test_create_pod_no_resources_is_retryable(monkeypatch):
    """A 500 'this machine does not have the resources' must be a retryable CapacityError.

    RunPod returns this when the machine it tried to place the pod on can't host it
    ("Please try a different machine") — a placement failure a fresh attempt escapes,
    so it must hop machines rather than fail the whole Run.
    """
    resp = _FakeResp(
        500,
        '{"error":"create pod: This machine does not have the resources to deploy '
        'your pod. Please try a different machine","status":500}',
    )

    class FakeHttpx:
        @staticmethod
        def Client(*a, **k):
            return _FakeHttpxClient(resp)

    monkeypatch.setattr(runpod_client, "_httpx", lambda: FakeHttpx)
    with pytest.raises(runpod_client.CapacityError):
        runpod_client.create_pod("rp_x", GpuSpec(), "ssh-rsa AAA grafux")


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

    def read(self, _n=-1):
        return self._store.get(self._path, b"")


class _FakeSftp:
    def __init__(self, store):
        self._store = store

    def open(self, path, _mode):
        return _FakeSftpFile(self._store, path)

    def mkdir(self, _path):
        pass

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
        "glob_listing": "",  # newline paths returned for the artifact-glob listing
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
        if command.startswith("bash -lc 'for f in"):  # the artifact-glob listing
            return 0, state["glob_listing"], ""
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


def test_run_remote_autodetects_python_when_language_default(fake_ssh):
    # The exact bug report: Python pasted into a block whose language port is
    # still the default "cuda" → must run via python3, not nvcc on /tmp/job.cu.
    source = "# Create two matrices\nimport torch\nprint(torch.zeros(2))\n"
    result = runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source=source,
        language="cuda",
        gpu_model="NVIDIA GeForce RTX 4090",
        compile_flags="-O3",
        args="",
        timeout=60,
    )
    assert result["status"] == "ok"
    # Routed to python, NOT compiled as CUDA.
    assert "/tmp/job.py" in fake_ssh["uploaded"]
    assert "/tmp/job.cu" not in fake_ssh["uploaded"]
    joined = "\n".join(fake_ssh["commands"])
    assert "nvcc" not in joined
    assert "python3 /tmp/job.py" in joined
    # The user is told we corrected the language.
    assert "language" in result["warnings"].lower()


def test_looks_like_python_does_not_misfire_on_cuda():
    # C/CUDA structure must suppress the Python guess even with a stray print(.
    assert not runpod_client.looks_like_python(
        "#include <cstdio>\nint main(){ print(1); return 0; }"
    )
    assert not runpod_client.looks_like_python("__global__ void k(){}")
    # Real Python is recognised.
    assert runpod_client.looks_like_python("import os\nprint(os.getcwd())")
    assert not runpod_client.looks_like_python("")


# ---------------------------------------------------------------------------
# Keep-warm (per-block) — fast re-runs without paying a cold start each time.
# ---------------------------------------------------------------------------

def test_run_keep_warm_reports_live_id_and_no_teardown(monkeypatch, fake_runpod):
    """A keep_warm_minutes>0 run keeps the pod alive and reports the live gpu_id."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}", keep_warm_minutes=5))
    assert out["status"] == "ok"
    assert out["kept_warm"] is True
    assert out["warm_until"] > 0
    # Pod NOT terminated, still registered, live id reported so the block reuses it.
    assert fake_runpod["terminate"] == []
    assert registry.get(gpu_id) is not None
    assert out["gpu_id"] == gpu_id
    # The warm window set a reaper deadline so it is freed eventually.
    assert registry.get(gpu_id).keep_warm_deadline > 0


def test_run_keep_warm_falls_back_to_spec(monkeypatch, fake_runpod):
    """With no per-run window, the block's GpuSpec.keep_warm_minutes is honored."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec(keep_warm_minutes=10))["gpu_id"]

    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}"))
    assert out["kept_warm"] is True
    assert fake_runpod["terminate"] == []
    assert registry.get(gpu_id) is not None


def test_run_keep_warm_failure_still_terminates(monkeypatch, fake_runpod):
    """A failed run is never kept warm — no orphan billing even with keep_warm set."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]

    def boom(*a, **k):
        raise OSError("ssh connection refused")

    monkeypatch.setattr(runpod_client, "run_remote", boom)
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}", keep_warm_minutes=5))
    assert out["status"] == "error"
    assert out["kept_warm"] is False
    assert ("rp_env_key", "pod-123") in fake_runpod["terminate"]
    assert registry.get(gpu_id) is None
    assert out["gpu_id"] == ""


def test_run_keep_warm_zero_is_ephemeral(monkeypatch, fake_runpod):
    """keep_warm_minutes=0 reproduces today's ephemeral teardown."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}", keep_warm_minutes=0))
    assert out["status"] == "ok"
    assert out["kept_warm"] is False
    assert ("rp_env_key", "pod-123") in fake_runpod["terminate"]
    assert registry.get(gpu_id) is None
    assert out["gpu_id"] == ""


def test_provision_prewarm_sets_warm_window(monkeypatch, fake_runpod):
    """create with keep_warm_minutes>0 doubles as pre-warm: a warm window is set."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    result = runtime.provision_gpu(GpuSpec(keep_warm_minutes=15))
    assert result["status"] == "ok"
    assert result["warm_until"] > 0
    assert registry.get(result["gpu_id"]).keep_warm_deadline > 0


# ---------------------------------------------------------------------------
# Reaper — honors per-record keep-warm deadline (monotonic), testable directly.
# ---------------------------------------------------------------------------

def test_select_stale_reaps_expired_warm_pod(monkeypatch):
    from GPU.registry import GpuRecord, GpuRegistry
    reg = GpuRegistry()
    rec = GpuRecord(spec=GpuSpec(), pod_id="p", public_ip="1.2.3.4", ssh_port=22)
    gid = reg.create(rec)
    # Warm until t=100 (monotonic).  At t=50 it is kept; at t=150 it is reaped.
    rec.keep_warm_deadline = 100.0
    assert reg._select_stale(50.0) == []
    assert reg.get(gid) is not None
    stale = reg._select_stale(150.0)
    assert [g for g, _ in stale] == [gid]
    assert reg.get(gid) is None


def test_select_stale_ignores_warm_pod_before_deadline(monkeypatch):
    from GPU.registry import GpuRecord, GpuRegistry
    reg = GpuRegistry()
    rec = GpuRecord(spec=GpuSpec(), pod_id="p", public_ip="1.2.3.4", ssh_port=22)
    reg.create(rec)
    rec.keep_warm_deadline = 1e9  # far future
    assert reg._select_stale(123.0) == []


# ---------------------------------------------------------------------------
# Pricing.
# ---------------------------------------------------------------------------

def test_price_for_known_and_unknown():
    assert runpod_client.price_for("NVIDIA GeForce RTX 4090") > 0
    assert runpod_client.price_for("totally-made-up") == 0.0


def test_list_gpu_types_includes_price():
    types = runpod_client.list_gpu_types()
    assert all("usd_per_hr" in g for g in types)
    rtx = next(g for g in types if g["id"] == "NVIDIA GeForce RTX 4090")
    assert rtx["usd_per_hr"] > 0


def test_run_reports_cost_estimate(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec(gpu_model="NVIDIA GeForce RTX 4090"))["gpu_id"]
    out = runtime.run_gpu(gpu_id, GpuRunRequest(code="int main(){}"))
    assert out["usd_per_hr"] > 0
    assert out["cost_estimate_usd"] >= 0


# ---------------------------------------------------------------------------
# Live status + phase derivation.
# ---------------------------------------------------------------------------

def test_status_unknown_id_is_none():
    assert runtime.gpu_status("nope") is None


def test_status_reports_phase_and_uptime(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec(gpu_model="NVIDIA GeForce RTX 4090"))["gpu_id"]
    st = runtime.gpu_status(gpu_id)
    assert st["phase"] == "ready"
    assert st["pod_status"] == "running"
    assert st["uptime_s"] >= 0
    assert st["usd_per_hr"] > 0


def test_phase_from_pod_shapes():
    # Ready: a public SSH endpoint is present.
    ready = {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {"22": 40022}}
    assert runpod_client.phase_from_pod(ready)[0] == "ready"
    # Running but no endpoint yet → still pulling image.
    pulling = {"desiredStatus": "RUNNING", "machineId": "m1", "publicIp": "", "portMappings": None}
    assert runpod_client.phase_from_pod(pulling)[0] == "pulling_image"
    # Not placed yet.
    creating = {"desiredStatus": "PENDING"}
    assert runpod_client.phase_from_pod(creating)[0] == "creating"
    # Terminal.
    dead = {"desiredStatus": "TERMINATED"}
    assert runpod_client.phase_from_pod(dead)[0] == "error"


def test_provision_async_returns_immediately(monkeypatch, fake_runpod):
    """create_async registers a record and returns a gpu_id without blocking."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    # Run the background "thread" inline so the test is deterministic.
    monkeypatch.setattr(runtime.threading, "Thread", _InlineThread)
    result = runtime.provision_gpu_async(GpuSpec())
    assert result["gpu_id"]
    assert result["status"] in ("creating", "ok")
    # With the inline thread, provisioning completed → record is ready.
    assert registry.get(result["gpu_id"]).phase == "ready"


def test_provision_async_no_key_errors(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    result = runtime.provision_gpu_async(GpuSpec())
    assert result["status"] == "error"
    assert result["gpu_id"] == ""


class _InlineThread:
    """A drop-in for threading.Thread that runs the target synchronously on start."""

    def __init__(self, target=None, name=None, daemon=None, **_kw):
        self._target = target

    def start(self):
        if self._target:
            self._target()


# ---------------------------------------------------------------------------
# File staging + artifact download (run_remote SFTP).
# ---------------------------------------------------------------------------

def test_run_remote_stages_input_files(fake_ssh):
    runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source="print(1)\n", language="python", gpu_model="", compile_flags="",
        args="", timeout=60,
        input_files=[{"path": "/workspace/data.txt", "content": "hello"}],
    )
    assert fake_ssh["uploaded"]["/workspace/data.txt"] == b"hello"


def test_run_remote_downloads_artifacts(fake_ssh):
    # The pod "produced" this file; the glob listing resolves it, then it's read back.
    fake_ssh["uploaded"]["/workspace/out.json"] = b'{"ok": true}'
    fake_ssh["glob_listing"] = "/workspace/out.json\n"
    result = runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source="print(1)\n", language="python", gpu_model="", compile_flags="",
        args="", timeout=60,
        output_globs=["/workspace/*.json"],
    )
    arts = result["artifacts"]
    assert len(arts) == 1
    assert arts[0]["path"] == "/workspace/out.json"
    import base64
    assert base64.b64decode(arts[0]["content"]) == b'{"ok": true}'
    assert arts[0]["truncated"] is False


def test_run_remote_no_globs_no_artifacts(fake_ssh):
    result = runpod_client.run_remote(
        "1.2.3.4", 40022, "PEM",
        source="print(1)\n", language="python", gpu_model="", compile_flags="",
        args="", timeout=60,
    )
    assert result["artifacts"] == []


def test_run_forwards_files_and_returns_artifacts_json(monkeypatch, fake_runpod):
    """run_gpu forwards input_files/output_globs and json-encodes artifacts."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")

    def fake_run_remote(host, port, key, **kwargs):
        assert kwargs["input_files"] == [{"path": "/workspace/a.txt", "content": "x"}]
        assert kwargs["output_globs"] == ["/workspace/*.png"]
        return {"status": "ok", "response": "", "errors": "", "warnings": "",
                "benchmark": {}, "artifacts": [{"path": "/workspace/a.png", "size": 3}]}

    monkeypatch.setattr(runpod_client, "run_remote", fake_run_remote)
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]
    out = runtime.run_gpu(gpu_id, GpuRunRequest(
        code="x",
        input_files=[{"path": "/workspace/a.txt", "content": "x"}],
        output_globs=["/workspace/*.png"],
    ))
    assert out["status"] == "ok"
    assert json.loads(out["artifacts"])[0]["path"] == "/workspace/a.png"


# ---------------------------------------------------------------------------
# SSH access (env-gated).
# ---------------------------------------------------------------------------

def test_ssh_disabled_by_default(monkeypatch, fake_runpod):
    monkeypatch.setattr(runtime, "_EXPOSE_SSH", False)
    assert runtime.ssh_exposed() is False


def test_ssh_returns_details_when_running(monkeypatch, fake_runpod):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
    gpu_id = runtime.provision_gpu(GpuSpec())["gpu_id"]
    details = runtime.gpu_ssh(gpu_id)
    assert details["host"] == "1.2.3.4"
    assert details["port"] == 40022
    assert details["username"] == "root"
    assert details["private_key_pem"] == "PRIVATE_PEM"
    assert "ssh -p 40022 root@1.2.3.4" in details["connect_hint"]


def test_ssh_unknown_id_is_none():
    assert runtime.gpu_ssh("nope") is None
