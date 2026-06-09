"""
runpod_client.py
Thin wrapper over the RunPod REST API + SSH, isolating the cloud-GPU vendor from
the rest of the runtime.

Provisioning uses RunPod's REST API (https://rest.runpod.io/v1) over ``httpx``
(already a devices-server dependency) rather than the heavier ``runpod`` SDK.
Command execution (compile + run + benchmark) uses ``paramiko`` over SSH.

Both ``httpx`` and ``paramiko`` are imported lazily so the devices server still
boots (and hardware-device endpoints keep working) on a host where either is
absent — a friendly RuntimeError is raised only when a GPU endpoint is actually
used.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gpu.runpod")

REST_BASE = "https://rest.runpod.io/v1"

# Prepended to compile/run commands so the CUDA toolkit resolves over SSH.  A
# non-login exec session does not inherit the image's Docker ENV PATH, so we add
# the standard CUDA bin/lib dirs explicitly (nvcc lives in /usr/local/cuda/bin).
_CUDA_ENV = (
    'export PATH="/usr/local/cuda/bin:$PATH"; '
    'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"; '
)

# Curated list of common RunPod GPU types offered in the creation-dialog dropdown.
# ``id`` is what RunPod expects in ``gpuTypeIds`` (and what the block stores in its
# gpu_model port); ``label`` is the human-friendly name shown to the user.
GPU_TYPES: List[Dict[str, str]] = [
    {"id": "NVIDIA GeForce RTX 4090", "label": "RTX 4090 (24 GB)"},
    {"id": "NVIDIA GeForce RTX 5090", "label": "RTX 5090 (32 GB)"},
    {"id": "NVIDIA GeForce RTX 3090", "label": "RTX 3090 (24 GB)"},
    {"id": "NVIDIA RTX 6000 Ada Generation", "label": "RTX 6000 Ada (48 GB)"},
    {"id": "NVIDIA RTX A6000", "label": "RTX A6000 (48 GB)"},
    {"id": "NVIDIA L4", "label": "L4 (24 GB)"},
    {"id": "NVIDIA L40S", "label": "L40S (48 GB)"},
    {"id": "NVIDIA A100 80GB PCIe", "label": "A100 PCIe (80 GB)"},
    {"id": "NVIDIA A100-SXM4-80GB", "label": "A100 SXM (80 GB)"},
    {"id": "NVIDIA H100 PCIe", "label": "H100 PCIe (80 GB)"},
    {"id": "NVIDIA H100 80GB HBM3", "label": "H100 SXM (80 GB)"},
    {"id": "NVIDIA H100 NVL", "label": "H100 NVL (94 GB)"},
    {"id": "NVIDIA H200", "label": "H200 (141 GB)"},
    {"id": "NVIDIA B200", "label": "B200 (180 GB)"},
]

# Best-effort CUDA compute-capability (-arch) per GPU type.  Used only when the
# model is confidently known; otherwise nvcc's default arch is used (PTX JIT keeps
# the binary forward-compatible, so an omitted/older arch still runs correctly).
_ARCH_BY_GPU: Dict[str, str] = {
    "NVIDIA GeForce RTX 4090": "sm_89",
    "NVIDIA GeForce RTX 5090": "sm_120",
    "NVIDIA GeForce RTX 3090": "sm_86",
    "NVIDIA RTX 6000 Ada Generation": "sm_89",
    "NVIDIA RTX A6000": "sm_86",
    "NVIDIA L4": "sm_89",
    "NVIDIA L40S": "sm_89",
    "NVIDIA A100 80GB PCIe": "sm_80",
    "NVIDIA A100-SXM4-80GB": "sm_80",
    "NVIDIA H100 PCIe": "sm_90",
    "NVIDIA H100 80GB HBM3": "sm_90",
    "NVIDIA H100 NVL": "sm_90",
    "NVIDIA H200": "sm_90",
    "NVIDIA B200": "sm_100",
}


def arch_for(gpu_model: str) -> str:
    """Return the ``-arch=sm_XX`` value for a GPU model, or '' if unknown."""
    sm = _ARCH_BY_GPU.get((gpu_model or "").strip())
    return sm or ""


def list_gpu_types() -> List[Dict[str, str]]:
    """Return the curated GPU dropdown list (id + label)."""
    return list(GPU_TYPES)


# ---------------------------------------------------------------------------
# Lazy dependency loaders
# ---------------------------------------------------------------------------

def _httpx():
    try:
        import httpx  # noqa: WPS433 (lazy import by design)
        return httpx
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "The 'httpx' package is required for the GPU runtime. Add httpx>=0.23 "
            "to requirements.txt."
        ) from exc


def _paramiko():
    try:
        import paramiko  # noqa: WPS433 (lazy import by design)
        return paramiko
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "The 'paramiko' package is required to run code on the GPU. Add "
            "paramiko>=3.0 to requirements.txt."
        ) from exc


# ---------------------------------------------------------------------------
# SSH keypair
# ---------------------------------------------------------------------------

def generate_keypair() -> Tuple[str, str]:
    """
    Generate an ephemeral RSA keypair for a pod.

    Returns ``(private_key_pem, openssh_public_key)``.  The public key is injected
    into the pod via the ``PUBLIC_KEY`` env var (RunPod images add it to
    authorized_keys); the private key is kept in the registry to SSH back in.
    """
    paramiko = _paramiko()
    key = paramiko.RSAKey.generate(2048)
    buf = io.StringIO()
    key.write_private_key(buf)
    private_pem = buf.getvalue()
    public_openssh = f"{key.get_name()} {key.get_base64()} grafux-gpu"
    return private_pem, public_openssh


# ---------------------------------------------------------------------------
# REST: provision / inspect / terminate
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def create_pod(api_key: str, spec, public_key: str) -> str:
    """
    Create an on-demand GPU pod and return its id.

    ``spec`` is a GpuSpec.  The public key is injected so we can SSH in.
    """
    httpx = _httpx()
    body: Dict[str, Any] = {
        "name": f"grafux-gpu-{(spec.name or 'job')}"[:60],
        "imageName": spec.image,
        "computeType": "GPU",
        "cloudType": (spec.cloud_type or "SECURE").upper(),
        "gpuTypeIds": [spec.gpu_model],
        "gpuCount": 1,
        "containerDiskInGb": int(spec.container_disk_gb or 20),
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": public_key},
        "interruptible": False,
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{REST_BASE}/pods", headers=_headers(api_key), json=body)
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"RunPod create_pod failed ({resp.status_code}): {resp.text[:500]}"
        )
    data = resp.json()
    pod_id = data.get("id") or data.get("podId")
    if not pod_id:
        raise RuntimeError(f"RunPod create_pod returned no id: {data}")
    return pod_id


def get_pod(api_key: str, pod_id: str) -> Dict[str, Any]:
    """Fetch a pod's current state (status, publicIp, portMappings)."""
    httpx = _httpx()
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{REST_BASE}/pods/{pod_id}", headers=_headers(api_key))
    if resp.status_code != 200:
        raise RuntimeError(
            f"RunPod get_pod failed ({resp.status_code}): {resp.text[:300]}"
        )
    return resp.json()


def terminate_pod(api_key: str, pod_id: str) -> None:
    """Terminate (delete) a pod, stopping all billing.  Best-effort."""
    httpx = _httpx()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(f"{REST_BASE}/pods/{pod_id}", headers=_headers(api_key))
        if resp.status_code not in (200, 204):
            logger.warning(
                "RunPod terminate_pod %s returned %s: %s",
                pod_id, resp.status_code, resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001 — teardown must never raise
        logger.warning("RunPod terminate_pod %s error: %s", pod_id, exc)


def _ssh_endpoint(pod: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Extract (public_ip, ssh_port) from a pod payload, or None if not ready."""
    public_ip = pod.get("publicIp") or pod.get("ip")
    mappings = pod.get("portMappings") or {}
    ssh_port = mappings.get("22") if isinstance(mappings, dict) else None
    if public_ip and ssh_port:
        return str(public_ip), int(ssh_port)
    return None


def wait_until_ready(
    api_key: str,
    pod_id: str,
    *,
    timeout_s: int = 240,
    poll_s: float = 5.0,
) -> Tuple[str, int]:
    """
    Poll a pod until it is RUNNING with an SSH endpoint, returning (ip, port).

    Raises RuntimeError on timeout or if the pod enters a terminal failure state.
    """
    deadline = time.monotonic() + timeout_s
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_pod(api_key, pod_id)
        status = (last.get("desiredStatus") or last.get("status") or "").upper()
        if status in ("TERMINATED", "FAILED"):
            raise RuntimeError(f"Pod {pod_id} entered status {status}")
        endpoint = _ssh_endpoint(last)
        if status == "RUNNING" and endpoint:
            return endpoint
        time.sleep(poll_s)
    raise RuntimeError(
        f"Pod {pod_id} not ready within {timeout_s}s (last status="
        f"{last.get('desiredStatus') or last.get('status')!r})"
    )


# ---------------------------------------------------------------------------
# SSH: compile + run + benchmark
# ---------------------------------------------------------------------------

def _connect_ssh(host: str, port: int, private_key_pem: str, *, timeout: float = 30.0):
    paramiko = _paramiko()
    key = paramiko.RSAKey.from_private_key(io.StringIO(private_key_pem))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # RunPod images log in as root over the exposed TCP port.
    client.connect(
        hostname=host,
        port=port,
        username="root",
        pkey=key,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
    )
    return client


def _exec(client, command: str, timeout: int) -> Tuple[int, str, str]:
    """Run a command, returning (exit_code, stdout, stderr)."""
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def run_remote(
    host: str,
    port: int,
    private_key_pem: str,
    *,
    source: str,
    language: str,
    gpu_model: str,
    compile_flags: str,
    args: str,
    timeout: int,
) -> Dict[str, Any]:
    """
    Upload source to the pod, compile it, run it, and capture output + timing.

    Returns a dict with keys: status, response, errors, warnings, benchmark(dict).
    Never raises for a compile/run failure — those are reported in the result.
    """
    is_cuda = (language or "cuda").lower() in ("cuda", "cu")
    ext = "cu" if is_cuda else "cpp"
    src_path = f"/tmp/job.{ext}"
    bin_path = "/tmp/job"

    client = _connect_ssh(host, port, private_key_pem, timeout=min(timeout, 60))
    try:
        # 1) Upload the source.
        sftp = client.open_sftp()
        try:
            with sftp.open(src_path, "w") as fh:
                fh.write(source or "")
        finally:
            sftp.close()

        # 2) Compile.  Pick the compiler + best-effort -arch.
        #
        # A non-login SSH exec session does NOT inherit the image's Docker ENV PATH,
        # so the CUDA toolkit at /usr/local/cuda/bin is invisible and ``nvcc`` fails
        # with "command not found" (exit 127).  Run everything through a login shell
        # (bash -lc) AND explicitly export the CUDA bin/lib dirs so nvcc + the
        # runtime libs resolve regardless of how the image sets up PATH.
        flags = (compile_flags or "").strip()
        if is_cuda:
            arch = arch_for(gpu_model)
            arch_flag = f"-arch={arch} " if arch else ""
            compile_inner = f"{_CUDA_ENV}nvcc {flags} {arch_flag}{src_path} -o {bin_path}"
        else:
            compile_inner = f"{_CUDA_ENV}g++ {flags} {src_path} -o {bin_path}"
        compile_cmd = "bash -lc '" + compile_inner + "'"

        t0 = time.monotonic()
        c_code, _c_out, c_err = _exec(client, compile_cmd, timeout=timeout)
        compile_ms = int((time.monotonic() - t0) * 1000)

        if c_code != 0:
            errors = c_err.strip() or f"compilation failed (exit {c_code})"
            # exit 127 = compiler not found even after the PATH fix → the image has
            # no CUDA toolkit.  Give the user an actionable message.
            if c_code == 127 and is_cuda and "not found" in errors.lower():
                errors = (
                    "nvcc not found on the pod — the selected image does not include "
                    "the CUDA toolkit. Point the gpu block's 'image' port at a CUDA "
                    "-devel image (e.g. nvidia/cuda:12.6.3-devel-ubuntu22.04) and "
                    "Regenerate. Underlying error: " + errors
                )
            return {
                "status": "error",
                "response": "",
                "errors": errors,
                "warnings": "",
                "benchmark": {
                    "compile_ms": compile_ms,
                    "exec_ms": 0,
                    "exit_code": c_code,
                    "gpu_model": gpu_model,
                    "stage": "compile",
                },
            }
        compile_warnings = c_err.strip()  # nvcc/g++ warnings go to stderr on success

        # 3) Run, timing execution on-device (nanosecond clock) to exclude network
        #    latency from the benchmark.  The program's own stdout/stderr come back
        #    on the exec channels; the wrapper writes timing + exit code to files.
        argv = (args or "").strip()
        run_cmd = (
            "bash -lc '"
            f"{_CUDA_ENV}"
            f"start=$(date +%s%N); {bin_path} {argv}; rc=$?; end=$(date +%s%N); "
            'echo $rc > /tmp/exit; echo $(( (end - start) / 1000000 )) > /tmp/ms'
            "'"
        )
        t1 = time.monotonic()
        _r_code, r_out, r_err = _exec(client, run_cmd, timeout=timeout)
        wall_ms = int((time.monotonic() - t1) * 1000)

        # Read on-device timing + exit code.
        _e1, exit_txt, _e2 = _exec(client, "cat /tmp/exit 2>/dev/null", timeout=30)
        _m1, ms_txt, _m2 = _exec(client, "cat /tmp/ms 2>/dev/null", timeout=30)
        try:
            exit_code = int(exit_txt.strip() or "0")
        except ValueError:
            exit_code = 0
        try:
            exec_ms = int(ms_txt.strip())
        except ValueError:
            exec_ms = wall_ms

        # 4) GPU info for the benchmark payload.
        _g1, gpu_info, _g2 = _exec(
            client,
            "nvidia-smi --query-gpu=name,memory.used,memory.total "
            "--format=csv,noheader 2>/dev/null",
            timeout=30,
        )

        benchmark = {
            "compile_ms": compile_ms,
            "exec_ms": exec_ms,
            "exit_code": exit_code,
            "gpu_model": gpu_model,
            "gpu_info": gpu_info.strip(),
        }
        if exit_code != 0:
            return {
                "status": "error",
                "response": r_out,
                "errors": r_err.strip() or f"program exited with code {exit_code}",
                "warnings": compile_warnings,
                "benchmark": benchmark,
            }
        return {
            "status": "ok",
            "response": r_out,
            "errors": r_err.strip(),
            "warnings": compile_warnings,
            "benchmark": benchmark,
        }
    finally:
        client.close()
