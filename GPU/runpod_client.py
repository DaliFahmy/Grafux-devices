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
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gpu.runpod")

REST_BASE = "https://rest.runpod.io/v1"

# How long to wait for a pod to come up with a public SSH endpoint.  Large CUDA
# -devel images can take several minutes to pull on a cold machine, so this is
# generous and overridable via env.  (With retries — see GPU_PROVISION_ATTEMPTS in
# runtime.py — this is the *per-attempt* cap, so it no longer needs to be huge.)
_PROVISION_TIMEOUT = int(os.environ.get("GPU_PROVISION_TIMEOUT", "300") or "300")

# A public IP + port-22 NAT mapping is a property of the *machine* a pod lands on,
# assigned at placement — well before the container image finishes pulling.  So
# once a pod is placed on a machine, if no public endpoint has appeared within this
# many seconds, that machine simply has no direct public-IP networking and never
# will: fail fast (NoEndpointError) so the caller can terminate and hop to a
# different machine instead of waiting out the full image-pull timeout.  Set to 0
# to disable early detection and always wait the full GPU_PROVISION_TIMEOUT.
_PUBLIC_IP_GRACE = int(os.environ.get("GPU_PUBLIC_IP_GRACE", "120") or "120")


class ProvisionError(RuntimeError):
    """
    Base for *retryable* provisioning failures — ones a fresh placement may fix.

    ``runtime.provision_gpu`` catches this (and only this) to terminate the pod and
    retry on a new machine; any other exception is treated as fatal.
    """


class CapacityError(ProvisionError):
    """RunPod had no instances available for the requested GPU type / cloud tier."""


class NoEndpointError(ProvisionError):
    """A pod came up but never exposed a public SSH endpoint (machine has no public IP)."""


# Substrings (case-insensitive) in a RunPod create error body that mean the failure
# is a transient *placement* problem a fresh attempt on another machine can fix —
# capacity scarcity or a machine that can't host the requested pod.  Matching any of
# these classifies the create as a retryable CapacityError instead of a fatal error.
_CAPACITY_MARKERS = (
    "no instances",                 # "There are no instances currently available"
    "does not have the resources",  # "This machine does not have the resources..."
    "try a different machine",      # "...Please try a different machine"
    "no longer any instances",
    "not enough free gpu",
    "out of capacity",
    "insufficient capacity",
)


def _is_capacity_error(text: str) -> bool:
    """True if a RunPod create error body indicates a retryable placement failure."""
    low = (text or "").lower()
    return any(marker in low for marker in _CAPACITY_MARKERS)

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
# Ordered cheapest -> priciest (typical RunPod $/hr) so cost-conscious users land
# on an affordable card first; the default selection is RTX 4090 (see
# DEFAULT_GPU_MODEL in models.py / the creation dialog), not index 0.
GPU_TYPES: List[Dict[str, str]] = [
    {"id": "NVIDIA RTX A4000", "label": "RTX A4000 (16 GB)"},
    {"id": "NVIDIA RTX 2000 Ada Generation", "label": "RTX 2000 Ada (16 GB)"},
    {"id": "NVIDIA RTX A4500", "label": "RTX A4500 (20 GB)"},
    {"id": "NVIDIA GeForce RTX 3090", "label": "RTX 3090 (24 GB)"},
    {"id": "NVIDIA GeForce RTX 4090", "label": "RTX 4090 (24 GB)"},
    {"id": "NVIDIA L4", "label": "L4 (24 GB)"},
    {"id": "NVIDIA RTX A6000", "label": "RTX A6000 (48 GB)"},
    {"id": "NVIDIA RTX 6000 Ada Generation", "label": "RTX 6000 Ada (48 GB)"},
    {"id": "NVIDIA L40S", "label": "L40S (48 GB)"},
    {"id": "NVIDIA GeForce RTX 5090", "label": "RTX 5090 (32 GB)"},
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
    "NVIDIA RTX A4000": "sm_86",
    "NVIDIA RTX 2000 Ada Generation": "sm_89",
    "NVIDIA RTX A4500": "sm_86",
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


# Heuristics for guessing the source language when the declared ``language`` does
# not match the code.  The gpu block's ``language`` port defaults to ``cuda`` (the
# creation dialog seeds it), so a user who pastes Python into the ``code`` port
# without also flipping ``language`` to ``python`` would otherwise have their
# Python written to ``/tmp/job.cu`` and fed to nvcc — which chokes on ``#``
# comments with "invalid preprocessing directive #...".  We detect that case and
# run the source through ``python3`` instead.
_PYTHON_SIGNALS = (
    re.compile(r"^\s*import\s+\w", re.MULTILINE),
    re.compile(r"^\s*from\s+\w[\w.]*\s+import\s", re.MULTILINE),
    re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+\w+\s*[\(:]", re.MULTILINE),
    re.compile(r"^\s*print\s*\(", re.MULTILINE),
    re.compile(r"^#!.*\bpython", re.MULTILINE),
)
# C / C++ / CUDA structure that means the source is NOT Python — these suppress
# the Python guess even if a stray ``print(`` slips through.
_C_SIGNALS = (
    re.compile(r"^\s*#\s*include\b", re.MULTILINE),
    re.compile(r"^\s*#\s*define\b", re.MULTILINE),
    re.compile(r"\b__global__\b|\b__device__\b|\b__host__\b"),
    re.compile(r"\b(?:int|void)\s+main\s*\("),
    re.compile(r"\busing\s+namespace\b"),
)


def looks_like_python(source: str) -> bool:
    """Best-effort guess: does this source read as Python (not C/C++/CUDA)?

    True only when there is at least one strong Python signal and no C-family
    structural signal.  Conservative on purpose — a false positive would route
    real CUDA to ``python3``, so when in doubt we say no.
    """
    if not source:
        return False
    if any(p.search(source) for p in _C_SIGNALS):
        return False
    return any(p.search(source) for p in _PYTHON_SIGNALS)


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
        # Guarantee a public IP + TCP port mapping for SSH.  Without this, pods —
        # especially on Community cloud — come up RUNNING with publicIp="" and
        # portMappings=null, so we can never SSH in.  On Secure cloud it's a no-op;
        # on Community it's required to expose a public IP. (RunPod REST v1 field.)
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": public_key},
        "interruptible": False,
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{REST_BASE}/pods", headers=_headers(api_key), json=body)
    if resp.status_code not in (200, 201):
        text = resp.text or ""
        # A 429/500 whose body matches one of these is a *placement* failure, not a
        # bad request: the requested GPU type is momentarily unavailable, or the
        # machine RunPod tried to place us on can't host the pod ("This machine does
        # not have the resources to deploy your pod. Please try a different
        # machine").  Both are escaped by a fresh placement, so raise a retryable
        # CapacityError and let provision_gpu back off and hop to another machine
        # instead of failing the whole Run.
        if resp.status_code in (429, 500) and _is_capacity_error(text):
            raise CapacityError(
                f"RunPod could not place a {(spec.cloud_type or 'SECURE').upper()} "
                f"'{spec.gpu_model}' pod right now ({resp.status_code}): "
                f"{text.strip()[:200]}"
            )
        raise RuntimeError(
            f"RunPod create_pod failed ({resp.status_code}): {text[:500]}"
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
    """
    Extract (public_ip, ssh_port) for port 22 from a pod payload.

    RunPod returns pod networking in several shapes depending on API version /
    machine, so we try them all:
      1. top-level ``portMappings`` dict: {"22": 40022} or {"22/tcp": 40022}
      2. nested ``runtime.ports`` list: [{ip, isIpPublic, privatePort, publicPort,
         type}] — the (old GraphQL-style) shape the REST API can still echo
      3. top-level ``ports`` list with the same per-entry fields

    Returns None until a public-IP TCP mapping for port 22 is available.
    """
    public_ip = pod.get("publicIp") or pod.get("ip")

    # Shape 1: portMappings dict.
    mappings = pod.get("portMappings")
    if isinstance(mappings, dict) and public_ip:
        ssh_port = mappings.get("22") or mappings.get("22/tcp")
        if ssh_port:
            try:
                return str(public_ip), int(ssh_port)
            except (TypeError, ValueError):
                pass

    # Shapes 2 & 3: a list of port entries (prefer a public one).
    runtime = pod.get("runtime") or {}
    port_lists = []
    if isinstance(runtime, dict) and isinstance(runtime.get("ports"), list):
        port_lists.append(runtime["ports"])
    if isinstance(pod.get("ports"), list):
        port_lists.append(pod["ports"])
    for ports in port_lists:
        for entry in ports:
            if not isinstance(entry, dict):
                continue
            if int(entry.get("privatePort") or 0) != 22:
                continue
            ip = entry.get("ip") or public_ip
            pub_port = entry.get("publicPort")
            is_public = entry.get("isIpPublic", True)
            if ip and pub_port and is_public:
                try:
                    return str(ip), int(pub_port)
                except (TypeError, ValueError):
                    continue
    return None


def _net_summary(pod: Dict[str, Any]) -> str:
    """A compact dump of a pod's networking fields, for timeout diagnostics."""
    runtime = pod.get("runtime") or {}
    parts = [
        f"publicIp={pod.get('publicIp')!r}",
        f"portMappings={pod.get('portMappings')!r}",
        f"ports={pod.get('ports')!r}",
        f"runtime.ports={runtime.get('ports') if isinstance(runtime, dict) else None!r}",
    ]
    return ", ".join(parts)


def _status_summary(pod: Dict[str, Any]) -> str:
    """Lifecycle + networking dump for diagnostics (image-pull vs no-public-IP)."""
    parts = [
        f"desiredStatus={pod.get('desiredStatus')!r}",
        f"lastStatusChange={pod.get('lastStatusChange')!r}",
        f"costPerHr={pod.get('costPerHr')!r}",
        f"machineId={pod.get('machineId')!r}",
        f"image={pod.get('image') or pod.get('imageName')!r}",
    ]
    return "Pod: " + ", ".join(parts) + ". Networking: " + _net_summary(pod)


def wait_until_ready(
    api_key: str,
    pod_id: str,
    *,
    timeout_s: Optional[int] = None,
    poll_s: float = 5.0,
) -> Tuple[str, int]:
    """
    Poll a pod until it is RUNNING with an SSH endpoint, returning (ip, port).

    The timeout defaults to ``GPU_PROVISION_TIMEOUT`` (env, default 600s) because a
    large CUDA -devel image can take several minutes to pull on a cold machine, and
    the public IP / port-22 mapping only populates once the container is actually
    up.  Raises RuntimeError on timeout or if the pod enters a terminal state.
    """
    if timeout_s is None:
        timeout_s = _PROVISION_TIMEOUT
    deadline = time.monotonic() + timeout_s
    placed_at: Optional[float] = None  # when we first saw the pod placed on a machine
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_pod(api_key, pod_id)
        status = (last.get("desiredStatus") or last.get("status") or "").upper()
        if status in ("TERMINATED", "FAILED"):
            raise NoEndpointError(
                f"Pod {pod_id} entered status {status}. {_status_summary(last)}"
            )
        endpoint = _ssh_endpoint(last)
        if endpoint:
            return endpoint
        # Early no-public-IP detection.  The public IP / port-22 mapping is assigned
        # when the pod is placed on a machine (it has a machineId / is RUNNING), so
        # if it has not appeared a short grace after placement, this machine has no
        # public-IP networking and never will — bail now so the caller can hop to a
        # different machine instead of burning the whole image-pull timeout here.
        if _PUBLIC_IP_GRACE and (last.get("machineId") or status == "RUNNING"):
            now = time.monotonic()
            if placed_at is None:
                placed_at = now
            elif now - placed_at >= _PUBLIC_IP_GRACE:
                raise NoEndpointError(
                    f"Pod {pod_id} has been placed on machine "
                    f"{last.get('machineId')!r} for ~{_PUBLIC_IP_GRACE}s with no public "
                    f"IP — this machine does not provide direct public-IP networking. "
                    f"{_status_summary(last)}"
                )
        time.sleep(poll_s)
    # Timed out.  Dump the full lifecycle + networking so the cause is visible:
    # still pulling the image? never allocated a public IP? wrong datacenter?
    raise NoEndpointError(
        f"Pod {pod_id} not ready within {timeout_s}s — no public SSH endpoint "
        f"appeared. If it is still pulling a large image, raise GPU_PROVISION_TIMEOUT "
        f"or use a smaller 'image'. If publicIp stays empty, the placement has no "
        f"public IP — Regenerate to retry, set 'cloud_type' to 'SECURE', or pick a "
        f"different GPU. {_status_summary(last)}"
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
    lang = (language or "cuda").lower()
    is_python = lang in ("python", "py")
    is_cuda = lang in ("cuda", "cu")

    # The ``language`` port defaults to ``cuda``; if the user pasted Python into
    # the ``code`` port without flipping it, compiling with nvcc fails on the
    # ``#`` comments ("invalid preprocessing directive #..."). Auto-correct when
    # the source unmistakably reads as Python so it runs via ``python3`` instead.
    lang_autodetected = False
    if not is_python and looks_like_python(source):
        is_python, is_cuda, lang_autodetected = True, False, True

    ext = "py" if is_python else ("cu" if is_cuda else "cpp")
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
        # NOTE on flag order: user ``compile_flags`` go AFTER the source + ``-o``
        # output so that ``-l`` libraries (e.g. ``-lcublas -lcurand``) link
        # correctly.  The linker resolves symbols left-to-right, so a library must
        # follow the object that references it — putting ``-lcublas`` before the
        # source yields "undefined reference" errors.  Compilation-only flags
        # (-O3, -std=…, -arch) work in this position too.
        # Python is interpreted — there is no compile step.  We run the source
        # directly with ``python3`` (PyTorch is preinstalled in the default image).
        # The compiled path (cuda/cpp) still nvcc/g++-compiles into ``bin_path``.
        if is_python:
            compile_ms = 0
            compile_warnings = (
                "Detected Python source but the 'language' port was not set to "
                "'python' — ran with python3. Set language=python to silence this."
                if lang_autodetected else ""
            )
            run_target = f"python3 {src_path}"
        else:
            flags = (compile_flags or "").strip()
            if is_cuda:
                arch = arch_for(gpu_model)
                arch_flag = f"-arch={arch} " if arch else ""
                compile_inner = f"{_CUDA_ENV}nvcc {arch_flag}{src_path} -o {bin_path} {flags}"
            else:
                compile_inner = f"{_CUDA_ENV}g++ {src_path} -o {bin_path} {flags}"
            compile_cmd = "bash -lc '" + compile_inner + "'"

            t0 = time.monotonic()
            c_code, _c_out, c_err = _exec(client, compile_cmd, timeout=timeout)
            compile_ms = int((time.monotonic() - t0) * 1000)

            if c_code != 0:
                errors = c_err.strip() or f"compilation failed (exit {c_code})"
                # exit 127 = compiler not found even after the PATH fix → the image
                # has no CUDA toolkit.  Give the user an actionable message.
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
            run_target = bin_path

        # 3) Run, timing execution on-device (nanosecond clock) to exclude network
        #    latency from the benchmark.  The program's own stdout/stderr come back
        #    on the exec channels; the wrapper writes timing + exit code to files.
        argv = (args or "").strip()
        run_cmd = (
            "bash -lc '"
            f"{_CUDA_ENV}"
            f"start=$(date +%s%N); {run_target} {argv}; rc=$?; end=$(date +%s%N); "
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
            errors = r_err.strip() or f"program exited with code {exit_code}"
            # exit 127 = python3 not found → the selected image has no Python.
            # The default image ships Python 3.11, so this only hits a custom image.
            if is_python and exit_code == 127 and "not found" in errors.lower():
                errors = (
                    "python3 not found on the pod — the selected image does not include "
                    "a Python interpreter. Point the gpu block's 'image' port at a "
                    "Python-capable image (the default "
                    "runpod/pytorch:...-cuda... image already is) and Regenerate. "
                    "Underlying error: " + errors
                )
            return {
                "status": "error",
                "response": r_out,
                "errors": errors,
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
