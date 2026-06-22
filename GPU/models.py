"""
models.py
Request/response schemas for the GPU (cloud-GPU) runtime.

A gpu block is assembled from the "gpu" block's input ports.  The *configuration*
ports (gpu_model, image, cloud_type, compile_flags, api_keys, credentials) define
the pod and are sent to ``POST /gpu/create`` (Regenerate).  The *run* ports
(code, language, args, timeout) are sent to ``POST /gpu/{id}/run`` (Run).

Every field is optional so a partially-wired block still works: the only field
that meaningfully changes the provisioned pod is ``gpu_model``; an empty ``code``
port simply compiles/runs nothing.

Port → field mapping
--------------------
gpu_model      -> GpuSpec.gpu_model       (RunPod GPU type id, e.g. "NVIDIA GeForce RTX 4090")
image          -> GpuSpec.image           (CUDA -devel docker image with nvcc)
cloud_type     -> GpuSpec.cloud_type      ("SECURE" | "COMMUNITY")
compile_flags  -> GpuSpec.compile_flags   (extra nvcc/g++ flags, e.g. "-O3")
api_keys       -> GpuSpec.api_keys        (optional RunPod key override; text or JSON)
credentials    -> GpuSpec.credentials     (optional RunPod key override; text or JSON)
code           -> GpuRunRequest.code      (C++/CUDA source)
language       -> GpuRunRequest.language  ("cuda" | "cpp")
args           -> GpuRunRequest.args      (argv passed to the compiled program)
timeout        -> GpuRunRequest.timeout   (per-run wall-clock limit, seconds)
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

# The default image is a RunPod CUDA *devel* image: it ships ``nvcc`` (devel),
# Python 3.11 + PyTorch, and RunPod's start script that installs ``PUBLIC_KEY``
# into authorized_keys and starts sshd — so SSH "just works" with no custom build.
#
# CUDA version note (the "cuda>=12.8 driver" gotcha): a CUDA container only starts
# if the *host* NVIDIA driver is new enough for the image's CUDA toolkit.  The
# previous default (cuda12.8.1) needs driver R570+, which a large share of RunPod
# machines — Community especially — don't have yet, so the container init fails
# with ``nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8``
# and the pod never comes up (RunPod itself advises "use an earlier cuda
# container").  CUDA 12.4 (driver R550) is far more widely deployed, so it lands on
# vastly more machines while still shipping a current PyTorch + nvcc.  Override via
# the gpu block's ``image`` port (or ``GPU_DEFAULT_IMAGE``) to pin a newer CUDA.
DEFAULT_IMAGE = os.environ.get(
    "GPU_DEFAULT_IMAGE",
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-cudnn-devel-ubuntu22.04",
)
DEFAULT_GPU_MODEL = "NVIDIA GeForce RTX 4090"


class GpuSpec(BaseModel):
    """The persistent definition of a gpu pod (everything except the live run)."""

    gpu_model: str = Field(
        DEFAULT_GPU_MODEL,
        description="RunPod GPU type id, e.g. 'NVIDIA GeForce RTX 4090' or 'NVIDIA A100 80GB PCIe'.",
    )
    image: str = Field(
        DEFAULT_IMAGE,
        description="Docker image for the pod — must be a CUDA -devel image so nvcc is present.",
    )
    cloud_type: str = Field(
        "SECURE",
        description="RunPod cloud tier: 'SECURE' (datacenter) or 'COMMUNITY' (cheaper).",
    )
    container_disk_gb: int = Field(20, description="Container disk size in GB.")
    compile_flags: str = Field(
        "-O3",
        description="Extra flags passed to nvcc/g++ when compiling the code.",
    )
    api_keys: str = Field(
        "",
        description="Optional RunPod API key override (bare 'rp_...' or JSON {\"runpod\": \"...\"}).",
    )
    credentials: str = Field(
        "",
        description="Optional RunPod API key (same shapes as api_keys), used if api_keys is empty.",
    )
    name: str = Field("", description="Optional human-friendly name for the gpu block.")


class GpuRunRequest(BaseModel):
    """The live inputs supplied on every run of an existing gpu pod."""

    code: str = Field("", description="C++/CUDA/Python source to run on the GPU.")
    language: str = Field(
        "cuda",
        description="'cuda' (nvcc), 'cpp' (g++), or 'python' (python3, PyTorch preinstalled).",
    )
    args: str = Field("", description="Command-line arguments passed to the compiled program.")
    timeout: int = Field(120, description="Per-run wall-clock limit in seconds.")


class CreateGpuResponse(BaseModel):
    """Returned by POST /gpu/create — the Regenerate action."""

    gpu_id: str
    status: str = "ok"          # "ok" | "error"
    pod_id: str = ""
    gpu_model: str = ""
    errors: str = ""


class GpuRunResponse(BaseModel):
    """Returned by POST /gpu/{id}/run — maps 1:1 onto the block's output ports."""

    gpu_id: str
    status: str = "ok"          # "ok" | "error" | "server_unreachable"
    response: str = ""          # program stdout
    errors: str = ""            # stderr / compile or run failure
    warnings: str = ""          # compiler warnings
    benchmark: str = ""         # JSON: {compile_ms, exec_ms, exit_code, gpu_model, gpu_info}


class GpuSummary(BaseModel):
    """A gpu entry as returned by the list endpoint (no secrets echoed back)."""

    gpu_id: str
    name: str = ""
    gpu_model: str = ""
    pod_id: str = ""
    pod_status: str = ""        # "running" once the SSH details are cached


class GpuModel(BaseModel):
    """One selectable GPU type for the creation-dialog dropdown."""

    id: str                     # RunPod gpu type id (passed back in GpuSpec.gpu_model)
    label: str = ""             # human-friendly display name


class GpuModelsResponse(BaseModel):
    models: list[GpuModel] = []
