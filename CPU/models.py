"""
models.py
Request schema for the post-silicon verification runtime.

A ``cpu`` block is assembled from its input ports.  The *configuration* ports
(instance_type, image, api_keys) define the pod and are sent to
``POST /cpu/create``; they use ``EDA.models.EdaSpec`` unchanged, because a pod is
a pod.  The *run* ports (code, language, args, ...) are sent to
``POST /cpu/{id}/run`` and are what this module defines.

Port -> field mapping
---------------------
code          -> CpuRunRequest.code          the program to compile and run
language      -> CpuRunRequest.language      c | cpp | python
spec          -> CpuRunRequest.spec          what the case claims to verify
args          -> CpuRunRequest.args          argv for the program
build_flags   -> CpuRunRequest.build_flags   extra compiler flags
defines       -> CpuRunRequest.defines       preprocessor defines
include_dirs  -> CpuRunRequest.include_dirs  -I paths
repetitions   -> CpuRunRequest.repetitions   measured runs
warmup        -> CpuRunRequest.warmup        discarded runs before those
files         -> _RunBase.input_files        extra files staged before the build

Every field is optional so a partially-wired block still works.
"""

from __future__ import annotations

import os

from pydantic import Field, PrivateAttr

from EDA.models import _RunBase, register_kind_image
from EDA.pod_client import register_instance_prices

# The image a cpu pod runs: a C/C++ toolchain, Python 3, and the sshd +
# PUBLIC_KEY contract every pod in this codebase is reached through.
#
# THIS IS THE VERIFY IMAGE, ON PURPOSE.  ``docker/Dockerfile`` beside this file
# builds a purpose-made grafux-cpu image, and this default deliberately does NOT
# point at it: that tag was referenced here before it had ever been built and
# pushed, so every Regenerate asked RunPod for an image that does not exist, and
# RunPod stopped the pod with a bare "Exited by Runpod" that named no cause.  A
# default must point at something that is actually published.
#
# The verify image carries everything CPU/flow.py invokes -- its runtime stage
# installs `g++ make perl python3 python3-venv` on ubuntu:22.04 (g++ pulls in gcc
# and libc6-dev), plus the same start.sh sshd contract -- and it is already public
# on GHCR, which matters because RunPod pulls ANONYMOUSLY.
#
# THE ONE THING IT LACKS is /usr/bin/time, so `benchmark.max_rss_kb` reads 0.
# Nothing else is affected: every timing comes from the pod's own nanosecond clock
# in build_bench_script, not from `time`, and run_cpu already tolerates the binary
# being absent.  To get peak RSS, run the "Build CPU image" workflow, make the new
# GHCR package PUBLIC, and set CPU_DEFAULT_IMAGE (or this default) to that tag.
#
# PIN THE TAG, for the same reason EDA_DEFAULT_IMAGE and GPU_DEFAULT_IMAGE are
# pinned: an unpinned image changes under you and breaks provisioning silently.
DEFAULT_CPU_IMAGE = os.environ.get(
    "CPU_DEFAULT_IMAGE",
    "ghcr.io/dalifahmy/grafux-verify:v5050-cocotb20-20260903",
)

# The languages a verification case can be written in.  Assembly is deliberately
# absent: it is not portable, so supporting it honestly means cross toolchains and
# a QEMU user-mode emulator, and an emulated duration is translation time rather
# than a silicon measurement — precisely the number this block exists not to lie
# about.  Native C, C++ and Python cover the bring-up cases that can be measured.
LANGUAGE_CHOICES = ("c", "cpp", "python")
DEFAULT_LANGUAGE = "cpp"

# No PDK applies to running a program on a CPU.  ``GET /cpu/pdks`` is part of the
# shared router surface, so it answers with this rather than offering an ORFS
# platform name that means nothing here.
CPU_PDK_CHOICES: tuple = ()
CPU_DEFAULT_PDK = ""

# Smaller than any EDA kind: a compiler, a Python runtime and one small binary.
CPU_DISK_GB = 25

# The machine dropdown for this block (``GET /cpu/instances``).  Wider than EDA's
# list on purpose: for a benchmark the machine IS the experiment, so every RunPod
# CPU flavour family is offered, not just the compute-optimised one synthesis
# wants.  Ids are EDA's ``<family>-<vcpus>`` convention, which ``create_pod``
# splits into ``cpuFlavorIds`` + ``vcpuCount``; the families are exactly the
# REST v1 ``cpuFlavorIds`` enum.  Families: c = compute (2 GB/vCPU), g = general
# (4 GB/vCPU), m = memory (8 GB/vCPU); 3 / 5 = hardware generation.
#
# ``usd_per_hr`` is advisory (the live pod costPerHr replaces it once polled).
# cpu3c-8 stays the default: it is the size every cpu block has run on so far.
CPU_INSTANCES = [
    {"id": "cpu3c-2", "label": "Compute 2 vCPU / 4 GB", "usd_per_hr": 0.06},
    {"id": "cpu3c-4", "label": "Compute 4 vCPU / 8 GB", "usd_per_hr": 0.12},
    {"id": "cpu3c-8", "label": "Compute 8 vCPU / 16 GB", "usd_per_hr": 0.24},
    {"id": "cpu3c-16", "label": "Compute 16 vCPU / 32 GB", "usd_per_hr": 0.48},
    {"id": "cpu3c-32", "label": "Compute 32 vCPU / 64 GB", "usd_per_hr": 0.96},
    {"id": "cpu3g-4", "label": "General 4 vCPU / 16 GB", "usd_per_hr": 0.16},
    {"id": "cpu3g-8", "label": "General 8 vCPU / 32 GB", "usd_per_hr": 0.33},
    {"id": "cpu3g-16", "label": "General 16 vCPU / 64 GB", "usd_per_hr": 0.64},
    {"id": "cpu3m-4", "label": "Memory 4 vCPU / 32 GB", "usd_per_hr": 0.24},
    {"id": "cpu3m-8", "label": "Memory 8 vCPU / 64 GB", "usd_per_hr": 0.48},
    {"id": "cpu5c-4", "label": "Compute (gen 5) 4 vCPU / 8 GB", "usd_per_hr": 0.14},
    {"id": "cpu5c-8", "label": "Compute (gen 5) 8 vCPU / 16 GB", "usd_per_hr": 0.28},
    {"id": "cpu5c-16", "label": "Compute (gen 5) 16 vCPU / 32 GB", "usd_per_hr": 0.56},
    {"id": "cpu5c-32", "label": "Compute (gen 5) 32 vCPU / 64 GB", "usd_per_hr": 1.12},
    {"id": "cpu5g-8", "label": "General (gen 5) 8 vCPU / 32 GB", "usd_per_hr": 0.40},
    {"id": "cpu5m-8", "label": "Memory (gen 5) 8 vCPU / 64 GB", "usd_per_hr": 0.56},
]
DEFAULT_CPU_INSTANCE = "cpu3c-8"


def list_cpu_instances() -> list:
    """The cpu block's machine dropdown (id + label + advisory usd_per_hr)."""
    return [dict(i) for i in CPU_INSTANCES]


# So the cost estimate knows these prices without EDA importing this package.
register_instance_prices(CPU_INSTANCES)

# Claim this kind's image and disk in EDA's maps.  Done at import of this module,
# which ``CPU.runtime`` (and therefore ``CPU.router``) pulls in, so it has always
# happened by the time a request can arrive.
register_kind_image("cpu", lambda: DEFAULT_CPU_IMAGE, CPU_DISK_GB)


class CpuRunRequest(_RunBase):
    """Live inputs for a post-silicon verification run."""

    # Set server-side by ``start_cpu_job`` from the pod's spec, never by a client:
    # a one-line note when the requested machine was substituted, so the run's
    # `warnings` say where the numbers actually came from.
    _machine_note: str = PrivateAttr("")

    code: str = Field(
        "",
        description=(
            "The verification case source, normally wired from a "
            "post_silicon_verification block's `code` port."
        ),
    )
    language: str = Field(
        DEFAULT_LANGUAGE,
        description="Source language: 'c', 'cpp' or 'python'.",
    )
    spec: str = Field(
        "",
        description=(
            "What the case claims to verify, wired from post_silicon_verification's "
            "`verifies` port. Never compiled or executed — it is evidence for the "
            "post-run review, so that the analysis can judge the run against what "
            "the case set out to prove rather than against nothing."
        ),
    )
    args: str = Field("", description="argv passed to the program.")
    build_flags: str = Field(
        "-O2",
        description=(
            "Extra compiler flags. '-O2' rather than '-O3': a verification case "
            "checks things the optimiser is free to prove redundant and delete, and "
            "a check that was optimised away passes without running."
        ),
    )
    defines: str = Field("", description="Preprocessor defines, e.g. 'WIDTH=8 DEBUG'.")
    include_dirs: str = Field("", description="Space- or newline-separated -I paths.")
    repetitions: str = Field(
        "5",
        description=(
            "How many measured runs to average. A string, not an int, because an "
            "unwired port must mean 'use the default' and an empty text port would "
            "otherwise arrive as 0 and measure nothing."
        ),
    )
    warmup: str = Field(
        "1",
        description=(
            "Runs executed and discarded before measuring, so page faults and a cold "
            "instruction cache land there instead of in the first reported sample."
        ),
    )
