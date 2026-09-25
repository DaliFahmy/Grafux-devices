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

from pydantic import Field

from EDA.models import _RunBase, register_kind_image

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

# Claim this kind's image and disk in EDA's maps.  Done at import of this module,
# which ``CPU.runtime`` (and therefore ``CPU.router``) pulls in, so it has always
# happened by the time a request can arrive.
register_kind_image("cpu", lambda: DEFAULT_CPU_IMAGE, CPU_DISK_GB)


class CpuRunRequest(_RunBase):
    """Live inputs for a post-silicon verification run."""

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
