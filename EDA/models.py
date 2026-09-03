"""
models.py
Request/response schemas for the EDA (chip-design) runtime.

An EDA block is assembled from its input ports.  The *configuration* ports
(instance_type, image, pdk, api_keys, credentials) define the pod and are sent to
``POST /{kind}/create`` (Regenerate).  The *run* ports (rtl, netlist, top, sdc, …)
are sent to ``POST /{kind}/{id}/run`` (Run).

Every field is optional so a partially-wired block still works.

Port -> field mapping (shared config, all three kinds)
-----------------------------------------------------
instance_type -> EdaSpec.instance_type   (RunPod CPU flavor id, or a GPU type id)
image         -> EdaSpec.image           (EDA image: yosys + openroad + verilator + PDK)
pdk           -> EdaSpec.pdk             ("sky130hd" | "sky130hs" | "asap7" | "nangate45")
api_keys      -> EdaSpec.api_keys        (optional RunPod key override; text or JSON)
credentials   -> EdaSpec.credentials     (optional RunPod key override; same shapes)

The three run requests mirror the three blocks' run ports one-for-one; see each
class.  Results come back as ``EdaResultResponse.outputs`` — a flat
``port_name -> text`` map — rather than three bespoke response models, because the
Qt executor writes those straight onto the block's output ports.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

# The EDA image must carry yosys, openroad, klayout, verilator, the OpenROAD
# flow-scripts (``$FLOW_HOME``) and the sky130 PDK.
#
# Why a *derived* image rather than ``openroad/orfs`` directly: RunPod's own
# images ship openssh-server plus a start script that installs the ``PUBLIC_KEY``
# env var into authorized_keys, which is how every pod in this codebase is reached.
# Upstream ORFS has neither, so a pod built from it comes up RUNNING with nothing
# listening on port 22 — wait_until_ready then reports "this machine does not
# provide direct public-IP networking", which is a completely misleading error.
# ``docker/Dockerfile`` here adds sshd + the PUBLIC_KEY contract on top of ORFS.
#
# PIN THE TAG.  An unpinned image changes under you and breaks provisioning
# silently — the gpu block learned this the hard way with CUDA host-driver
# mismatches (see GPU_DEFAULT_IMAGE in ../GPU/models.py).
DEFAULT_IMAGE = os.environ.get(
    "EDA_DEFAULT_IMAGE",
    "ghcr.io/dalifahmy/grafux-eda:sky130-20260901",
)

# sky130hd ("high density") is the standard OpenROAD-flow-scripts sky130 variant
# and the one every public example targets.
DEFAULT_PDK = os.environ.get("EDA_DEFAULT_PDK", "sky130hd")

# The PDKs OpenROAD-flow-scripts ships under $FLOW_HOME/platforms.  Offered to the
# creation dialog by ``GET /{kind}/pdks``.
PDK_CHOICES = ["sky130hd", "sky130hs", "asap7", "nangate45", "ihp-sg13g2"]

# EDA tools are CPU-bound — synthesis and place-and-route never touch a GPU — so
# the default pod is a RunPod CPU instance, which is far cheaper than renting an
# idle GPU.  ``EDA_COMPUTE_TYPE=GPU`` falls back to the gpu block's proven code
# path (pick a cheap type such as "NVIDIA RTX A4000") if CPU pods prove awkward.
DEFAULT_COMPUTE_TYPE = os.environ.get("EDA_COMPUTE_TYPE", "CPU").upper()
DEFAULT_INSTANCE = os.environ.get("EDA_DEFAULT_INSTANCE", "cpu3c-8")

# The ORFS flow stages, in order.  These are literal ``make`` targets, which is
# why stage progress here is exact rather than scraped out of a log.
ORFS_STAGES = ("synth", "floorplan", "place", "cts", "route", "final")

# The kinds this package serves, one per Grafux block type.
EDA_KINDS = ("verilator", "yosys", "openroad")

# The light verification image: Verilator + cocotb + iverilog, no PDK and no
# OpenROAD.  It exists because pod placement and image pull dominate a simulation
# that itself takes seconds -- pulling a multi-gigabyte ORFS image to run a
# 200-cycle FIFO test is most of the wall clock and most of the cost, and the fix
# loop pays that price on every iteration.  Built from EDA/docker/Dockerfile.verify.
#
# PIN THE TAG, for the same reason DEFAULT_IMAGE is pinned.  Verilator is held at
# the SAME version in both images: a design that simulates in one and fails in the
# other is a bug report nobody can reproduce.
DEFAULT_VERIFY_IMAGE = os.environ.get(
    "EDA_VERIFY_IMAGE",
    "ghcr.io/dalifahmy/grafux-verify:v5050-cocotb20-20260902",
)


def image_for_kind(kind: str) -> str:
    """
    The default image for an EDA kind.

    Only synthesis and place-and-route need the PDK and the OpenROAD toolchain;
    verilator needs a simulator and cocotb, which is a tenth of the size.
    """
    return DEFAULT_VERIFY_IMAGE if (kind or "") == "verilator" else DEFAULT_IMAGE


def disk_for_kind(kind: str) -> int:
    """Container disk in GB for an EDA kind -- the verify image needs far less."""
    return 20 if (kind or "") == "verilator" else 60


class EdaSpec(BaseModel):
    """The persistent definition of an EDA pod (everything except the live run)."""

    kind: str = Field(
        "yosys",
        description="Which tool this block runs: 'verilator' | 'yosys' | 'openroad'.",
    )
    image: str = Field(
        DEFAULT_IMAGE,
        description="Docker image carrying yosys + openroad + verilator + the PDK.",
    )
    compute_type: str = Field(
        DEFAULT_COMPUTE_TYPE,
        description="RunPod compute tier: 'CPU' (default, EDA is CPU-bound) or 'GPU'.",
    )
    instance_type: str = Field(
        DEFAULT_INSTANCE,
        description="RunPod CPU flavor id (compute_type=CPU) or GPU type id (compute_type=GPU).",
    )
    cloud_type: str = Field(
        "SECURE",
        description="RunPod cloud tier: 'SECURE' (datacenter) or 'COMMUNITY' (cheaper).",
    )
    container_disk_gb: int = Field(
        60,
        description="Container disk in GB — the ORFS image plus PDK and run results are large.",
    )
    pdk: str = Field(DEFAULT_PDK, description="Process design kit / ORFS platform name.")
    api_keys: str = Field(
        "",
        description="Optional RunPod API key override (bare 'rp_...' or JSON with a 'runpod' key).",
    )
    credentials: str = Field(
        "",
        description="Optional RunPod API key (same shapes as api_keys), used if api_keys is empty.",
    )
    name: str = Field("", description="Optional human-friendly name for the block.")
    keep_warm_minutes: int = Field(
        0,
        description=(
            "Keep the pod alive this many minutes after a run for instant re-runs; "
            "0 falls back to EDA_DEFAULT_KEEP_WARM_MIN. Iterating on a floorplan or "
            "re-running a testbench is miserable if every attempt re-pulls a multi-GB "
            "image, so this defaults higher than the gpu block's."
        ),
    )


class _RunBase(BaseModel):
    """Fields every run request shares."""

    timeout: int = Field(900, description="Per-run wall-clock limit in seconds.")
    keep_warm_minutes: int = Field(
        0, description="Keep the pod warm this long after the run; 0 = use the spec/env default."
    )
    input_files: list = Field(
        default_factory=list,
        description="Extra files staged into the pod before the run: [{path, content, b64}].",
    )


class VerilatorRunRequest(_RunBase):
    """Live inputs for a verilator run — simulation or lint."""

    rtl: str = Field("", description="Verilog/SystemVerilog design source.")
    testbench: str = Field(
        "",
        description=(
            "The test harness. A C++ file (uses the generated V<top> class) drives "
            "mode='sim'; empty with mode='sim' falls back to a generated harness that "
            "just evaluates the design, which is enough to prove it elaborates."
        ),
    )
    top: str = Field("", description="Top module name; inferred from the RTL when empty.")
    mode: str = Field(
        "sim",
        description=(
            "'sim' (build + run a C++ harness), 'lint' (lint only), or 'cocotb' "
            "(run a Python cocotb testbench). A Python testbench is detected and "
            "run as cocotb even when mode is left at 'sim'."
        ),
    )
    defines: str = Field("", description="Preprocessor defines, e.g. 'WIDTH=8 DEBUG'.")
    include_dirs: str = Field("", description="Space- or newline-separated +incdir paths.")
    trace: str = Field("1", description="'1' to build with --trace and emit a VCD waveform.")
    sim_args: str = Field("", description="argv passed to the compiled simulation binary.")
    verilator_flags: str = Field("", description="Extra flags passed to verilator.")
    sva: str = Field(
        "",
        description=(
            "Optional SystemVerilog assertions compiled alongside the design "
            "(cocotb mode; enables --assert)."
        ),
    )
    simulator: str = Field(
        "verilator",
        description=(
            "cocotb mode only: 'verilator' (default) or 'icarus' -- the escape "
            "hatch when a Verilator/cocotb version pair misbehaves."
        ),
    )
    tests: str = Field(
        "", description="cocotb mode: comma-separated testcase names to run; empty runs all."
    )
    seed: str = Field("", description="cocotb mode: RNG seed, for a reproducible run.")
    coverage: str = Field(
        "1", description="cocotb mode: '1' to build with --coverage and report line/branch coverage."
    )


class YosysRunRequest(_RunBase):
    """Live inputs for a yosys synthesis run."""

    rtl: str = Field("", description="Verilog/SystemVerilog design source to synthesize.")
    top: str = Field("", description="Top module name; inferred from the RTL when empty.")
    pdk: str = Field("", description="Platform whose liberty file to map onto; falls back to the spec.")
    liberty: str = Field("", description="Explicit .lib path override inside the container.")
    synth_flags: str = Field("", description="Extra flags for the yosys synth command.")
    defines: str = Field("", description="Preprocessor defines, e.g. 'WIDTH=8'.")
    include_dirs: str = Field("", description="Space- or newline-separated include paths.")


class OpenRoadRunRequest(_RunBase):
    """Live inputs for an OpenROAD (flow-scripts) physical-design run."""

    netlist: str = Field(
        "",
        description=(
            "Gate-level Verilog from a yosys block. May instead be a bare filename "
            "when the upstream block wrote a sidecar file (large netlists)."
        ),
    )
    rtl: str = Field(
        "",
        description="RTL fallback — with netlist empty, ORFS runs its own synth first.",
    )
    top: str = Field("", description="Top module / design name.")
    pdk: str = Field("", description="ORFS platform; falls back to the spec's pdk.")
    sdc: str = Field("", description="Timing constraints (SDC). Generated from clock_* when empty.")
    clock_port: str = Field("clk", description="Name of the clock port in the design.")
    clock_period: str = Field("10", description="Clock period in nanoseconds.")
    core_utilization: str = Field("45", description="Target core utilization percentage.")
    aspect_ratio: str = Field("1", description="Core aspect ratio.")
    die_area: str = Field("", description="Explicit die area 'x1 y1 x2 y2' (overrides utilization).")
    core_area: str = Field("", description="Explicit core area 'x1 y1 x2 y2'.")
    place_density: str = Field("", description="Global placement target density, e.g. '0.60'.")
    from_stage: str = Field("synth", description="First ORFS stage to run.")
    to_stage: str = Field("final", description="Last ORFS stage to run.")
    extra_config: str = Field("", description="Raw extra lines appended to the ORFS config.mk.")


class CreateEdaResponse(BaseModel):
    """Returned by POST /{kind}/create and /create_async — the Regenerate action."""

    eda_id: str
    kind: str = ""
    status: str = "ok"          # "ok" | "creating" | "error"
    pod_id: str = ""
    pdk: str = ""
    errors: str = ""
    usd_per_hr: float = 0.0
    warm_until: float = 0.0


class EdaRunAccepted(BaseModel):
    """Returned by POST /{kind}/{id}/run — the job has STARTED, not finished."""

    eda_id: str
    kind: str = ""
    status: str = "running"     # "running" | "error"
    stage: str = ""
    errors: str = ""


class EdaStatusResponse(BaseModel):
    """Live status of one pod/job, polled by the block while it runs."""

    eda_id: str
    kind: str = ""
    phase: str = ""             # creating | pulling_image | ready | running | done | error
    phase_detail: str = ""
    stage: str = ""             # verilate/build/sim, or an ORFS stage
    stage_detail: str = ""      # "running" | "done" | "failed"
    stages_done: list = []      # stages completed so far, in order
    log_tail: str = ""          # last few hundred lines of tool output
    elapsed_s: float = 0.0      # seconds since the job started (0 when not running)
    done: bool = False          # True once a result is available at /result
    pod_id: str = ""
    pod_status: str = ""        # "running" | "pending"
    uptime_s: float = 0.0
    warm_until: float = 0.0
    usd_per_hr: float = 0.0
    cost_estimate_usd: float = 0.0


class EdaResultResponse(BaseModel):
    """
    The finished run, fetched once ``/status`` reports ``done``.

    ``outputs`` is a flat ``port_name -> text`` map that the Qt executor writes
    straight onto the block's output ports; ``artifacts`` carries the binary and
    oversized files (GDS, DEF, VCD, layout PNG) as base64 for the client to save
    and upload.  One model serves all three kinds because the block, not the
    server, decides which ports exist.
    """

    eda_id: str
    kind: str = ""
    status: str = "ok"          # "ok" | "error" | "running"
    stage: str = ""             # the last stage reached
    done: bool = True
    outputs: dict = {}          # port_name -> text value
    artifacts: list = []        # [{path, size, content, b64, truncated}]
    errors: str = ""
    warnings: str = ""
    log: str = ""
    usd_per_hr: float = 0.0
    cost_estimate_usd: float = 0.0


class EdaSummary(BaseModel):
    """An EDA entry as returned by the list endpoint (no secrets echoed back)."""

    eda_id: str
    kind: str = ""
    name: str = ""
    pdk: str = ""
    instance_type: str = ""
    pod_id: str = ""
    pod_status: str = ""
    phase: str = ""
    stage: str = ""
    uptime_s: float = 0.0
    usd_per_hr: float = 0.0
    warm_until: float = 0.0


class EdaInstance(BaseModel):
    """One selectable machine type for the creation-dialog dropdown."""

    id: str                     # RunPod instance id (passed back in EdaSpec.instance_type)
    label: str = ""
    usd_per_hr: float = 0.0     # advisory hourly rate, 0 = unknown
    compute_type: str = "CPU"


class EdaInstancesResponse(BaseModel):
    instances: list[EdaInstance] = []


class EdaPdksResponse(BaseModel):
    pdks: list[str] = []
    default: str = DEFAULT_PDK
