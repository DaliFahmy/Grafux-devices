"""
pod_client.py
Thin wrapper over the RunPod REST API + SSH for the EDA runtime, isolating the
cloud vendor from the rest of the package.

This is a deliberate sibling of ``GPU/runpod_client.py`` rather than an import of
it.  ``device/app.py`` mounts the GPU and EDA routers in independent try/except
blocks precisely so one can fail without taking the other down, and a cross-package
import would couple them again — it would also leave the env vars ``GPU_``-named
and leak GPU-flavored error text ("point the gpu block's image port at a CUDA
-devel image") into EDA errors.  Once both consumers have settled, the genuinely
shared half belongs in a ``cloudpods`` package; until then the duplication is the
cheaper mistake.

Two things differ substantively from the GPU client:

1. ``create_pod`` is compute-type aware.  EDA tools are CPU-bound, so the default
   pod is a RunPod CPU instance; the GPU path is kept intact as a fallback.
2. ``exec_stream`` streams output line by line.  The GPU client's ``_exec`` blocks
   on ``stdout.read()``, which is fine for a 2-minute compile and useless for a
   45-minute route where the user needs to see progress.

Both ``httpx`` and ``paramiko`` are imported lazily so the devices server still
boots on a host where either is absent — a friendly RuntimeError is raised only
when an EDA endpoint is actually used.
"""

from __future__ import annotations

import io
import logging
import os
import shlex
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("eda.pod")

REST_BASE = "https://rest.runpod.io/v1"

# How long to wait, per attempt, for a pod to come up with a public SSH endpoint.
# Far more generous than the GPU default: the EDA image carries a full PDK and
# toolchain, so a cold-machine pull is measured in many minutes.
_PROVISION_TIMEOUT = int(os.environ.get("EDA_PROVISION_TIMEOUT", "900") or "900")

# A public IP + port-22 NAT mapping is a property of the *machine* a pod lands on,
# assigned at placement — well before the image finishes pulling.  So once a pod is
# placed, if no public endpoint has appeared within this many seconds, that machine
# has no public-IP networking and never will: fail fast so the caller can hop to a
# different machine instead of burning the whole pull timeout.  0 disables.
_PUBLIC_IP_GRACE = int(os.environ.get("EDA_PUBLIC_IP_GRACE", "180") or "180")

# Caps on artifact bytes pulled back per file and in total.  Much larger than the
# GPU block's: a sky130 GDS for a small design is already several MB.
_ARTIFACT_MAX_FILE_BYTES = int(
    os.environ.get("EDA_ARTIFACT_MAX_FILE_BYTES", str(32 * 1024 * 1024))
)
_ARTIFACT_MAX_TOTAL_BYTES = int(
    os.environ.get("EDA_ARTIFACT_MAX_TOTAL_BYTES", str(64 * 1024 * 1024))
)

# Prepended to every remote command.  A non-login SSH exec does not inherit the
# image's Docker ENV, so the OpenROAD flow-scripts paths must be set explicitly —
# the exact bug class as the GPU client's _CUDA_ENV, and just as silent when wrong
# ("make: command not found" or a mysterious empty PLATFORM).
_EDA_ENV = (
    'export FLOW_HOME="${FLOW_HOME:-/OpenROAD-flow-scripts/flow}"; '
    'export PATH="/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:'
    '/OpenROAD-flow-scripts/tools/install/yosys/bin:/usr/local/bin:$PATH"; '
)

# The working directory inputs are staged into and results are read from.
WORK_DIR = "/workspace/grafux"

# Pod lifecycle states that mean the pod is not coming up.  RunPod's desiredStatus
# enum is RUNNING | EXITED | TERMINATED; EXITED is what an image that cannot be
# pulled lands in, and leaving it out is how such a pod used to burn the whole
# _PUBLIC_IP_GRACE and then get blamed on the machine's networking.  An EDA pod
# never exits on its own — start.sh runs sshd in the foreground — so EXITED here
# always means failure.
_TERMINAL_STATUSES = ("TERMINATED", "FAILED", "EXITED")


class ProvisionError(RuntimeError):
    """
    Base for *retryable* provisioning failures — ones a fresh placement may fix.

    ``runtime.provision_eda`` catches this (and only this) to terminate the pod and
    retry on a new machine; any other exception is treated as fatal.
    """


class CapacityError(ProvisionError):
    """RunPod had no instances available for the requested type / cloud tier."""


class NoEndpointError(ProvisionError):
    """A pod came up but never exposed a public SSH endpoint (machine has no public IP)."""


class ImagePullError(RuntimeError):
    """
    The pod's container image cannot be pulled — a broken *configuration*, not a
    placement failure.

    Deliberately NOT a ProvisionError.  A fresh machine pulls the same missing or
    private image and fails identically, so ``provision_eda`` must not hop: falling
    through to its generic ``except Exception`` gives one terminate and an immediate
    fatal error carrying the fix, instead of three pods and a misleading
    "no public-IP networking" message several minutes later.
    """


# Substrings (case-insensitive) in a RunPod create error body that mean the failure
# is a transient *placement* problem a fresh attempt on another machine can fix.
_CAPACITY_MARKERS = (
    "no instances",
    "does not have the resources",
    "try a different machine",
    "no longer any instances",
    "not enough free",
    "out of capacity",
    "insufficient capacity",
)


def _is_capacity_error(text: str) -> bool:
    """True if a RunPod create error body indicates a retryable placement failure."""
    low = (text or "").lower()
    return any(marker in low for marker in _CAPACITY_MARKERS)


# Substrings (case-insensitive) that mean the *image* cannot be pulled.  Unlike a
# capacity error this is never fixed by a fresh placement — every machine pulls the
# same missing or private image — so it must be classified separately and fatally.
# Sources: RunPod's own pod lifecycle text ("IMAGE_AUTH_ERROR", "problem pulling
# the image"), Docker/containerd registry errors, and Kubernetes-style pull states.
_IMAGE_ERROR_MARKERS = (
    "image_auth_error",
    "imagepullbackoff",
    "errimagepull",
    "manifest unknown",
    "was not found on the registry",
    "problem pulling the image",
    "invalid reference format",
    "pull access denied",
    "repository does not exist",
    "unauthorized: authentication required",
)

# "denied" is the tail of the registry's "denied: requested access to the resource
# is denied" — but it is ALSO how RunPod words a rejection of our own API key, and
# telling someone with a bad rp_... key to go change a GHCR package's visibility
# sends them a long way in the wrong direction.  So these are only trusted when the
# same text is visibly talking about an image.
_IMAGE_ERROR_WEAK_MARKERS = ("denied", "not found")
_IMAGE_CONTEXT_MARKERS = ("image", "pull", "registry", "manifest", "repository", "docker")


def _is_image_error(text: str, image: str = "") -> bool:
    """
    True if ``text`` describes a failure to pull the pod image.

    ``image``, when given, widens the weak tier: its repository path appearing in
    the text is itself proof the message is about the image.
    """
    low = (text or "").lower()
    if not low:
        return False
    if any(marker in low for marker in _IMAGE_ERROR_MARKERS):
        return True
    if any(marker in low for marker in _IMAGE_ERROR_WEAK_MARKERS):
        context = list(_IMAGE_CONTEXT_MARKERS)
        if image:
            # Tag-insensitive: the reason text may name the repo without the tag.
            context.append(image.lower().split(":")[0])
        return any(marker in low for marker in context)
    return False


def _pod_image(pod: Dict[str, Any]) -> str:
    """The image a pod payload reports (REST v1 says ``image``; be tolerant)."""
    return str(pod.get("image") or pod.get("imageName") or "")


def _pod_image_error(pod: Dict[str, Any]) -> str:
    """
    The pod's ``lastStatusChange`` when it describes a pull failure, else "".

    ``lastStatusChange`` is RunPod's "text describing final lifecycle event" — it is
    where IMAGE_AUTH_ERROR shows up, and the only place the reason is available.
    """
    reason = str(pod.get("lastStatusChange") or "")
    return reason if _is_image_error(reason, _pod_image(pod)) else ""


# RunPod's CPU flavour families, from the REST v1 schema's ``cpuFlavorIds`` enum.
# The suffix is the family's RAM-per-vCPU ratio: c = compute (2 GB), g = general
# (4 GB), m = memory (8 GB); the digit is the hardware generation.
CPU_FLAVOR_FAMILIES = ("cpu3c", "cpu3g", "cpu3m", "cpu5c", "cpu5g", "cpu5m")

# Curated machine list for the creation-dialog dropdown.  ``id`` is
# ``<family>-<vcpus>`` — a Grafux-side convention, NOT something RunPod accepts:
# ``create_pod`` splits it into the ``cpuFlavorIds`` family and the separate
# ``vcpuCount`` field.  Keeping it as one string means the block needs only one
# port and the dialog only one combo box.
#
# Compute-optimised is the default because synthesis and place-and-route are
# CPU-bound and not especially memory-hungry at these design sizes.
# ``usd_per_hr`` is an *advisory* reference price, not billing-authoritative — the
# real rate is the live pod costPerHr.
EDA_INSTANCES: List[Dict[str, Any]] = [
    {"id": "cpu3c-4", "label": "Compute 4 vCPU / 8 GB", "usd_per_hr": 0.12},
    {"id": "cpu3c-8", "label": "Compute 8 vCPU / 16 GB", "usd_per_hr": 0.24},
    {"id": "cpu3c-16", "label": "Compute 16 vCPU / 32 GB", "usd_per_hr": 0.48},
    {"id": "cpu3g-8", "label": "General 8 vCPU / 32 GB", "usd_per_hr": 0.33},
    {"id": "cpu5c-8", "label": "Compute (gen 5) 8 vCPU / 16 GB", "usd_per_hr": 0.28},
    {"id": "cpu5c-16", "label": "Compute (gen 5) 16 vCPU / 32 GB", "usd_per_hr": 0.56},
]

_PRICE_BY_INSTANCE: Dict[str, float] = {
    i["id"]: float(i.get("usd_per_hr", 0.0)) for i in EDA_INSTANCES
}

# Fallback when the instance id carries no usable vCPU count.  4 is RunPod's
# smallest useful compute size; their own default of 2 is slow enough for a route
# to feel broken.
_DEFAULT_VCPUS = 4


def split_instance_type(instance_type: str) -> Tuple[str, int]:
    """
    Split a ``<family>-<vcpus>`` dropdown id into RunPod's two separate fields.

    Returns ``(cpuFlavorId, vcpuCount)``.  Tolerant on purpose — the value comes
    from a user-editable port, and a malformed one should degrade to a working
    default rather than fail the create:

        "cpu3c-8"  -> ("cpu3c", 8)
        "cpu3c"    -> ("cpu3c", 4)      # bare family, default size
        "cpu3c-4-8"-> ("cpu3c", 4)      # the old three-part form still parses
        ""         -> ("cpu3c", 4)
    """
    raw = (instance_type or "").strip().lower()
    if not raw:
        return CPU_FLAVOR_FAMILIES[0], _DEFAULT_VCPUS
    parts = raw.split("-")
    family = parts[0]
    if family not in CPU_FLAVOR_FAMILIES:
        logger.warning(
            "unknown CPU flavour %r; falling back to %s. Valid families: %s",
            family, CPU_FLAVOR_FAMILIES[0], ", ".join(CPU_FLAVOR_FAMILIES),
        )
        family = CPU_FLAVOR_FAMILIES[0]
    vcpus = _DEFAULT_VCPUS
    if len(parts) > 1:
        try:
            vcpus = int(parts[1])
        except ValueError:
            pass
    return family, max(1, vcpus)

# Advisory prices for the GPU fallback path (compute_type=GPU), so a cost estimate
# still appears if a deployment runs EDA on GPU pods.  Only the cheap end is listed
# — renting an H100 to run yosys would be a mistake worth not making easy.
_PRICE_BY_GPU: Dict[str, float] = {
    "NVIDIA RTX A4000": 0.17,
    "NVIDIA RTX 2000 Ada Generation": 0.23,
    "NVIDIA RTX A4500": 0.26,
    "NVIDIA GeForce RTX 3090": 0.43,
    "NVIDIA GeForce RTX 4090": 0.69,
}


def list_instances() -> List[Dict[str, Any]]:
    """Return the curated machine dropdown list (id + label + usd_per_hr)."""
    return list(EDA_INSTANCES)


def price_for(instance_type: str, compute_type: str = "CPU") -> float:
    """Advisory hourly price for an instance, or 0.0 if unknown."""
    key = (instance_type or "").strip()
    if (compute_type or "CPU").upper() == "GPU":
        return _PRICE_BY_GPU.get(key, 0.0)
    return _PRICE_BY_INSTANCE.get(key, 0.0)


def cost_per_hr_of(pod: Dict[str, Any]) -> float:
    """Live hourly rate from a pod payload (RunPod ``costPerHr``), or 0.0."""
    try:
        return float(pod.get("costPerHr") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def phase_from_pod(pod: Dict[str, Any]) -> Tuple[str, str]:
    """
    Derive a (phase, detail) for live status from a RunPod pod payload.

    creating       — placed but not yet RUNNING (still starting the container)
    pulling_image  — RUNNING but no public SSH endpoint yet (image still pulling)
    ready          — RUNNING with a public SSH endpoint
    error          — TERMINATED / FAILED / EXITED, or an image that cannot be pulled

    The image check comes first: an unpullable image otherwise reports
    ``pulling_image`` — technically true, permanently — so the block's live status
    sits on a spinner instead of showing the one thing that would fix it.
    """
    reason = _pod_image_error(pod)
    if reason:
        return "error", (
            f"image {_pod_image(pod)} could not be pulled — it must exist and be "
            f"PUBLIC (RunPod pulls anonymously): {reason.strip()[:160]}"
        )
    status = (pod.get("desiredStatus") or pod.get("status") or "").upper()
    if status in _TERMINAL_STATUSES:
        detail = f"pod entered status {status}"
        last = str(pod.get("lastStatusChange") or "").strip()
        return "error", (f"{detail}: {last[:160]}" if last else detail)
    if _ssh_endpoint(pod):
        return "ready", ""
    if status == "RUNNING" or pod.get("machineId"):
        return "pulling_image", "container starting / image pulling"
    return "creating", "placing the pod on a machine"


# ---------------------------------------------------------------------------
# Lazy dependency loaders
# ---------------------------------------------------------------------------

def _httpx():
    try:
        import httpx  # noqa: WPS433 (lazy import by design)
        return httpx
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "The 'httpx' package is required for the EDA runtime. Add httpx>=0.23 "
            "to requirements.txt."
        ) from exc


def _paramiko():
    try:
        import paramiko  # noqa: WPS433 (lazy import by design)
        return paramiko
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "The 'paramiko' package is required to run EDA tools. Add paramiko>=3.0 "
            "to requirements.txt."
        ) from exc


# ---------------------------------------------------------------------------
# SSH keypair
# ---------------------------------------------------------------------------

def generate_keypair() -> Tuple[str, str]:
    """
    Generate an ephemeral RSA keypair for a pod.

    Returns ``(private_key_pem, openssh_public_key)``.  The public key is injected
    via the ``PUBLIC_KEY`` env var, which the EDA image's start.sh appends to
    authorized_keys (see docker/Dockerfile — upstream ORFS does NOT do this).
    """
    paramiko = _paramiko()
    key = paramiko.RSAKey.generate(2048)
    buf = io.StringIO()
    key.write_private_key(buf)
    private_pem = buf.getvalue()
    public_openssh = f"{key.get_name()} {key.get_base64()} grafux-eda"
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
    Create an on-demand pod and return its id.

    ``spec`` is an EdaSpec.  Unlike the GPU client this branches on
    ``spec.compute_type``: EDA work is CPU-bound, so the default body asks for a
    RunPod CPU instance.  The GPU branch is kept byte-compatible with the gpu
    block's so ``EDA_COMPUTE_TYPE=GPU`` is a safe fallback if CPU pods misbehave.
    """
    httpx = _httpx()
    compute = (getattr(spec, "compute_type", "CPU") or "CPU").upper()
    body: Dict[str, Any] = {
        "name": f"grafux-eda-{(spec.kind or 'job')}-{(spec.name or '')}"[:60],
        "imageName": spec.image,
        "computeType": compute,
        "cloudType": (spec.cloud_type or "SECURE").upper(),
        "containerDiskInGb": int(spec.container_disk_gb or 60),
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        # Guarantee a public IP + TCP port mapping for SSH.  Without this, pods can
        # come up RUNNING with publicIp="" and portMappings=null, so we could never
        # SSH in.  (RunPod REST v1 field.)
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": public_key},
        "interruptible": False,
    }
    if compute == "GPU":
        body["gpuTypeIds"] = [spec.instance_type]
        body["gpuCount"] = 1
    else:
        # A CPU pod is selected by FLAVOUR FAMILY plus a separate vCPU count — not
        # by a compound instance id, and there is no ``instanceIds`` property in
        # REST v1 at all.  Sending one is not harmlessly ignored: the create fails
        # with a plain 400, which is not a capacity error, so provision_eda treats
        # it as fatal and does not even hop machines.
        flavor, vcpus = split_instance_type(spec.instance_type)
        body["cpuFlavorIds"] = [flavor]
        body["vcpuCount"] = vcpus

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{REST_BASE}/pods", headers=_headers(api_key), json=body)
    if resp.status_code not in (200, 201):
        text = resp.text or ""
        # An image RunPod cannot resolve is a permanently broken config, so it is
        # classified BEFORE capacity: RunPod answers both a bad image and genuine
        # scarcity with a 500, and a body that matches both must be fatal — hopping
        # machines over an unpullable image is pure spend.  401/403 are excluded on
        # purpose: those are OUR api key being rejected, and an "access denied" there
        # must never be blamed on the image.
        if resp.status_code in (400, 404, 422, 500) and _is_image_error(text, spec.image):
            raise ImagePullError(
                _image_error_message(
                    spec.image,
                    f"RunPod rejected the create ({resp.status_code}): {text.strip()[:300]}",
                )
            )
        # A 429/500 whose body matches one of these is a *placement* failure, not a
        # bad request — escaped by a fresh placement, so raise a retryable error and
        # let provision_eda hop to another machine instead of failing the Run.
        if resp.status_code in (429, 500) and _is_capacity_error(text):
            raise CapacityError(
                f"RunPod could not place a {compute} '{spec.instance_type}' pod right "
                f"now ({resp.status_code}): {text.strip()[:200]}"
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


def terminate_pod(api_key: str, pod_id: str) -> bool:
    """
    Terminate (delete) a pod, stopping all billing.  Never raises.

    Returns True when the pod is known to be gone (deleted, or already absent).
    Callers MUST NOT discard their handle on a False return: that is the only
    record of a pod that is still billing, so dropping it strands the pod until
    someone notices it in the RunPod console.
    """
    httpx = _httpx()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(f"{REST_BASE}/pods/{pod_id}", headers=_headers(api_key))
        if resp.status_code in (200, 204, 404):
            return True   # 404 = already gone, which is the outcome we wanted
        logger.warning(
            "RunPod terminate_pod %s returned %s: %s",
            pod_id, resp.status_code, resp.text[:200],
        )
        return False
    except Exception as exc:  # noqa: BLE001 — teardown must never raise
        logger.warning("RunPod terminate_pod %s error: %s", pod_id, exc)
        return False


def _ssh_endpoint(pod: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """
    Extract (public_ip, ssh_port) for port 22 from a pod payload.

    RunPod returns pod networking in several shapes depending on API version /
    machine, so we try them all:
      1. top-level ``portMappings`` dict: {"22": 40022} or {"22/tcp": 40022}
      2. nested ``runtime.ports`` list of {ip, isIpPublic, privatePort, publicPort}
      3. top-level ``ports`` list with the same per-entry fields

    Returns None until a public-IP TCP mapping for port 22 is available.
    """
    public_ip = pod.get("publicIp") or pod.get("ip")

    mappings = pod.get("portMappings")
    if isinstance(mappings, dict) and public_ip:
        ssh_port = mappings.get("22") or mappings.get("22/tcp")
        if ssh_port:
            try:
                return str(public_ip), int(ssh_port)
            except (TypeError, ValueError):
                pass

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


def _image_error_message(
    image: str,
    reason: str,
    pod_id: str = "",
    pod: Optional[Dict[str, Any]] = None,
) -> str:
    """
    One wording for an unpullable image, shared by create and wait.

    It has to say *both* halves of the contract, because either one alone produces
    the identical registry "denied": the tag must actually exist, and the package
    must be readable without credentials.
    """
    where = f"Pod {pod_id} " if pod_id else ""
    tail = f" {_status_summary(pod)}" if pod else ""
    return (
        f"{where}could not pull its image {image!r}: {reason.strip()[:300]} "
        "This is NOT a placement problem and retrying will not help. RunPod pulls "
        "ANONYMOUSLY, so the image must (a) exist at that exact tag — build and push "
        "it (EDA/docker/Dockerfile.verify for the verify image, EDA/docker/Dockerfile "
        "for the full EDA image; the 'Build verify image' GitHub Actions workflow does "
        "both steps) — and (b) be readable with no credentials: on GitHub open the "
        "package -> Package settings -> Change visibility -> PUBLIC. Or point the "
        "block's 'image' port (or EDA_VERIFY_IMAGE / EDA_DEFAULT_IMAGE on the devices "
        f"server) at an image that is already published.{tail}"
    )


def wait_until_ready(
    api_key: str,
    pod_id: str,
    *,
    timeout_s: Optional[int] = None,
    poll_s: float = 5.0,
) -> Tuple[str, int]:
    """
    Poll a pod until it is RUNNING with an SSH endpoint, returning (ip, port).

    Raises ImagePullError (fatal, never retried) as soon as the pod reports that it
    cannot pull its image, and NoEndpointError (retryable) on timeout, on a terminal
    pod state, or as soon as it is clear the machine has no public-IP networking.
    """
    if timeout_s is None:
        timeout_s = _PROVISION_TIMEOUT
    deadline = time.monotonic() + timeout_s
    placed_at: Optional[float] = None  # when we first saw the pod placed on a machine
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_pod(api_key, pod_id)
        # The image check runs FIRST, and on every poll rather than only on a
        # terminal status.  First, because RunPod writes IMAGE_AUTH_ERROR into
        # lastStatusChange at the same moment it sets desiredStatus=EXITED: if the
        # terminal branch below won the race the caller would get a *retryable*
        # NoEndpointError and provision_eda would hop machines three times over an
        # image that can never be pulled.  Every poll, because a backing-off pull can
        # sit at RUNNING and never reach a terminal status at all — and because this
        # way the failure is reported on the first poll instead of at _PUBLIC_IP_GRACE.
        image_reason = _pod_image_error(last)
        if image_reason:
            raise ImagePullError(
                _image_error_message(_pod_image(last), image_reason, pod_id, last)
            )
        status = (last.get("desiredStatus") or last.get("status") or "").upper()
        if status in _TERMINAL_STATUSES:
            raise NoEndpointError(
                f"Pod {pod_id} entered status {status}. {_status_summary(last)}"
            )
        endpoint = _ssh_endpoint(last)
        if endpoint:
            return endpoint
        # Early no-public-IP detection — see _PUBLIC_IP_GRACE.
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
    raise NoEndpointError(
        f"Pod {pod_id} not ready within {timeout_s}s — no public SSH endpoint appeared. "
        f"The EDA image is large; if it is still pulling, raise EDA_PROVISION_TIMEOUT. "
        f"If publicIp stays empty the placement has no public IP — Regenerate to retry "
        f"or set 'cloud_type' to 'SECURE'. If the image is a stock openroad/orfs build "
        f"it has no sshd at all and will NEVER become ready — use the derived grafux-eda "
        f"image. {_status_summary(last)}"
    )


# ---------------------------------------------------------------------------
# SSH: connect, exec (streaming), stage files, download artifacts
# ---------------------------------------------------------------------------

def connect_ssh(host: str, port: int, private_key_pem: str, *, timeout: float = 30.0):
    """
    Open an SSH client to a pod.

    Public (unlike the GPU client's ``_connect_ssh``) because an EDA run holds ONE
    session open across many stages — reconnecting per stage would multiply
    handshake latency and, worse, lose the shell's working state.
    """
    paramiko = _paramiko()
    key = paramiko.RSAKey.from_private_key(io.StringIO(private_key_pem))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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


def exec_simple(client, command: str, timeout: int) -> Tuple[int, str, str]:
    """Run a short command, returning (exit_code, stdout, stderr)."""
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def exec_stream(
    client,
    command: str,
    *,
    timeout: int,
    on_line: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    tail_lines: int = 400,
) -> Tuple[int, str, str]:
    """
    Run a long command, delivering stdout/stderr line by line as it happens.

    Returns ``(exit_code, stdout_tail, stderr_tail)`` — only the last
    ``tail_lines`` of each are retained, because an ORFS route emits tens of
    thousands of lines and the whole thing has no business sitting in memory or in
    a port file.  ``on_line`` receives every line as it arrives (that is what makes
    "5/6 Routing… detail iteration 7" possible); ``should_cancel`` is polled so a
    user pressing Stop kills the command instead of waiting out the timeout.

    An exit code of -1 means the command was cancelled; -2 means it exceeded
    ``timeout``.
    """
    from collections import deque

    channel = client.get_transport().open_session()
    channel.settimeout(1.0)
    channel.exec_command(command)

    out_tail: "deque[str]" = deque(maxlen=tail_lines)
    err_tail: "deque[str]" = deque(maxlen=tail_lines)
    out_buf = ""
    err_buf = ""
    deadline = time.monotonic() + max(1, timeout)
    cancelled = False
    timed_out = False

    def _drain(buf: str, sink: "deque[str]", is_err: bool) -> str:
        """Split a buffer into whole lines, record and report them, keep the rest."""
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            sink.append(line)
            if on_line and not is_err:
                on_line(line)
            elif on_line and is_err and line.strip():
                # Most EDA tools write progress to stderr, so surface it too.
                on_line(line)
        return buf

    while True:
        if channel.recv_ready():
            out_buf += channel.recv(65536).decode("utf-8", "replace")
            out_buf = _drain(out_buf, out_tail, False)
            continue
        if channel.recv_stderr_ready():
            err_buf += channel.recv_stderr(65536).decode("utf-8", "replace")
            err_buf = _drain(err_buf, err_tail, True)
            continue
        if channel.exit_status_ready() and not channel.recv_ready() \
                and not channel.recv_stderr_ready():
            break
        if should_cancel and should_cancel():
            cancelled = True
            break
        if time.monotonic() > deadline:
            timed_out = True
            break
        time.sleep(0.2)

    # Flush whatever did not end in a newline.
    for buf, sink in ((out_buf, out_tail), (err_buf, err_tail)):
        if buf.strip():
            sink.append(buf.rstrip("\r"))

    if cancelled or timed_out:
        try:
            channel.close()
        except Exception:  # noqa: BLE001 — the channel may already be gone
            pass
        return (-1 if cancelled else -2), "\n".join(out_tail), "\n".join(err_tail)

    code = channel.recv_exit_status()
    try:
        channel.close()
    except Exception:  # noqa: BLE001
        pass
    return code, "\n".join(out_tail), "\n".join(err_tail)


def sftp_makedirs(sftp, directory: str) -> None:
    """mkdir -p over SFTP — create each path component, ignoring 'already exists'."""
    parts = [p for p in directory.split("/") if p]
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.mkdir(cur)
        except Exception:  # noqa: BLE001 — exists or no perms; open() reports real failures
            pass


def stage_input_files(sftp, input_files: List[Dict[str, Any]]) -> None:
    """Write caller-supplied files into the pod before the run (best-effort dirs)."""
    import base64 as _b64
    for item in input_files or []:
        if not isinstance(item, dict):
            continue
        path = (item.get("path") or "").strip()
        if not path:
            continue
        content = item.get("content") or ""
        data = _b64.b64decode(content) if item.get("b64") else (
            content.encode("utf-8") if isinstance(content, str) else content
        )
        parent = path.rsplit("/", 1)[0]
        if parent and parent != path:
            sftp_makedirs(sftp, parent)
        with sftp.open(path, "wb") as fh:
            fh.write(data)


def download_artifacts(client, output_globs: List[str]) -> List[Dict[str, Any]]:
    """
    Resolve globs on-device and SFTP-read each match, capped in size.

    Returns a list of {path, size, content(base64), b64:true, truncated}.  Best-
    effort: a glob that matches nothing, or a file that can't be read, is skipped —
    which matters here because artifacts are collected even after a FAILED stage,
    so that a route failure still hands back the placement results.
    """
    import base64 as _b64
    globs = [g for g in (output_globs or []) if isinstance(g, str) and g.strip()]
    if not globs:
        return []
    # The EDA environment must be exported here too: the openroad globs are written
    # in terms of $FLOW_HOME, which a bare non-login shell does not have. Without
    # it every glob expands against an empty prefix, matches nothing, and a
    # successful run silently returns no GDS at all.
    listing_cmd = (
        "bash -lc " + shlex.quote(
            _EDA_ENV + "for f in " + " ".join(globs) + '; do [ -f "$f" ] && echo "$f"; done'
        )
    )
    _code, out, _err = exec_simple(client, listing_cmd, timeout=60)
    paths: List[str] = []
    for line in out.splitlines():
        p = line.strip()
        if p and p not in paths:
            paths.append(p)
    artifacts: List[Dict[str, Any]] = []
    total = 0
    sftp = client.open_sftp()
    try:
        for path in paths:
            try:
                with sftp.open(path, "rb") as fh:
                    data = fh.read(_ARTIFACT_MAX_FILE_BYTES + 1)
            except Exception:  # noqa: BLE001 — unreadable file, skip it
                continue
            truncated = len(data) > _ARTIFACT_MAX_FILE_BYTES
            data = data[:_ARTIFACT_MAX_FILE_BYTES]
            if total + len(data) > _ARTIFACT_MAX_TOTAL_BYTES:
                artifacts.append({"path": path, "size": len(data), "content": "",
                                  "b64": True, "truncated": True})
                continue
            total += len(data)
            artifacts.append({
                "path": path,
                "size": len(data),
                "content": _b64.b64encode(data).decode("ascii"),
                "b64": True,
                "truncated": truncated,
            })
    finally:
        sftp.close()
    return artifacts
