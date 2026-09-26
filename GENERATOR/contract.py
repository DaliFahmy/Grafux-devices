"""
contract.py
What the agent must hand back, how Grafux judges it, and what Grafux adds.

THE RULE THAT SHAPES THIS FILE: the agent never touches its judge (see the
agent-contract-integrity decision).  The agent writes, under /workspace/gen:

    grafux-block.json   the manifest (runtime.image is Grafux's to set)
    Dockerfile          builds the tool and puts the adapter at runtime.entry
    run.sh              the adapter ($GRAFUX_IN/<port> -> $GRAFUX_OUT/<port>)
    ...                 anything else its Dockerfile COPYs

Grafux then, and only Grafux:

* validates the manifest with the SAME parser the runtime uses;
* picks the image tag (content-addressed, so a rebuild of identical files is
  recognisably identical);
* appends a TRAILER to the Dockerfile that installs the SSH entry point every
  pod needs and runs a SELF-TEST generated here from the manifest's
  ``selftest`` block.  An agent that could edit the self-test could make it
  pass; this one it never sees until the build log tells it what failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
from typing import Dict, List, Optional, Tuple

from CUSTOM.manifest import Manifest, ManifestError, parse_manifest

WORKSPACE = "/workspace"
SRC_DIR = f"{WORKSPACE}/src"
GEN_DIR = f"{WORKSPACE}/gen"
MANIFEST_FILE = "grafux-block.json"
REQUIRED_FILES = (MANIFEST_FILE, "Dockerfile", "run.sh")

# Names Grafux adds to the build context; the agent may not ship files by these
# names (they would be silently replaced, and the agent would debug the wrong file).
GRAFUX_START = "grafux-start.sh"
GRAFUX_SELFTEST = "grafux-selftest.sh"
RESERVED_FILES = (GRAFUX_START, GRAFUX_SELFTEST)

# The build context is committed to a git branch: keep it source-sized.
MAX_FILES = 200
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024

IMAGE_REPO = os.environ.get("GENERATOR_IMAGE_REPO", "ghcr.io/dalifahmy/grafux-gen")
PENDING_IMAGE = f"{IMAGE_REPO}:pending"

_START_SH = os.path.join(os.path.dirname(__file__), "..", "EDA", "docker", "start.sh")
_OWNER_RE = re.compile(r"[^a-z0-9]+")


def owner_slug(owner: str) -> str:
    """The owner part of a tag: lowercase alnum and dashes, at most 24 chars."""
    s = _OWNER_RE.sub("-", (owner or "user").lower()).strip("-")
    return (s or "user")[:24].strip("-") or "user"


def _safe_path(path: str) -> Optional[str]:
    p = posixpath.normpath((path or "").replace("\\", "/")).lstrip("/")
    if not p or p.startswith("..") or "/../" in f"/{p}/" or p.startswith(".git/") or p == ".git":
        return None
    return p


def with_pending_image(text: str) -> str:
    """
    Fill runtime.image if the agent left it out, so validation can run.

    The agent is told the image is Grafux's to set; a manifest without one is
    the expected shape, not a defect to send back for repair.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        rt = data.setdefault("runtime", {})
        if isinstance(rt, dict) and not (rt.get("image") or "").strip():
            rt["image"] = PENDING_IMAGE
    return json.dumps(data, indent=2)


def validate(files: Dict[str, str]) -> Tuple[Optional[Manifest], List[str]]:
    """
    Judge the agent's output.  Returns ``(manifest, problems)``.

    Every problem is a sentence the agent can act on -- this list IS the repair
    order sent back to it -- so each names the file and the fix.
    """
    problems: List[str] = []
    if len(files) > MAX_FILES:
        problems.append(f"/workspace/gen holds {len(files)} files; keep it under {MAX_FILES} "
                        "(the build clones the upstream repo itself -- do not vendor it).")
    total = 0
    for path, text in files.items():
        size = len(text.encode("utf-8"))
        total += size
        if size > MAX_FILE_BYTES:
            problems.append(f"{path} is {size} bytes; files must be under {MAX_FILE_BYTES}. "
                            "Download large assets inside the Dockerfile instead.")
        if _safe_path(path) is None:
            problems.append(f"'{path}' is not a path inside /workspace/gen.")
        if posixpath.basename(path) in RESERVED_FILES:
            problems.append(f"{path}: that name is reserved for the file Grafux adds; rename yours.")
    if total > MAX_TOTAL_BYTES:
        problems.append(f"/workspace/gen totals {total} bytes; keep it under {MAX_TOTAL_BYTES}.")

    for name in REQUIRED_FILES:
        if name not in files:
            problems.append(f"/workspace/gen/{name} is missing.")

    manifest = None
    if MANIFEST_FILE in files:
        try:
            manifest = parse_manifest(with_pending_image(files[MANIFEST_FILE]))
        except ManifestError as exc:
            problems.append(f"{MANIFEST_FILE}: {exc}")
    if manifest is not None:
        if not manifest.selftest.expect_outputs:
            problems.append(f"{MANIFEST_FILE}: selftest.expect_outputs is empty. Name the outputs a "
                            "small, fast run must produce (they are checked when the image is built).")
        missing = [n for n in manifest.status.require_outputs
                   if n not in manifest.selftest.expect_outputs]
        if missing:
            problems.append(f"{MANIFEST_FILE}: selftest.expect_outputs must include every "
                            f"status.require_outputs port; add {', '.join(missing)}.")
    docker = files.get("Dockerfile", "")
    if docker and not re.search(r"^\s*FROM\s", docker, re.IGNORECASE | re.MULTILINE):
        problems.append("Dockerfile has no FROM line.")
    if docker and re.search(r"^\s*(CMD|ENTRYPOINT)\s", docker, re.IGNORECASE | re.MULTILINE):
        problems.append("Dockerfile must not set CMD or ENTRYPOINT: Grafux appends the pod's own "
                        "SSH entry point, and a second one would leave the pod unreachable.")
    return manifest, problems


def content_hash(files: Dict[str, str]) -> str:
    """sha256 over the agent's files, excluding the image field Grafux itself sets."""
    h = hashlib.sha256()
    for path in sorted(files):
        text = files[path]
        if path == MANIFEST_FILE:
            try:
                data = json.loads(text)
                data.get("runtime", {}).pop("image", None)
                text = json.dumps(data, sort_keys=True)
            except (json.JSONDecodeError, AttributeError):
                pass
        h.update(path.encode() + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()


def image_tag(owner: str, slug: str, files: Dict[str, str]) -> str:
    """``<owner>-<slug>-<sha12>``: one tag per distinct build, in ONE package."""
    return f"{owner_slug(owner)}-{slug}-{content_hash(files)[:12]}"


def finalize_manifest(files: Dict[str, str], image: str) -> str:
    """The manifest the user's block will carry: the agent's, with Grafux's image."""
    data = json.loads(with_pending_image(files[MANIFEST_FILE]))
    data["runtime"]["image"] = image
    return json.dumps(data, indent=2)


def generate_selftest(manifest: Manifest) -> str:
    """
    A POSIX-sh self-test built from the manifest -- no python, no jq, so it runs
    on any base image the agent chose.  Inputs are the port defaults overlaid
    with selftest.inputs; each expected output must exist and be non-empty.
    """
    values = {p.name: p.default for p in manifest.inputs}
    values.update(manifest.selftest.inputs)
    lines = [
        "#!/bin/sh",
        "# Generated by Grafux from grafux-block.json. The agent never sees this file.",
        "set -u",
        'WORK=$(mktemp -d)',
        'export GRAFUX_WORK="$WORK" GRAFUX_IN="$WORK/in" GRAFUX_OUT="$WORK/out"',
        'mkdir -p "$GRAFUX_IN" "$GRAFUX_OUT/files"',
        "command -v sshd >/dev/null 2>&1 || [ -x /usr/sbin/sshd ] || "
        "{ echo 'GRAFUX SELFTEST FAILED: no sshd in the image (install openssh-server)'; exit 1; }",
        f"[ -x {shlex.quote(manifest.runtime.entry)} ] || "
        f"{{ echo 'GRAFUX SELFTEST FAILED: {manifest.runtime.entry} is missing or not executable'; exit 1; }}",
    ]
    for name, value in values.items():
        lines.append(f'printf %s {shlex.quote(value)} > "$GRAFUX_IN/{name}"')
    lines += [
        f'cd "$WORK" && bash -lc {shlex.quote(manifest.runtime.entry)}',
        "rc=$?",
        'echo "GRAFUX SELFTEST: entry exited $rc"',
        "fail=0",
    ]
    ports = {p.name: p for p in manifest.outputs}
    for name in manifest.selftest.expect_outputs:
        p = ports[name]
        if p.kind == "artifact":
            lines.append(
                f'found=0; for f in "$GRAFUX_OUT"/{p.glob}; do [ -s "$f" ] && found=1; done; '
                f'[ $found = 1 ] || {{ echo "GRAFUX SELFTEST FAILED: no file matched {p.glob} '
                f'for output \'{name}\'"; fail=1; }}')
        else:
            lines.append(
                f'[ -s "$GRAFUX_OUT/{name}" ] || {{ echo "GRAFUX SELFTEST FAILED: output '
                f'\'{name}\' was not written to \\$GRAFUX_OUT/{name}"; fail=1; }}')
    lines += [
        '[ "$rc" = 0 ] || { echo "GRAFUX SELFTEST FAILED: the entry must exit 0 on the selftest inputs"; '
        '[ -f "$GRAFUX_OUT/errors" ] && cat "$GRAFUX_OUT/errors"; fail=1; }',
        '[ $fail = 0 ] || exit 1',
        'echo "GRAFUX SELFTEST OK"',
        'rm -rf "$WORK"',
        "",
    ]
    return "\n".join(lines)


def dockerfile_trailer(manifest: Manifest) -> str:
    return "\n".join([
        "",
        "# ---- appended by Grafux: the pod contract and the self-test ----",
        f"COPY {MANIFEST_FILE} /opt/grafux/{MANIFEST_FILE}",
        f"COPY {GRAFUX_START} /start.sh",
        f"COPY {GRAFUX_SELFTEST} /opt/grafux/{GRAFUX_SELFTEST}",
        f"RUN chmod +x /start.sh /opt/grafux/{GRAFUX_SELFTEST} && /opt/grafux/{GRAFUX_SELFTEST}",
        "EXPOSE 22",
        'CMD ["/start.sh"]',
        "",
    ])


def build_context(files: Dict[str, str], manifest_text: str) -> Dict[str, bytes]:
    """Everything the builder commits: the agent's files plus Grafux's three."""
    manifest = parse_manifest(manifest_text)
    ctx: Dict[str, bytes] = {}
    for path, text in files.items():
        safe = _safe_path(path)
        if safe:
            ctx[safe] = text.encode("utf-8")
    ctx[MANIFEST_FILE] = manifest_text.encode("utf-8")
    ctx["Dockerfile"] = (files["Dockerfile"].rstrip() + "\n" + dockerfile_trailer(manifest)).encode("utf-8")
    with open(_START_SH, "rb") as fh:
        ctx[GRAFUX_START] = fh.read()
    ctx[GRAFUX_SELFTEST] = generate_selftest(manifest).encode("utf-8")
    return ctx


def is_executable(path: str) -> bool:
    return path.endswith(".sh") or posixpath.basename(path) in ("run", "entry")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "generator.md")
_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "CUSTOM", "examples", "openram")


def system_prompt() -> str:
    """The contract the agent works to, with the openram example inlined."""
    with open(_PROMPT_FILE, encoding="utf-8") as fh:
        text = fh.read()
    example = []
    for name in ("grafux-block.json", "Dockerfile", "run.sh", "adapter.py"):
        with open(os.path.join(_EXAMPLE_DIR, name), encoding="utf-8") as fh:
            body = fh.read()
        if name == MANIFEST_FILE:
            # Shown as the agent should write it: the image is Grafux's to set.
            data = json.loads(body)
            data.get("runtime", {}).pop("image", None)
            body = json.dumps(data, indent=2)
        elif name == "Dockerfile":
            # The hand-written example carries its own self-test and entry point;
            # a generated one gets Grafux's, so those lines are not shown.
            body = "\n".join(ln for ln in body.splitlines()
                             if not re.match(r"\s*(CMD|EXPOSE|RUN /opt/grafux/selftest)", ln))
        example.append(f"### {name}\n```\n{body.rstrip()}\n```")
    return text.replace("{{EXAMPLE}}", "\n\n".join(example))


def first_prompt(idea: str, repo_url: str, ref: str, mode: str) -> str:
    where = (f"The upstream project is cloned at {SRC_DIR} (from {repo_url}"
             + (f", ref {ref}" if ref else "") + ").") if repo_url else \
        f"No repository was given; {SRC_DIR} is empty."
    task = ("Read the project and PROPOSE the block: its input and output ports (names, types, "
            "defaults), the base image and build steps, how run.sh maps ports to the tool, and a "
            "small selftest. Do NOT write any files in this turn." if mode == "plan" else
            f"Build the block now: write {', '.join(REQUIRED_FILES)} and any files your Dockerfile "
            f"needs into {GEN_DIR}. Stop when they are complete; Grafux builds and tests the image.")
    return f"The user wants a Grafux block:\n\n{idea.strip()}\n\n{where}\n\n{task}"


def repair_prompt(stage: str, detail: str) -> str:
    heading = {
        "validate": "Your files in /workspace/gen were rejected before building:",
        "build": "The image build failed. The tail of the build log:",
    }.get(stage, "The last attempt failed:")
    return (f"{heading}\n\n{detail.strip()}\n\nFix the files in {GEN_DIR} (edit them in place) and "
            "stop. Do not weaken the selftest to make it pass -- make the block produce its outputs.")
