"""
manifest.py
The block manifest: the whole definition of a ``custom`` block, as data.

A native block type (openram, cpu, ...) is defined by Python here plus C++ in the
app.  A ``custom`` block is defined by ONE JSON document, usually written by the
Generator (Claude Code / Codex in a sandbox pod), and interpreted at run time by
``CUSTOM/flow.py``.  Nothing in Grafux is generated -- only the manifest and the
image it names -- so a bad generation can produce a broken block but never a
broken Grafux.

Schema v1 (every key other than those marked required has a default)::

    {
      "schema": 1,
      "slug": "openram",                      required, [a-z][a-z0-9_-]{1,47}
      "name": "OpenRAM", "version": "1.0.0", "description": "...",
      "source":  {"repo": "...", "ref": "stable", "commit": "d909181"},
      "runtime": {"compute": "cpu"|"gpu", "instance_type": "cpu3c-8",
                  "disk_gb": 30, "image": "ghcr.io/...", required
                  "timeout_s": 1800, "entry": "/opt/grafux/run.sh"},
      "inputs":  [{"name", "type", "default", "choices", "required", "description"}],
      "outputs": [{"name", "kind": "text"|"artifact", "glob", "inline_max", "description"}],
      "status":  {"require_outputs": ["gds"]},
      "selftest": {"inputs": {...}, "expect_outputs": [...]}
    }

THE IMAGE CONTRACT the runner relies on (the Generator's prompt states it too):

* sshd + the ``PUBLIC_KEY`` start.sh every pod here is reached through
  (``EDA/docker/start.sh``);
* ``runtime.entry`` (default ``/opt/grafux/run.sh``) reads one file per input
  port from ``$GRAFUX_IN/<port>`` and writes one file per TEXT output port to
  ``$GRAFUX_OUT/<port>``; files for ARTIFACT ports go under ``$GRAFUX_OUT/files/``
  and are matched by each port's ``glob`` (relative to ``$GRAFUX_OUT``);
* anything the entry needs in its environment lives in ``/etc/profile.d/`` --
  Docker ``ENV`` does not reach an SSH exec, ``bash -lc`` sources profile.d.

STANDARD PORTS are added by the runner and may not be declared: see
``RESERVED_INPUTS`` / ``RESERVED_OUTPUTS``.  They are the ports every pod-backed
block already has, so the app's shared EDA executor drives a custom block with
no special cases.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = 1

# Ports the runtime owns.  An input of one of these names would be sent to the
# pod spec, not to run.sh, and an output would be overwritten by the runtime --
# either way the manifest's meaning would silently not be what ran.
RESERVED_INPUTS = frozenset({
    "timeout", "instance_type", "image", "api_keys", "credentials", "files",
    "keep_warm_minutes", "manifest", "block_description",
})
RESERVED_OUTPUTS = frozenset({
    "status", "errors", "warnings", "log", "artifacts", "eda_id", "cost",
})

# The order the app lays the standard ports out in, after the manifest's own.
STANDARD_INPUTS = ("files", "timeout", "instance_type", "image", "api_keys")
STANDARD_OUTPUTS = ("status", "errors", "warnings", "log", "artifacts", "eda_id", "cost")

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{1,47}$")
# A port name is also a file name on the pod and a file name in the block
# folder, so it is held to the narrowest shape both accept.
_PORT_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
# An image reference: no whitespace, no shell metacharacters.  It reaches the
# RunPod API, not a shell, but it is also shown and copied -- keep it boring.
_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._\-/:@]{2,255}$")
# The entry is run inside `bash -lc`; an absolute path of plain characters only.
_ENTRY_RE = re.compile(r"^/[A-Za-z0-9._\-/]{1,200}$")
# A glob relative to $GRAFUX_OUT.  No `..`, no absolute paths: the runner quotes
# nothing inside it (it must expand), so its alphabet is what keeps it safe.
_GLOB_RE = re.compile(r"^[A-Za-z0-9._\-/*?\[\]]{1,200}$")

INLINE_MAX_CAP = 4 * 1024 * 1024
DEFAULT_INLINE_MAX = 1024 * 1024


class ManifestError(ValueError):
    """A manifest that cannot define a block.  The message names what to fix."""


class Source(BaseModel):
    repo: str = ""
    ref: str = ""
    commit: str = ""


class Runtime(BaseModel):
    compute: Literal["cpu", "gpu"] = "cpu"
    instance_type: str = ""
    disk_gb: int = Field(20, ge=5, le=500)
    image: str
    timeout_s: int = Field(1800, ge=10, le=24 * 3600)
    entry: str = "/opt/grafux/run.sh"

    @field_validator("image")
    @classmethod
    def _image_shape(cls, v: str) -> str:
        v = (v or "").strip()
        if not _IMAGE_RE.match(v):
            raise ValueError(
                "runtime.image must be a registry reference like "
                "ghcr.io/owner/name:tag (lowercase, no spaces)")
        return v

    @field_validator("entry")
    @classmethod
    def _entry_shape(cls, v: str) -> str:
        v = (v or "").strip() or "/opt/grafux/run.sh"
        if not _ENTRY_RE.match(v) or ".." in v:
            raise ValueError("runtime.entry must be an absolute path such as /opt/grafux/run.sh")
        return v


class InputPort(BaseModel):
    name: str
    type: Literal["text", "int", "float", "enum", "bool", "json", "file"] = "text"
    default: str = ""
    choices: List[str] = Field(default_factory=list)
    required: bool = False
    description: str = ""

    @field_validator("default", mode="before")
    @classmethod
    def _default_as_text(cls, v: Any) -> str:
        # Ports are text.  A generator that writes `"default": 2` means "2".
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    @field_validator("choices", mode="before")
    @classmethod
    def _choices_as_text(cls, v: Any) -> List[str]:
        return [str(c) for c in (v or [])]


class OutputPort(BaseModel):
    name: str
    kind: Literal["text", "artifact"] = "text"
    glob: str = ""
    inline_max: int = Field(DEFAULT_INLINE_MAX, ge=1, le=INLINE_MAX_CAP)
    description: str = ""

    @field_validator("glob")
    @classmethod
    def _glob_shape(cls, v: str) -> str:
        v = (v or "").strip()
        if v and (not _GLOB_RE.match(v) or v.startswith("/") or ".." in v):
            raise ValueError(
                "an output glob is relative to $GRAFUX_OUT, e.g. files/*.gds")
        return v


class StatusRule(BaseModel):
    require_outputs: List[str] = Field(default_factory=list)


class SelfTest(BaseModel):
    inputs: Dict[str, str] = Field(default_factory=dict)
    expect_outputs: List[str] = Field(default_factory=list)

    @field_validator("inputs", mode="before")
    @classmethod
    def _inputs_as_text(cls, v: Any) -> Dict[str, str]:
        return {str(k): ("" if x is None else x if isinstance(x, str) else json.dumps(x)
                         if isinstance(x, (dict, list)) else str(x))
                for k, x in (v or {}).items()}


class Manifest(BaseModel):
    schema_: int = Field(SCHEMA_VERSION, alias="schema")
    slug: str
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    source: Source = Field(default_factory=Source)
    runtime: Runtime
    inputs: List[InputPort] = Field(default_factory=list)
    outputs: List[OutputPort] = Field(default_factory=list)
    status: StatusRule = Field(default_factory=StatusRule)
    selftest: SelfTest = Field(default_factory=SelfTest)

    model_config = {"populate_by_name": True}

    @field_validator("slug")
    @classmethod
    def _slug_shape(cls, v: str) -> str:
        v = (v or "").strip()
        if not _SLUG_RE.match(v):
            raise ValueError("slug must match [a-z][a-z0-9_-]{1,47}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Manifest":
        if self.schema_ != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema {self.schema_}; this server speaks {SCHEMA_VERSION}")
        problems: List[str] = []

        def _ports(ports, reserved, side):
            seen = set()
            for p in ports:
                if not _PORT_RE.match(p.name):
                    problems.append(f"{side} port '{p.name}' must match [a-z][a-z0-9_]{{0,47}}")
                if p.name in reserved:
                    problems.append(
                        f"{side} port '{p.name}' is a standard port the runtime adds itself; rename it")
                if p.name in seen:
                    problems.append(f"{side} port '{p.name}' is declared twice")
                seen.add(p.name)
            return seen

        in_names = _ports(self.inputs, RESERVED_INPUTS, "input")
        out_names = _ports(self.outputs, RESERVED_OUTPUTS, "output")
        if not self.outputs:
            problems.append("a block with no outputs cannot report anything; declare at least one")

        for p in self.inputs:
            if p.type == "enum" and not p.choices:
                problems.append(f"enum input '{p.name}' needs choices")
            if p.choices and p.default and p.default not in p.choices:
                problems.append(f"input '{p.name}' default '{p.default}' is not one of its choices")
        for p in self.outputs:
            if p.kind == "artifact" and not p.glob:
                problems.append(f"artifact output '{p.name}' needs a glob, e.g. files/*.gds")

        for n in self.status.require_outputs:
            if n not in out_names:
                problems.append(f"status.require_outputs names '{n}', which is not an output")
        for n in self.selftest.inputs:
            if n not in in_names:
                problems.append(f"selftest.inputs names '{n}', which is not an input")
        for n in self.selftest.expect_outputs:
            if n not in out_names:
                problems.append(f"selftest.expect_outputs names '{n}', which is not an output")

        if problems:
            raise ValueError("; ".join(problems))
        return self

    # ---- helpers the runner and the app-facing API use -------------------

    def input_names(self) -> List[str]:
        return [p.name for p in self.inputs]

    def output_names(self) -> List[str]:
        return [p.name for p in self.outputs]

    def all_input_ports(self) -> List[str]:
        """The block's full input list: its own ports, then the standard ones."""
        return self.input_names() + list(STANDARD_INPUTS)

    def all_output_ports(self) -> List[str]:
        """The block's full output list: its own ports, then the standard ones."""
        return self.output_names() + list(STANDARD_OUTPUTS)


def parse_manifest(raw: Union[str, bytes, Dict[str, Any], Manifest, None]) -> Manifest:
    """
    Parse and validate a manifest from a JSON string or a dict.

    Raises ``ManifestError`` whose message lists EVERY problem found, one line
    each -- this text is fed straight back to the Generator as its repair order,
    so "field required" with no path would be useless to it.
    """
    if isinstance(raw, Manifest):
        return raw
    if raw is None or (isinstance(raw, (str, bytes)) and not raw.strip()):
        raise ManifestError("no manifest: a custom block needs its manifest to run")
    data: Any = raw
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
            msg = err.get("msg", "")
            msg = msg.removeprefix("Value error, ")
            lines.append(f"{loc}: {msg}" if loc else msg)
        raise ManifestError("invalid manifest:\n" + "\n".join(lines)) from exc


def json_schema() -> Dict[str, Any]:
    """The manifest's JSON Schema (served to the app and baked into the generator image)."""
    return Manifest.model_json_schema(by_alias=True)


def manifest_summary(m: Manifest) -> Dict[str, Any]:
    """What the app needs to lay a block out: both full port lists plus the pod defaults."""
    return {
        "slug": m.slug,
        "name": m.name or m.slug,
        "version": m.version,
        "description": m.description,
        "inputs": m.all_input_ports(),
        "outputs": m.all_output_ports(),
        "image": m.runtime.image,
        "compute": m.runtime.compute,
        "instance_type": m.runtime.instance_type,
        "timeout": str(m.runtime.timeout_s),
        "defaults": {p.name: p.default for p in m.inputs if p.default},
    }


__all__ = [
    "Manifest", "ManifestError", "parse_manifest", "json_schema", "manifest_summary",
    "RESERVED_INPUTS", "RESERVED_OUTPUTS", "STANDARD_INPUTS", "STANDARD_OUTPUTS",
    "SCHEMA_VERSION",
]
