"""
test_custom_manifest.py
The manifest is the whole definition of a custom block, and its validation text
is fed back to the Generator as a repair order -- so these tests check both that
bad manifests are refused and that the refusal NAMES what to fix.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from CUSTOM.manifest import (  # noqa: E402
    STANDARD_INPUTS,
    STANDARD_OUTPUTS,
    ManifestError,
    json_schema,
    manifest_summary,
    parse_manifest,
)

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "CUSTOM", "examples", "openram",
                       "grafux-block.json")


def _minimal(**over):
    m = {
        "schema": 1,
        "slug": "demo",
        "runtime": {"image": "ghcr.io/dalifahmy/grafux-gen:demo-1"},
        "inputs": [{"name": "text", "type": "text"}],
        "outputs": [{"name": "result", "kind": "text"}],
    }
    m.update(over)
    return m


def test_the_openram_example_is_a_valid_manifest():
    m = parse_manifest(open(EXAMPLE, encoding="utf-8").read())
    assert m.slug == "openram"
    assert m.status.require_outputs == ["gds"]
    assert set(m.selftest.expect_outputs) <= set(m.output_names())


def test_the_openram_example_mirrors_the_native_openram_ports():
    """The custom block is judged against the native one: same design ports."""
    from EDA.models import OpenRamRunRequest
    m = parse_manifest(open(EXAMPLE, encoding="utf-8").read())
    native_run_fields = set(OpenRamRunRequest.model_fields) - {"timeout", "keep_warm_minutes",
                                                               "input_files"}
    assert set(m.input_names()) == native_run_fields


def test_a_minimal_manifest_gets_defaults_and_standard_ports():
    m = parse_manifest(_minimal())
    assert m.runtime.entry == "/opt/grafux/run.sh"
    assert m.runtime.compute == "cpu"
    assert m.all_input_ports() == ["text", *STANDARD_INPUTS]
    assert m.all_output_ports() == ["result", *STANDARD_OUTPUTS]


def test_non_string_defaults_are_ports_text():
    m = parse_manifest(_minimal(inputs=[
        {"name": "n", "type": "int", "default": 2},
        {"name": "b", "type": "bool", "default": True},
        {"name": "j", "type": "json", "default": {"a": 1}},
    ]))
    assert [p.default for p in m.inputs] == ["2", "true", '{"a": 1}']


@pytest.mark.parametrize("mutate,needle", [
    (lambda m: m.update(slug="Bad Slug"), "slug"),
    (lambda m: m["runtime"].update(image="not an image"), "runtime.image"),
    (lambda m: m["runtime"].update(entry="run.sh"), "absolute path"),
    (lambda m: m["inputs"].append({"name": "timeout"}), "standard port"),
    (lambda m: m["outputs"].append({"name": "status"}), "standard port"),
    (lambda m: m["inputs"].append({"name": "text"}), "declared twice"),
    (lambda m: m["inputs"].append({"name": "Has-Dash"}), "must match"),
    (lambda m: m["inputs"].append({"name": "e", "type": "enum"}), "needs choices"),
    (lambda m: m["inputs"].append({"name": "e", "type": "enum", "choices": ["a"], "default": "b"}),
     "not one of its choices"),
    (lambda m: m["outputs"].append({"name": "g", "kind": "artifact"}), "needs a glob"),
    (lambda m: m["outputs"].append({"name": "g", "kind": "artifact", "glob": "../etc/*"}),
     "relative to $GRAFUX_OUT"),
    (lambda m: m["outputs"].append({"name": "g", "kind": "artifact", "glob": "files/$(id)"}),
     "relative to $GRAFUX_OUT"),
    (lambda m: m.update(status={"require_outputs": ["nope"]}), "not an output"),
    (lambda m: m.update(selftest={"inputs": {"nope": "1"}}), "not an input"),
    (lambda m: m.update(outputs=[]), "at least one"),
    (lambda m: m.update(schema=2), "unsupported manifest schema"),
])
def test_bad_manifests_are_refused_with_a_message_naming_the_fix(mutate, needle):
    m = copy.deepcopy(_minimal())
    mutate(m)
    with pytest.raises(ManifestError) as exc:
        parse_manifest(m)
    assert needle in str(exc.value)


def test_every_problem_is_reported_at_once():
    """The generator repairs from this text; one error per round-trip wastes pods."""
    m = _minimal(outputs=[{"name": "status"}, {"name": "g", "kind": "artifact"}])
    with pytest.raises(ManifestError) as exc:
        parse_manifest(m)
    text = str(exc.value)
    assert "standard port" in text and "needs a glob" in text


@pytest.mark.parametrize("raw,needle", [
    ("", "no manifest"), ("{", "not valid JSON"), ("[1]", "JSON object"), (None, "no manifest"),
])
def test_unparseable_manifests(raw, needle):
    with pytest.raises(ManifestError) as exc:
        parse_manifest(raw)
    assert needle in str(exc.value)


def test_summary_and_schema():
    m = parse_manifest(_minimal(inputs=[{"name": "n", "type": "int", "default": "3"}]))
    s = manifest_summary(m)
    assert s["inputs"][0] == "n" and "api_keys" in s["inputs"]
    assert s["defaults"] == {"n": "3"}
    assert s["image"] == "ghcr.io/dalifahmy/grafux-gen:demo-1"
    schema = json_schema()
    assert "schema" in schema["properties"] and "runtime" in schema["required"]
    json.dumps(schema)
