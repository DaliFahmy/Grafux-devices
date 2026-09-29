"""
test_generator_contract.py
How Grafux judges the agent's files and what it adds to them.  The self-test
it generates is also EXECUTED here (on a POSIX host) against fake entries, since
it is the gate every generated image passes through.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from CUSTOM.manifest import parse_manifest  # noqa: E402
from GENERATOR import contract  # noqa: E402

MANIFEST = {
    "schema": 1, "slug": "demo", "name": "Demo",
    "runtime": {"compute": "cpu"},
    "inputs": [{"name": "n", "type": "int", "default": "3"}, {"name": "msg", "type": "text"}],
    "outputs": [{"name": "summary", "kind": "text"},
                {"name": "gds", "kind": "artifact", "glob": "files/*.gds"}],
    "status": {"require_outputs": ["gds"]},
    "selftest": {"inputs": {"msg": "it's \"quoted\" $HOME"}, "expect_outputs": ["summary", "gds"]},
}


def _files(**over):
    f = {
        "grafux-block.json": json.dumps(MANIFEST),
        "Dockerfile": "FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y openssh-server\n"
                      "COPY run.sh /opt/grafux/run.sh\n",
        "run.sh": "#!/bin/bash\necho hi\n",
    }
    f.update(over)
    return f


def test_valid_output_needs_no_image_from_the_agent():
    manifest, problems = contract.validate(_files())
    assert problems == [] and manifest.runtime.image == contract.PENDING_IMAGE


@pytest.mark.parametrize("files,needle", [
    (lambda f: f.pop("run.sh"), "run.sh is missing"),
    (lambda f: f.update({"grafux-block.json": "{"}), "not valid JSON"),
    (lambda f: f.update({"Dockerfile": "RUN true\n"}), "no FROM"),
    (lambda f: f.update({"Dockerfile": "FROM x\nCMD [\"x\"]\n"}), "must not set CMD"),
    (lambda f: f.update({"grafux-selftest.sh": "exit 0"}), "reserved"),
    (lambda f: f.update({"big.bin": "x" * (contract.MAX_FILE_BYTES + 1)}), "files must be under"),
])
def test_rejections_name_the_fix(files, needle):
    f = _files()
    files(f)
    _m, problems = contract.validate(f)
    assert any(needle in p for p in problems), problems


def test_selftest_must_cover_required_outputs():
    m = dict(MANIFEST, selftest={"inputs": {}, "expect_outputs": ["summary"]})
    _m, problems = contract.validate(_files(**{"grafux-block.json": json.dumps(m)}))
    assert any("must include every status.require_outputs" in p for p in problems)
    m2 = dict(MANIFEST, status={"require_outputs": []}, selftest={})
    _m, problems = contract.validate(_files(**{"grafux-block.json": json.dumps(m2)}))
    assert any("expect_outputs is empty" in p for p in problems)


def test_tag_is_content_addressed_and_ignores_the_image_field():
    a = contract.image_tag("Ahmed Fahmy!", "demo", _files())
    m = dict(MANIFEST, runtime={"compute": "cpu", "image": "ghcr.io/x/y:z"})
    b = contract.image_tag("Ahmed Fahmy!", "demo", _files(**{"grafux-block.json": json.dumps(m)}))
    c = contract.image_tag("Ahmed Fahmy!", "demo", _files(**{"run.sh": "#!/bin/bash\necho bye\n"}))
    assert a == b != c
    assert a.startswith("ahmed-fahmy-demo-") and len(a.rsplit("-", 1)[1]) == 12


def test_build_context_appends_the_trailer_and_grafux_files():
    files = _files()
    image = f"{contract.IMAGE_REPO}:t1"
    manifest_text = contract.finalize_manifest(files, image)
    assert json.loads(manifest_text)["runtime"]["image"] == image
    ctx = contract.build_context(files, manifest_text)
    docker = ctx["Dockerfile"].decode()
    assert docker.startswith("FROM ubuntu:22.04")
    assert docker.rstrip().endswith('CMD ["/start.sh"]')
    assert "/opt/grafux/grafux-selftest.sh" in docker
    assert set(ctx) >= {"run.sh", "grafux-block.json", "grafux-start.sh", "grafux-selftest.sh"}
    assert b"PUBLIC_KEY" in ctx["grafux-start.sh"]


def test_start_sh_creates_the_sshd_privsep_dir_before_sshd():
    """
    A generated Dockerfile only installs openssh-server, which does not create
    /run/sshd in a container; without it the pod's sshd dies at boot with
    "Missing privilege separation directory: /run/sshd".
    """
    start = contract.build_context(_files(), contract.finalize_manifest(
        _files(), f"{contract.IMAGE_REPO}:t1"))["grafux-start.sh"].decode()
    assert "mkdir -p /run/sshd" in start
    assert start.index("mkdir -p /run/sshd") < start.index("exec /usr/sbin/sshd")


def test_tag_changes_when_grafux_changes_its_half(monkeypatch):
    """A fix to start.sh/the trailer must not be hidden behind an old, cached tag."""
    a = contract.image_tag("o", "demo", _files())
    monkeypatch.setattr(contract, "_contract_salt", lambda: b"a newer start.sh")
    assert contract.image_tag("o", "demo", _files()) != a


def test_prompts():
    assert "{{EXAMPLE}}" not in contract.system_prompt()
    p = contract.first_prompt("an SRAM compiler", "https://github.com/x/y.git", "stable", "plan")
    assert "Do NOT write any files" in p and "ref stable" in p
    assert "Build the block now" in contract.first_prompt("x", "", "", "edit")
    assert "Do not weaken the selftest" in contract.repair_prompt("build", "boom")


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"),
                    reason="executes the generated POSIX self-test")
@pytest.mark.parametrize("entry_body,ok,needle", [
    ('cat "$GRAFUX_IN/msg" > "$GRAFUX_OUT/summary"; echo x > "$GRAFUX_OUT/files/a.gds"', True, "OK"),
    ('cat "$GRAFUX_IN/msg" > "$GRAFUX_OUT/summary"', False, "no file matched files/*.gds"),
    ('echo x > "$GRAFUX_OUT/files/a.gds"; exit 3', False, "must exit 0"),
])
def test_the_generated_selftest_really_judges(tmp_path, entry_body, ok, needle):
    entry = tmp_path / "entry.sh"
    entry.write_text("#!/bin/bash\n" + entry_body + "\n")
    entry.chmod(0o755)
    m = dict(MANIFEST, runtime={"image": "ghcr.io/x/y:z", "entry": str(entry)})
    script = contract.generate_selftest(parse_manifest(m))
    # sshd may be absent on a CI host: stub it onto PATH.
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "sshd").write_text("#!/bin/sh\n")
    (stub / "sshd").chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}")
    proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env)
    assert (proc.returncode == 0) == ok, proc.stdout + proc.stderr
    assert needle in proc.stdout
