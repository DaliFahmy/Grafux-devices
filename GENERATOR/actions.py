"""
actions.py
Runs one agent TURN as a GitHub Actions job -- no RunPod pod, ever.

Pods are rented only when a user presses Regenerate on a block (the cpu /
openram / verilator rule).  The agent CLIs cannot run on the devices server
(Render, Python runtime, no Node), so each turn is a job in the builds repo
(``builds-repo/.github/workflows/agent.yml``):

    server                                        Actions job
    ------                                        -----------
    commit branch session/<sid>:                  checkout (no GITHUB_TOKEN kept)
      gen/**            the block so far          unseal keys + callback (masked)
      .grafux/state/**  CLI transcripts (resume)  install the pinned CLI
      .grafux/run.sh    the exact command         clone the upstream repo -> /workspace/src
      .grafux/prompt.md / system.md / turn.json   run.sh | forwarder --POST--> /ingest (live)
    dispatch agent.yml {session_id, turn, sealed} commit gen/, state, agent.jsonl back
    poll the run, then read the branch back  <--  push session/<sid>

``sealed`` is Fernet-encrypted with GENERATOR_WRAP_KEY (a secret on both sides):
workflow inputs are readable by anyone who can read the repo, so the model key
and the callback token never travel in clear.

The command in run.sh is built HERE by ``agents.build_command`` -- the workflow
just executes it -- so there is one definition of how each CLI is invoked.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import contract
from .builder import GitHubBuilder

logger = logging.getLogger("generator.actions")

AGENT_WORKFLOW = os.environ.get("GENERATOR_AGENT_WORKFLOW", "agent.yml")
TURN_TIMEOUT_S = float(os.environ.get("GENERATOR_TURN_TIMEOUT_S", "3900"))

GEN_PREFIX = "gen/"
STATE_PREFIX = ".grafux/state/"
OUT_PREFIX = ".grafux/out/"
# Transcripts grow every turn; past this, resume is dropped rather than pushing
# tens of MB through the git API on every turn.
MAX_STATE_BYTES = 16 * 1024 * 1024


@dataclass
class TurnSpec:
    session_id: str
    turn: int
    command: str                       # the agent command line (-> .grafux/run.sh)
    prompt: str
    system: str
    mode: str
    agent: str
    repo_url: str = ""
    ref: str = ""
    cli_versions: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)       # gen/ contents
    state: Dict[str, bytes] = field(default_factory=dict)     # .grafux/state/ contents
    sealed: str = ""


@dataclass
class TurnOutcome:
    ok: bool
    infra: bool = False                # Grafux/GitHub failed, not the agent
    detail: str = ""
    exit_code: Optional[int] = None
    files: Dict[str, str] = field(default_factory=dict)
    binaries: List[str] = field(default_factory=list)
    state: Dict[str, bytes] = field(default_factory=dict)
    lines: List[str] = field(default_factory=list)            # the whole agent.jsonl
    run_url: str = ""


def turn_title(session_id: str, turn: int) -> str:
    """The workflow's run-name -- how the dispatched run is found again."""
    return f"agent {session_id}-{turn}"


def turn_branch(session_id: str) -> str:
    return f"session/{session_id}"


def build_turn_tree(spec: TurnSpec) -> Dict[str, bytes]:
    """Everything the job needs, as the branch's whole tree."""
    tree: Dict[str, bytes] = {}
    for path, text in spec.files.items():
        tree[GEN_PREFIX + path] = text.encode("utf-8")
    for path, data in spec.state.items():
        tree[STATE_PREFIX + path] = data
    tree[".grafux/prompt.md"] = spec.prompt.encode("utf-8")
    tree[".grafux/system.md"] = spec.system.encode("utf-8")
    tree[".grafux/run.sh"] = ("#!/bin/bash\n" + spec.command + "\n").encode("utf-8")
    tree[".grafux/turn.json"] = json.dumps({
        "session_id": spec.session_id, "turn": spec.turn, "mode": spec.mode, "agent": spec.agent,
        "repo_url": spec.repo_url, "ref": spec.ref, "cli_versions": spec.cli_versions,
    }, indent=2).encode("utf-8")
    return tree


class ActionsAgentRunner:
    """Runs a turn as an Actions job; the GitHub plumbing is ``GitHubBuilder``'s."""

    def __init__(self, gh: GitHubBuilder, *, workflow: str = AGENT_WORKFLOW,
                 poll_s: Optional[float] = None, timeout_s: float = TURN_TIMEOUT_S) -> None:
        self.gh = gh
        self.workflow = workflow
        self.poll_s = gh.poll_s if poll_s is None else poll_s
        self.timeout_s = timeout_s

    # ---- read the branch back ---------------------------------------------

    def read_back(self, branch: str) -> Dict[str, bytes]:
        """The branch's files (gen/, state, out) as bytes, via the git data API."""
        r = f"/repos/{self.gh.repo}"
        ref = self.gh._json("GET", f"{r}/git/ref/heads/{branch}")
        commit = self.gh._json("GET", f"{r}/git/commits/{ref['object']['sha']}")
        tree = self.gh._json("GET", f"{r}/git/trees/{commit['tree']['sha']}",
                             params={"recursive": "1"})
        out: Dict[str, bytes] = {}
        for entry in tree.get("tree", []):
            path = entry.get("path", "")
            if entry.get("type") != "blob" or not path.startswith((GEN_PREFIX, STATE_PREFIX, OUT_PREFIX)):
                continue
            blob = self.gh._json("GET", f"{r}/git/blobs/{entry['sha']}")
            out[path] = base64.b64decode(blob.get("content", "")) if blob.get("encoding") == "base64" \
                else (blob.get("content", "") or "").encode("utf-8")
        return out

    def head_sha(self, branch: str) -> str:
        ref = self.gh._json("GET", f"/repos/{self.gh.repo}/git/ref/heads/{branch}")
        return ref["object"]["sha"]

    def _step_name(self, run_id: int) -> str:
        jobs = self.gh._json("GET", f"/repos/{self.gh.repo}/actions/runs/{run_id}/jobs").get("jobs", [])
        for job in jobs:
            for step in job.get("steps", []):
                if step.get("status") == "in_progress":
                    return step.get("name", "")
        return ""

    # ---- the turn -----------------------------------------------------------

    def run_turn(self, spec: TurnSpec, *, on_status: Callable[..., None] = lambda *a: None,
                 should_cancel: Callable[[], bool] = lambda: False) -> TurnOutcome:
        branch = turn_branch(spec.session_id)
        title = turn_title(spec.session_id, spec.turn)
        try:
            on_status("uploading the turn to GitHub")
            pushed = self.gh.push_context(branch, build_turn_tree(spec),
                                          f"Grafux generator {spec.session_id} turn {spec.turn}")
            on_status("starting the agent job")
            self.gh.dispatch_workflow(self.workflow, {
                "session_id": spec.session_id, "turn": str(spec.turn), "sealed": spec.sealed})
            run = self.gh.find_run_named(self.workflow, title)
            if run is None:
                return TurnOutcome(False, infra=True, detail=(
                    "The agent job was dispatched but never appeared. Check that "
                    f"{self.workflow} exists on the builds repo's default branch."))
            run_id, url = run["id"], run.get("html_url", "")
            if url:
                on_status("agent run: " + url, url)
            deadline = time.monotonic() + self.timeout_s
            last_step = ""
            while run.get("status") != "completed":
                if should_cancel():
                    self.gh.cancel(run_id)
                    return TurnOutcome(False, infra=True, detail="stopped", run_url=url)
                if time.monotonic() > deadline:
                    self.gh.cancel(run_id)
                    return TurnOutcome(False, infra=True, run_url=url,
                                       detail=f"The agent job exceeded {int(self.timeout_s)}s.")
                try:
                    step = self._step_name(run_id)
                except Exception:  # noqa: BLE001 -- status is cosmetic
                    step = ""
                if step and step != last_step:
                    on_status(step)
                    last_step = step
                self.gh._sleep(self.poll_s)
                run = self.gh.get_run(run_id)

            if self.head_sha(branch) == pushed:
                # The job never committed back: it died before or during setup,
                # which is ours (or GitHub's) to fix, not the agent's.
                return TurnOutcome(False, infra=True, run_url=url,
                                   detail="The agent job failed before it could run the agent:\n"
                                          + self.gh.failure_log(run_id))
            files_b = self.read_back(branch)
            out = TurnOutcome(ok=run.get("conclusion") == "success", run_url=url)
            for path, data in files_b.items():
                if path.startswith(GEN_PREFIX):
                    rel = path[len(GEN_PREFIX):]
                    if contract.is_junk(rel):
                        continue
                    try:
                        out.files[rel] = data.decode("utf-8")
                    except UnicodeDecodeError:
                        out.binaries.append(rel)
                elif path.startswith(STATE_PREFIX):
                    out.state[path[len(STATE_PREFIX):]] = data
            if sum(len(v) for v in out.state.values()) > MAX_STATE_BYTES:
                out.state = {}
                out.detail = "The agent's transcript grew too large to keep; the next turn starts fresh."
            jsonl = files_b.get(OUT_PREFIX + "agent.jsonl", b"").decode("utf-8", "replace")
            out.lines = jsonl.splitlines()
            code_text = files_b.get(OUT_PREFIX + "exit_code", b"").decode().strip()
            out.exit_code = int(code_text) if code_text.lstrip("-").isdigit() else None
            if out.exit_code not in (None, 0):
                out.ok = False
            return out
        except Exception as exc:  # noqa: BLE001 -- reported, never raised into the loop
            logger.warning("generator turn %s failed: %s", title, exc)
            return TurnOutcome(False, infra=True, detail=f"Grafux could not run the agent job: {exc}")
