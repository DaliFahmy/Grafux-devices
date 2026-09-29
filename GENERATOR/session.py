"""
session.py
One Generator session: an agent conversation and the loop that turns the
agent's files into a built block.  It rents NO RunPod pod -- only a block's
Regenerate does that (the cpu / openram / verilator rule).

    create ──► agent turn (a GitHub Actions job, actions.py)
                   │
      plan mode ◄──┤  (the turn's files are discarded; the plan is the answer)
      awaiting_user│
                   ▼ edit mode
               validate ──► build (GitHub Actions: docker build + Grafux self-test + push)
                  │              │
                  └── any failure: a repair turn (the failure text is the agent's
                      next prompt), at most MAX_REPAIRS; stop early if a repair
                      changed nothing
                   ▼
                 done  (result = manifest + image; the block's Regenerate rents its pod)

The agent never judges itself: validation and the build's self-test are
Grafux's, and a failure reaches the agent only as text.

Two seams, so the whole loop is tested with fakes: a runner
(``actions.ActionsAgentRunner``) and a builder (``builder.GitHubBuilder``).
Live output arrives through ``ingest`` (the job's forwarder POSTs it); whatever
did not arrive is replayed from the job's agent.jsonl when the turn ends.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import agents, contract
from .actions import TurnSpec
from .models import AGENT_PROVIDER, CLI_VERSIONS, callback_url, resolve_agent_key, seal

logger = logging.getLogger("generator.session")

MAX_REPAIRS = int(os.environ.get("GENERATOR_MAX_REPAIRS", "4"))
MAX_EVENTS = 5000

STATES = ("starting", "agent", "validating", "building", "awaiting_user", "done", "error", "stopped")
BUSY_STATES = ("starting", "agent", "validating", "building")

# Paths inside the agent job (agent.yml makes /workspace and puts these there).
KEYS_FILE = f"{contract.WORKSPACE}/.grafux/keys.env"
SYSTEM_FILE = f"{contract.WORKSPACE}/.grafux/system.md"
PROMPT_FILE = f"{contract.WORKSPACE}/.grafux/prompt.md"

_GH_TREE = re.compile(r"^(https://github\.com/[^/]+/[^/]+?)(?:\.git)?/(?:tree|blob)/([^?#]+?)/?$")


def parse_repo_url(url: str) -> Tuple[str, str]:
    """
    Split a pasted URL into (clone URL, ref).

    ``https://github.com/VLSIDA/OpenRAM/tree/stable`` -> (``.../OpenRAM.git``, ``stable``).
    A ref containing a slash is ambiguous in that form; the first segment is taken.
    """
    url = (url or "").strip()
    if not url:
        return "", ""
    m = _GH_TREE.match(url)
    if m:
        return m.group(1) + ".git", m.group(2).split("/")[0]
    if not re.match(r"^(https://|git@)[\w.@:/~+-]+$", url):
        raise ValueError("repo_url must be an https:// (or git@) repository URL")
    return url, ""


class GeneratorSession:
    def __init__(self, req, *, runner, builder=None, max_repairs: int = MAX_REPAIRS) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.agent = req.agent
        self.model = (req.model or agents.default_model(req.agent)).strip()
        self.owner = req.owner or "user"
        self.idea = req.prompt
        self.repo_url, self.ref = parse_repo_url(req.repo_url)
        # The user's key if they gave one, else Grafux's (GENERATOR_* then the
        # general server key) -- see models.resolve_agent_key.
        anthropic, anthropic_src = resolve_agent_key("anthropic", req.anthropic_api_key)
        openai, openai_src = resolve_agent_key("openai", req.openai_api_key)
        self._keys = {"ANTHROPIC_API_KEY": anthropic, "OPENAI_API_KEY": openai}
        self.key_source = {"anthropic": anthropic_src, "openai": openai_src}
        self._announced_key = False
        # Proves an /ingest POST came from THIS session's job; sealed into it.
        self.callback_token = secrets.token_urlsafe(24)
        self.runner, self.builder = runner, builder
        self.max_repairs = max_repairs
        self.state = "starting"
        self.events: List[Dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()
        self.cancel = threading.Event()
        self.agent_session_id = ""
        self.files: Dict[str, str] = {}
        self.state_files: Dict[str, bytes] = {}
        self.result: Optional[Dict[str, Any]] = None
        self.turns = 0
        self.cost_usd = 0.0
        self.created = time.time()
        self.updated = self.created
        self._thread: Optional[threading.Thread] = None
        # The live turn: its number, the last line number ingested, its parsed events.
        self._turn_no = 0
        self._ingested_n = -1
        self._turn_events: List[Dict[str, Any]] = []
        self._line_lock = threading.Lock()

    # ---- events ------------------------------------------------------------

    def _redact(self, text: str) -> str:
        for key in list(self._keys.values()) + [self.callback_token]:
            if key and len(key) > 8:
                text = text.replace(key, "***")
        return text

    def emit(self, kind: str, text: str = "", **data: Any) -> None:
        with self._lock:
            self._seq += 1
            safe = json.loads(self._redact(json.dumps(data))) if data else {}
            self.events.append({"seq": self._seq, "ts": time.time(), "kind": kind,
                                "text": self._redact(text or ""), "data": safe})
            if len(self.events) > MAX_EVENTS:
                del self.events[: len(self.events) - MAX_EVENTS]
            self.updated = time.time()

    def set_state(self, state: str, note: str = "") -> None:
        self.state = state
        self.emit("state", note or state, state=state)

    def events_after(self, after: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e["seq"] > after]

    def summary(self) -> Dict[str, Any]:
        return {"session_id": self.id, "agent": self.agent, "model": self.model, "state": self.state,
                "repo_url": self.repo_url, "ref": self.ref, "turns": self.turns,
                "cost_usd": round(self.cost_usd, 4), "verified": bool(self.result),
                "busy": self.busy, "last_seq": self._seq}

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- live output from the job -----------------------------------------

    def ingest(self, token: str, turn: int, lines: List[Dict[str, Any]]) -> int:
        """
        Accept numbered output lines from the running job.  Returns how many were
        new.  A wrong token raises PermissionError; a stale turn is ignored.
        """
        if not hmac.compare_digest(str(token or ""), self.callback_token):
            raise PermissionError("bad token")
        accepted = 0
        with self._line_lock:
            if turn != self._turn_no:
                return 0
            for item in sorted(lines or [], key=lambda x: int(x.get("n", -1))):
                n = int(item.get("n", -1))
                if n <= self._ingested_n:
                    continue
                self._ingested_n = n
                self._on_line(str(item.get("line", "")))
                accepted += 1
        return accepted

    def _on_line(self, line: str) -> None:
        for ev in agents.parse_line(self.agent, line):
            self._turn_events.append(ev)
            if ev["kind"] == "result":
                # The agent's final answer for this turn.  Emitted as `answer`:
                # `result` is reserved for "the block is built".
                self.cost_usd += float(ev["data"].get("cost_usd") or 0.0)
                if ev["text"]:
                    self.emit("answer", ev["text"], **ev["data"])
                continue
            self.emit(ev["kind"], ev["text"], **ev["data"])

    # ---- entry points (each runs a turn on a worker thread) ---------------

    def start(self, mode: str) -> None:
        # The panel shows what the user typed; the agent gets it wrapped in the task.
        self._spawn(contract.first_prompt(self.idea, self.repo_url, self.ref, mode), mode,
                    shown=self.idea)

    def message(self, text: str, mode: str) -> None:
        if self.busy:
            raise RuntimeError("the generator is still working on the last request")
        if self.state == "stopped":
            raise RuntimeError("this session was stopped; start a new one")
        self.cancel.clear()
        self._spawn(text, mode)

    def stop(self) -> None:
        # The running job (agent or build) sees cancel on its next poll and is
        # cancelled on GitHub; there is no pod to release.
        self.cancel.set()
        self.set_state("stopped", "stopped by the user")

    def _spawn(self, text: str, mode: str, shown: str = "") -> None:
        if mode not in agents.MODES:
            raise ValueError("mode must be plan or edit")
        self.emit("user", shown or text, mode=mode)
        if not self._announced_key:
            self._announced_key = True
            provider = AGENT_PROVIDER.get(self.agent, "anthropic")
            label = "Anthropic" if provider == "anthropic" else "OpenAI"
            src = self.key_source.get(provider, "")
            if src:
                whose = "your" if src == "user" else "Grafux's"
                self.emit("info", f"Using {whose} {label} key.", key_source=src, provider=provider)
        self._thread = threading.Thread(target=self._turn, args=(text, mode),
                                        name=f"generator-{self.id}", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---- the turn ----------------------------------------------------------

    def _turn(self, text: str, mode: str) -> None:
        try:
            if not self._agent_turn(text, mode):
                return
            if mode == "plan":
                plan = self._last_answer
                self.emit("plan", plan or "(the agent returned no plan text)")
                self.set_state("awaiting_user", "plan ready: reply to refine it, or switch to Edit to build")
                return
            self._verify_loop()
        except Exception as exc:  # noqa: BLE001 -- a dead worker would hang the panel
            logger.exception("generator %s: turn failed", self.id)
            self.emit("error", f"Grafux hit an internal error: {exc}")
            self.set_state("error")

    _last_answer = ""
    _last_binaries: List[str] = []

    def _agent_turn(self, text: str, mode: str) -> bool:
        provider = AGENT_PROVIDER.get(self.agent, "anthropic")
        key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        if not self._keys[key_name]:
            label, env = (("Anthropic", "GENERATOR_ANTHROPIC_API_KEY") if provider == "anthropic"
                          else ("OpenAI", "GENERATOR_OPENAI_API_KEY"))
            self.emit("error", f"No {label} key: add yours in the Generator settings (⚙), or ask "
                               f"the admin to set {env} on the devices server.")
            self.set_state("awaiting_user")
            return False

        with self._line_lock:
            self._turn_no += 1
            self._ingested_n = -1
            self._turn_events = []
        turn_no = self._turn_no
        self.set_state("agent", f"{self.agent} is working ({mode} mode) on GitHub Actions")
        command = agents.build_command(self.agent, mode=mode, model=self.model, prompt_file=PROMPT_FILE,
                                       system_file=SYSTEM_FILE, resume_id=self.agent_session_id,
                                       cwd=contract.WORKSPACE, env_file=KEYS_FILE)
        cb = callback_url()
        sealed = seal({
            key_name: self._keys[key_name],
            "callback_url": f"{cb}/generator/sessions/{self.id}/ingest" if cb else "",
            "callback_token": self.callback_token,
        })
        spec = TurnSpec(session_id=self.id, turn=turn_no, command=command, prompt=text,
                        system=contract.system_prompt(), mode=mode, agent=self.agent,
                        repo_url=self.repo_url, ref=self.ref,
                        cli_versions={self.agent: CLI_VERSIONS.get(self.agent, "")},
                        files=dict(self.files), state=dict(self.state_files), sealed=sealed)
        def on_status(step: str, url: str = "") -> None:
            if url:
                self.emit("build_log", step, url=url, phase="agent")
            else:
                self.emit("build_log", step, phase="agent")

        out = self.runner.run_turn(spec, on_status=on_status, should_cancel=self.cancel.is_set)

        # Whatever the forwarder did not deliver live, from the job's full log.
        with self._line_lock:
            for n, line in enumerate(out.lines):
                if n > self._ingested_n:
                    self._ingested_n = n
                    self._on_line(line)
            turn_events = list(self._turn_events)
        self.turns += 1
        self.agent_session_id = agents.session_id_from(turn_events, self.agent_session_id)
        self._last_answer = agents.final_text(self.agent, turn_events)

        if self.cancel.is_set():
            return False
        if out.infra:
            self.emit("error", out.detail)
            self.set_state("error")
            return False
        if out.state or not out.detail:
            self.state_files = out.state
        elif out.detail:
            self.state_files = {}
            self.emit("warning", out.detail)
        # Plan turns must not change the block: only an Edit turn's files are kept.
        if mode == "edit":
            self.files = out.files
            self._last_binaries = out.binaries
        elif out.files != self.files:
            self.emit("warning", "The agent changed files during a Plan turn; they were discarded. "
                                 "Switch to Edit to let it write.")
        failed = not out.ok or any(e["kind"] == "result" and e["data"].get("is_error")
                                   for e in turn_events)
        if failed:
            why = self._last_answer or out.detail or f"exit code {out.exit_code}"
            self.emit("error", f"The agent's turn failed: {why}")
            self.set_state("awaiting_user")
            return False
        return True

    # ---- validate -> build -> repair ----------------------------------------

    def _verify_loop(self) -> None:
        last_failed = ""
        for attempt in range(self.max_repairs + 1):
            if self.cancel.is_set():
                return
            self.set_state("validating", "checking the generated files")
            files = self.files
            for path in sorted(files):
                self.emit("file_snapshot", path, path=path, size=len(files[path]))
            manifest, problems = contract.validate(files)
            problems += [f"{b} is a binary file; download or build it inside the Dockerfile instead."
                         for b in self._last_binaries]
            # Binaries are in the digest too: they never travel back to the agent,
            # so a repair that only deleted one leaves the text files identical,
            # and without this the fixed result would be mistaken for "no change".
            digest = contract.content_hash(files) + "|" + ",".join(sorted(self._last_binaries))
            if digest == last_failed:
                # Checked BEFORE building: re-running an identical build re-buys
                # the same failure (the verifyloop rule).
                self.emit("error", "The agent's repair did not change any file, so trying again would "
                                   "reproduce the same failure. Reply with guidance to continue.")
                self.set_state("awaiting_user")
                return
            stage, detail = "", ""
            if problems:
                stage, detail = "validate", "\n".join(f"- {p}" for p in problems)
            else:
                tag = contract.image_tag(self.owner, manifest.slug, files)
                image = f"{contract.IMAGE_REPO}:{tag}"
                manifest_text = contract.finalize_manifest(files, image)
                if self.builder is None:
                    self.emit("error", "The image builder is not configured on this server "
                                       "(GENERATOR_GITHUB_TOKEN / GENERATOR_BUILDS_REPO).")
                    self.set_state("error")
                    return
                self.set_state("building", f"building {image}")
                res = self.builder.build(tag, contract.build_context(files, manifest_text),
                                         on_status=lambda s: self.emit("build_log", s),
                                         should_cancel=self.cancel.is_set)
                if res.run_url:
                    self.emit("build_log", f"build run: {res.run_url}", url=res.run_url)
                if not res.ok and res.infra:
                    if not self.cancel.is_set():
                        self.emit("error", res.detail)
                        self.set_state("error")
                    return
                if not res.ok:
                    stage, detail = "build", res.detail
                else:
                    # The self-test ran the tool inside the image during the build.
                    # The first POD is the user's Regenerate on the placed block.
                    self.emit("build_log", f"image pushed: {image}", image=image)
                    self.result = {"manifest": manifest_text, "image": image, "tag": tag,
                                   "files": files, "verified": True, "summary": self._last_answer}
                    self.emit("result", f"Block '{manifest.name or manifest.slug}' is ready. "
                                        "Add it to the canvas; Regenerate on the block rents its machine.",
                              image=image, slug=manifest.slug)
                    self.set_state("done", "the block is built and its self-test passed")
                    return

            self.emit("test", f"{stage} failed", stage=stage, detail=detail[-4000:])
            last_failed = digest
            if attempt == self.max_repairs:
                self.emit("error", f"Still failing after {self.max_repairs} repairs. The last failure "
                                   f"is above; reply with guidance to continue.")
                self.set_state("awaiting_user")
                return
            self.emit("repair", f"asking the agent to fix the {stage} failure "
                                f"(repair {attempt + 1}/{self.max_repairs})", stage=stage)
            if not self._agent_turn(contract.repair_prompt(stage, detail), "edit"):
                return


# ---------------------------------------------------------------------------
# Registry of live sessions
# ---------------------------------------------------------------------------

class SessionRegistry:
    def __init__(self, idle_timeout_s: float = float(os.environ.get("GENERATOR_SESSION_IDLE_S", "3600"))):
        self._sessions: Dict[str, GeneratorSession] = {}
        self._lock = threading.Lock()
        self.idle_timeout_s = idle_timeout_s

    def add(self, s: GeneratorSession) -> None:
        with self._lock:
            self._sessions[s.id] = s
        self.sweep()

    def get(self, sid: str) -> Optional[GeneratorSession]:
        with self._lock:
            return self._sessions.get(sid)

    def remove(self, sid: str) -> Optional[GeneratorSession]:
        with self._lock:
            return self._sessions.pop(sid, None)

    def sweep(self) -> None:
        """Drop sessions idle past the timeout (nothing to tear down: no pods)."""
        now = time.time()
        with self._lock:
            stale = [s for s in self._sessions.values()
                     if not s.busy and now - s.updated > self.idle_timeout_s]
            for s in stale:
                self._sessions.pop(s.id, None)


sessions = SessionRegistry()
