"""
session.py
One Generator session: a sandbox pod, an agent conversation, and the loop that
turns the agent's files into a verified block.

    create ──► provision sandbox ──► clone repo ──► agent turn
                                                     │
                     plan mode ◄─────────────────────┤ (workspace reverted; the
                     (awaiting_user)                 │  plan is the answer)
                                                     ▼ edit mode
                     validate ──► build (GitHub) ──► smoke run (real pod)
                        │             │                  │
                        └──── any failure: a repair turn (the failure text is
                              the agent's next prompt), at most MAX_REPAIRS;
                              stop early if a repair changed nothing
                                                     ▼
                                                   done  (result = manifest + image)

The agent never judges itself: validation, the build's self-test and the smoke
run are all Grafux's, and a failure reaches the agent only as text.

The outside world is reached only through three seams, so the whole loop is
tested with fakes: a ``Sandbox`` (the pod), a builder (``builder.GitHubBuilder``)
and a smoke runner.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import shlex
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from CUSTOM.manifest import parse_manifest

from . import agents, contract

logger = logging.getLogger("generator.session")

MAX_REPAIRS = int(os.environ.get("GENERATOR_MAX_REPAIRS", "4"))
AGENT_TURN_TIMEOUT_S = int(os.environ.get("GENERATOR_TURN_TIMEOUT_S", "2700"))
SMOKE_ENABLED = os.environ.get("GENERATOR_SMOKE_RUN", "1").strip().lower() not in ("0", "false", "no", "off")
MAX_EVENTS = 5000

STATES = ("starting", "provisioning", "preparing", "agent", "validating", "building", "smoke",
          "awaiting_user", "done", "error", "stopped")
BUSY_STATES = ("starting", "provisioning", "preparing", "agent", "validating", "building", "smoke")

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


class Sandbox(Protocol):
    fresh: bool

    def ensure(self) -> str: ...
    def exec(self, cmd: str, timeout: int = 120) -> Tuple[int, str, str]: ...
    def stream(self, cmd: str, *, timeout: int, on_line: Callable[[str], None],
               should_cancel: Callable[[], bool]) -> Tuple[int, str, str]: ...
    def write_file(self, path: str, text: str, mode: int = 0o644) -> None: ...
    def read_tree(self, directory: str) -> Tuple[Dict[str, str], List[str]]: ...
    def keepalive(self) -> None: ...
    def terminate(self) -> None: ...


SmokeRunner = Callable[..., Tuple[bool, str, bool]]


class GeneratorSession:
    def __init__(self, req, *, sandbox: Sandbox, builder=None, smoke: Optional[SmokeRunner] = None,
                 max_repairs: int = MAX_REPAIRS) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.agent = req.agent
        self.model = (req.model or agents.default_model(req.agent)).strip()
        self.owner = req.owner or "user"
        self.idea = req.prompt
        self.repo_url, self.ref = parse_repo_url(req.repo_url)
        self.api_keys = req.api_keys
        self._keys = {"ANTHROPIC_API_KEY": (req.anthropic_api_key or "").strip(),
                      "OPENAI_API_KEY": (req.openai_api_key or "").strip()}
        self.sandbox, self.builder, self.smoke = sandbox, builder, smoke
        self.max_repairs = max_repairs
        self.state = "starting"
        self.events: List[Dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()
        self.cancel = threading.Event()
        self.agent_session_id = ""
        self.files: Dict[str, str] = {}
        self.result: Optional[Dict[str, Any]] = None
        self.turns = 0
        self.cost_usd = 0.0
        self.created = time.time()
        self.updated = self.created
        self._thread: Optional[threading.Thread] = None

    # ---- events ------------------------------------------------------------

    def _redact(self, text: str) -> str:
        for key in self._keys.values():
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
        self.cancel.set()
        self.set_state("stopped", "stopped by the user")
        try:
            self.sandbox.terminate()
        except Exception:  # noqa: BLE001
            logger.exception("generator %s: sandbox teardown failed", self.id)

    def _spawn(self, text: str, mode: str, shown: str = "") -> None:
        if mode not in agents.MODES:
            raise ValueError("mode must be plan or edit")
        self.emit("user", shown or text, mode=mode)
        self._thread = threading.Thread(target=self._turn, args=(text, mode),
                                        name=f"generator-{self.id}", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---- the turn ----------------------------------------------------------

    def _turn(self, text: str, mode: str) -> None:
        try:
            if not self._ensure_sandbox():
                return
            if not self._agent_turn(text, mode):
                return
            if mode == "plan":
                self._revert_plan_edits()
                plan = self._last_answer
                self.emit("plan", plan or "(the agent returned no plan text)")
                self.set_state("awaiting_user", "plan ready: reply to refine it, or switch to Edit to build")
                return
            self._verify_loop()
        except Exception as exc:  # noqa: BLE001 -- a dead worker would hang the panel
            logger.exception("generator %s: turn failed", self.id)
            self.emit("error", f"Grafux hit an internal error: {exc}")
            self.set_state("error")

    def _ensure_sandbox(self) -> bool:
        self.set_state("provisioning", "renting the sandbox machine")
        err = self.sandbox.ensure()
        if err:
            self.emit("error", err)
            self.set_state("error")
            return False
        if self.sandbox.fresh:
            self.set_state("preparing", "preparing the workspace")
            self._prepare()
        return True

    def _prepare(self) -> None:
        sb = self.sandbox
        sb.exec(f"mkdir -p {contract.SRC_DIR} {contract.GEN_DIR} {posixpath.dirname(KEYS_FILE)}")
        if self.repo_url:
            branch = f"--branch {shlex.quote(self.ref)} " if self.ref else ""
            code, out, err = sb.exec(
                f"git clone --depth 1 {branch}{shlex.quote(self.repo_url)} {contract.SRC_DIR} 2>&1",
                timeout=600)
            if code != 0:
                raise RuntimeError(f"could not clone {self.repo_url}: {(out or err).strip()[-800:]}")
            _c, sha, _e = sb.exec(f"git -C {contract.SRC_DIR} rev-parse HEAD")
            self.emit("tool", f"cloned {self.repo_url}" + (f" @ {self.ref}" if self.ref else "")
                      + (f" ({sha.strip()[:12]})" if sha.strip() else ""), tool="git")
        # Files from before a sandbox was reaped between turns come back.
        for path, body in self.files.items():
            sb.write_file(f"{contract.GEN_DIR}/{path}", body,
                          0o755 if contract.is_executable(path) else 0o644)
        sb.exec(f"cd {contract.GEN_DIR} && git init -q 2>/dev/null; "
                "git -c user.email=gen@grafux -c user.name=grafux add -A && "
                "git -c user.email=gen@grafux -c user.name=grafux commit -qm start --allow-empty")
        sb.write_file(SYSTEM_FILE, contract.system_prompt())
        env = "".join(f"{k}={shlex.quote(v)}\n" for k, v in self._keys.items() if v)
        sb.write_file(KEYS_FILE, env, 0o600)
        sb.fresh = False

    def _agent_turn(self, text: str, mode: str) -> bool:
        if self.agent == "claude_code" and not self._keys["ANTHROPIC_API_KEY"]:
            self.emit("error", "Claude Code needs your Anthropic API key (Generator settings).")
            self.set_state("awaiting_user")
            return False
        if self.agent == "codex" and not self._keys["OPENAI_API_KEY"]:
            self.emit("error", "Codex needs your OpenAI API key (Generator settings).")
            self.set_state("awaiting_user")
            return False
        self.set_state("agent", f"{self.agent} is working ({mode} mode)")
        self.sandbox.write_file(PROMPT_FILE, text)
        # A checkpoint, so a plan turn's stray edits can be undone exactly.
        self.sandbox.exec(f"cd {contract.GEN_DIR} && git -c user.email=gen@grafux -c user.name=grafux "
                          "add -A && git -c user.email=gen@grafux -c user.name=grafux commit -qm turn "
                          "--allow-empty")
        cmd = agents.build_command(self.agent, mode=mode, model=self.model, prompt_file=PROMPT_FILE,
                                   system_file=SYSTEM_FILE, resume_id=self.agent_session_id,
                                   cwd=contract.WORKSPACE, env_file=KEYS_FILE)
        turn_events: List[Dict[str, Any]] = []
        last_touch = [time.monotonic()]

        def on_line(line: str) -> None:
            if time.monotonic() - last_touch[0] > 30:
                self.sandbox.keepalive()
                last_touch[0] = time.monotonic()
            for ev in agents.parse_line(self.agent, line):
                turn_events.append(ev)
                if ev["kind"] == "result":
                    # The agent's final answer for this turn.  Emitted as `answer`:
                    # `result` is reserved for "the block is built and verified".
                    self.cost_usd += float(ev["data"].get("cost_usd") or 0.0)
                    if ev["text"]:
                        self.emit("answer", ev["text"], **ev["data"])
                    continue
                self.emit(ev["kind"], ev["text"], **ev["data"])

        stop_ticker = threading.Event()

        def ticker() -> None:     # an agent may think for minutes without printing
            while not stop_ticker.wait(60):
                self.sandbox.keepalive()

        t = threading.Thread(target=ticker, daemon=True)
        t.start()
        try:
            code, _out, err = self.sandbox.stream(
                "bash -lc " + shlex.quote(cmd), timeout=AGENT_TURN_TIMEOUT_S, on_line=on_line,
                should_cancel=self.cancel.is_set)
        finally:
            stop_ticker.set()
        self.turns += 1
        self.agent_session_id = agents.session_id_from(turn_events, self.agent_session_id)
        self._last_answer = agents.final_text(self.agent, turn_events)
        if self.cancel.is_set():
            return False
        failed = code != 0 or any(e["kind"] == "result" and e["data"].get("is_error") for e in turn_events)
        if failed:
            why = (err or "").strip()[-2000:] or self._last_answer or f"exit code {code}"
            self.emit("error", f"The agent's turn failed: {why}")
            self.set_state("awaiting_user")
            return False
        return True

    _last_answer = ""

    def _revert_plan_edits(self) -> None:
        code, out, _ = self.sandbox.exec(f"cd {contract.GEN_DIR} && git status --porcelain")
        if code == 0 and out.strip():
            self.sandbox.exec(f"cd {contract.GEN_DIR} && git reset -q --hard && git clean -fdq")
            self.emit("warning", "The agent changed files during a Plan turn; they were reverted. "
                                 "Switch to Edit to let it write.")

    # ---- validate -> build -> smoke -> repair -----------------------------

    def _verify_loop(self) -> None:
        last_failed = ""
        for attempt in range(self.max_repairs + 1):
            if self.cancel.is_set():
                return
            self.set_state("validating", "checking the generated files")
            files, binaries = self.sandbox.read_tree(contract.GEN_DIR)
            self.files = files
            for path in sorted(files):
                self.emit("file_snapshot", path, path=path, size=len(files[path]))
            manifest, problems = contract.validate(files)
            problems += [f"{b} is a binary file; download or build it inside the Dockerfile instead."
                         for b in binaries]
            digest = contract.content_hash(files)
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
                    self.emit("build_log", f"image pushed: {image}", image=image)
                    ok, why, infra = True, "", False
                    if self.smoke is not None:
                        self.set_state("smoke", "running the block once on a real pod")
                        ok, why, infra = self.smoke(manifest_text, api_keys=self.api_keys,
                                                    should_cancel=self.cancel.is_set,
                                                    on_status=lambda s: self.emit("test", s))
                    if ok:
                        self.result = {"manifest": manifest_text, "image": image, "tag": tag,
                                       "files": files, "verified": self.smoke is not None,
                                       "summary": self._last_answer}
                        self.emit("result", f"Block '{manifest.name or manifest.slug}' is ready.",
                                  image=image, slug=manifest.slug)
                        self.set_state("done", "the block is built and verified")
                        return
                    if infra:
                        self.emit("error", why)
                        self.set_state("error")
                        return
                    stage, detail = "smoke", why

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
# The real seams
# ---------------------------------------------------------------------------

class PodSandbox:
    """A RunPod pod from the generator image, held through EDA's registry."""

    def __init__(self, api_keys: str = "") -> None:
        self.api_keys = api_keys
        self.eda_id = ""
        self.fresh = True
        self._client = None

    def _record(self):
        from EDA.registry import registry
        return registry.get(self.eda_id) if self.eda_id else None

    def ensure(self) -> str:
        from EDA import runtime as eda_runtime
        from EDA.models import EdaSpec
        from .models import GENERATOR_DISK_GB, GENERATOR_IMAGE, GENERATOR_INSTANCE, image_published

        rec = self._record()
        if rec is not None and rec.public_ip:
            return ""
        if not image_published(GENERATOR_IMAGE):
            return (f"The generator sandbox image {GENERATOR_IMAGE} is not published (or not public). "
                    "Run the 'Build generator image' workflow in Grafux-devices and make the GHCR "
                    "package public, or set GENERATOR_IMAGE.")
        self._client = None
        self.fresh = True
        spec = EdaSpec(kind="generator", image=GENERATOR_IMAGE, instance_type=GENERATOR_INSTANCE,
                       container_disk_gb=GENERATOR_DISK_GB, api_keys=self.api_keys,
                       name="grafux-generator")
        res = eda_runtime.provision_eda(spec)
        if res.get("status") == "error" or not res.get("eda_id"):
            return f"Could not start the sandbox: {res.get('errors') or 'unknown error'}"
        self.eda_id = res["eda_id"]
        return ""

    def _ssh(self):
        from EDA import pod_client
        if self._client is None:
            rec = self._record()
            if rec is None:
                raise RuntimeError("the sandbox pod is gone")
            self._client = pod_client.connect_ssh(rec.public_ip, rec.ssh_port, rec.private_key_pem)
        return self._client

    def exec(self, cmd: str, timeout: int = 120) -> Tuple[int, str, str]:
        from EDA import pod_client
        return pod_client.exec_simple(self._ssh(), "bash -lc " + shlex.quote(cmd), timeout=timeout)

    def stream(self, cmd, *, timeout, on_line, should_cancel):
        from EDA import pod_client
        return pod_client.exec_stream(self._ssh(), cmd, timeout=timeout, on_line=on_line,
                                      should_cancel=should_cancel, tail_lines=200)

    def write_file(self, path: str, text: str, mode: int = 0o644) -> None:
        from EDA import pod_client
        sftp = self._ssh().open_sftp()
        try:
            pod_client.sftp_makedirs(sftp, posixpath.dirname(path))
            with sftp.open(path, "wb") as fh:
                fh.write(text.encode("utf-8"))
            sftp.chmod(path, mode)
        finally:
            sftp.close()

    def read_tree(self, directory: str) -> Tuple[Dict[str, str], List[str]]:
        code, out, _ = self.exec(
            f"cd {shlex.quote(directory)} && find . -type f -not -path './.git/*' | head -{contract.MAX_FILES + 1}")
        files: Dict[str, str] = {}
        binaries: List[str] = []
        sftp = self._ssh().open_sftp()
        try:
            for rel in (out or "").splitlines():
                rel = rel.strip()[2:] if rel.strip().startswith("./") else rel.strip()
                if not rel:
                    continue
                with sftp.open(f"{directory}/{rel}", "rb") as fh:
                    data = fh.read(contract.MAX_FILE_BYTES + 1)
                try:
                    files[rel] = data.decode("utf-8")
                except UnicodeDecodeError:
                    binaries.append(rel)
        finally:
            sftp.close()
        return files, binaries

    def keepalive(self) -> None:
        from EDA.registry import registry
        if self.eda_id:
            registry.touch(self.eda_id)

    def terminate(self) -> None:
        from EDA import runtime as eda_runtime
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        if self.eda_id:
            eda_runtime.terminate_eda(self.eda_id)
            self.eda_id = ""
        self.fresh = True


def run_smoke(manifest_text: str, *, api_keys: str = "", should_cancel=lambda: False,
              on_status=lambda s: None, poll_s: float = 3.0) -> Tuple[bool, str, bool]:
    """
    Run the new block ONCE, for real: a fresh pod from the pushed image, the
    selftest inputs, the generic runner.  Returns ``(ok, detail, infra)``.

    This is the step no hand-written block had before the first user run: it
    proves the pull, SSH, the entry and the port mapping, not just the build.
    """
    from CUSTOM.models import CustomRunRequest
    from CUSTOM.runtime import start_custom_job
    from EDA import runtime as eda_runtime
    from EDA.models import EdaSpec
    from EDA.registry import registry

    m = parse_manifest(manifest_text)
    spec_kw: Dict[str, Any] = dict(kind="custom", image=m.runtime.image,
                                   compute_type="GPU" if m.runtime.compute == "gpu" else "CPU",
                                   container_disk_gb=m.runtime.disk_gb, api_keys=api_keys,
                                   name=f"grafux-smoke-{m.slug}")
    if m.runtime.instance_type:
        spec_kw["instance_type"] = m.runtime.instance_type
    on_status("provisioning a test pod from the new image")
    res = eda_runtime.provision_eda(EdaSpec(**spec_kw))
    eda_id = res.get("eda_id", "")
    try:
        if res.get("status") == "error" or not eda_id:
            msg = res.get("errors") or "provisioning failed"
            # An image RunPod cannot pull or start is the generated block's fault;
            # capacity, keys and networking are not.
            image_fault = any(w in msg.lower() for w in ("image", "exited", "pull"))
            return False, f"The test pod did not start: {msg}", not image_fault
        on_status("running the selftest inputs through the block")
        start_custom_job(eda_id, CustomRunRequest(manifest=manifest_text,
                                                  inputs=dict(m.selftest.inputs),
                                                  timeout=m.runtime.timeout_s))
        result = None
        while result is None:
            if should_cancel():
                return False, "stopped", True
            rec = registry.get(eda_id)
            if rec is None:
                return False, "the test pod disappeared before reporting", True
            result = rec.result
            if result is None:
                time.sleep(poll_s)
        outputs = result.get("outputs") or {}
        empty = [n for n in m.selftest.expect_outputs if not str(outputs.get(n, "")).strip()]
        if result.get("status") == "ok" and not empty:
            return True, "", False
        detail = [f"status: {result.get('status')}"]
        if empty:
            detail.append("empty outputs: " + ", ".join(empty))
        if result.get("errors"):
            detail.append("errors:\n" + str(result["errors"])[-3000:])
        if result.get("log"):
            detail.append("log tail:\n" + str(result["log"])[-3000:])
        return False, "\n".join(detail), False
    finally:
        if eda_id:
            eda_runtime.terminate_eda(eda_id)


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
        """Drop sessions idle past the timeout, tearing their sandboxes down."""
        now = time.time()
        with self._lock:
            stale = [s for s in self._sessions.values()
                     if not s.busy and now - s.updated > self.idle_timeout_s]
            for s in stale:
                self._sessions.pop(s.id, None)
        for s in stale:
            try:
                s.sandbox.terminate()
            except Exception:  # noqa: BLE001
                pass


sessions = SessionRegistry()
