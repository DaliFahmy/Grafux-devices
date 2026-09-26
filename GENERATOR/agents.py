"""
agents.py
Headless drivers for the two code-generation agents the Generator offers.

Pure: each driver turns (mode, model, prompt, resume id) into ONE shell command,
and turns each JSONL line the agent prints into zero or more normalized events.
Nothing here opens a socket, so both halves are tested with captured lines.

Normalized event kinds (what the app renders):

    session   the agent's session/thread id (data.session_id) -- kept for resume
    thought   prose from the agent (assistant text, reasoning, agent_message)
    tool      a tool call or shell command it ran (data.tool, data.input / command)
    file      a file it wrote or edited (data.path)
    result    the turn's final answer (data.is_error, data.cost_usd)
    error     something the agent itself reported as failed

Facts relied on (checked against the vendors' docs 2026-09; the Codex CLI's
JSONL schema is not frozen upstream, so its driver is marked EXPERIMENTAL and
parses defensively -- an unknown event is dropped, never raised):

Claude Code  ``claude -p PROMPT --output-format stream-json --verbose``
             ``--permission-mode plan|acceptEdits``, ``--model``, ``--resume ID``,
             ``--append-system-prompt``, ``--allowedTools``.  Events: ``system``
             (subtype init, session_id), ``assistant``/``user`` (message.content
             blocks: text / tool_use / tool_result), ``result`` (result, is_error,
             total_cost_usd, session_id).
Codex        ``codex exec --json --skip-git-repo-check -m MODEL
             --sandbox danger-full-access PROMPT``; resume is
             ``codex exec resume ID PROMPT``.  Its own sandbox (bubblewrap +
             landlock) cannot start in an unprivileged container, so the Actions
             runner is the sandbox and plan mode is enforced by the caller (the
             session discards a plan turn's files), not by the flag.  Events:
             ``thread.started`` (thread_id), ``item.completed`` with item.type
             agent_message / reasoning / command_execution / file_change,
             ``turn.completed``, ``turn.failed``, ``error``.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Optional

AGENTS = ("claude_code", "codex")
MODES = ("plan", "edit")

# What the model dropdowns offer.  The first entry is the default.  Any id is
# accepted server-side (the CLI validates it); these are only the suggestions.
MODELS = {
    "claude_code": ["claude-opus-5-5", "claude-sonnet-5", "claude-fable-5-1", "claude-haiku-4-5-20251001"],
    "codex": ["gpt-5-codex", "gpt-5"],
}

# Tools Claude Code may use without asking in EDIT mode.  The Actions runner is a
# throwaway machine holding only the cloned repo and the output directory, so Bash is
# allowed whole: exploring a build system means running it.
CLAUDE_EDIT_TOOLS = ("Bash", "Read", "Edit", "Write", "Glob", "Grep", "WebFetch", "WebSearch")

# Everything an event's text may carry; longer is cut (the full log is on the pod).
EVENT_TEXT_MAX = 8000


def default_model(agent: str) -> str:
    return MODELS.get(agent, [""])[0]


def build_command(
    agent: str,
    *,
    mode: str,
    prompt_file: str,
    system_file: str,
    model: str = "",
    resume_id: str = "",
    cwd: str = "/workspace",
    env_file: str = "",
) -> str:
    """
    The one shell command for a turn, to be run under ``bash -lc``.

    The prompt and system prompt are read from FILES on the pod (written over
    SFTP) rather than inlined: a user's idea plus a repair order can be many KB,
    and quoting that through two shells is where injection bugs live.  The key
    file is sourced, never interpolated, for the same reason and so the key never
    appears in a command line (``ps``) or in anything this server logs.
    """
    if agent not in AGENTS:
        raise ValueError(f"unknown agent '{agent}'; expected one of {', '.join(AGENTS)}")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; expected plan or edit")
    model = (model or default_model(agent)).strip()
    prefix = f"cd {shlex.quote(cwd)} && "
    if env_file:
        prefix = f"set -a; . {shlex.quote(env_file)}; set +a; " + prefix
    prompt = f'"$(cat {shlex.quote(prompt_file)})"'
    system = f'"$(cat {shlex.quote(system_file)})"'

    if agent == "claude_code":
        parts = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
                 "--append-system-prompt", system]
        if model:
            parts += ["--model", shlex.quote(model)]
        if mode == "plan":
            parts += ["--permission-mode", "plan"]
        else:
            parts += ["--permission-mode", "acceptEdits",
                      "--allowedTools", shlex.quote(" ".join(CLAUDE_EDIT_TOOLS))]
        if resume_id:
            parts += ["--resume", shlex.quote(resume_id)]
        return prefix + " ".join(parts) + " < /dev/null"

    # codex: the system prompt is prepended to the prompt -- `exec` has no flag
    # for it, and AGENTS.md would be read from the repo the user pointed at.
    full = f'"$(cat {shlex.quote(system_file)}; printf \'\\n\\n---\\n\\n\'; cat {shlex.quote(prompt_file)})"'
    base = ["codex", "exec"]
    if resume_id:
        base += ["resume", shlex.quote(resume_id)]
    base += ["--json", "--skip-git-repo-check", "--sandbox", "danger-full-access"]
    if model:
        base += ["-m", shlex.quote(model)]
    return prefix + " ".join(base + [full]) + " < /dev/null"


def _cut(text: Any) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False) if text is not None else ""
    return text if len(text) <= EVENT_TEXT_MAX else text[:EVENT_TEXT_MAX] + " …"


def _ev(kind: str, text: str = "", **data: Any) -> Dict[str, Any]:
    return {"kind": kind, "text": _cut(text), "data": data}


def _claude_tool_event(block: Dict[str, Any]) -> Dict[str, Any]:
    name = block.get("name", "")
    inp = block.get("input") or {}
    path = inp.get("file_path") or inp.get("path") or ""
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit") and path:
        return _ev("file", f"{name} {path}", tool=name, path=path)
    if name == "Bash":
        return _ev("tool", inp.get("command", ""), tool="Bash", command=inp.get("command", ""))
    summary = path or inp.get("pattern") or inp.get("url") or inp.get("query") or ""
    return _ev("tool", f"{name} {summary}".strip(), tool=name, input=inp)


def parse_claude(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    t = obj.get("type")
    if t == "system" and obj.get("subtype") == "init":
        return [_ev("session", "", session_id=obj.get("session_id", ""), model=obj.get("model", ""))]
    if t == "assistant":
        out = []
        for block in (obj.get("message") or {}).get("content") or []:
            bt = block.get("type")
            if bt == "text" and (block.get("text") or "").strip():
                out.append(_ev("thought", block["text"]))
            elif bt == "tool_use":
                out.append(_claude_tool_event(block))
        return out
    if t == "result":
        is_error = bool(obj.get("is_error")) or obj.get("subtype", "success") != "success"
        return [_ev("result", obj.get("result") or "", is_error=is_error,
                    cost_usd=obj.get("total_cost_usd") or 0.0,
                    session_id=obj.get("session_id", ""), turns=obj.get("num_turns") or 0)]
    return []


def parse_codex(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    t = obj.get("type")
    if t == "thread.started":
        return [_ev("session", "", session_id=obj.get("thread_id", ""))]
    if t == "item.completed":
        item = obj.get("item") or {}
        it = item.get("type")
        if it in ("agent_message", "reasoning") and (item.get("text") or "").strip():
            return [_ev("thought", item["text"], reasoning=(it == "reasoning"),
                        final=(it == "agent_message"))]
        if it == "command_execution":
            return [_ev("tool", item.get("command", ""), tool="shell", command=item.get("command", ""),
                        exit_code=item.get("exit_code"))]
        if it == "file_change":
            return [_ev("file", f"{c.get('kind', 'edit')} {c.get('path', '')}", path=c.get("path", ""))
                    for c in item.get("changes") or [] if c.get("path")]
        return []
    if t == "turn.failed":
        return [_ev("error", ((obj.get("error") or {}).get("message")) or "the turn failed")]
    if t == "error":
        return [_ev("error", obj.get("message") or "codex reported an error")]
    if t == "turn.completed":
        return [_ev("result", "", is_error=False, usage=obj.get("usage") or {})]
    return []


def parse_line(agent: str, line: str) -> List[Dict[str, Any]]:
    """
    Normalize one output line.  A line that is not JSON is the CLI talking outside
    its protocol (a warning, an npm notice, a crash trace) and is surfaced as-is:
    swallowing it would hide exactly the message that explains a dead turn.
    """
    line = (line or "").strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return [_ev("log", line)]
    if not isinstance(obj, dict):
        return []
    try:
        return parse_claude(obj) if agent == "claude_code" else parse_codex(obj)
    except Exception:  # noqa: BLE001 -- a schema drift must not kill a turn
        return [_ev("log", line)]


def final_text(agent: str, events: List[Dict[str, Any]]) -> str:
    """The turn's answer: Claude's result text, or Codex's last agent_message."""
    if agent == "claude_code":
        for e in reversed(events):
            if e["kind"] == "result" and e["text"]:
                return e["text"]
    for e in reversed(events):
        if e["kind"] == "thought" and not e["data"].get("reasoning"):
            return e["text"]
    return ""


def session_id_from(events: List[Dict[str, Any]], previous: Optional[str] = "") -> str:
    for e in reversed(events):
        sid = (e.get("data") or {}).get("session_id")
        if sid:
            return sid
    return previous or ""
