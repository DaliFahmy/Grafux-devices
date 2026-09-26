"""
builder.py
Builds a generated block's image with GitHub Actions and pushes it to GHCR.

Why Actions and not a RunPod pod: RunPod containers are unprivileged, so
neither docker nor rootful buildah runs there, and kaniko is only supported
inside its own image.  Actions runs real ``docker build`` -- the path every
Grafux image already takes -- and the push credential lives in the runner, never
on a machine that executes the repo's code (``docker build`` RUN steps do not see
the job's GITHUB_TOKEN).

The flow, all over the REST API with one token:

    1. commit the build context to a fresh branch ``build/<tag>`` of the builds
       repo (git data API: blobs -> tree -> orphan commit -> ref);
    2. dispatch the builds repo's workflow with {branch, tag};
    3. find the run by its ``run-name`` (the workflow sets ``build <tag>``;
       a dispatch returns no run id);
    4. poll until it completes; on failure, return the failing job's log tail --
       which becomes the agent's repair order.

Configuration (devices server env):

    GENERATOR_GITHUB_TOKEN    token with contents:write + actions:write on the builds repo
    GENERATOR_BUILDS_REPO     owner/name, e.g. DaliFahmy/grafux-gen-builds
    GENERATOR_BUILD_WORKFLOW  workflow file name (default build.yml)
    GENERATOR_BUILDS_REF      branch holding the workflow (default main)

``GENERATOR/builds-repo/`` holds that repo's workflow and setup notes.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .contract import IMAGE_REPO, is_executable

logger = logging.getLogger("generator.builder")

API = "https://api.github.com"
LOG_TAIL_LINES = 150


@dataclass
class BuildResult:
    ok: bool
    image: str = ""
    detail: str = ""        # the log tail on failure, a note on success
    run_url: str = ""
    # True when the failure is Grafux's or GitHub's, not the generated files':
    # the loop must not hand it to the agent as a repair order.
    infra: bool = False
    extra: Dict[str, str] = field(default_factory=dict)


class BuilderNotConfigured(RuntimeError):
    pass


class GitHubBuilder:
    def __init__(self, token: str, repo: str, *, workflow: str = "build.yml", ref: str = "main",
                 image_repo: str = IMAGE_REPO, http=None, poll_s: float = 10.0,
                 find_timeout_s: float = 120.0, build_timeout_s: float = 3600.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not token or not repo:
            raise BuilderNotConfigured(
                "The image builder is not configured: set GENERATOR_GITHUB_TOKEN and "
                "GENERATOR_BUILDS_REPO on the devices server (see GENERATOR/builds-repo/README.md).")
        self.repo, self.workflow, self.ref, self.image_repo = repo, workflow, ref, image_repo
        self.poll_s, self.find_timeout_s, self.build_timeout_s = poll_s, find_timeout_s, build_timeout_s
        self._sleep = sleep
        if http is None:
            import httpx
            http = httpx.Client(base_url=API, timeout=60.0, follow_redirects=True, headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            })
        self.http = http

    @classmethod
    def from_env(cls) -> "GitHubBuilder":
        return cls(os.environ.get("GENERATOR_GITHUB_TOKEN", "").strip(),
                   os.environ.get("GENERATOR_BUILDS_REPO", "").strip(),
                   workflow=os.environ.get("GENERATOR_BUILD_WORKFLOW", "build.yml").strip(),
                   ref=os.environ.get("GENERATOR_BUILDS_REF", "main").strip())

    # ---- REST helpers ------------------------------------------------------

    def _json(self, method: str, path: str, **kw):
        resp = self.http.request(method, path, **kw)
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub {method} {path} failed ({resp.status_code}): {resp.text[:400]}")
        return resp.json() if resp.content else {}

    # ---- steps -------------------------------------------------------------

    def push_context(self, branch: str, files: Dict[str, bytes], message: str) -> str:
        """Commit ``files`` as the whole tree of ``branch`` (an orphan commit)."""
        r = f"/repos/{self.repo}"
        tree = []
        for path, data in sorted(files.items()):
            blob = self._json("POST", f"{r}/git/blobs", json={
                "content": base64.b64encode(data).decode("ascii"), "encoding": "base64"})
            tree.append({"path": path, "mode": "100755" if is_executable(path) else "100644",
                         "type": "blob", "sha": blob["sha"]})
        tree_sha = self._json("POST", f"{r}/git/trees", json={"tree": tree})["sha"]
        commit = self._json("POST", f"{r}/git/commits",
                            json={"message": message, "tree": tree_sha, "parents": []})["sha"]
        resp = self.http.request("POST", f"{r}/git/refs",
                                 json={"ref": f"refs/heads/{branch}", "sha": commit})
        if resp.status_code == 422:        # the branch exists: this is a rebuild
            self._json("PATCH", f"{r}/git/refs/heads/{branch}", json={"sha": commit, "force": True})
        elif resp.status_code >= 400:
            raise RuntimeError(f"GitHub could not create branch {branch} ({resp.status_code}): "
                               f"{resp.text[:400]}")
        return commit

    def dispatch_workflow(self, workflow: str, inputs: Dict[str, str]) -> None:
        self._json("POST", f"/repos/{self.repo}/actions/workflows/{workflow}/dispatches",
                   json={"ref": self.ref, "inputs": inputs})

    def find_run_named(self, workflow: str, title: str) -> Optional[dict]:
        """
        A dispatched run, matched by its run-name.  A dispatch returns no run id,
        so every workflow the Generator starts sets a unique ``run-name``.
        """
        deadline = time.monotonic() + self.find_timeout_s
        while True:
            runs = self._json("GET", f"/repos/{self.repo}/actions/workflows/{workflow}/runs",
                              params={"event": "workflow_dispatch", "per_page": 30}).get("workflow_runs", [])
            for run in runs:
                if (run.get("display_title") or run.get("name") or "").strip() == title:
                    return run
            if time.monotonic() > deadline:
                return None
            self._sleep(min(5.0, self.poll_s))

    def get_run(self, run_id: int) -> dict:
        return self._json("GET", f"/repos/{self.repo}/actions/runs/{run_id}")

    def dispatch(self, branch: str, tag: str) -> None:
        self.dispatch_workflow(self.workflow, {"branch": branch, "tag": tag})

    def find_run(self, tag: str) -> Optional[dict]:
        """The dispatched build run, matched by its run-name (``build <tag>``)."""
        return self.find_run_named(self.workflow, f"build {tag}")

    def failure_log(self, run_id: int) -> str:
        jobs = self._json("GET", f"/repos/{self.repo}/actions/runs/{run_id}/jobs").get("jobs", [])
        for job in jobs:
            if job.get("conclusion") not in ("success", "skipped", None):
                resp = self.http.request("GET", f"/repos/{self.repo}/actions/jobs/{job['id']}/logs")
                if resp.status_code < 400:
                    lines = resp.text.splitlines()
                    # Keep the GRAFUX SELFTEST lines even when they scrolled out of
                    # the tail: they are the most actionable text in the log.
                    marked = [ln for ln in lines if "GRAFUX SELFTEST" in ln]
                    tail = lines[-LOG_TAIL_LINES:]
                    extra = [m for m in marked if m not in tail]
                    return "\n".join(extra + tail)
                failed = [s["name"] for s in job.get("steps", []) if s.get("conclusion") == "failure"]
                return f"job '{job.get('name')}' failed at step(s): {', '.join(failed) or 'unknown'}"
        return "the build failed, but GitHub returned no failed job to read a log from"

    def cancel(self, run_id: int) -> None:
        try:
            self.http.request("POST", f"/repos/{self.repo}/actions/runs/{run_id}/cancel")
        except Exception:  # noqa: BLE001 -- best effort
            pass

    # ---- the whole build ---------------------------------------------------

    def build(self, tag: str, files: Dict[str, bytes], *,
              on_status: Optional[Callable[[str], None]] = None,
              should_cancel: Optional[Callable[[], bool]] = None) -> BuildResult:
        say = on_status or (lambda _s: None)
        image = f"{self.image_repo}:{tag}"
        branch = f"build/{tag}"
        try:
            say("uploading build context")
            self.push_context(branch, files, f"Grafux generated block {tag}")
            say("dispatching build")
            self.dispatch(branch, tag)
            run = self.find_run(tag)
            if run is None:
                return BuildResult(False, image, "The build was dispatched but its run never appeared. "
                                   "Check the builds repo's Actions tab.", infra=True)
            run_id, url = run["id"], run.get("html_url", "")
            deadline = time.monotonic() + self.build_timeout_s
            while run.get("status") != "completed":
                if should_cancel and should_cancel():
                    self.cancel(run_id)
                    return BuildResult(False, image, "stopped", url, infra=True)
                if time.monotonic() > deadline:
                    self.cancel(run_id)
                    return BuildResult(False, image, f"The build exceeded {int(self.build_timeout_s)}s.",
                                       url, infra=True)
                say(f"building ({run.get('status', 'queued')})")
                self._sleep(self.poll_s)
                run = self._json("GET", f"/repos/{self.repo}/actions/runs/{run_id}")
            if run.get("conclusion") == "success":
                return BuildResult(True, image, "built and pushed", url)
            return BuildResult(False, image, self.failure_log(run_id), url)
        except Exception as exc:  # noqa: BLE001 -- reported, never raised into the loop
            logger.warning("generator build %s failed: %s", tag, exc)
            return BuildResult(False, image, f"Grafux could not run the build: {exc}", infra=True)
