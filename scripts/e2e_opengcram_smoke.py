#!/usr/bin/env python3
"""
End-to-end smoke test for the opengcram (gain-cell memory compiler) path.

Drives a REAL opengcram block through the REAL devices API -- provision, run,
poll, collect.  A SCRIPT, not a pytest, because it rents a machine; run it on the
free local-container path first (see scripts/e2e_openram_smoke.py for the
EDA_LOCAL_SSH=1 recipe, with the grafux-opengcram image).

Two cases, and which one you can run depends on what you hold:

  refusal  (default, needs nothing)  -- no technology archive.  Proves the pod
           comes up, the gain-cell probe runs under the real environment, and the
           block gets the readable "supply one on tech_archive" error instead of
           a compiler traceback.  This is the most that can be verified without
           a gain-cell technology, because none is public.

  compile  (--tech-archive PATH)     -- a .tar.gz/.zip of a technology directory
           whose gds_lib/sp_lib carry os_gc (or --gc-type Si/hybrid).  Uploads it
           exactly as the Qt block does (input_files, base64) and asserts a macro
           comes back with every view.

Exit code is 0 only if the case produced the outcome it was supposed to.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = os.environ.get("EDA_E2E_BASE", "http://127.0.0.1:8000")
PROVISION_TIMEOUT_S = int(os.environ.get("EDA_E2E_PROVISION_TIMEOUT", "900"))
RUN_TIMEOUT_S = int(os.environ.get("EDA_E2E_RUN_TIMEOUT", "3600"))
POLL_S = 3.0
WORK_DIR = "/workspace/grafux"

REQUIRED_TEXT_PORTS = ("status", "top", "tech_name", "verilog_model", "config",
                       "stats", "log")
REQUIRED_ARTIFACT_EXTS = (".gds", ".lef", ".lib", ".v", ".sp")


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def poll(client: httpx.Client, base: str, eda_id: str, *, until: str, timeout_s: int) -> dict:
    started = time.monotonic()
    last: dict = {}
    last_stage = ""
    while time.monotonic() - started < timeout_s:
        resp = client.get(f"{base}/opengcram/{eda_id}/status",
                          params={"live": "true" if until == "ready" else "false"})
        resp.raise_for_status()
        last = resp.json()
        stage = f"{last.get('phase', '')}/{last.get('stage', '')}"
        if stage != last_stage:
            print(f"    {stage}")
            last_stage = stage
        if until == "ready":
            if last.get("phase") == "ready":
                return last
            if last.get("phase") == "error":
                raise Failure(f"provisioning failed: {last.get('errors')}")
        elif last.get("done"):
            return last
        time.sleep(POLL_S)
    raise Failure(f"timed out after {timeout_s}s (last status: {last})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--tech-archive", default="",
                        help="archive of a technology that carries a gain cell")
    parser.add_argument("--tech-name", default="")
    parser.add_argument("--gc-type", default="OS")
    args = parser.parse_args()

    body = {"word_size": "2", "num_words": "16", "gc_type": args.gc_type,
            "tech_name": args.tech_name, "timeout": RUN_TIMEOUT_S}
    expect_macro = bool(args.tech_archive)
    if expect_macro:
        with open(args.tech_archive, "rb") as handle:
            data = handle.read()
        base_name = os.path.basename(args.tech_archive)
        body["tech_archive"] = args.tech_archive
        body["input_files"] = [{"path": f"{WORK_DIR}/{base_name}",
                                "content": base64.b64encode(data).decode("ascii"),
                                "b64": True}]
        print(f"uploading {base_name} ({len(data) / 1024 / 1024:.1f} MB)")

    try:
        with httpx.Client(timeout=30.0) as client:
            created = client.post(f"{args.base}/opengcram/create_async",
                                  json={"name": "e2e-opengcram"})
            created.raise_for_status()
            eda_id = created.json()["eda_id"]
            print(f"eda_id: {eda_id}")
            poll(client, args.base, eda_id, until="ready", timeout_s=PROVISION_TIMEOUT_S)

            started = client.post(f"{args.base}/opengcram/{eda_id}/run", json=body,
                                  timeout=300.0)
            started.raise_for_status()
            poll(client, args.base, eda_id, until="done", timeout_s=RUN_TIMEOUT_S)
            result = client.get(f"{args.base}/opengcram/{eda_id}/result", timeout=120.0).json()
    except (Failure, httpx.HTTPError) as exc:
        print(f"FAILED: {exc}")
        return 1

    outputs = result.get("outputs", {})
    artifacts = [a.get("path", "").lower() for a in result.get("artifacts", [])]
    print(f"status={outputs.get('status')} stage={result.get('stage')} "
          f"cost=${result.get('cost_estimate_usd', 0):.4f}")
    print(f"errors: {outputs.get('errors', '')}")

    try:
        if not expect_macro:
            check(outputs.get("status") == "error", "a run with no technology did not fail")
            check(result.get("stage") == "tech",
                  f"it failed at {result.get('stage')!r}, not at the tech stage")
            check("tech_archive" in outputs.get("errors", ""),
                  "the error does not point the user at tech_archive")
        else:
            check(outputs.get("status") == "ok", "the compile did not succeed")
            for port in REQUIRED_TEXT_PORTS:
                check(bool((outputs.get(port) or "").strip()), f"the {port} port came back empty")
            for ext in REQUIRED_ARTIFACT_EXTS:
                check(any(p.endswith(ext) for p in artifacts), f"no {ext} artifact came back")
            print(f"stats: {json.dumps(json.loads(outputs.get('stats') or '{}'), sort_keys=True)}")
    except Failure as exc:
        print(f"FAILED: {exc}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
