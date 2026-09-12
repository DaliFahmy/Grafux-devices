#!/usr/bin/env python3
"""
End-to-end smoke test for the openram (memory compiler) path.

Drives a REAL openram block through the REAL devices API -- provision, run,
poll, collect -- and asserts that a tiny SRAM comes back with every view the
block's ports expect.

This is a SCRIPT, not a pytest, on purpose: it rents a machine and costs money,
and nothing that does that should be one `pytest` away from running in CI.

Two ways to run it:

  Local container (no RunPod account, no spend). Needs the openram image pulled
  and a devices server pointed at it -- DO THIS ONE FIRST:

      docker pull ghcr.io/dalifahmy/grafux-openram:<tag>
      ssh-keygen -t rsa -b 2048 -f /tmp/eda_key -N ''
      docker run -d --name grafux-openram -p 2222:22 \
          -e PUBLIC_KEY="$(cat /tmp/eda_key.pub)" ghcr.io/dalifahmy/grafux-openram:<tag>
      EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/eda_key uvicorn device.app:app --port 8000
      python scripts/e2e_openram_smoke.py

  Real RunPod. The image tag in EDA/models.py (or EDA_OPENRAM_IMAGE) must
  already be pushed to GHCR **as a public package** -- RunPod pulls anonymously,
  and a private one fails with a misleading "no public-IP networking" error:

      RUNPOD_API_KEY=rp_... uvicorn device.app:app --port 8000
      python scripts/e2e_openram_smoke.py

The default case is deliberately the smallest useful macro (2 x 16, scn4m_subm):
this checks the PATH, not the compiler, and a bigger macro only buys minutes of
waiting and dollars of pod time.  `--big` runs a second, realistic case for when
the question is how long a real one takes.

Exit code is 0 only if every case produced the outcome it was supposed to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = os.environ.get("EDA_E2E_BASE", "http://127.0.0.1:8000")
PROVISION_TIMEOUT_S = int(os.environ.get("EDA_E2E_PROVISION_TIMEOUT", "900"))
RUN_TIMEOUT_S = int(os.environ.get("EDA_E2E_RUN_TIMEOUT", "3600"))
POLL_S = 3.0

# The ports the block promises to fill on a successful run.  Asserted by NAME
# here rather than by filename, exactly as run_openram classifies them -- if
# this list and EdaPorts::kOpenRamOutputs ever disagree, the block shows an
# empty port and nothing says why.
REQUIRED_TEXT_PORTS = ("status", "top", "tech_name", "verilog_model", "config",
                       "stats", "log")
# Extensions the artifact download must bring back.  The `.lib` is the one whose
# NAME carries the process corner and moves between OpenRAM releases, which is
# why every check here is on the extension.
REQUIRED_ARTIFACT_EXTS = (".gds", ".lef", ".lib", ".v", ".sp")


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def poll(client: httpx.Client, base: str, eda_id: str, *, until: str,
         timeout_s: int, label: str) -> dict:
    """Poll /status until `until` ('ready' phase, or done), or give up."""
    started = time.monotonic()
    last: dict = {}
    last_stage = ""
    while time.monotonic() - started < timeout_s:
        resp = client.get(f"{base}/openram/{eda_id}/status",
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
                raise Failure(f"{label}: provisioning failed: {last.get('errors')}")
        elif last.get("done"):
            return last
        time.sleep(POLL_S)
    raise Failure(f"{label}: timed out after {timeout_s}s (last status: {last})")


def run_case(client: httpx.Client, base: str, *, name: str, word_size: str,
             num_words: str, tech: str) -> dict:
    """Provision, compile one macro, and assert the outcome."""
    print(f"\n=== {name} ({word_size} x {num_words}, {tech}) ===")
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    created = client.post(f"{base}/openram/create_async", json={"name": f"e2e-{name}"})
    created.raise_for_status()
    eda_id = created.json()["eda_id"]
    print(f"  eda_id: {eda_id}")

    poll(client, base, eda_id, until="ready",
         timeout_s=PROVISION_TIMEOUT_S, label=name)
    timings["provision_s"] = time.monotonic() - t0
    print(f"  provisioned in {timings['provision_s']:.1f}s")

    t1 = time.monotonic()
    started = client.post(
        f"{base}/openram/{eda_id}/run",
        json={
            "word_size": word_size,
            "num_words": num_words,
            "num_banks": "1",
            "num_rw_ports": "1",
            "num_r_ports": "0",
            "num_w_ports": "0",
            "tech_name": tech,
            "timeout": RUN_TIMEOUT_S,
        },
        timeout=60.0,
    )
    started.raise_for_status()

    poll(client, base, eda_id, until="done", timeout_s=RUN_TIMEOUT_S, label=name)
    timings["run_s"] = time.monotonic() - t1

    result = client.get(f"{base}/openram/{eda_id}/result", timeout=120.0).json()
    outputs = result.get("outputs", {})
    artifacts = [a.get("path", "") for a in result.get("artifacts", [])]

    print(f"  compiled in {timings['run_s']:.1f}s   "
          f"cost: ${result.get('cost_estimate_usd', 0):.4f}")
    print(f"  status={outputs.get('status')}  top={outputs.get('top')}")
    for path in artifacts:
        print(f"    artifact: {path.rsplit('/', 1)[-1]}")

    check(outputs.get("status") == "ok",
          f"{name}: status was {outputs.get('status')!r}: {outputs.get('errors')}")
    for port in REQUIRED_TEXT_PORTS:
        check(bool((outputs.get(port) or "").strip()),
              f"{name}: the {port} port came back empty")
    check("module" in outputs.get("verilog_model", ""),
          f"{name}: verilog_model does not look like a Verilog module")

    lower = [p.lower() for p in artifacts]
    for ext in REQUIRED_ARTIFACT_EXTS:
        check(any(p.endswith(ext) for p in lower),
              f"{name}: no {ext} artifact came back; the matching port stays empty")

    # The resolved config is the only place OpenRAM's own filled-in defaults are
    # visible, so a run that cannot report it has lost the single most useful
    # thing for debugging a macro that came out wrong.
    check("word_size" in outputs.get("config", ""),
          f"{name}: the config port does not carry the resolved configuration")

    stats = json.loads(outputs.get("stats") or "{}")
    print(f"  stats: {json.dumps(stats, sort_keys=True)}")
    check(stats.get("total_bits") == int(word_size) * int(num_words),
          f"{name}: stats.total_bits disagrees with the requested geometry")

    print(f"  OK  ({name})")
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="devices server base URL")
    parser.add_argument("--tech", default=os.environ.get("EDA_OPENRAM_TECH", "scn4m_subm"),
                        help="OpenRAM technology to compile for")
    parser.add_argument("--big", action="store_true",
                        help="also compile a realistic macro (slow, and it bills)")
    args = parser.parse_args()

    cases = [dict(name="tiny", word_size="2", num_words="16", tech=args.tech)]
    if args.big:
        cases.append(dict(name="realistic", word_size="8", num_words="256",
                          tech=args.tech))

    failures = []
    with httpx.Client(timeout=30.0) as client:
        for case in cases:
            try:
                run_case(client, args.base, **case)
            except (Failure, httpx.HTTPError) as exc:
                print(f"  FAILED: {exc}")
                failures.append(case["name"])

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
