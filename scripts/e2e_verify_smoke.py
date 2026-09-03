#!/usr/bin/env python3
"""
End-to-end smoke test for the cocotb verification path.

Drives a REAL verilator block through the REAL devices API — provision, run,
poll, collect — against the good FIFO and the deliberately buggy one, and asserts
the outcome each should produce.

This is a SCRIPT, not a pytest, on purpose: it rents a machine and costs money,
and nothing that does that should be one `pytest` away from running in CI.

Two ways to run it:

  Local container (no RunPod account, no spend). Needs the verify image pulled
  and a devices server pointed at it:

      docker pull ghcr.io/dalifahmy/grafux-verify:<tag>
      ssh-keygen -t rsa -b 2048 -f /tmp/eda_key -N ''
      docker run -d --name grafux-verify -p 2222:22 \
          -e PUBLIC_KEY="$(cat /tmp/eda_key.pub)" ghcr.io/dalifahmy/grafux-verify:<tag>
      EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/eda_key uvicorn device.app:app --port 8000
      python scripts/e2e_verify_smoke.py

  Real RunPod. The image tag in EDA/models.py (or EDA_VERIFY_IMAGE) must already
  be pushed to GHCR **as a public package** — RunPod pulls anonymously, and a
  private one fails with a misleading "no public-IP networking" error:

      RUNPOD_API_KEY=rp_... EDA_DEFAULT_KEEP_WARM_MIN=15 uvicorn device.app:app --port 8000
      python scripts/e2e_verify_smoke.py

Exit code is 0 only if every case produced the outcome it was supposed to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eda"
DEFAULT_BASE = os.environ.get("EDA_E2E_BASE", "http://127.0.0.1:8000")

# Provisioning dominates a run that itself takes seconds; that is the whole
# reason the light verify image exists, so the script reports it rather than
# hiding it inside a total.
PROVISION_TIMEOUT_S = int(os.environ.get("EDA_E2E_PROVISION_TIMEOUT", "900"))
RUN_TIMEOUT_S = int(os.environ.get("EDA_E2E_RUN_TIMEOUT", "900"))
POLL_S = 3.0


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def poll(client: httpx.Client, base: str, eda_id: str, *, until: str,
         timeout_s: int, label: str) -> dict:
    """Poll /status until `until` ('ready' phase, or done), or give up."""
    started = time.monotonic()
    last = {}
    while time.monotonic() - started < timeout_s:
        resp = client.get(f"{base}/verilator/{eda_id}/status",
                          params={"live": "true" if until == "ready" else "false"})
        resp.raise_for_status()
        last = resp.json()
        if until == "ready":
            if last.get("phase") == "ready":
                return last
            if last.get("phase") == "error":
                raise Failure(f"{label}: provisioning failed: {last.get('errors')}")
        else:
            if last.get("done"):
                return last
        time.sleep(POLL_S)
    raise Failure(f"{label}: timed out after {timeout_s}s (last status: {last})")


def run_case(client: httpx.Client, base: str, *, name: str, rtl_file: str,
             expect_pass: bool, must_name: str = "") -> dict:
    """Provision, run one cocotb job, and assert the outcome."""
    print(f"\n=== {name} ===")
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    created = client.post(f"{base}/verilator/create_async",
                          json={"name": f"e2e-{name}"})
    created.raise_for_status()
    eda_id = created.json()["eda_id"]
    print(f"  eda_id: {eda_id}")

    ready = poll(client, base, eda_id, until="ready",
                 timeout_s=PROVISION_TIMEOUT_S, label=name)
    timings["provision_s"] = time.monotonic() - t0
    print(f"  provisioned in {timings['provision_s']:.1f}s "
          f"(image: {ready.get('image', '?')})")

    t1 = time.monotonic()
    started = client.post(
        f"{base}/verilator/{eda_id}/run",
        json={
            "rtl": fixture(rtl_file),
            "testbench": fixture("test_sync_fifo.py"),
            "top": "sync_fifo",
            # Left at "sim" ON PURPOSE: a cocotb testbench must be recognised and
            # run as cocotb without anyone changing a dropdown. If this stops
            # working, every existing verilator block silently breaks.
            "mode": "sim",
            "coverage": "1",
            "trace": "1",
            "timeout": RUN_TIMEOUT_S,
        },
        timeout=60.0,
    )
    started.raise_for_status()

    poll(client, base, eda_id, until="done", timeout_s=RUN_TIMEOUT_S, label=name)
    timings["run_s"] = time.monotonic() - t1

    result = client.get(f"{base}/verilator/{eda_id}/result", timeout=60.0).json()
    outputs = result.get("outputs", {})
    artifacts = [a.get("path", "") for a in result.get("artifacts", [])]

    print(f"  ran in {timings['run_s']:.1f}s   "
          f"cost: ${result.get('cost_estimate_usd', 0):.4f}")
    print(f"  passed={outputs.get('passed')}  status={outputs.get('status')}")

    results = json.loads(outputs.get("results") or "{}")
    print(f"  tests: {results.get('passed', 0)} passed, "
          f"{results.get('failed', 0)} failed, of {results.get('total', 0)}")
    if outputs.get("coverage"):
        print(f"  coverage: {outputs['coverage']}")
    if outputs.get("failures"):
        first = outputs["failures"].splitlines()[:4]
        print("  failures: " + " / ".join(line.strip() for line in first if line.strip()))

    # The testbench must actually have run. Zero tests reported as a pass is the
    # single worst thing this pipeline could do, so it is checked on both cases.
    check(results.get("total", 0) > 0,
          f"{name}: no tests were collected — the testbench did not run")

    if expect_pass:
        check(outputs.get("passed") == "true",
              f"{name}: expected a pass, got passed={outputs.get('passed')} "
              f"errors={outputs.get('errors')!r}")
        check(any(a.endswith((".vcd", ".fst")) for a in artifacts),
              f"{name}: no waveform came back (artifacts: {artifacts})")
        check(any(a.endswith("coverage.info") for a in artifacts),
              f"{name}: no coverage report came back (artifacts: {artifacts})")
    else:
        check(outputs.get("passed") == "false",
              f"{name}: the buggy design PASSED — the testbench cannot tell the "
              f"good FIFO from the broken one, which makes it worthless")
        check(must_name in (outputs.get("failures") or ""),
              f"{name}: expected '{must_name}' among the failures, got:\n"
              f"{outputs.get('failures')}")

    client.delete(f"{base}/verilator/{eda_id}", timeout=60.0)
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="devices server base URL (default: %(default)s)")
    parser.add_argument("--case", choices=["good", "buggy", "both"], default="both")
    args = parser.parse_args()

    if not FIXTURES.is_dir():
        print(f"fixtures not found: {FIXTURES}", file=sys.stderr)
        return 2

    cases = []
    if args.case in ("good", "both"):
        cases.append(dict(name="good-fifo", rtl_file="sync_fifo_good.v",
                          expect_pass=True))
    if args.case in ("buggy", "both"):
        # The mutation check: an LLM-written testbench that passes this fixture is
        # not verifying anything, and this is the one assertion that catches it.
        cases.append(dict(name="buggy-fifo", rtl_file="sync_fifo_bug_full_flag.v",
                          expect_pass=False,
                          must_name="test_full_asserts_at_depth"))

    failures: list[str] = []
    with httpx.Client(timeout=30.0) as client:
        for case in cases:
            try:
                run_case(client, args.base, **case)
                print(f"  OK: {case['name']}")
            except (Failure, httpx.HTTPError) as exc:
                print(f"  FAIL: {case['name']}: {exc}", file=sys.stderr)
                failures.append(case["name"])

    print("\n" + ("=" * 60))
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(cases)} case(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
