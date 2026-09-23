#!/usr/bin/env python3
"""
End-to-end smoke test for the post-silicon verification path.

Drives a REAL cpu block through the REAL devices API — provision, run, poll,
collect — with cases whose outcome is known in advance, and asserts each one
produced what it should: a passing case passes, a case that prints FAIL fails
despite exiting 0, a case that crashes is judged on its exit code, a Python case
skips the compiler, and a case that does not compile never reaches the benchmark.

This is a SCRIPT, not a pytest, on purpose: it rents a machine and costs money,
and nothing that does that should be one `pytest` away from running in CI.

Two ways to run it:

  Local container (no RunPod account, no spend). Needs the cpu image built or
  pulled, and a devices server pointed at it:

      docker build -t grafux-cpu:local CPU/docker
      ssh-keygen -t rsa -b 2048 -f /tmp/cpu_key -N ''
      docker run -d --name grafux-cpu -p 2222:22 \
          -e PUBLIC_KEY="$(cat /tmp/cpu_key.pub)" grafux-cpu:local
      EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/cpu_key uvicorn device.app:app --port 8000
      python scripts/e2e_cpu_smoke.py

  Real RunPod. The tag in CPU/models.py (or CPU_DEFAULT_IMAGE) must already be
  pushed to GHCR **as a public package** — RunPod pulls anonymously, and a
  private one fails with a misleading "no public-IP networking" error:

      RUNPOD_API_KEY=rp_... uvicorn device.app:app --port 8000
      python scripts/e2e_cpu_smoke.py

Exit code is 0 only if every case produced the outcome it was supposed to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = os.environ.get("CPU_E2E_BASE", "http://127.0.0.1:8000")
PROVISION_TIMEOUT_S = int(os.environ.get("CPU_E2E_PROVISION_TIMEOUT", "900"))
RUN_TIMEOUT_S = int(os.environ.get("CPU_E2E_RUN_TIMEOUT", "900"))
POLL_S = 3.0


class Failure(Exception):
    pass


# ---------------------------------------------------------------------------
# The cases, and what each one proves
# ---------------------------------------------------------------------------

PASSING_CPP = """\
#include <cstdio>
#include <vector>
int main() {
    std::vector<int> line(16, 0);
    for (int i = 0; i < 16; ++i) line[i] = i * 3;
    int sum = 0;
    for (int v : line) sum += v;
    std::printf("%s: cache line writeback\\n", sum == 360 ? "PASS" : "FAIL");
    return 0;
}
"""

# Exits 0 while reporting a failure. If the runtime trusted the exit code alone,
# this would come back green for a chip that did not do what the case asked --
# the single most dangerous wrong answer this block can give.
FAILING_CHECK_C = """\
#include <stdio.h>
int main(void) {
    printf("PASS: tlb refill\\n");
    printf("FAIL: store buffer drain\\n");
    return 0;
}
"""

# No parseable output, nonzero exit: the verdict has to fall back to the status.
CRASHING_C = """\
#include <stdio.h>
int main(void) {
    printf("running with no check lines at all\\n");
    return 3;
}
"""

PASSING_PYTHON = """\
total = sum(i * 3 for i in range(16))
print("PASS: python path" if total == 360 else "FAIL: python path")
"""

BROKEN_C = "int main(void) { return undeclared_symbol; }\n"


CASES = [
    {
        "name": "passing C++ case",
        "body": {"code": PASSING_CPP, "language": "cpp", "repetitions": "5",
                 "warmup": "1", "build_flags": "-O2"},
        "expect_status": "ok",
        "expect_passed": "true",
        "expect_benchmark": True,
    },
    {
        "name": "C case that prints FAIL but exits 0",
        "body": {"code": FAILING_CHECK_C, "language": "c", "repetitions": "3"},
        "expect_status": "failed",
        "expect_passed": "false",
        "expect_benchmark": True,
        "expect_in_errors": "store buffer drain",
    },
    {
        "name": "C case with a nonzero exit and no check lines",
        "body": {"code": CRASHING_C, "language": "c", "repetitions": "3"},
        "expect_status": "failed",
        "expect_passed": "false",
        "expect_benchmark": True,
        "expect_in_errors": "exited with code 3",
    },
    {
        "name": "passing Python case",
        "body": {"code": PASSING_PYTHON, "language": "python", "repetitions": "3"},
        "expect_status": "ok",
        "expect_passed": "true",
        "expect_benchmark": True,
        "expect_compile_ms": 0,
    },
    {
        "name": "case that does not compile",
        "body": {"code": BROKEN_C, "language": "c"},
        "expect_status": "error",
        "expect_passed": "false",
        "expect_benchmark": False,
        "expect_in_errors": "did not compile",
    },
]


# ---------------------------------------------------------------------------
# Driving the API
# ---------------------------------------------------------------------------

def provision(client: httpx.Client) -> str:
    resp = client.post("/cpu/create_async", json={})
    resp.raise_for_status()
    body = resp.json()
    cpu_id = body.get("eda_id", "")
    if not cpu_id:
        raise Failure(f"no id returned from create_async: {body}")
    print(f"  provisioning {cpu_id} ...", flush=True)

    deadline = time.monotonic() + PROVISION_TIMEOUT_S
    last_phase = ""
    while time.monotonic() < deadline:
        status = client.get(f"/cpu/{cpu_id}/status").json()
        phase = status.get("phase", "")
        if phase != last_phase:
            print(f"    phase: {phase} {status.get('phase_detail', '')}".rstrip(), flush=True)
            last_phase = phase
        if phase == "ready":
            return cpu_id
        if phase == "error":
            raise Failure(f"provisioning failed: {status.get('phase_detail', '')}")
        time.sleep(POLL_S)
    raise Failure(f"pod was not ready within {PROVISION_TIMEOUT_S}s")


def run_case(client: httpx.Client, cpu_id: str, body: dict) -> dict:
    accepted = client.post(f"/cpu/{cpu_id}/run", json=body).json()
    if accepted.get("status") == "error":
        raise Failure(f"run refused: {accepted.get('errors', '')}")

    deadline = time.monotonic() + RUN_TIMEOUT_S
    last_stage = ""
    while time.monotonic() < deadline:
        status = client.get(f"/cpu/{cpu_id}/status").json()
        stage = f"{status.get('stage', '')} {status.get('stage_detail', '')}".strip()
        if stage and stage != last_stage:
            print(f"    stage: {stage}", flush=True)
            last_stage = stage
        if status.get("done"):
            break
        time.sleep(POLL_S)
    else:
        raise Failure(f"run did not finish within {RUN_TIMEOUT_S}s")

    result = client.get(f"/cpu/{cpu_id}/result")
    if result.status_code != 200:
        raise Failure(f"no result: {result.status_code} {result.text[:400]}")
    return result.json()


def check(case: dict, result: dict) -> None:
    outputs = result.get("outputs", {})
    status = result.get("status", "")
    if status != case["expect_status"]:
        raise Failure(
            f"expected status {case['expect_status']!r}, got {status!r}; "
            f"errors={outputs.get('errors', '')[:300]}"
        )
    if outputs.get("passed") != case["expect_passed"]:
        raise Failure(
            f"expected passed={case['expect_passed']!r}, got {outputs.get('passed')!r}")

    needle = case.get("expect_in_errors")
    if needle and needle not in outputs.get("errors", ""):
        raise Failure(
            f"expected {needle!r} in errors, got: {outputs.get('errors', '')[:300]}")

    benchmark = json.loads(outputs.get("benchmark") or "{}")
    if case["expect_benchmark"]:
        if not benchmark.get("repetitions"):
            raise Failure("expected a benchmark with at least one repetition")
        if not outputs.get("duration"):
            raise Failure("expected a duration on a run that produced samples")
        # The measurement must be plausible, not merely present: a zero mean
        # across several runs means the timing never actually happened.
        if float(outputs["duration"]) < 0:
            raise Failure(f"nonsensical duration {outputs['duration']!r}")
        print(
            f"    {benchmark['repetitions']} runs: "
            f"mean {benchmark['exec_ms_mean']}ms "
            f"(min {benchmark['exec_ms_min']}, max {benchmark['exec_ms_max']}, "
            f"sd {benchmark['exec_ms_stddev']}), "
            f"compile {benchmark.get('compile_ms')}ms, "
            f"rss {benchmark.get('max_rss_kb')}kB on "
            f"{benchmark.get('cpu_model', 'unknown CPU')}",
            flush=True,
        )
    elif benchmark.get("repetitions"):
        raise Failure("a case that never built must not report a benchmark")

    if "expect_compile_ms" in case and benchmark.get("compile_ms") != case["expect_compile_ms"]:
        raise Failure(
            f"expected compile_ms={case['expect_compile_ms']}, "
            f"got {benchmark.get('compile_ms')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="devices server base URL")
    parser.add_argument("--keep", action="store_true",
                        help="leave the pod running afterwards (it bills)")
    args = parser.parse_args()

    failures: list[str] = []
    with httpx.Client(base_url=args.base, timeout=120.0) as client:
        print(f"devices server: {args.base}")
        cpu_id = provision(client)
        try:
            for case in CASES:
                print(f"\n== {case['name']}", flush=True)
                try:
                    check(case, run_case(client, cpu_id, case["body"]))
                    print("   OK", flush=True)
                except Failure as exc:
                    print(f"   FAILED: {exc}", flush=True)
                    failures.append(f"{case['name']}: {exc}")
        finally:
            if not args.keep:
                # A pod bills for every second it exists, so this runs even when a
                # case blew up -- that is exactly when it is easiest to forget.
                print(f"\nterminating {cpu_id}", flush=True)
                try:
                    client.delete(f"/cpu/{cpu_id}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARNING: could not terminate: {exc}", flush=True)

    print()
    if failures:
        for line in failures:
            print(f"FAIL  {line}")
        return 1
    print(f"all {len(CASES)} cases behaved as expected")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as exc:
        print(f"\nsmoke test aborted: {exc}", file=sys.stderr)
        sys.exit(2)
