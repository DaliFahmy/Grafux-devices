#!/usr/bin/env python3
"""
End-to-end smoke test for the analogue_simulator (ngspice) path.

Drives a REAL analogue_simulator block through the REAL devices API -- provision,
run, poll, collect.  A SCRIPT, not a pytest, because on RunPod it rents a machine.
Run it on the free local-container path first:

    ssh-keygen -t rsa -b 2048 -f /tmp/eda_key -N ''
    docker run -d --name grafux-ngspice -p 2222:22 \\
        -e PUBLIC_KEY="$(cat /tmp/eda_key.pub)" ghcr.io/dalifahmy/grafux-ngspice:<tag>
    EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/eda_key uvicorn device.app:app --port 8000
    python scripts/e2e_ngspice_smoke.py

Three cases, each asserting the ports the block would show:

  sky130   a sky130A inverter transient at the ss corner, 85 C, with a propagation
           delay .meas and two probes -- the PDK library injection, the corner,
           the temperature, the measurement and the plotter-shaped waveforms;
  gf180    a gf180mcuD inverter DC sweep plus an operating point -- the second
           PDK and the op/dc multi-plot rawfile;
  broken   a deck naming a device that does not exist -- must come back RED with
           the device-name hint, not green with empty ports.

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
RUN_TIMEOUT_S = int(os.environ.get("EDA_E2E_RUN_TIMEOUT", "900"))
POLL_S = 2.0
KIND = "analogue_simulator"

SKY130_INVERTER = """sky130 inverter
Vdd vdd 0 {vdd}
Vin in 0 PULSE(0 {vdd} 0 50p 50p 1n 2n)
XM1 out in 0 0 sky130_fd_pr__nfet_01v8 W=1 L=0.15
XM2 out in vdd vdd sky130_fd_pr__pfet_01v8 W=2 L=0.15
C1 out 0 10f
.tran 1p 4n
.end
"""

GF180_INVERTER = """gf180 inverter
Vdd vdd 0 {vdd}
Vin in 0 0
M1 out in 0 0 nfet_03v3 W=1u L=0.28u
M2 out in vdd vdd pfet_03v3 W=2u L=0.28u
"""

CASES = {
    "sky130": {
        "body": {"netlist": SKY130_INVERTER, "pdk": "sky130A", "corner": "ss",
                 "temperature": "85", "supply_voltage": "1.8", "probes": "v(in) v(out)",
                 "meas_statements": ".meas tran tpd TRIG v(in) VAL=0.9 RISE=1 "
                                    "TARG v(out) VAL=0.9 FALL=1"},
        "expect": "ok", "measure": "tpd", "header": "time,v(in),v(out)",
    },
    "gf180": {
        "body": {"netlist": GF180_INVERTER, "pdk": "gf180mcuD", "supply_voltage": "3.3",
                 "analyses": "op\ndc Vin 0 3.3 0.01",
                 "meas_statements": ".meas dc vm WHEN v(out)=v(in)"},
        "expect": "ok", "measure": "vm", "header": "v(v-sweep),",
    },
    "broken": {
        "body": {"netlist": "broken\nVdd vdd 0 1.8\n"
                            "XM1 vdd vdd 0 0 sky130_fd_pr__nfet_99v9 W=1 L=0.15\n.op\n",
                 "pdk": "sky130A"},
        "expect": "error", "error_fragment": "unknown subckt",
    },
}


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
        resp = client.get(f"{base}/{KIND}/{eda_id}/status",
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


def verify(name: str, case: dict, result: dict) -> None:
    outputs = result.get("outputs", {})
    artifacts = [a.get("path", "").rsplit("/", 1)[-1] for a in result.get("artifacts", [])]
    print(f"  status={outputs.get('status')} stage={result.get('stage')} artifacts={artifacts}")
    if case["expect"] == "error":
        check(outputs.get("status") == "error", f"{name}: a broken deck came back green")
        check(case["error_fragment"] in outputs.get("errors", ""),
              f"{name}: errors does not name the cause: {outputs.get('errors')!r}")
        check("Likely cause" in outputs.get("errors", ""), f"{name}: no hint on errors")
        return
    check(outputs.get("status") == "ok", f"{name}: {outputs.get('errors')}")
    measurements = json.loads(outputs.get("measurements") or "{}")
    print(f"  measurements={measurements}")
    check(isinstance(measurements.get(case["measure"]), float),
          f"{name}: measurement {case['measure']} has no value")
    waveforms = outputs.get("waveforms", "")
    check(waveforms.startswith(case["header"]),
          f"{name}: waveforms header is {waveforms.splitlines()[:1]}")
    check(len(waveforms.splitlines()) > 10, f"{name}: waveforms has almost no rows")
    check(outputs.get("netlist", "").rstrip().endswith(".end"), f"{name}: netlist echo missing")
    check("sim.raw" in artifacts and "sim.log" in artifacts, f"{name}: run files not downloaded")
    stats = json.loads(outputs.get("stats") or "{}")
    print(f"  stats: pdk={stats.get('pdk')} corner={stats.get('corner')} "
          f"version={stats.get('version')} plots={[p['name'] for p in stats.get('plots', [])]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--case", choices=sorted(CASES), action="append",
                        help="run only these cases (default: all)")
    args = parser.parse_args()

    failures = 0
    try:
        with httpx.Client(timeout=30.0) as client:
            created = client.post(f"{args.base}/{KIND}/create_async", json={"name": "e2e-ngspice"})
            created.raise_for_status()
            eda_id = created.json()["eda_id"]
            print(f"eda_id: {eda_id}")
            poll(client, args.base, eda_id, until="ready", timeout_s=PROVISION_TIMEOUT_S)
            for name in args.case or list(CASES):
                case = CASES[name]
                print(f"[{name}]")
                body = dict(case["body"], timeout=RUN_TIMEOUT_S)
                started = client.post(f"{args.base}/{KIND}/{eda_id}/run", json=body,
                                      timeout=120.0)
                started.raise_for_status()
                poll(client, args.base, eda_id, until="done", timeout_s=RUN_TIMEOUT_S)
                result = client.get(f"{args.base}/{KIND}/{eda_id}/result", timeout=120.0).json()
                try:
                    verify(name, case, result)
                    print("  OK")
                except Failure as exc:
                    failures += 1
                    print(f"  FAILED: {exc}")
    except (Failure, httpx.HTTPError) as exc:
        print(f"FAILED: {exc}")
        return 1
    print("OK" if failures == 0 else f"{failures} case(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
