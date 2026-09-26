#!/usr/bin/env python3
"""
End-to-end smoke test for the CUSTOM (manifest-defined) block path.

Drives the hand-written OpenRAM example (CUSTOM/examples/openram) through the
REAL /custom API -- provision from the manifest's image, run with the manifest
and an inputs map, poll, collect -- and asserts the same views the native
openram block produces.  This is the Phase-1 acceptance test: if the generic
runner cannot reproduce openram, nothing the Generator writes will work either.

A SCRIPT, not a pytest: it rents a machine.

  Local container (free). Needs the example image built or pulled:

      docker build -t grafux-gen-example CUSTOM/examples/openram
        (or: docker pull ghcr.io/dalifahmy/grafux-gen:example-openram-1)
      ssh-keygen -t rsa -b 2048 -f /tmp/eda_key -N ''
      docker run -d --name grafux-custom -p 2222:22 \\
          -e PUBLIC_KEY="$(cat /tmp/eda_key.pub)" grafux-gen-example
      EDA_LOCAL_SSH=1 EDA_LOCAL_KEY=/tmp/eda_key uvicorn device.app:app --port 8000
      python scripts/e2e_custom_smoke.py

  Real RunPod (the image must be PUBLIC on GHCR -- run the
  "Build custom example image" workflow first):

      RUNPOD_API_KEY=rp_... uvicorn device.app:app --port 8000
      python scripts/e2e_custom_smoke.py

Exit code 0 only if the block came back with every expected port.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

BASE = os.environ.get("EDA_E2E_BASE", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "..", "CUSTOM", "examples", "openram", "grafux-block.json")
TEXT_PORTS = ("status", "top", "tech_name", "verilog_model", "config", "stats", "log")
ARTIFACT_PORTS = ("gds", "lef", "lib", "spice")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    manifest_text = open(MANIFEST_PATH, encoding="utf-8").read()
    m = json.loads(manifest_text)

    with httpx.Client(timeout=60.0) as c:
        v = c.post(f"{args.base}/custom/manifest/validate", json={"manifest": manifest_text}).json()
        if not v["ok"]:
            print("manifest invalid:", v["errors"])
            return 1
        r = c.post(f"{args.base}/custom/create_async", json={
            "image": m["runtime"]["image"], "compute_type": "CPU",
            "container_disk_gb": m["runtime"]["disk_gb"], "name": "e2e-custom"})
        r.raise_for_status()
        eda_id = r.json()["eda_id"]
        print("eda_id", eda_id)
        t0 = time.monotonic()
        while True:
            s = c.get(f"{args.base}/custom/{eda_id}/status", params={"live": "true"}).json()
            if s.get("phase") == "ready":
                break
            if s.get("phase") == "error":
                print("provisioning failed:", s)
                return 1
            if time.monotonic() - t0 > 900:
                print("provisioning timed out")
                return 1
            time.sleep(3)
        print(f"ready in {time.monotonic() - t0:.1f}s")

        c.post(f"{args.base}/custom/{eda_id}/run", json={
            "manifest": manifest_text,
            "inputs": {"word_size": "2", "num_words": "16", "tech_name": "scn4m_subm"},
        }).raise_for_status()
        t1 = time.monotonic()
        while not c.get(f"{args.base}/custom/{eda_id}/status").json().get("done"):
            if time.monotonic() - t1 > args.timeout:
                print("run timed out")
                return 1
            time.sleep(3)
        res = c.get(f"{args.base}/custom/{eda_id}/result", timeout=120.0).json()
        c.delete(f"{args.base}/custom/{eda_id}")

    out = res.get("outputs", {})
    print(f"ran in {time.monotonic() - t1:.1f}s  status={out.get('status')}")
    problems = [f"text port {p} empty" for p in TEXT_PORTS if not str(out.get(p, "")).strip()]
    problems += [f"artifact port {p} empty" for p in ARTIFACT_PORTS if not str(out.get(p, "")).strip()]
    if out.get("status") != "ok":
        problems.insert(0, f"status {out.get('status')}: {out.get('errors')}")
    if "module" not in out.get("verilog_model", ""):
        problems.append("verilog_model does not look like Verilog")
    for a in res.get("artifacts", []):
        print("  artifact:", a.get("path", "").rsplit("/", 1)[-1])
    if problems:
        print("FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("OK -- the custom openram block reproduced the native block's views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
