#!/usr/bin/env bash
# Run the entry against the manifest's selftest.inputs and check expect_outputs.
set -eu
WORK=$(mktemp -d)
export GRAFUX_WORK="$WORK" GRAFUX_IN="$WORK/in" GRAFUX_OUT="$WORK/out"
mkdir -p "$GRAFUX_IN" "$GRAFUX_OUT/files"
python3 - <<'PY'
import json, os
m = json.load(open("/opt/grafux/grafux-block.json"))
vals = {p["name"]: p.get("default", "") for p in m["inputs"]}
vals.update(m.get("selftest", {}).get("inputs", {}))
for k, v in vals.items():
    open(os.path.join(os.environ["GRAFUX_IN"], k), "w").write(str(v))
PY
/opt/grafux/run.sh
python3 - <<'PY'
import glob, json, os, sys
m = json.load(open("/opt/grafux/grafux-block.json"))
out = os.environ["GRAFUX_OUT"]
ports = {p["name"]: p for p in m["outputs"]}
missing = []
for name in m.get("selftest", {}).get("expect_outputs", []):
    p = ports[name]
    if p.get("kind") == "artifact":
        ok = bool(glob.glob(os.path.join(out, p["glob"])))
    else:
        f = os.path.join(out, name)
        ok = os.path.isfile(f) and open(f).read().strip() != ""
    if not ok:
        missing.append(name)
if missing:
    sys.exit("selftest FAILED: no " + ", ".join(missing))
print("selftest OK")
PY
rm -rf "$WORK"
