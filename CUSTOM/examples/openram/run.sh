#!/usr/bin/env bash
# The custom-block entry: $GRAFUX_IN/<port> in, $GRAFUX_OUT/<port> out.
set -u
export OPENRAM_HOME="${OPENRAM_HOME:-/opt/openram/compiler}"
export OPENRAM_TECH="${OPENRAM_TECH:-/opt/openram/technology}"
export PYTHONPATH="$OPENRAM_HOME:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=/opt/openram-venv/bin/python3
[ -x "$PY" ] || PY=python3
exec "$PY" /opt/grafux/adapter.py
