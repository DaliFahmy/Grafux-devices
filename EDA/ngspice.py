"""
ngspice.py
The analogue_simulator block's domain knowledge: turn block ports into an ngspice
deck, and turn ngspice's rawfile and log back into ports.

STDLIB ONLY, and that is load-bearing, not tidiness.  This file is used in three
places that must agree byte for byte:

* the devices server imports it to build the deck (``build_ngspice_deck``);
* ``run_analogue_simulator`` UPLOADS THIS FILE to the pod and runs
  ``python3 grafux_ngspice.py postprocess`` there, so a million-point transient
  is decimated beside the rawfile instead of crossing SSH -- the image's python3
  has no pip packages;
* the image's CI smoke test imports it to generate its decks and to check what
  ngspice wrote.

Nothing here opens a socket or imports paramiko/pydantic.  Requests are read by
``getattr`` so any object with the port names works (the pydantic model, a
SimpleNamespace in a test).

Facts this file encodes, read from the pinned releases rather than assumed
(open_pdks 1689ac3f via ciel-releases, ngspice 47):

* sky130A: ``libs.tech/ngspice/sky130.lib.spice`` has sections tt ss ff sf fs,
  ll hh hl lh and the ``*_mm`` mismatch variants; it ``.include``s
  ``../../libs.ref/sky130_fd_pr/spice/...`` by relative path.  MOSFETs are
  SUBCIRCUITS (``XM1 d g s b sky130_fd_pr__nfet_01v8 W=1 L=0.15``).
* gf180mcuD: ``libs.tech/ngspice/sm141064.ngspice`` has MOS sections typical ff
  ss fs sf plus separate res_/mimcap_/moscap_/bjt_/diode_ corner sections, and
  ``design.ngspice`` must be included first (it sets the statistical switches).
  MOSFETs are MODELS (``M1 d g s b nfet_03v3 W=1u L=0.28u``).
* both PDKs' own spinit asks for ``set ngbehavior=hsa`` and ``set ng_nomodcheck``.
"""

import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Files on the pod.  All relative to WORK_DIR, which is the simulation's cwd, so
# a model file staged through the `files` port is `.include "mymodel.lib"`.
# ---------------------------------------------------------------------------

DECK_FILE = "deck.sp"
LOG_FILE = "sim.log"
RAW_FILE = "sim.raw"
SPICEINIT_FILE = ".spiceinit"
SCRIPT_FILE = "grafux_ngspice.py"
WAVEFORMS_FILE = "waveforms.csv"

DEFAULT_PDK_ROOT = "/opt/pdks"

DEFAULT_MAX_POINTS = 2000
MIN_POINTS = 10
MAX_POINTS = 20000

# The plotter block draws at most 8 series (PlotData::kMaxSeries in the app).
# More columns than that would be silently dropped THERE; dropping them here is
# the same outcome with a warning that says which.
MAX_SERIES = 8

# ---------------------------------------------------------------------------
# PDKs
# ---------------------------------------------------------------------------

PDK_NONE = "none"

NGSPICE_PDKS: Dict[str, Dict[str, Any]] = {
    "sky130A": {
        "default_corner": "tt",
        "corners": (
            "tt", "ss", "ff", "sf", "fs", "ll", "hh", "hl", "lh",
            "tt_mm", "ss_mm", "ff_mm", "sf_mm", "fs_mm",
        ),
        "nominal_vdd": "1.8",
        # (kind, path relative to $PDK_ROOT, section or "")
        "lines": (("lib", "sky130A/libs.tech/ngspice/sky130.lib.spice", "{corner}"),),
        "library_marker": "sky130.lib.spice",
        "spiceinit": ("set skywaterpdk",),
        "device_hint": "sky130_fd_pr__nfet_01v8 / sky130_fd_pr__pfet_01v8 (subcircuits: XM1 ...)",
    },
    "gf180mcuD": {
        "default_corner": "typical",
        "corners": ("typical", "ff", "ss", "fs", "sf"),
        "nominal_vdd": "3.3",
        "lines": (
            ("include", "gf180mcuD/libs.tech/ngspice/design.ngspice", ""),
            ("lib", "gf180mcuD/libs.tech/ngspice/sm141064.ngspice", "{corner}"),
            ("lib", "gf180mcuD/libs.tech/ngspice/sm141064.ngspice", "res_{passive}"),
            ("lib", "gf180mcuD/libs.tech/ngspice/sm141064.ngspice", "mimcap_{passive}"),
            ("lib", "gf180mcuD/libs.tech/ngspice/sm141064.ngspice", "moscap_{passive}"),
            ("lib", "gf180mcuD/libs.tech/ngspice/sm141064.ngspice", "diode_{passive}"),
            ("lib", "gf180mcuD/libs.tech/ngspice/sm141064.ngspice", "bjt_{passive}"),
        ),
        "library_marker": "sm141064",
        "spiceinit": (),
        "device_hint": "nfet_03v3 / pfet_03v3 (models: M1 ...)",
    },
}

_PDK_ALIASES = {
    # sky130hd/hs are the ORFS platform names every other EDA block on the canvas
    # uses; to a chat or a user they mean "the sky130 process", and refusing them
    # here would be pedantry about a naming split the user never chose.
    "": "sky130A", "sky130": "sky130A", "sky130a": "sky130A",
    "sky130hd": "sky130A", "sky130hs": "sky130A",
    "gf180": "gf180mcuD", "gf180mcu": "gf180mcuD", "gf180mcud": "gf180mcuD",
    "none": PDK_NONE, "generic": PDK_NONE, "no": PDK_NONE,
}

PDK_CHOICES = ["sky130A", "gf180mcuD", PDK_NONE]
DEFAULT_PDK = "sky130A"

# gf180's passive sections exist only as typical/ff/ss; a skewed MOS corner keeps
# typical passives, which is what the PDK's own xschem testbenches do.
_GF180_PASSIVE = {"typical": "typical", "ff": "ff", "ss": "ss", "fs": "typical", "sf": "typical"}


def normalize_pdk(text: str) -> Tuple[str, str]:
    """``(pdk, error)``: the canonical PDK name, case-insensitive, with aliases."""
    raw = (text or "").strip()
    if raw in NGSPICE_PDKS or raw == PDK_NONE:
        return raw, ""
    key = raw.lower()
    if key in _PDK_ALIASES:
        return _PDK_ALIASES[key], ""
    return "", ("Unknown pdk '{0}'. This image carries sky130A and gf180mcuD device "
                "models; use 'none' for a deck that brings its own models."
                .format(raw))


def normalize_corner(pdk: str, text: str) -> Tuple[str, str]:
    """``(corner, error)`` for a PDK; empty means the PDK's typical corner."""
    if pdk not in NGSPICE_PDKS:
        return (text or "").strip(), ""
    info = NGSPICE_PDKS[pdk]
    raw = (text or "").strip().lower()
    if not raw:
        return info["default_corner"], ""
    if pdk == "gf180mcuD" and raw in ("tt", "typ", "nom", "nominal"):
        return "typical", ""
    if pdk == "sky130A" and raw in ("typical", "typ", "nom", "nominal"):
        return "tt", ""
    if raw in info["corners"]:
        return raw, ""
    return "", "Corner '{0}' does not exist in {1}. Valid corners: {2}.".format(
        text.strip(), pdk, ", ".join(info["corners"]))


def pdk_library_lines(pdk: str, corner: str, pdk_root: str) -> List[str]:
    """The ``.include``/``.lib`` lines that load a PDK corner, with absolute paths."""
    info = NGSPICE_PDKS.get(pdk)
    if not info:
        return []
    root = (pdk_root or DEFAULT_PDK_ROOT).rstrip("/")
    passive = _GF180_PASSIVE.get(corner, "typical")
    out = []
    for kind, rel, section in info["lines"]:
        path = '"{0}/{1}"'.format(root, rel)
        if kind == "include":
            out.append(".include {0}".format(path))
        else:
            out.append(".lib {0} {1}".format(
                path, section.format(corner=corner, passive=passive)))
    return out


def pdk_library_paths(pdk: str, pdk_root: str) -> List[str]:
    """Every PDK file the injected lines reference, for an on-pod existence check."""
    info = NGSPICE_PDKS.get(pdk)
    if not info:
        return []
    root = (pdk_root or DEFAULT_PDK_ROOT).rstrip("/")
    seen: List[str] = []
    for _kind, rel, _section in info["lines"]:
        path = "{0}/{1}".format(root, rel)
        if path not in seen:
            seen.append(path)
    return seen


def spiceinit_for(pdk: str, threads: int = 0) -> str:
    """
    The ``.spiceinit`` written beside the deck.

    ngspice reads ``.spiceinit`` from the CURRENT directory before the deck, and
    ``ngbehavior`` must be set there -- set inside the deck's ``.control`` block it
    is too late, the model library has already been parsed.  ``filetype=ascii``
    is what makes the rawfile parseable by this module with no numpy.
    """
    lines = [
        "* written by Grafux for the analogue_simulator block",
        "set ngbehavior=hsa",
        "set ng_nomodcheck",
        "set filetype=ascii",
        "set noaskquit",
        # KLU is the sparse solver both PDKs' own spinit files select; the
        # built-in SPARSE 1.3 is markedly slower on a netlist of any size.  The
        # trade, measured on ngspice-47: SPARSE recovers a capacitor-only
        # floating node through transient-op where KLU fails.  `.option sparse`
        # in the deck switches back (verified), and the failure hint says so.
        "option klu",
    ]
    info = NGSPICE_PDKS.get(pdk)
    if info:
        lines.extend(info["spiceinit"])
    if threads > 0:
        lines.append("set num_threads={0}".format(int(threads)))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Deck construction
# ---------------------------------------------------------------------------

_ANALYSIS_WORDS = ("tran", "ac", "dc", "op", "noise", "disto", "tf", "sens", "pz", "sp", "pss")
_DOT_ANALYSIS_RE = re.compile(
    r"^\s*\.(" + "|".join(_ANALYSIS_WORDS) + r")\b(.*)$", re.IGNORECASE)
_CONTROL_START_RE = re.compile(r"^\s*\.control\b", re.IGNORECASE)
_CONTROL_END_RE = re.compile(r"^\s*\.endc\b", re.IGNORECASE)
_END_RE = re.compile(r"^\s*\.end\s*$", re.IGNORECASE)
_TEMP_RE = re.compile(r"^\s*\.temp\b", re.IGNORECASE | re.MULTILINE)
_VDD_PARAM_RE = re.compile(r"^\s*\.param\b[^\n]*\bvdd\s*=", re.IGNORECASE | re.MULTILINE)
_MEAS_RE = re.compile(
    r"^\s*\.meas(?:ure)?\s+(tran|ac|dc|op|noise|sp|tf)\s+([A-Za-z_][\w.]*)",
    re.IGNORECASE | re.MULTILINE)
_NUMBER_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_SI_NUMBER_RE = re.compile(
    r"^\s*(" + _NUMBER_RE + r")\s*(meg|[tgkmunpfa])?\s*[a-z]*\s*$", re.IGNORECASE)
_SI = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6,
       "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}


def parse_spice_number(text: str) -> Optional[float]:
    """A SPICE number with an optional scale suffix (``1.8``, ``10m``, ``2meg``)."""
    m = _SI_NUMBER_RE.match(text or "")
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower()
    return value * _SI.get(suffix, 1.0)


def _int_or(text: str, fallback: int) -> int:
    try:
        return int(float((text or "").strip()))
    except (TypeError, ValueError):
        return fallback


def _port(req: Any, name: str) -> str:
    value = getattr(req, name, "") or ""
    return value if isinstance(value, str) else str(value)


def _lines_of(text: str) -> List[str]:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def split_analyses(text: str) -> List[str]:
    """
    The ``analyses`` port, one analysis per line (``;`` also separates).

    A leading dot is accepted and removed: ``.tran 1n 10m`` and ``tran 1n 10m``
    mean the same thing on this port, because both spellings appear in every
    tutorial and refusing one would be pedantry.
    """
    out = []
    for chunk in re.split(r"[\n;]", text or ""):
        line = chunk.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("."):
            line = line[1:]
        out.append(line)
    return out


def has_control_block(netlist: str) -> bool:
    return any(_CONTROL_START_RE.match(line) for line in _lines_of(netlist))


def dot_analyses(netlist: str) -> List[str]:
    """Analysis cards in the deck OUTSIDE any .control block, without the dot."""
    out = []
    inside = False
    for line in _lines_of(netlist):
        if _CONTROL_START_RE.match(line):
            inside = True
            continue
        if _CONTROL_END_RE.match(line):
            inside = False
            continue
        m = None if inside else _DOT_ANALYSIS_RE.match(line)
        if m:
            out.append((m.group(1).lower() + m.group(2)).strip())
    return out


def measure_names(deck: str) -> List[str]:
    """The names every ``.meas`` statement in a deck declares, in order, unique."""
    seen: List[str] = []
    for m in _MEAS_RE.finditer(deck or ""):
        name = m.group(2).lower()
        if name not in seen:
            seen.append(name)
    return seen


def references_pdk_library(netlist: str, pdk: str) -> bool:
    """Whether the user's deck already loads this PDK's models itself."""
    info = NGSPICE_PDKS.get(pdk)
    if not info:
        return False
    marker = info["library_marker"].lower()
    for line in _lines_of(netlist):
        low = line.strip().lower()
        if (low.startswith(".lib") or low.startswith(".include") or low.startswith(".inc ")) \
                and marker in low:
            return True
    return False


def analogue_preflight(req: Any) -> str:
    """
    Everything decidable from the request alone, refused before a pod is touched.

    Returns the refusal text, or "" when the run may proceed.
    """
    netlist = _port(req, "netlist")
    if not netlist.strip():
        return ("The netlist port is empty. Wire a SPICE netlist into it, or write "
                "what the circuit should be on block_description so one is generated.")
    pdk, error = normalize_pdk(_port(req, "pdk"))
    if error:
        return error
    _corner, error = normalize_corner(pdk, _port(req, "corner"))
    if error:
        return error
    for field, label in (("temperature", "temperature"), ("supply_voltage", "supply_voltage")):
        text = _port(req, field).strip()
        if text and parse_spice_number(text) is None:
            return "{0} must be a number, got '{1}'.".format(label, text)
    max_points = _port(req, "max_points").strip()
    if max_points:
        n = _int_or(max_points, -1)
        if n < MIN_POINTS or n > MAX_POINTS:
            return "max_points must be between {0} and {1}, got '{2}'.".format(
                MIN_POINTS, MAX_POINTS, max_points)
    if not has_control_block(netlist) and not dot_analyses(netlist) \
            and not split_analyses(_port(req, "analyses")):
        return ("The deck has no analysis, so there is nothing to simulate. Add a "
                ".tran/.ac/.dc/.op card to the netlist, or put one on the analyses "
                "port (for example: tran 10p 5n).")
    return ""


def _title_needed(first_line: str) -> bool:
    """
    SPICE throws away the first line of a deck as its title.

    A netlist that starts straight into a dot-card -- a bare ``.subckt`` wired from
    openram's `spice` port is the common case -- would lose that card silently, so
    a title is prepended.  Only a leading dot or continuation is treated as
    "not a title": an element line and a free-text title are indistinguishable
    ("Vin in 0 1" vs "Voltage divider test"), and guessing wrong in that case
    would turn a real title into a broken element.
    """
    stripped = first_line.strip()
    return stripped.startswith(".") or stripped.startswith("+")


def build_ngspice_deck(req: Any, *, pdk_root: str = DEFAULT_PDK_ROOT,
                       raw_file: str = RAW_FILE) -> Tuple[str, List[str], List[str]]:
    """
    The deck ngspice actually runs: ``(deck, analyses_run, notes)``.

    The user's text is kept verbatim except for three edits, each noted:

    1. the PDK's ``.lib`` lines are added -- only when the deck does not already
       load that library itself (a wired deck from xschem usually does);
    2. dot-analysis cards are commented out and re-issued as ``.control``
       commands, each followed by ``write`` with ``appendwrite`` -- because a
       plain ``write`` after ``run`` saves only the LAST plot, so a deck with
       ``.op`` and ``.tran`` would silently lose its operating point;
    3. ``.end`` is moved to the end, after the control block.

    A deck that brings its own ``.control`` block is not rewritten at all beyond
    the library lines: the user is scripting ngspice themselves, and the notes say
    that waveforms then depend on that script writing a rawfile.
    """
    netlist = _port(req, "netlist")
    notes: List[str] = []
    pdk, _ = normalize_pdk(_port(req, "pdk"))
    corner, _ = normalize_corner(pdk, _port(req, "corner"))

    lines = _lines_of(netlist)
    while lines and not lines[-1].strip():
        lines.pop()
    # Drop the final .end; it is re-added after the control block.
    for i in range(len(lines) - 1, -1, -1):
        if _END_RE.match(lines[i]):
            del lines[i]
            break

    first_real = next((ln for ln in lines if ln.strip()), "")
    if not lines or _title_needed(first_real):
        lines.insert(0, "* Grafux analogue_simulator deck")

    own_control = has_control_block(netlist)
    port_analyses = split_analyses(_port(req, "analyses"))
    deck_analyses = dot_analyses(netlist)

    header: List[str] = []
    if pdk in NGSPICE_PDKS:
        if references_pdk_library(netlist, pdk):
            notes.append("The netlist loads the {0} library itself, so no .lib line "
                         "was added and the corner port was not applied.".format(pdk))
        else:
            header.append("* --- {0} models, corner {1} (added by Grafux) ---".format(pdk, corner))
            header.extend(pdk_library_lines(pdk, corner, pdk_root))

    temperature = _port(req, "temperature").strip()
    if temperature:
        if _TEMP_RE.search(netlist):
            notes.append("The netlist sets .temp itself; the temperature port was not applied.")
        else:
            header.append(".temp {0}".format(temperature))

    supply = _port(req, "supply_voltage").strip()
    if supply:
        if _VDD_PARAM_RE.search(netlist):
            notes.append("The netlist defines the vdd parameter itself; the "
                         "supply_voltage port was not applied.")
        else:
            header.append(".param vdd={0}".format(supply))

    meas = [ln for ln in _lines_of(_port(req, "meas_statements")) if ln.strip()]
    for i, ln in enumerate(meas):
        stripped = ln.strip()
        if not stripped.startswith(".") and not stripped.startswith("*") \
                and not stripped.startswith("+"):
            meas[i] = "." + stripped

    body = [lines[0]] + header + lines[1:]
    if meas:
        body.append("* --- measurements (meas_statements port) ---")
        body.extend(meas)

    analyses_run: List[str] = []
    if own_control:
        if port_analyses:
            notes.append("The netlist has its own .control block, so the analyses "
                         "port was ignored.")
        notes.append("The netlist has its own .control block. It runs as written; "
                     "waveforms appear only if it writes a rawfile (write sim.raw).")
        analyses_run = deck_analyses
    else:
        if port_analyses:
            analyses_run = port_analyses
            if deck_analyses:
                notes.append("The analyses port replaced the netlist's own analysis "
                             "cards ({0}).".format("; ".join(deck_analyses)))
        else:
            analyses_run = deck_analyses
        # Comment the dot-analysis cards out: they are issued below instead, and
        # leaving them in would run every analysis twice.
        for i, ln in enumerate(body):
            if _DOT_ANALYSIS_RE.match(ln):
                body[i] = "* (run from .control by Grafux) " + ln.strip()

        probes = split_probes(_port(req, "probes"))
        # The runner deletes the previous rawfile before ngspice starts; with
        # appendwrite on, a stale one would be read back as this run's result.
        control = [".control", "set filetype=ascii", "set appendwrite"]
        if probes:
            # `save` must precede the analyses.  The scale vector (time,
            # frequency, sweep) is always kept by ngspice regardless.
            control.append("save " + " ".join(probes))
        for analysis in analyses_run:
            control.append(analysis)
            control.append("write {0}".format(raw_file))
        extra = [ln for ln in _lines_of(_port(req, "extra_control")) if ln.strip()]
        control.extend(ln.strip() for ln in extra)
        control.extend(["quit", ".endc"])
        body.append("* --- simulation control (added by Grafux) ---")
        body.extend(control)

    body.append(".end")
    return "\n".join(body) + "\n", analyses_run, notes


def split_probes(text: str) -> List[str]:
    """``probes``: vectors to save and plot, separated by commas, spaces or lines."""
    out: List[str] = []
    for token in re.split(r"[\s,;]+", text or ""):
        token = token.strip()
        if token and token.lower() not in (p.lower() for p in out):
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Rawfile parsing (ASCII)
# ---------------------------------------------------------------------------

def _parse_value(token: str) -> Any:
    if "," in token:
        re_part, _, im_part = token.partition(",")
        return complex(float(re_part), float(im_part))
    return float(token)


def iter_raw_plots(lines: Iterable[str], keep_points: Optional[int] = None):
    """
    Stream the plots of an ASCII rawfile.

    Yields ``{"title", "name", "flags", "variables": [(name, type)], "npoints",
    "points": [[v0, v1, ...]], "stride"}``.  ``keep_points`` decimates while
    reading: every ``stride``-th point is kept plus the last one, where
    ``stride = ceil(npoints / keep_points)`` from the header's point count -- so
    memory is bounded by ``keep_points`` however long the transient.

    A file cut short (ngspice killed by the timeout) yields what was read, which
    is the useful behaviour: a partial waveform explains a hang better than none.
    """
    it = iter(lines)
    header: Dict[str, str] = {}
    for raw_line in it:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key_l = key.strip().lower()
        if not sep:
            continue
        if key_l == "variables":
            nvars = _int_or(header.get("no. variables", "0"), 0)
            variables = []
            while len(variables) < nvars:
                var_line = next(it, None)
                if var_line is None:
                    break
                parts = var_line.split()
                if len(parts) >= 2:
                    variables.append((parts[1], parts[2] if len(parts) > 2 else ""))
            header["_variables"] = json.dumps(variables)
            continue
        if key_l in ("values", "binary"):
            if key_l == "binary":
                raise ValueError("binary rawfile; set filetype=ascii")
            variables = [tuple(v) for v in json.loads(header.get("_variables", "[]"))]
            nvars = len(variables)
            npoints = _int_or(header.get("no. points", "0"), 0)
            stride = 1
            if keep_points and npoints > keep_points:
                stride = int(math.ceil(npoints / float(keep_points)))
            plot = {
                "title": header.get("title", ""),
                "name": header.get("plotname", ""),
                "flags": header.get("flags", ""),
                "variables": variables,
                "npoints": npoints,
                "stride": stride,
                "points": [],
            }
            index = -1
            current: List[Any] = []
            last: List[Any] = []
            read = 0
            while read < npoints:
                values: List[Any] = []
                while len(values) < nvars:
                    value_line = next(it, None)
                    if value_line is None:
                        break
                    parts = value_line.split()
                    if not parts:
                        continue
                    if len(values) == 0 and len(parts) >= 2:
                        parts = parts[1:]          # leading point index
                    for tok in parts:
                        values.append(_parse_value(tok))
                if len(values) < nvars:
                    break
                index += 1
                read += 1
                last = values
                if index % stride == 0:
                    plot["points"].append(values)
                    current = values
            if last and last is not current:
                plot["points"].append(last)
            plot["npoints_read"] = read
            yield plot
            header = {}
            continue
        header[key_l] = value.strip()


def parse_ascii_raw(text: str, keep_points: Optional[int] = None) -> List[Dict[str, Any]]:
    return list(iter_raw_plots(_lines_of(text), keep_points=keep_points))


# ---------------------------------------------------------------------------
# Waveforms, operating point
# ---------------------------------------------------------------------------

def _plot_kind(plot: Dict[str, Any]) -> str:
    name = (plot.get("name") or "").lower()
    for word, kind in (("operating point", "op"), ("transient", "tran"), ("ac ", "ac"),
                       ("dc transfer", "dc"), ("noise", "noise"), ("transfer function", "tf")):
        if word in name + " ":
            return kind
    return "other"


def _norm_vec(name: str) -> str:
    return name.strip().lower()


def _probe_matches(var: str, probe: str) -> bool:
    v, p = _norm_vec(var), _norm_vec(probe)
    if v == p:
        return True
    # A probe written as a bare node name matches v(node); v(x) matches a bare x.
    inner_v = v[2:-1] if v.startswith("v(") and v.endswith(")") else v
    inner_p = p[2:-1] if p.startswith("v(") and p.endswith(")") else p
    return inner_v == inner_p and not v.startswith("i(") and not p.startswith("i(")


def _fmt(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        return ""
    return "{0:.6g}".format(value)


def operating_point(plots: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """``{vector: value}`` from the first operating-point plot, real parts only."""
    for plot in plots:
        if _plot_kind(plot) == "op" and plot.get("points"):
            row = plot["points"][0]
            out: Dict[str, float] = {}
            for (name, _type), value in zip(plot["variables"], row):
                out[name] = float(value.real if isinstance(value, complex) else value)
            return out
    return {}


def choose_waveform_plot(plots: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The plot the `waveforms` port shows: the first swept (non-op) plot with points."""
    for plot in plots:
        if _plot_kind(plot) != "op" and len(plot.get("points") or []) > 1:
            return plot
    return None


def waveforms_csv(plot: Optional[Dict[str, Any]], probes: Sequence[str],
                  max_series: int = MAX_SERIES) -> Tuple[str, List[str]]:
    """
    ``(csv, notes)`` for one plot, in the shape the plotter block parses: a header
    row, the scale in the first column, one series per remaining column.

    Complex plots (AC) become magnitude in dB and phase in degrees -- two series
    per vector, which is what a Bode plot is -- and a real scale is taken from the
    real part of the complex frequency vector.
    """
    notes: List[str] = []
    if not plot:
        return "", notes
    variables = plot["variables"]
    if not variables:
        return "", notes
    is_complex = "complex" in (plot.get("flags") or "").lower()
    scale_name = variables[0][0]

    candidates = list(range(1, len(variables)))
    if probes:
        chosen = [i for i in candidates
                  if any(_probe_matches(variables[i][0], p) for p in probes)]
        missing = [p for p in probes
                   if not any(_probe_matches(variables[i][0], p) for i in candidates)]
        if missing:
            notes.append("Probe(s) not found in the {0} results: {1}.".format(
                plot.get("name", "simulation"), ", ".join(missing)))
        if not chosen:
            chosen = candidates
    else:
        # Node voltages first, then branch currents, then anything else.
        def rank(i: int) -> int:
            vtype = (variables[i][1] or "").lower()
            return 0 if vtype == "voltage" else 1 if vtype == "current" else 2
        chosen = sorted(candidates, key=rank)

    per_vector = 2 if is_complex else 1
    capacity = max(1, max_series // per_vector)
    if len(chosen) > capacity:
        dropped = [variables[i][0] for i in chosen[capacity:]]
        chosen = chosen[:capacity]
        notes.append("The waveforms port carries at most {0} series (the plotter's "
                     "limit); not shown: {1}. Name the vectors you want on the probes "
                     "port.".format(max_series, ", ".join(dropped)))

    header = [scale_name]
    for i in chosen:
        name = variables[i][0]
        if is_complex:
            # ngspice's own spelling: vdb(out), not vdb(v(out)).
            low = name.lower()
            inner = name[2:-1] if low.startswith("v(") and low.endswith(")") else name
            header.extend(["vdb({0})".format(inner), "vp({0})".format(inner)])
        else:
            header.append(name)
    rows = [",".join(header)]
    for row in plot["points"]:
        scale = row[0]
        cells = [_fmt(scale.real if isinstance(scale, complex) else scale)]
        for i in chosen:
            value = row[i]
            if is_complex:
                c = value if isinstance(value, complex) else complex(value, 0)
                # Floored rather than blank: an empty cell would shift a CSV
                # row's columns in a naive parser.  -400 dB is "zero" on any plot.
                cells.append(_fmt(20.0 * math.log10(max(abs(c), 1e-20))))
                cells.append(_fmt(math.degrees(math.atan2(c.imag, c.real))))
            else:
                cells.append(_fmt(value.real if isinstance(value, complex) else value))
        rows.append(",".join(cells))
    if plot.get("stride", 1) > 1:
        notes.append("{0}: {1} points decimated to {2} (1 in {3} kept; raise max_points "
                     "for more, the full data is on the raw port).".format(
                         plot.get("name", "plot"), plot.get("npoints_read", plot["npoints"]),
                         len(plot["points"]), plot["stride"]))
    return "\n".join(rows) + "\n", notes


def postprocess(raw_text_lines: Iterable[str], *, probes: Sequence[str],
                max_points: int) -> Dict[str, Any]:
    """Everything the ports need from a rawfile, as one small JSON-able dict."""
    plots = list(iter_raw_plots(raw_text_lines, keep_points=max_points))
    wave_plot = choose_waveform_plot(plots)
    csv_text, notes = waveforms_csv(wave_plot, probes)
    return {
        "plots": [{"name": p["name"], "kind": _plot_kind(p),
                   "points": p.get("npoints_read", p["npoints"]),
                   "vectors": [v[0] for v in p["variables"]]} for p in plots],
        "waveform_plot": wave_plot["name"] if wave_plot else "",
        "waveforms": csv_text,
        "operating_point": operating_point(plots),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_MEAS_VALUE_RE_TMPL = r"^\s*{name}\s*=\s*(" + _NUMBER_RE + r")"

# What makes a run RED.  Read off real ngspice-47 logs, and two of the choices
# are the opposite of what the words suggest:
#
# * "singular matrix", "gmin stepping failed" and "source stepping failed" are
#   NOT fatal on their own.  ngspice prints them as Warning lines while it falls
#   back through its homotopy methods, and then "Transient op finished
#   successfully" and the run completes.  They only matter when an Error line
#   (or no rawfile) follows, which the Error pattern already catches.
# * "... is not a valid resistor instance line, ignored!" IS fatal, although it
#   is printed as a Warning and ngspice exits 0: the element is dropped and a
#   different circuit is simulated.  A green run of the wrong circuit is the
#   worst outcome this block can produce.
_FATAL_PATTERNS = (
    re.compile(r"^\s*error\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*simulation\(s\) aborted.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*timestep too small.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*no simulations run.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*is not a valid .* line, ignored.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*fatal\b.*$", re.IGNORECASE | re.MULTILINE),
)
_WARNING_RE = re.compile(r"^\s*warning\b.*$", re.IGNORECASE | re.MULTILINE)

# Printed on EVERY deck in the hsa compatibility mode the PDKs require, whether or
# not the deck uses m= at all.  Left on the warnings port it would teach the
# reader to ignore that port.
_NOISE = ("m=xx on .subckt line will override multiplier m hierarchy",)

_HINTS = (
    (re.compile(r"unknown subckt|unable to find definition of model|could not find",
                re.IGNORECASE),
     "A device name does not exist in the loaded models. Check the pdk and the device "
     "names: sky130A uses subcircuits such as XM1 d g s b sky130_fd_pr__nfet_01v8, "
     "gf180mcuD uses models such as M1 d g s b nfet_03v3."),
    (re.compile(r"timestep too small", re.IGNORECASE),
     "The transient did not converge. Common causes: an ideal source driving a "
     "capacitor-only node, a step with zero rise time, or a floating node. Give "
     "pulses a finite rise/fall time and add a large resistor to ground on floating nodes."),
    (re.compile(r"singular matrix|gmin stepping failed|source stepping failed", re.IGNORECASE),
     "The operating point could not be solved -- usually a node with no DC path to "
     "ground (a gate or a capacitor-only node) or a loop of ideal voltage sources. "
     "The real fix is a DC path (e.g. 1G to ground). As a workaround, .option sparse "
     "in the deck selects ngspice's older solver, which recovers from some of these "
     "where the default KLU solver gives up."),
    (re.compile(r"is not a valid .* line, ignored", re.IGNORECASE),
     "An element line has too few nodes or no value, and ngspice DROPPED it -- the "
     "circuit simulated is not the one written. Fix the line quoted above."),
    (re.compile(r"incomplete or empty netlist", re.IGNORECASE),
     "ngspice stopped before simulating: the netlist failed to load. The first "
     "Error line above is the cause; everything after it is a consequence."),
    (re.compile(r"no such vector|vector .* not found", re.IGNORECASE),
     "A probe or measurement names a vector the analysis did not produce. Node "
     "names are case-insensitive; currents are i(vsource)."),
)


def parse_measurements(log_text: str, names: Sequence[str]) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """
    ``({name: value | None}, notes)`` for the measurements a deck declared.

    Only DECLARED names are looked up: ngspice prints plenty of other
    ``name = value`` lines (parameter echoes, op prints), and harvesting them all
    would put noise on the port.  A declared name with no value is kept as
    ``None`` with a note, never dropped -- a missing key reads as "not measured",
    which is a different statement from "measured and failed".
    """
    results: Dict[str, Optional[float]] = {}
    notes: List[str] = []
    text = log_text or ""
    for name in names:
        pattern = re.compile(_MEAS_VALUE_RE_TMPL.format(name=re.escape(name)),
                             re.IGNORECASE | re.MULTILINE)
        matches = pattern.findall(text)
        if matches:
            try:
                results[name] = float(matches[-1])
                continue
            except ValueError:
                pass
        results[name] = None
        reason = ""
        for line in text.splitlines():
            low = line.lower()
            if name in low and ("fail" in low or "out of interval" in low or "error" in low):
                reason = line.strip()
                break
        notes.append("Measurement {0} produced no value{1}.".format(
            name, " -- ngspice said: " + reason if reason else
            " (its trigger/target condition never occurred in the simulated window)"))
    return results, notes


def classify_ngspice_log(log_text: str, exit_code: int) -> Tuple[List[str], List[str], List[str]]:
    """
    ``(errors, warnings, hints)`` from an ngspice log.

    ngspice exits 0 after many failures (a missing model, an aborted transient),
    so the log is the verdict's evidence, not the exit code.  An ``Error`` line
    about a MEASUREMENT is demoted to a warning: a measure whose trigger never
    fires is a statement about the circuit, and the simulation itself succeeded.
    """
    text = log_text or ""
    errors: List[str] = []
    warnings: List[str] = []
    for pattern in _FATAL_PATTERNS:
        for m in pattern.finditer(text):
            line = m.group(0).strip()
            low = line.lower()
            if "meas" in low:
                if line not in warnings:
                    warnings.append(line)
            elif line not in errors:
                errors.append(line)
    for m in _WARNING_RE.finditer(text):
        line = m.group(0).strip()
        if line in errors or any(noise in line.lower() for noise in _NOISE):
            continue
        if line not in warnings:
            warnings.append(line)
    if exit_code not in (0, None) and not errors:
        errors.append("ngspice exited with code {0}.".format(exit_code))
    # Hints explain a FAILURE.  On a green run the same words are ngspice
    # recovering (see _FATAL_PATTERNS), and advice about a problem that solved
    # itself is noise.
    hints = [hint for pattern, hint in _HINTS if pattern.search(text)] if errors else []
    return errors[:20], warnings[:20], hints


_VERSION_RE = re.compile(r"ngspice-(\d+(?:\.\d+)*)", re.IGNORECASE)


def parse_version(text: str) -> str:
    m = _VERSION_RE.search(text or "")
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# On-pod entry point
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str]) -> int:
    """
    ``python3 grafux_ngspice.py postprocess RAWFILE MAX_POINTS PROBES_JSON``

    Prints one JSON object.  A missing rawfile is an answer (``plots: []``), not
    a crash: a deck that failed to load writes none, and the log explains why.
    """
    if len(argv) < 2 or argv[1] != "postprocess":
        sys.stderr.write(_main.__doc__ or "")
        return 2
    raw_path = argv[2] if len(argv) > 2 else RAW_FILE
    max_points = _int_or(argv[3] if len(argv) > 3 else "", DEFAULT_MAX_POINTS)
    probes = json.loads(argv[4]) if len(argv) > 4 and argv[4] else []
    try:
        with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
            result = postprocess(fh, probes=probes, max_points=max_points)
        result["raw_bytes"] = os.path.getsize(raw_path)
    except FileNotFoundError:
        result = {"plots": [], "waveform_plot": "", "waveforms": "",
                  "operating_point": {}, "notes": [], "missing_raw": True}
    except ValueError as exc:
        result = {"plots": [], "waveform_plot": "", "waveforms": "",
                  "operating_point": {}, "notes": ["Could not read the rawfile: {0}".format(exc)]}
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
