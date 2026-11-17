"""Synthetic motor datasheets with known truth and logged traps (B6, 2026-11-16). No LLM is run here or on the output.

Each sheet describes a fictitious motor of a fictitious manufacturer ("Synthetic sheet — not a real product"), printed in one
of the six document styles of docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md §2:
  A variants in columns · B variants in rows · C single article · D web page without constants ·
  E catalogue page with curves · F rating plate
and carries traps from §3 of that survey (basis, waveform, Ke units, temperature, US units, unitless inertia, multiple
ratings, continuous vs peak, wrong label, internal inconsistency, decimal comma, curve-only value, missing value, unnamed
manufacturer). For each sheet it writes the HTML (and a PDF when an interpreter with Playwright is given), the expected
file in the golden schema (golden-expected/1, with must_ask as {field, reason, item, text}) plus the true SI values, and a
trap log line.

    python experiments/generate_synthetic_datasheets.py                      # 30 sheets in data/synthetic_datasheets/
    python experiments/generate_synthetic_datasheets.py --pdf-python /path/to/python-with-playwright

Everything is deterministic for a given --seed (default SEED). The output directory is outside git; the generator, the
seed, the trap log and three examples are committed (docs/audit/assistant/synthetic_datasheets/).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEED = 20261116
N_SHEETS = 30
STYLES = ("A", "B", "C", "D", "E", "F")
OUT = REPO / "data" / "synthetic_datasheets"
#: experiments/score_golden_extraction.py REASONS (closed list, decision D8)
REASONS = ("missing", "variant_unspecified", "ambiguous_basis", "ambiguous_waveform", "inconsistent", "ambiguous_meaning",
           "condition_unspecified", "machine_type_unspecified", "manufacturer_unnamed")
TRAPS = ("line_line_resistance", "ke_waveform_undeclared", "ke_units", "resistance_temperature", "us_units", "unitless_inertia",
         "multiple_ratings", "continuous_vs_peak", "wrong_label", "kt_ke_inconsistent", "decimal_comma", "curve_only", "missing_inductance",
         "manufacturer_unnamed")
OZIN_PER_NM = 141.611932  # 1 N·m = 141.6119 oz-in
LBFT2_PER_KGM2 = 23.730360  # 1 kg·m² = 23.73 lb-ft² (WK²)
FICTITIOUS = ["Northwind Synthetic Drives (fictitious)", "Atlas Test Motors (fictitious)", "Kestrel Demo Machines (fictitious)",
              "Orion Sample Motors (fictitious)", "Lumen Fictional Drives (fictitious)"]


def _truth(rng: random.Random, kind: str) -> dict:
    """Physically consistent SI values of one fictitious motor."""
    if kind == "dc":
        u = rng.choice([24.0, 36.0, 48.0, 60.0])
        kt = round(rng.uniform(0.05, 0.25), 4)  # N·m/A = V·s/rad (DC: Ke = Kt in SI)
        r = round(rng.uniform(0.02, 0.4), 4)
        i = round(rng.uniform(10, 80), 1)
        w = (u - r * i) / kt
        return {"machine": "dc", "u_v": u, "i_a": i, "kt_nm_per_a": kt, "ke_v_s_per_rad": kt, "r_ohm": r, "l_h": round(rng.uniform(50e-6, 800e-6), 6),
                "j_kgm2": round(rng.uniform(2e-4, 2e-2), 6), "speed_rpm": round(w * 60 / (2 * math.pi)), "torque_nm": round(kt * i * 0.92, 3), "pairs": None}
    p = rng.choice([2, 3, 4, 5])
    psi = round(rng.uniform(0.01, 0.08), 4)  # V·s, peak phase flux
    kt = round(1.5 * p * psi, 4)  # N·m/A per peak phase current
    r = round(rng.uniform(0.05, 1.5), 4)  # phase
    i = round(rng.uniform(3, 30), 2)
    return {"machine": "pmsm", "u_v": rng.choice([48.0, 230.0, 325.0, 400.0]), "i_a": i, "pole_pairs": p, "psi_v_s": psi, "kt_nm_per_a": kt,
            "ke_v_s_per_rad_phase_peak": round(p * psi, 5), "r_ohm": r, "l_h": round(rng.uniform(0.3e-3, 8e-3), 6),
            "j_kgm2": round(rng.uniform(1e-5, 3e-3), 7), "speed_rpm": rng.choice([1500, 2000, 3000, 4500]), "torque_nm": round(kt * i, 3), "pairs": p}


def _num(x: float, comma: bool, digits: int = 3) -> str:
    """Three significant digits, never in exponent notation (as datasheets print them)."""
    if x == 0:
        s = "0"
    else:
        decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
        s = f"{round(x, decimals):.{decimals}f}"
    return s.replace(".", ",") if comma else s


def _field(field, value, unit, label, basis="not_applicable", waveform="not_applicable", temperature=None, condition=None, variant=None):
    return {"field": field, "value_printed": value, "unit_printed": unit, "label_printed": label, "basis": basis, "waveform": waveform,
            "temperature_c": temperature, "condition": condition, "variant": variant}


def make_sheet(n: int, rng: random.Random) -> dict:
    """One sheet: printed rows, expected fields, must_ask, must_warn, traps, truth."""
    style = STYLES[n % len(STYLES)]
    kind = "dc" if style in ("A", "C", "D") and rng.random() < 0.6 else ("pmsm" if style != "F" else rng.choice(["dc", "pmsm"]))
    t = _truth(rng, kind)
    traps = set(rng.sample([x for x in TRAPS if x != "manufacturer_unnamed"], k=3))
    if style == "D":
        traps.add("missing_inductance")
    if style == "E":
        traps.add("curve_only")
    if style == "F":
        traps.add("multiple_ratings")
        traps.discard("curve_only")
    if rng.random() < 0.15:
        traps.add("manufacturer_unnamed")
    if kind == "dc":
        traps.discard("line_line_resistance")
        traps.discard("ke_waveform_undeclared")
    comma = "decimal_comma" in traps
    variants = [None]
    if style in ("A", "B"):
        variants = [f"{v:g} V" for v in sorted({t["u_v"], t["u_v"] * 2})]
    rows, fields, ask, warn, trap_log = [], [], [], [], []

    def add(field, value_si, unit, label, variant=None, **kw):
        txt = _num(value_si, comma) if isinstance(value_si, float) else str(value_si)
        rows.append((label, txt, unit, variant))
        fields.append(_field(field, txt, unit, label, variant=variant, **kw))

    for v in variants:
        scale = 1.0 if v is None or v.startswith(f"{t['u_v']:g} ") else 2.0
        add("voltage", t["u_v"] * scale, "V", "Rated voltage", v, waveform="dc_equivalent" if kind == "dc" else "rms")
        if "continuous_vs_peak" in traps:
            add("current", t["i_a"], "A", "Continuous current (S1)", v, condition="S1")
            add("current_peak", t["i_a"] * 2.5, "A", "Peak current (10 s)", v, condition="peak, 10 s")
        else:
            add("current", t["i_a"], "A", "Rated current", v)
        add("speed", float(t["speed_rpm"] * scale), "rpm", "Rated speed", v)
    if "continuous_vs_peak" in traps:
        warn.append("continuous (S1) and peak (10 s) currents are separate values; the peak is a limit, not a rating")
        trap_log.append(("continuous_vs_peak", "both printed, labelled"))
    if len(variants) > 1:
        ask.append({"field": "voltage", "reason": "variant_unspecified", "text": "which voltage variant (column/row) is the user's motor"})
    # torque, constants
    if "us_units" in traps:
        add("torque", t["torque_nm"] * OZIN_PER_NM, "oz-in", "Rated torque")
        add("torque_constant", t["kt_nm_per_a"] * OZIN_PER_NM, "oz-in/A", "Torque constant",
            basis="unknown" if kind == "pmsm" else "not_applicable", waveform="unknown" if kind == "pmsm" else "dc_equivalent")
        warn.append("US units (oz-in, oz-in/A): converted with fixed factors in Python")
        trap_log.append(("us_units", f"torque and Kt in oz-in (1 N·m = {OZIN_PER_NM} oz-in)"))
    else:
        add("torque", t["torque_nm"], "N·m", "Rated torque")
        if not (style == "D" and rng.random() < 0.5):
            add("torque_constant", t["kt_nm_per_a"], "N·m/A", "Torque constant", basis="unknown" if kind == "pmsm" else "not_applicable",
                waveform="unknown" if kind == "pmsm" else "dc_equivalent")
        else:
            ask.append({"field": "torque_constant", "reason": "missing", "text": "the torque constant Kt (not printed)"})
            trap_log.append(("missing_kt", "Kt not printed (style D)"))
    if style != "D" or "kt_ke_inconsistent" in traps:
        ke_si = t["ke_v_s_per_rad"] if kind == "dc" else t["ke_v_s_per_rad_phase_peak"] * math.sqrt(3)  # line-line peak
        if "kt_ke_inconsistent" in traps:
            ke_si *= 1.12
            ask.append({"field": "voltage_constant", "reason": "inconsistent", "text": "Kt and Ke disagree by about 12 %: which one to use"})
            trap_log.append(("kt_ke_inconsistent", "Ke printed 12 % above the value consistent with Kt"))
        unit, value = "V/krpm", ke_si * 1000 * 2 * math.pi / 60
        if "ke_units" in traps:
            choice = rng.choice(["mV/rpm", "V/rpm", "rpm/V"])
            unit, value = choice, {"mV/rpm": ke_si * 2 * math.pi / 60 * 1000, "V/rpm": ke_si * 2 * math.pi / 60, "rpm/V": 1 / (ke_si * 2 * math.pi / 60)}[choice]
            trap_log.append(("ke_units", f"Ke printed in {choice}" + (" (a speed constant: the inverse)" if choice == "rpm/V" else "")))
            if choice == "rpm/V":
                warn.append("the voltage constant is printed as a speed constant in rpm/V (its inverse)")
        wave = "dc_equivalent" if kind == "dc" else ("unknown" if "ke_waveform_undeclared" in traps else "peak")
        basis = "not_applicable" if kind == "dc" else "line-line"
        label = "Back-EMF constant" + ("" if kind == "dc" or wave == "unknown" else " (line-line, peak)")
        add("voltage_constant", float(value), unit, label, basis=basis, waveform=wave)
        if wave == "unknown":
            ask.append({"field": "voltage_constant", "reason": "ambiguous_waveform", "text": "is the back-EMF constant rms or peak"})
            trap_log.append(("ke_waveform_undeclared", "Ke printed line-line without rms/peak"))
    # resistance
    r_label, r_value, r_basis, r_temp = "Terminal resistance", t["r_ohm"], "not_applicable" if kind == "dc" else "phase", None
    if "line_line_resistance" in traps:
        r_label, r_value, r_basis = "Resistance (line-line)", t["r_ohm"] * 2, "line-line"
        warn.append("resistance printed line-line: R_phase = R_LL / 2 for a star connection")
        trap_log.append(("line_line_resistance", "R printed line-line (2 × phase)"))
    if "resistance_temperature" in traps:
        r_temp = 20
        r_label += " at 20 °C"
        add("resistance_hot", float(r_value * 1.39), "Ω", "Resistance at 120 °C", basis=r_basis, temperature=120)
        warn.append("two resistances (20 °C and 120 °C): the cold value is the default and is said")
        trap_log.append(("resistance_temperature", "R at 20 °C and at 120 °C (+39 %)"))
    if "wrong_label" in traps:
        r_label = "Thermal resistance (Ra" + (", line-line" if r_basis == "line-line" else "") + ")" + (" at 20 °C" if r_temp else "")
        ask.append({"field": "resistance", "reason": "ambiguous_meaning", "text": "'Thermal resistance (Ra)' is in ohms: is it the armature resistance"})
        trap_log.append(("wrong_label", "armature resistance labelled 'Thermal resistance (Ra)'"))
    add("resistance", float(r_value), "Ω", r_label, basis=r_basis, temperature=r_temp)
    # inductance
    if "missing_inductance" in traps:
        ask.append({"field": "inductance", "reason": "missing", "text": "the inductance (not printed)"})
        trap_log.append(("missing_inductance", "L not printed"))
    else:
        add("inductance", t["l_h"] * 1e3, "mH", "Inductance" + (" (line-line)" if r_basis == "line-line" else ""), basis=r_basis)
    # inertia
    if "unitless_inertia" in traps:
        add("inertia", float(t["j_kgm2"] * 1e4), "", "Rotor inertia")
        ask.append({"field": "inertia", "reason": "ambiguous_meaning", "text": "the rotor inertia has no unit"})
        trap_log.append(("unitless_inertia", "J printed without unit (value in kg·cm²)"))
    elif "us_units" in traps:
        add("inertia", float(t["j_kgm2"] * LBFT2_PER_KGM2), "lb-ft²", "Rotor inertia (WK²)")
    else:
        add("inertia", float(t["j_kgm2"] * 1e4), "kg·cm²", "Rotor inertia")
    if kind == "pmsm":
        add("pole_pairs", str(t["pole_pairs"] * 2), "", "Poles")
    if "multiple_ratings" in traps:
        if kind == "pmsm":
            add("voltage_alt", f"{t['u_v']:g}/{2 * t['u_v']:g}", "V", "Voltage (Δ/Y)")
            ask.append({"field": "voltage", "reason": "variant_unspecified", "text": "which connection / rating (Δ or Y)"})
            trap_log.append(("multiple_ratings", "two ratings on one line (Δ/Y)"))
        else:
            add("voltage_alt", f"{t['u_v']:g} // {2 * t['u_v']:g}", "V", "Voltage (dual rating)")
            ask.append({"field": "voltage", "reason": "variant_unspecified", "text": "which of the two printed supply voltages"})
            trap_log.append(("multiple_ratings", "two supply voltages on one line ('//')"))
    if "curve_only" in traps:
        warn.append("efficiency is only in a curve: not a parameter")
        trap_log.append(("curve_only", "efficiency shown only as a curve"))
    if comma:
        warn.append("decimal commas")
        trap_log.append(("decimal_comma", "decimal comma in every number"))
    manufacturer = None if "manufacturer_unnamed" in traps else FICTITIOUS[n % len(FICTITIOUS)]
    if manufacturer is None:
        ask.append({"field": "manufacturer", "reason": "manufacturer_unnamed", "text": "who makes the motor (no name printed)"})
        trap_log.append(("manufacturer_unnamed", "no manufacturer printed"))
    ask.append({"field": "load_torque", "reason": "missing", "text": "the load (never in a motor datasheet)"})
    for k, a in enumerate(ask):
        a["item"] = k
    assert all(a["reason"] in REASONS for a in ask)
    sid = f"S{n + 1:02d}"
    return {"id": sid, "style": style, "machine": kind, "manufacturer": manufacturer, "rows": rows, "variants": [v for v in variants if v],
            "fields": fields, "must_ask": ask, "must_warn": warn, "traps": [{"trap": a, "detail": b} for a, b in trap_log], "truth": t}


# ------------------------------------------------------------------------------------------------------------ html
CSS = """body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111;background:#fff}h1{font-size:20px;margin:0 0 4px}
.banner{font-size:11px;color:#a00;border:1px solid #a00;padding:4px 8px;margin-bottom:12px;display:inline-block}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #888;padding:4px 8px;text-align:left}th{background:#eee}
.plate{width:520px;border:6px double #444;border-radius:10px;padding:14px;background:#d9d9d2;font-family:'Courier New',monospace;
transform:rotate(-1.5deg);filter:blur(0.4px)}.plate td{border:none;padding:2px 6px}.web{max-width:640px}.web p{line-height:1.5}"""


def _curve_svg(rng: random.Random) -> str:
    pts = " ".join(f"{20 + 4 * k},{180 - (150 * (1 - math.exp(-k / 9)) - 0.5 * max(0, k - 50) + rng.uniform(-1.5, 1.5)):.0f}" for k in range(85))
    return ('<svg width="380" height="200" viewBox="0 0 380 200"><rect x="20" y="10" width="340" height="170" fill="none" stroke="#999"/>'
            f'<polyline points="{pts}" fill="none" stroke="#06c" stroke-width="2"/><text x="24" y="196" font-size="11">speed →</text>'
            '<text x="24" y="24" font-size="11">efficiency (%)</text></svg>')


def render_html(s: dict, rng: random.Random) -> str:
    e = html.escape
    title = f"{'Motor' if s['manufacturer'] is None else e(s['manufacturer'])} — model SX-{s['id']}"
    banner = '<div class="banner">SYNTHETIC SHEET — fictitious motor, generated for testing (SPADE B6). Not a real product.</div>'
    body = ""
    if s["style"] == "A" and s["variants"]:  # variants in columns
        labels = list(dict.fromkeys(r[0] for r in s["rows"]))
        body = "<table><tr><th>Parameter</th><th>Unit</th>" + "".join(f"<th>{e(v)}</th>" for v in s["variants"]) + "</tr>"
        for lab in labels:
            cells = {r[3]: r[1] for r in s["rows"] if r[0] == lab}
            unit = next(r[2] for r in s["rows"] if r[0] == lab)
            body += f"<tr><td>{e(lab)}</td><td>{e(unit)}</td>" + "".join(f"<td>{e(cells.get(v, cells.get(None, '')))}</td>" for v in s["variants"]) + "</tr>"
        body += "</table>"
    elif s["style"] == "B" and s["variants"]:  # variants in rows
        labels = list(dict.fromkeys(r[0] for r in s["rows"]))
        body = "<table><tr><th>Variant</th>" + "".join(f"<th>{e(l)} [{e(next(r[2] for r in s['rows'] if r[0] == l))}]</th>" for l in labels) + "</tr>"
        for v in s["variants"]:
            cells = {r[0]: r[1] for r in s["rows"] if r[3] in (v, None)}
            body += f"<tr><td>{e(v)}</td>" + "".join(f"<td>{e(cells.get(l, ''))}</td>" for l in labels) + "</tr>"
        body += "</table>"
    elif s["style"] == "D":  # web page prose, no table
        body = '<div class="web"><h2>Product overview</h2><p>A compact motor for mobile and industrial use. ' + " ".join(
            f"{e(r[0])}: <b>{e(r[1])} {e(r[2])}</b>." for r in s["rows"]) + "</p><p>Contact your distributor for the full manual.</p></div>"
    elif s["style"] == "F":  # rating plate
        body = '<div class="plate"><table>' + "".join(f"<tr><td>{e(r[0][:14].upper())}</td><td>{e(r[1])} {e(r[2])}</td></tr>" for r in s["rows"]) + "</table></div>"
    else:  # C single article, E catalogue with a curve
        body = "<table><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>" + "".join(
            f"<tr><td>{e(r[0])}</td><td>{e(r[1])}</td><td>{e(r[2])}</td></tr>" for r in s["rows"]) + "</table>"
        if s["style"] == "E":
            body = "<h3>Catalogue page 7 of 22</h3>" + body + "<h3>Characteristic curves</h3>" + _curve_svg(rng)
    if "curve_only" in [t["trap"] for t in s["traps"]] and s["style"] != "E":
        body += "<h3>Efficiency</h3>" + _curve_svg(rng)
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{banner}<h1>{title}</h1>{body}</body></html>\n'


def expected(s: dict) -> dict:
    return {"id": s["id"], "written_by": "experiments/generate_synthetic_datasheets.py (B6, 2026-11-16), from the known truth", "schema": "golden-expected/1",
            "synthetic": True, "document": f"synthetic sheet SX-{s['id']} (style {s['style']})",
            "manufacturer_printed": s["manufacturer"] or "not printed", "motor_type": "permanent magnet DC" if s["machine"] == "dc" else "PMSM (synthetic)",
            "simulable_today": s["machine"] == "dc", "variants": s["variants"], "variant_policy": "every variant written" if s["variants"] else "single article",
            "fields": s["fields"], "must_ask": s["must_ask"], "must_warn": s["must_warn"], "traps": s["traps"], "truth_si": s["truth"],
            "notes": "Synthetic: the truth is known exactly; numbers are printed with 3 significant digits."}


def generate(out: Path, seed: int = SEED, n: int = N_SHEETS) -> list[dict]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expected").mkdir(exist_ok=True)
    log = []
    for k in range(n):
        s = make_sheet(k, rng)
        page = render_html(s, rng)
        (out / f"{s['id']}.html").write_text(page, encoding="utf-8")
        (out / "expected" / f"{s['id']}.json").write_text(json.dumps(expected(s), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        log.append({"id": s["id"], "style": s["style"], "machine": s["machine"], "traps": [t["trap"] for t in s["traps"]],
                    "must_ask_reasons": [a["reason"] for a in s["must_ask"]], "html_sha256": hashlib.sha256(page.encode()).hexdigest()[:16]})
    (out / "trap_log.json").write_text(json.dumps({"seed": seed, "n": n, "generator": "experiments/generate_synthetic_datasheets.py", "sheets": log},
                                                  indent=1) + "\n", encoding="utf-8")
    return log


PDF_SCRIPT = r'''
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
out = Path(sys.argv[1])
with sync_playwright() as pw:
    kw = {"executable_path": "/usr/bin/google-chrome"} if Path("/usr/bin/google-chrome").exists() else {}
    b = pw.chromium.launch(args=["--no-sandbox"], **kw)
    page = b.new_page()
    for h in sorted(out.glob("S*.html")):
        page.goto(h.resolve().as_uri())
        page.pdf(path=str(h.with_suffix(".pdf")), format="A4")
    b.close()
print("pdf ok")
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_SHEETS)
    ap.add_argument("--pdf-python", default=None, help="an interpreter with Playwright, to print each HTML to PDF (Chromium)")
    args = ap.parse_args(argv)
    log = generate(args.out, args.seed, args.n)
    print(f"{len(log)} sheets in {args.out}; styles {sorted({x['style'] for x in log})}")
    if args.pdf_python:
        p = subprocess.run([args.pdf_python, "-c", PDF_SCRIPT, str(args.out)], capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip()[-300:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
