"""T8 (round C, decision C10): the golden-set catalogue and its evidence.

    python experiments/report_golden_set.py

Reads data/upstream/assistant_golden/fetch_log.json (experiments/fetch_golden_datasheets.py) and
data/upstream/assistant_golden/photos/manifest_photos.json (fetch_golden_photos.py + the hand-written photo expected values),
writes tests/assistant/golden/manifest.yaml (committed; the files themselves are not) and
docs/audit/assistant/golden_set.{md,json}. Style (A-F) and traps follow RELEVAMIENTO_DATASHEETS_2026-11-02.md §2-§3.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.assistant.golden import conventions  # noqa: E402

UP = REPO / "data" / "upstream" / "assistant_golden"
COPY = "public document; copyright of its manufacturer or publisher; downloaded for internal evaluation only, not redistributed"
#: id -> (manufacturer, motor type, simulable today, style, traps, star, note)
META = {
    "D1": ("perm-motor (Heinzmann PMG 132 in annex A; 'Heinzmann' not printed)", "PM DC", True, "A", ["multiple ratings (5 voltage columns)", "continuous vs short time (S1 / S2 10 min)", "data only in curves", "decimal comma"], True, ""),
    "D2": ("Parvalux", "PM DC", True, "C", ["mislabelled field ('Thermal resistance (Ra)')"], True, ""),
    "D3": ("Motenergy", "PM DC", True, "D", ["internal inconsistency (Kt vs Ke, 12 %)", "missing R, L, J"], True, "the title reads ME1513"),
    "D4": ("Bodine Electric", "PM DC", True, "D", ["inertia without unit", "paired values '90/130' (multiple ratings)", "US units"], False, "web page, printed to PDF"),
    "D5": ("Portescap", "PM DC (coreless)", True, "A", ["many windings in columns", "Ke unit (mV/rpm)", "inertia unit (g-cm2)", "two units in one cell"], True,
           "DEVIATION: one catalog page (16N78, 6 windings) instead of the ~120-variant catalog of annex A"),
    "D6": ("not named in the document (annex A: do not invent it)", "PM DC", True, "C", ["manufacturer not named", "Ncm/A and g.cm2", "two voltages (12 V, 24 V)"], True, ""),
    "D7": ("Portescap", "PM DC (coreless)", True, "A", ["to verify when downloaded"], False, "only third-party viewer pages found, no PDF"),
    "P1": ("Siemens", "PMSM servo", True, "C", ["temperature of R and Ke (20 °C) vs ratings at 60 K / 100 K", "phase or line-line (not printed)", "poles vs pole pairs"], True, ""),
    "P2": ("Kollmorgen", "PMSM servo", True, "A", ["14 windings and 5 bus voltages", "rms units", "a note '25 °C' with no clear parameter"], True, ""),
    "P3": ("Industrial Devices Corporation (annex A: Kollmorgen B23)", "PMSM servo", True, "C", ["peak, line-to-line (Kt, Kv)", "internal inconsistency (tau_el vs L/R, 16 %)", "US units", "temperature 118 [155] °F [°C]"], True, ""),
    "P4": ("Delta Electronics", "PMSM servo (ECM-B3)", True, "A", ["Ke in mV/rpm", "number of poles missing"], True, ""),
    "P5": ("Nanotec", "BLDC", True, "C", ["line-to-line R and L", "L at 1 kHz", "Vrms/krpm", "poles, not pairs"], True, ""),
    "P6": ("Nanotec", "BLDC", True, "C", ["same family at 48 V: consistency between variants", "Ke missing"], False, "expected values written although not starred (to reach >= 12)"),
    "P7": ("EMRAX", "axial-flux PMSM", True, "A", ["Ld/Lq", "6 windings x cooling columns", "Kt/Ke convention ambiguous", "phase R = line-line / 2", "decimal comma"], True, ""),
    "P8": ("maxon", "BLDC", True, "A", ["6 variants", "R and L phase-to-phase", "g.cm2"], False, ""),
    "P9": ("IMTT (annex A: TI BLDC17F; hosted on TI E2E)", "BLDC", True, "B", ["variants in rows", "oz-in/A, oz-in-s2", "poles vs pairs"], False, "expected values written although not starred (to reach >= 12)"),
    "P10": ("Bosch Rexroth (reseller sheet, IBA-Retzer GbR)", "PMSM servo", True, "D", ["not enough to simulate", "thousands separator '7.500'", "frame contradicts model", "German labels"], True, ""),
    "P11": ("Yaskawa", "PMSM servo", True, "D", ["no electrical constants", "decimal comma"], False, "web page, printed to PDF"),
    "I1": ("WEG", "induction (squirrel cage)", False, "E", ["4 ratings", "NEMA units", "bearings", "not simulable"], True, "WEG: reprinting not allowed without authorisation"),
    "I2": ("Baldor (ABB)", "induction (inverter duty)", False, "E", ["R line-to-line at 25 °C", "'//' for 50 and 60 Hz"], False, ""),
    "R1": ("Kollmorgen", "reference (not extracted)", None, "-", [], False, "source of configs/assistant_conversions.yaml"),
}


def main() -> None:
    log = json.loads((UP / "fetch_log.json").read_text())
    docs = []
    for doc_id, (maker, mtype, sim, style, traps, star, note) in META.items():
        r = log.get(doc_id, {"status": "unavailable", "reason": "not attempted"})
        d = {"id": doc_id, "star": star, "status": r["status"], "url": r.get("url"), "final_url": r.get("final_url"), "consulted_utc": r.get("consulted_utc"),
             "sha256": r.get("sha256"), "content_type": r.get("content_type"), "bytes": r.get("bytes"), "manufacturer": maker, "motor_type": mtype,
             "style": style, "traps": traps, "licence": COPY, "simulable_today": sim, "committed": False,
             "expected": (REPO / "tests/assistant/golden/expected" / f"{doc_id}.json").exists(), "local_path": f"data/upstream/assistant_golden/{doc_id}/"}
        if r.get("printed_pdf"):
            d["printed_pdf_sha256"] = r["printed_pdf"]["sha256"]
        if r["status"] != "downloaded":
            d["unavailable_reason"] = r.get("reason")
        if note:
            d["note"] = note
        docs.append(d)
    photos = json.loads((UP / "photos" / "manifest_photos.json").read_text())
    manifest = {"about": "Golden set of phase 2 (decision C10, round C, 2026-11-09). Third-party files live in data/upstream/assistant_golden/ and are never "
                         "committed (committed: false), except photos under CC0 / CC BY in tests/assistant/golden/photos/. No model has been run on them.",
                "documents": docs, "photos": photos}
    (REPO / "tests/assistant/golden/manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=160))

    checks = conventions.ratio_checks()
    count = lambda key, items: dict(collections.Counter(str(i[key]) for i in items))
    stats = {"documents": len(docs), "downloaded": sum(d["status"] == "downloaded" for d in docs),
             "unavailable": {d["id"]: d["unavailable_reason"] for d in docs if d["status"] != "downloaded"},
             "with_expected": [d["id"] for d in docs if d["expected"]], "by_type": count("motor_type", docs), "by_style": count("style", docs),
             "by_trap": dict(collections.Counter(t for d in docs for t in d["traps"])), "photos": len(photos),
             "photos_by_licence": count("licence_class", photos), "photos_committed": sum(p["committed"] for p in photos),
             "photos_by_type": count("machine_type", photos)}
    out = {"task": "T8, round C (decision C10)", "commands": ["python experiments/fetch_golden_datasheets.py", "python experiments/fetch_golden_photos.py search|download ...",
                                                               "python experiments/fetch_golden_ingenium.py 1992.2220.001 1992.2292.001", "python experiments/report_golden_set.py"],
           "stats": stats, "ratio_checks": checks}
    (REPO / "docs/audit/assistant/golden_set.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=1, ensure_ascii=False)[:2500])
    print(json.dumps(checks, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
