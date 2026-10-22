"""E11a (round 2026-10-22, decision D3): audit of the NASA PCoE IGBT accelerated-aging archive.
Which devices ran to failure, when, under which protocol, and whether a common precursor can be
computed for all of them. No model is built.

Inputs (not committed, see DATA.md):
  --archive   the extracted archive (…/IGBTAgingData_04022009)
  --log-csv   the stem of the CSVs written by scripts/oneoff/convert_igbt_aging_log.py
              (one per sheet: <stem>__Device_N.csv)

    python experiments/report_e11a_igbt_data_audit.py --archive data/upstream/fund/igbt/extracted/IGBTAgingData_04022009 \
        --log-csv data/upstream/fund/igbt/device_aging_log
Writes docs/audit/ai/e11a_igbt_data_audit.{md,json}.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "ai"
SQ_SMU = "Data/Thermal Overstress Aging with Square Signal at gate and SMU data"
SQ = "Data/Thermal Overstress Aging with Square Signal at gate"
DC = "Data/Thermal Overstress Aging with DC at gate"

#: What the sources say about each device's end of test. Each entry quotes its source.
END_OF_TEST = {
    "DC-gate (20080429T135531)": ("failed: latch-up and thermal runaway", "DC folder Readme: 'Experiments stopped after latch-up and thermal runaway'; ~4 h"),
    "A17 (square gate)": ("failed: latch-up (recovered at room temperature)", "Sonnenfeld, Goebel, Celaya, Autotestcon 2008, Fig. 8 and text: 'latch up … The experiment was ended after the "
                          "failure and the IGBT was found to be functional when returned to room temperature'; the folder Readme says this file is that paper's experiment"),
    "Device 2": ("no failure recorded", "Device Aging log: sessions 2 and 2b end with no failure note, then a low-temperature check (Device2check)"),
    "Device 3": ("failed: latch-up", "Device Aging log, session 3b, 04:02: 'end experiment…device latchup'"),
    "Device 4": ("no failure recorded", "Device Aging log: sessions 4 and 4b, no failure note"),
    "Device 5": ("failed: no longer able to draw current", "Device Aging log, after 12:54: 'The device is no longer able to draw current.'"),
}
PROTOCOL = {
    "DC-gate (20080429T135531)": "DC gate 10 V; package held at 268-270 °C (stepped 220/240/270 °C), runaway cut-off 305 °C; low-speed data only",
    "A17 (square gate)": "square gate 10 V, 10 kHz, 40 % duty; package band 329-340 °C (Autotestcon 2008); low-speed + oscilloscope transients",
    "Device 2": "square gate + SMU set-up; package set point stepped 100→280 °C and supply 2.5→6 V by hand within 20-40 min sessions (log)",
    "Device 3": "same as Device 2",
    "Device 4": "same as Device 2",
    "Device 5": "same as Device 2",
}
PROTOCOL_GROUP = {"DC-gate (20080429T135531)": "DC gate", "A17 (square gate)": "square gate, fixed 329-340 °C band",
                  "Device 2": "square gate + SMU, stepped", "Device 3": "square gate + SMU, stepped", "Device 4": "square gate + SMU, stepped",
                  "Device 5": "square gate + SMU, stepped"}


def _transient_stats(paths) -> dict:
    n = fall = 0
    span_h = 0.0
    for p in paths:
        m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["measurement"]
        tr = np.atleast_1d(m.transient)
        n += len(tr)
        for t in tr:
            g = np.asarray(t.timeDomain.gateSignalVoltage, float)
            if g.size > 10 and np.ptp(g) > 1:
                hi = g > (g.max() + g.min()) / 2
                fall += bool(np.any(hi[:-1] & ~hi[1:]))
        pw = np.atleast_1d(m.pwmTempControllerState)
        span_h += (float(np.atleast_1d(pw[-1].timeEpoch)[0]) - float(np.atleast_1d(pw[0].timeEpoch)[0])) / 3600
    return {"files": [Path(p).name for p in paths], "transients": n, "transients_with_turn_off_edge": fall, "recorded_hours": round(span_h, 2)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--log-csv", type=Path, required=True)
    a = ap.parse_args(argv)
    arch = a.archive

    devices = {}
    dc = sio.loadmat(arch / DC / "20080429T135531.mat")
    t = np.asarray(dc["TIME"]).ravel()
    devices["DC-gate (20080429T135531)"] = {"files": ["20080429T135531.mat"], "transients": 0, "transients_with_turn_off_edge": 0,
                                           "recorded_hours": round(float(t.max() - t.min()) / 3600, 2) if t.size else None,
                                           "low_speed_samples": int(t.size)}
    devices["A17 (square gate)"] = _transient_stats([str(arch / SQ / "april22nd-23rdIgbtIRCG40BC30kd-A17.mat")])
    for k in (2, 3, 4, 5):
        files = sorted(p for p in glob.glob(str(arch / SQ_SMU / "Aging Data" / f"Device {k}" / "*.mat")) if "check" not in p)
        devices[f"Device {k}"] = _transient_stats(files)
    log_sheets = sorted(a.log_csv.parent.glob(f"{a.log_csv.name}__Device_*.csv"))

    rows = []
    for name, st in devices.items():
        end, source = END_OF_TEST[name]
        rows.append({"device": name, "protocol": PROTOCOL[name], "protocol_group": PROTOCOL_GROUP[name], "end_of_test": end, "source": source,
                     "run_to_failure": end.startswith("failed"), "failure_mode": end.split(": ", 1)[1] if end.startswith("failed") else None,
                     "turn_off_tail_computable": st["transients_with_turn_off_edge"] > 0, **st})
    rtf = [r for r in rows if r["run_to_failure"]]
    by_group = {}
    for r in rtf:
        by_group.setdefault(r["protocol_group"], []).append(r["device"])
    tail_rtf = [r["device"] for r in rtf if r["turn_off_tail_computable"]]
    best_group = max(by_group.items(), key=lambda kv: len(kv[1]))
    n_comparable = len(best_group[1])
    rep = {
        "command": "python experiments/report_e11a_igbt_data_audit.py --archive data/upstream/fund/igbt/extracted/IGBTAgingData_04022009 --log-csv data/upstream/fund/igbt/device_aging_log",
        "log_conversion": "uv run --with xlrd --with pandas python scripts/oneoff/convert_igbt_aging_log.py (throwaway environment; xlrd not added to pyproject.toml)",
        "log_sheets": [p.name for p in log_sheets], "devices": rows,
        "run_to_failure_devices": [r["device"] for r in rtf],
        "run_to_failure_by_protocol": by_group,
        "run_to_failure_with_computable_turn_off_tail": tail_rtf,
        "max_run_to_failure_on_one_comparable_protocol": n_comparable,
        "conclusion": ("NEGATIVE" if n_comparable < 3 else "AT LEAST 3"),
        "finding": f"the NASA IGBT accelerated-aging archive supports at most {n_comparable} run-to-failure devices on comparable protocols; a validated RUL model is not feasible"
                   if n_comparable < 3 else None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e11a_igbt_data_audit.json").write_text(json.dumps(rep, indent=2) + "\n")
    L = ["# E11a — IGBT data audit (round 2026-10-22, decision D3)", "",
         f"- **Command:** `{rep['command']}`. The aging log was converted once with {rep['log_conversion']}.",
         "- **Data:** the NASA PCoE archive `IGBTAgingData_04022009` (U.S. Government work, see DATA.md). It is not committed; the CSVs stay in `data/upstream/fund/igbt/`.",
         "- **What was not done:** no model and no Prognostics page.", "",
         "| Device | Protocol | End of test | Run to failure | Recorded hours | Transients (with a turn-off edge) | Source |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['device']} | {r['protocol']} | {r['end_of_test']} | {'yes' if r['run_to_failure'] else 'no'} | {r['recorded_hours']} | "
                 f"{r['transients']} ({r['transients_with_turn_off_edge']}) | {r['source']} |")
    L += ["", "## Reading", "",
          f"- **Run to failure:** {len(rtf)} devices ({', '.join(rep['run_to_failure_devices'])}), spread over {len(by_group)} protocols:",
          *[f"  - {g}: {', '.join(d)}" for g, d in by_group.items()],
          f"- **On one comparable protocol:** at most **{n_comparable}** ({', '.join(best_group[1])}).",
          "  - Their failure modes differ: latch-up for Device 3; loss of conduction for Device 5.",
          "  - The steady-state data of that set-up are not scaled, and the collector current drifts by ~600 mA (folder Readme).",
          f"- **Turn-off current tail** (the precursor named by NASA for latch-up), computable from high-speed transients: {', '.join(tail_rtf)}.",
          "  - The DC-gate device has low-speed data only.",
          "  - A17 ran a different thermal protocol (a fixed 329–340 °C band against a stepped 140→280 °C ramp) and recovered after latch-up.",
          "  - The tail is a latch-up precursor, which is not Device 5's failure mode.",
          "- **On-state V_CE:** not comparable across set-ups. The Device 2–5 steady-state voltages are unscaled.", "",
          "## Conclusion", ""]
    if rep["conclusion"] == "NEGATIVE":
        L += [f"**E11 closes as a negative result.** Finding: \"{rep['finding']}\".", "",
              "- **Looser reading, stated so it can be checked.** If protocol and failure mode are ignored, three run-to-failure devices have a computable turn-off tail (A17, Device 3, Device 5).",
              "- **Why that pooled set is rejected.** Their times to failure come from different thermal schedules (hours at 329–340 °C against tens of minutes on a manual ramp), and one of them fails by another mechanism. A leave-one-device-out RUL model on that set would measure the protocol, not the degradation.",
              "- **Option 3** (another NASA PCoE dataset, e.g. MOSFET thermal-overstress aging) stays **postponed**: not rejected, not done in this round."]
    else:
        L += ["**At least 3 run-to-failure devices on one comparable protocol with a common precursor.** E11b is authorized but is NOT started; it waits for the user to read this table."]
    (OUT / "e11a_igbt_data_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
