"""Round Assistant, task F0.7: S-20, the PMG 132 datasheet vs the stock DC motor's GEM values.

Documentation only. The datasheet values below were READ from the PDF (PMG-132-e.pdf, Perm-Motor
GmbH, https://www.thierry-lequeu.fr/data/karting/PMG-132-e.pdf, downloaded 2026-11-02 to
data/upstream/pmg132/, not committed); this script checks the file's hash when it is present and
recomputes the flux constant with the commands of docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md §8.

    python experiments/report_f0_7_s20_pmg132.py
Writes docs/audit/assistant/f0_7_s20_pmg132.{md,json}.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from driveflow.sim import DcPermanentlyExcitedMotor

REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "data" / "upstream" / "pmg132" / "PMG-132-e.pdf"
PDF_SHA256 = "4e3647904079a6e474eec571fbaa37901b1b3aae7459f3e1121aaaad51d423ed"
OUT = REPO / "docs" / "audit" / "assistant" / "f0_7_s20_pmg132"
RPM = 2 * math.pi / 60
#: page 1 table, as read: voltage -> (S1 current A, power kW, speed min^-1)
VARIANTS = {24: (110, 2.20, 1080), 36: (110, 3.50, 1700), 48: (110, 4.74, 2300), 60: (110, 5.97, 2870), 72: (110, 7.22, 3480)}
COMMON = {"torque_nm": 20.0, "inertia_kg_m2": 0.025, "inductance_h": 0.019e-3, "resistance_ohm": 16e-3, "short_time_current_a": 200.0,
          "short_time_mode": "S2 - 10 min", "short_time_torque_nm": 38.0, "weight_kg": 11.0, "protection": "IP 20"}


def main() -> None:
    pdf = {"path": str(PDF.relative_to(REPO)), "present": PDF.exists()}
    if PDF.exists():
        pdf["sha256_matches"] = hashlib.sha256(PDF.read_bytes()).hexdigest() == PDF_SHA256
    r = COMMON["resistance_ohm"]
    derived = {}
    for u, (i, p_kw, n) in VARIANTS.items():
        w = n * RPM
        t = p_kw * 1e3 / w
        derived[u] = {"omega_rad_s": w, "torque_from_power_nm": t, "psi_from_torque": t / i, "psi_from_emf": (u - i * r) / w}
    gem = {"motor_parameter": dict(DcPermanentlyExcitedMotor._default_motor_parameter), "nominal": dict(DcPermanentlyExcitedMotor._default_nominal_values),
           "limits": dict(DcPermanentlyExcitedMotor._default_limits)}
    result = {"finding": "S-20", "command": "python experiments/report_f0_7_s20_pmg132.py", "datasheet": {"source": "Perm-Motor GmbH, PMG 132 (PMG-132-e.pdf)", "sha256": PDF_SHA256, **pdf},
              "read": {"variants": {str(k): {"current_s1_a": v[0], "power_kw": v[1], "speed_min-1": v[2]} for k, v in VARIANTS.items()}, **COMMON},
              "derived": {str(k): v for k, v in derived.items()}, "gem": gem}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    d60 = derived[60]
    lines = [
        "# F0.7 — S-20: the PMG 132 datasheet vs the stock DC motor's GEM values",
        "",
        "**Documentation only. No default changed.**",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0_7_s20_pmg132.json`](f0_7_s20_pmg132.json).",
        f"- **Datasheet:** Perm-Motor GmbH, PMG 132, `PMG-132-e.pdf` (2 pages), downloaded 2026-11-02 to `data/upstream/pmg132/` (not committed), SHA-256 `{PDF_SHA256[:16]}…`"
        + (f"; hash checked now: {'matches' if pdf.get('sha256_matches') else '**does not match**'}." if pdf["present"] else "; not present on this machine, values below as read on 2026-11-02."),
        "- **Read by me from page 1** (the table): S1 at 24/36/48/60/72 V, each 110 A; 2.20/3.50/4.74/5.97/7.22 kW; 1080/1700/2300/2870/3480 min⁻¹; torque 20 N·m; "
        "short-time operation 200 A, S2 10 min, torque 38 N·m; inertia 0.025 kg·m²; inductance 0.019 mH; resistance 16 mΩ; 11 kg; IP 20. "
        "Page 2 has curves for 24, 36, 48 and 72 V only (no 60 V curve).",
        "",
        "## Comparison at 60 V (the variant GEM's u = 60 V corresponds to)",
        "",
        "| Quantity | Datasheet | GEM (`DcPermanentlyExcitedMotor`) | Agreement |",
        "|---|---|---|---|",
        "| Armature resistance | 16 mΩ | 16 mΩ | match |",
        "| Inductance | 0.019 mH | 0.019 mH | match |",
        "| Inertia | 0.025 kg·m² | 0.025 kg·m² | match |",
        f"| Rated speed | 2870 min⁻¹ = {d60['omega_rad_s']:.1f} rad/s | 300 rad/s (nominal omega) | match (0.2 %) |",
        "| Rated current (S1) | 110 A | 97 A (nominal i) | −12 % |",
        "| Rated torque | 20 N·m | 16 N·m (nominal torque) | −20 % |",
        f"| Flux constant ψ_e | {d60['psi_from_torque']:.3f} V·s/rad from P/(ω·I); {d60['psi_from_emf']:.3f} from (U − I·R)/ω | 0.165 | −9 % / −15 % |",
        "| Short-time current / torque | 200 A, 38 N·m (S2 10 min) | limits i = 210 A, torque = 38 N·m | close: GEM's limits look like the short-time values |",
        "",
        "Note on the instruction's list: the rated speed is **not** a mismatch. 2870 min⁻¹ is 300.5 rad/s.",
        "",
        "## Flux constant per voltage variant (§8 of the survey)",
        "",
        "| Variant | ω (rad/s) | T = P/ω (N·m) | ψ = T/I | ψ = (U − I·R)/ω |",
        "|---|---|---|---|---|",
    ] + [f"| {u} V | {v['omega_rad_s']:.1f} | {v['torque_from_power_nm']:.2f} | {v['psi_from_torque']:.4f} | {v['psi_from_emf']:.4f} |" for u, v in derived.items()] + [
        "",
        "The two estimates differ by about 7 % (P is the output power: losses and brush drop separate them); every variant gives 0.177–0.197, all above GEM's 0.165.",
        "",
        "## Statement (S-20, Info, Documented)",
        "",
        "- SPADE validates the fidelity of its port of gym-electric-motor (V4–V6), **not** that the model represents a real PMG 132.",
        "- **No default is changed**; that needs the user's confirmation.",
        "- The future assistant will use this datasheet as a test case. The correct behaviour is to extract the datasheet's values (110 A, 20 N·m, ψ_e ≈ 0.18–0.19), not GEM's.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
