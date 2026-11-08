"""dc_perm_ex_v2 step 1: convergence of the efficient discontinuous-conduction converter (variant B,
``dc_converter_model="dcm_exact"``; variant A only if B fails). Pre-registration:
docs/audit/assistant/dc_perm_ex_v2_convergence_preregistration.md and its amendment 1, committed before
any code of B and before this script.

The six cases of S21-2 (report_s21_2_dcm_convergence.CASES). REF = "dcm" x10000 (recomputed), D1000 =
"dcm" x1000, B = "dcm_exact". Section 4 of the pre-registration, per variant:
1. NRMS(variant vs REF) < 1e-4 for omega and i_a in every case;
2. control cases (5, 6): NRMS(variant vs D1000) <= 1e-4 for omega and i_a;
3. min i_a >= -1e-6 A in every case;
4. median over the cases of CPU per simulated second at least 10x lower than D1000's.

    python experiments/report_dc_perm_ex_v2_convergence.py [--workers 6]
Writes docs/audit/assistant/dc_perm_ex_v2_convergence.{md,json} (simulated).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_s21_2_dcm_convergence import CASES, CONTROLS, _run, nrms  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "dc_perm_ex_v2_convergence"
TOL = 1e-4
CONTROL_TOL = 1e-4
MIN_CURRENT_A = -1e-6
SPEEDUP_MIN = 10.0
VARIANTS = {"REF": {"dc_converter_model": "dcm", "dc_plant_substeps": 10000}, "D1000": {"dc_converter_model": "dcm", "dc_plant_substeps": 1000},
            "B": {"dc_converter_model": "dcm_exact"}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    jobs = [(n, v, kw) for n in CASES for v, kw in VARIANTS.items()]
    jobs.sort(key=lambda j: (-(j[1] == "REF"), -CASES[j[0]]["duration_s"]))  # longest first
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_run, jobs)
    R = {(n, v): (w, i, cpu) for n, v, w, i, cpu in res}
    out = {"command": "python experiments/report_dc_perm_ex_v2_convergence.py" + (f" --workers {args.workers}" if args.workers != 6 else ""), "simulated": True,
           "preregistration": ["docs/audit/assistant/dc_perm_ex_v2_convergence_preregistration.md", "docs/audit/assistant/dc_perm_ex_v2_convergence_preregistration_amendment1.md"],
           "host": platform.platform(), "python": platform.python_version(), "workers": args.workers, "variants": VARIANTS, "cases": {}}
    for n, case in CASES.items():
        w_ref, i_ref, c_ref = R[(n, "REF")]
        w_1k, i_1k, c_1k = R[(n, "D1000")]
        w_b, i_b, c_b = R[(n, "B")]
        d = case["duration_s"]
        out["cases"][n] = {
            "B_vs_REF": {"omega": nrms(w_b, w_ref), "i_a": nrms(i_b, i_ref)},
            "D1000_vs_REF": {"omega": nrms(w_1k, w_ref), "i_a": nrms(i_1k, i_ref)},
            "B_vs_D1000": {"omega": nrms(w_b, w_1k), "i_a": nrms(i_b, i_1k)},
            "min_i_a": {"B": float(i_b.min()), "D1000": float(i_1k.min()), "REF": float(i_ref.min())},
            "final_speed": {"B": float(w_b[-1]), "D1000": float(w_1k[-1]), "REF": float(w_ref[-1]), "reference": case["omega_ref_rad_s"]},
            "cpu_s_per_sim_s": {"B": c_b / d, "D1000": c_1k / d, "REF": c_ref / d}, "duration_s": d,
        }
    C = out["cases"]
    checks = {
        "1_convergence_vs_REF": all(c["B_vs_REF"]["omega"] < TOL and c["B_vs_REF"]["i_a"] < TOL for c in C.values()),
        "2_controls_vs_D1000": all(C[n]["B_vs_D1000"]["omega"] <= CONTROL_TOL and C[n]["B_vs_D1000"]["i_a"] <= CONTROL_TOL for n in CONTROLS),
        "3_no_negative_current": all(c["min_i_a"]["B"] >= MIN_CURRENT_A for c in C.values()),
    }
    med_b = statistics.median(c["cpu_s_per_sim_s"]["B"] for c in C.values())
    med_1k = statistics.median(c["cpu_s_per_sim_s"]["D1000"] for c in C.values())
    out["cost"] = {"median_cpu_s_per_sim_s": {"B": med_b, "D1000": med_1k}, "speedup": med_1k / med_b}
    checks["4_cost_10x"] = med_1k / med_b >= SPEEDUP_MIN
    out["checks"] = checks
    accurate = checks["1_convergence_vs_REF"] and checks["2_controls_vs_D1000"] and checks["3_no_negative_current"]
    out["verdict_B"] = "MEETS THE CRITERION" if all(checks.values()) else ("ACCURATE, NOT EFFICIENT ENOUGH" if accurate else "DOES NOT MEET THE CRITERION")
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    L = ["# dc_perm_ex_v2 step 1 — convergence of variant B (`dcm_exact`)", "",
         "**Simulated. Pre-registered** in `dc_perm_ex_v2_convergence_preregistration.md` and its amendment 1, both committed before any code of B.", "",
         f"- **Command:** `{out['command']}`. Raw values: `dc_perm_ex_v2_convergence.json`. Host: {out['host']}, Python {out['python']}.",
         "- **REF** = `dcm` ×10000 (the converged reference, S21-2 amendment 1); **D1000** = `dcm` ×1000; **B** = `dcm_exact`.",
         f"- **Verdict for B: {out['verdict_B']}.**", "",
         "| Case | B vs REF (ω / i_a) | D1000 vs REF (ω / i_a) | B vs D1000 (ω / i_a) | min i_a B (A) | final ω B / REF (ref) | CPU s per sim s B / D1000 |",
         "|---|---|---|---|---|---|---|"]
    for n, c in C.items():
        L.append(f"| {n} | {c['B_vs_REF']['omega']:.1e} / {c['B_vs_REF']['i_a']:.1e} | {c['D1000_vs_REF']['omega']:.1e} / {c['D1000_vs_REF']['i_a']:.1e} | "
                 f"{c['B_vs_D1000']['omega']:.1e} / {c['B_vs_D1000']['i_a']:.1e} | {c['min_i_a']['B']:.3g} | "
                 f"{c['final_speed']['B']:.3f} / {c['final_speed']['REF']:.3f} ({c['final_speed']['reference']:g}) | "
                 f"{c['cpu_s_per_sim_s']['B']:.2f} / {c['cpu_s_per_sim_s']['D1000']:.1f} |")
    L += ["", "| Check (§4) | Result |", "|---|---|"]
    names = {"1_convergence_vs_REF": f"1. NRMS(B vs REF) < {TOL:g}, ω and i_a, every case", "2_controls_vs_D1000": f"2. Controls: NRMS(B vs D1000) ≤ {CONTROL_TOL:g}",
             "3_no_negative_current": f"3. min i_a ≥ {MIN_CURRENT_A:g} A", "4_cost_10x": f"4. Median CPU per simulated second ≥ {SPEEDUP_MIN:g}× lower than D1000 (measured {out['cost']['speedup']:.1f}×)"}
    for k, v in checks.items():
        L.append(f"| {names[k]} | {'met' if v else '**not met**'} |")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
