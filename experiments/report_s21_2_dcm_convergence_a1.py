"""S21-2, amendment 1 (docs/audit/assistant/ai_n27_preregistration_amendment1.md, committed before this ran):
the same six cases of report_s21_2_dcm_convergence.py with "dcm" at x1000 and x10000 substeps.
Converged if NRMS(x1000 vs x10000) < 1e-4 for omega and i_a in every case (threshold unchanged).

    python experiments/report_s21_2_dcm_convergence_a1.py [--workers N]
Writes docs/audit/assistant/s21_2_dcm_convergence_a1.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_s21_2_dcm_convergence import CASES, CONV_TOL, _run, nrms  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "s21_2_dcm_convergence_a1"
ORIGINAL = OUT.parent / "s21_2_dcm_convergence.json"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    jobs = [(n, f"dcm_x{s}", {"dc_converter_model": "dcm", "dc_plant_substeps": s}) for n in CASES for s in (10000, 1000)]
    jobs.sort(key=lambda j: (-("x10000" in j[1]), -CASES[j[0]]["duration_s"]))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_run, jobs)
    R = {(n, v): (w, i, cpu) for n, v, w, i, cpu in res}
    orig = json.loads(ORIGINAL.read_text())["cases"]
    out = {"command": "python experiments/report_s21_2_dcm_convergence_a1.py", "simulated": True, "preregistration": "docs/audit/assistant/ai_n27_preregistration_amendment1.md",
           "conv_tol": CONV_TOL, "cases": {}}
    ok_all = True
    for n in CASES:
        w1k, i1k, c1k = R[(n, "dcm_x1000")]
        w10k, i10k, c10k = R[(n, "dcm_x10000")]
        c = {"nrms_x1000_vs_x10000": {"omega": nrms(w1k, w10k), "i_a": nrms(i1k, i10k)}, "nrms_x100_vs_x1000_original": orig[n]["nrms_x100_vs_x1000"],
             "final_speed_x10000": float(w10k[-1]), "reference": CASES[n]["omega_ref_rad_s"], "cpu_s": {"x1000": c1k, "x10000": c10k},
             "x1000_reproduces_original": abs(float(w1k[-1]) - orig[n]["final_speed"]["dcm_x1000"]) == 0.0}
        c["converged"] = c["nrms_x1000_vs_x10000"]["omega"] < CONV_TOL and c["nrms_x1000_vs_x10000"]["i_a"] < CONV_TOL
        ok_all &= c["converged"]
        out["cases"][n] = c
    out["converged"] = ok_all
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    L = ["# S21-2, amendment 1 — convergence of `\"dcm\"` between ×1000 and ×10000", "",
         "**Simulated. Pre-registered** in [`ai_n27_preregistration_amendment1.md`](ai_n27_preregistration_amendment1.md), committed before this ran. "
         "The original criterion at ×100 vs ×1000 was not met in 2 of 6 cases ([`s21_2_dcm_convergence.md`](s21_2_dcm_convergence.md)); that result stands.", "",
         f"- **Command:** `{out['command']}`. Raw values: [`s21_2_dcm_convergence_a1.json`](s21_2_dcm_convergence_a1.json).",
         f"- **Criterion (threshold unchanged):** NRMS(×1000 vs ×10000) < {CONV_TOL:g} for ω and i_a in every case → **{'CONVERGED' if ok_all else 'NOT CONVERGED'}**.", "",
         "| Case | ×100 vs ×1000 (original; ω / i_a) | ×1000 vs ×10000 (ω / i_a) | Converged | Final speed ×10000 (ref) | CPU s ×10000 |", "|---|---|---|---|---|---|"]
    for n, c in out["cases"].items():
        o, a = c["nrms_x100_vs_x1000_original"], c["nrms_x1000_vs_x10000"]
        L.append(f"| {n} | {o['omega']:.1e} / {o['i_a']:.1e} | {a['omega']:.1e} / {a['i_a']:.1e} | {'yes' if c['converged'] else '**no**'} | "
                 f"{c['final_speed_x10000']:.3f} ({c['reference']:g}) | {c['cpu_s']['x10000']:.0f} |")
    L += ["", "×1000 re-run deterministically: same final speed as in the original run in every case: "
          + ("yes" if all(c["x1000_reproduces_original"] for c in out["cases"].values()) else "**no**") + "."]
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
