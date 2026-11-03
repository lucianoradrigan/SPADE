"""S21-2: convergence of the discontinuous-conduction converter model and the control against GEM.

Pre-registered in docs/audit/assistant/ai_n27_preregistration.md §1 (committed before this ran):
  - "dcm" with x10, x100, x1000 substeps; converged if NRMS(x100 vs x1000) < 1e-4 for omega and i_a
    in every case;
  - control (cases where GEM's current never goes negative): "dcm" xN vs "gem" with
    dc_plant_substeps = N, N = 1 and 100: NRMS <= 1e-9 for omega and i_a.
NRMS = RMS(difference over the run) / range of the finer (or reference) solution over the run.

    python experiments/report_s21_2_dcm_convergence.py [--workers N]
Writes docs/audit/assistant/s21_2_dcm_convergence.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "s21_2_dcm_convergence"
CONV_TOL = 1e-4
CONTROL_TOL = 1e-9
STOCK_MP = {"r_a": 0.016, "l_a": 1.9e-05, "psi_e": 0.165, "j_rotor": 0.025}
STOCK_NOMINAL = {"omega": 300.0, "torque": 16.0, "i": 97.0, "u": 60.0}
CASES = {
    "1_pmg132_120_no_load": dict(plant_config_id="dc_pmg132_v1", omega_ref_rad_s=120.0, duration_s=1.0),
    "2_stock_D1_friction_120": dict(omega_ref_rad_s=120.0, duration_s=1.0, motor_parameter_overrides=STOCK_MP, dc_nominal_overrides=STOCK_NOMINAL),
    "3_outer_race_x2_80": dict(omega_ref_rad_s=80.0, duration_s=2.5, fault_type="outer_race", electrical_severity=16.0, mechanical_severity=0.1),
    "4_cage_x2_80": dict(omega_ref_rad_s=80.0, duration_s=2.5, fault_type="cage", electrical_severity=16.0, mechanical_severity=0.04),
    "5_control_healthy_220": dict(omega_ref_rad_s=220.0, duration_s=1.0),
    "6_control_load5_120": dict(omega_ref_rad_s=120.0, duration_s=1.0, load_torque_nm=5.0),
}
CONTROLS = ("5_control_healthy_220", "6_control_load5_120")


def _run(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario

    name, variant, kw = job
    t0 = time.process_time()
    recs = run_scenario(Scenario("s21_2", **CASES[name], **kw))
    return name, variant, np.array([r["rpm"] * 2 * np.pi / 60 for r in recs]), np.array([r["current_r"] for r in recs]), time.process_time() - t0


def nrms(a, ref):
    rng = float(ref.max() - ref.min())
    return float(np.sqrt(np.mean((a - ref) ** 2)) / rng) if rng > 0 else float(np.sqrt(np.mean((a - ref) ** 2)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    jobs = [(n, f"dcm_x{s}", {"dc_converter_model": "dcm", "dc_plant_substeps": s}) for n in CASES for s in (10, 100, 1000)]
    jobs += [(n, "gem_default", {}) for n in CASES]
    jobs += [(n, f"dcm_x1", {"dc_converter_model": "dcm", "dc_plant_substeps": 1}) for n in CONTROLS]
    jobs += [(n, "gem_x100", {"dc_converter_model": "gem", "dc_plant_substeps": 100}) for n in CONTROLS]
    jobs.sort(key=lambda j: -("x1000" in j[1]) * CASES[j[0]]["duration_s"])  # longest first
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_run, jobs)
    R = {(n, v): (w, i, cpu) for n, v, w, i, cpu in res}
    out = {"command": "python experiments/report_s21_2_dcm_convergence.py", "simulated": True, "preregistration": "docs/audit/assistant/ai_n27_preregistration.md §1",
           "conv_tol": CONV_TOL, "control_tol": CONTROL_TOL, "cases": {}}
    converged_all, control_all = True, True
    for n in CASES:
        c = {"scenario": {k: v for k, v in CASES[n].items()}}
        w10, i10, _ = R[(n, "dcm_x10")]
        w100, i100, cpu100 = R[(n, "dcm_x100")]
        w1k, i1k, cpu1k = R[(n, "dcm_x1000")]
        wg, ig, cpug = R[(n, "gem_default")]
        c["nrms_x10_vs_x1000"] = {"omega": nrms(w10, w1k), "i_a": nrms(i10, i1k)}
        c["nrms_x100_vs_x1000"] = {"omega": nrms(w100, w1k), "i_a": nrms(i100, i1k)}
        c["converged"] = c["nrms_x100_vs_x1000"]["omega"] < CONV_TOL and c["nrms_x100_vs_x1000"]["i_a"] < CONV_TOL
        converged_all &= c["converged"]
        c["final_speed"] = {"gem_default": float(wg[-1]), "dcm_x100": float(w100[-1]), "dcm_x1000": float(w1k[-1]), "reference": CASES[n]["omega_ref_rad_s"]}
        c["mean_speed_second_half"] = {"gem_default": float(wg[len(wg) // 2:].mean()), "dcm_x100": float(w100[len(w100) // 2:].mean())}
        c["min_current"] = {"gem_default": float(ig.min()), "dcm_x100": float(i100.min())}
        c["gem_default_vs_dcm_x1000"] = {"omega": nrms(wg, w1k), "i_a": nrms(ig, i1k)}
        c["cpu_s"] = {"gem_default": cpug, "dcm_x100": cpu100, "dcm_x1000": cpu1k}
        if n in CONTROLS:
            w1, i1, _ = R[(n, "dcm_x1")]
            wg100, ig100, _ = R[(n, "gem_x100")]
            c["control"] = {"dcm_x1_vs_gem_x1": {"omega": nrms(w1, wg), "i_a": nrms(i1, ig)}, "dcm_x100_vs_gem_x100": {"omega": nrms(w100, wg100), "i_a": nrms(i100, ig100)},
                            "gem_negative_current_samples": int((ig < 0).sum()),
                            "info_dcm_x100_vs_gem_default": {"omega": nrms(w100, wg), "i_a": nrms(i100, ig)}}
            ok = all(v <= CONTROL_TOL for k in ("dcm_x1_vs_gem_x1", "dcm_x100_vs_gem_x100") for v in c["control"][k].values()) and c["control"]["gem_negative_current_samples"] == 0
            c["control"]["pass"] = ok
            control_all &= ok
        out["cases"][n] = c
    out["converged"] = converged_all
    out["control_pass"] = control_all
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    L = ["# S21-2 — convergence of the discontinuous-conduction converter model (`dc_converter_model=\"dcm\"`)", "",
         "**Simulated. Pre-registered** in [`ai_n27_preregistration.md`](ai_n27_preregistration.md) §1, committed before this ran.", "",
         f"- **Command:** `{out['command']}`. Raw values: [`s21_2_dcm_convergence.json`](s21_2_dcm_convergence.json).",
         f"- **Criterion:** NRMS(×100 vs ×1000) < {CONV_TOL:g} for ω and i_a in every case → **{'CONVERGED' if converged_all else 'NOT CONVERGED'}**.",
         f"- **Control:** \"dcm\" ×N vs \"gem\" with N substeps (N = 1, 100), NRMS ≤ {CONTROL_TOL:g}, where GEM's current is never negative → **{'PASS' if control_all else 'FAIL'}**.",
         "", "## Convergence", "",
         "| Case | NRMS ×10 vs ×1000 (ω / i_a) | NRMS ×100 vs ×1000 (ω / i_a) | Converged | Final speed GEM → DCM ×1000 (ref) | Min current GEM → DCM | GEM vs DCM ×1000 (ω / i_a) |",
         "|---|---|---|---|---|---|---|"]
    for n, c in out["cases"].items():
        a, b, g = c["nrms_x10_vs_x1000"], c["nrms_x100_vs_x1000"], c["gem_default_vs_dcm_x1000"]
        f = c["final_speed"]
        L.append(f"| {n} | {a['omega']:.1e} / {a['i_a']:.1e} | {b['omega']:.1e} / {b['i_a']:.1e} | {'yes' if c['converged'] else '**no**'} | "
                 f"{f['gem_default']:.2f} → {f['dcm_x1000']:.2f} ({f['reference']:g}) | {c['min_current']['gem_default']:.1f} → {c['min_current']['dcm_x100']:.1f} A | {g['omega']:.2e} / {g['i_a']:.2e} |")
    L += ["", "## Control (no artifact)", "", "| Case | GEM negative-current samples | DCM ×1 vs GEM ×1 (ω / i_a) | DCM ×100 vs GEM ×100 (ω / i_a) | Pass | Info: DCM ×100 vs default plant (ω / i_a) |", "|---|---|---|---|---|---|"]
    for n in CONTROLS:
        c = out["cases"][n]["control"]
        L.append(f"| {n} | {c['gem_negative_current_samples']} | {c['dcm_x1_vs_gem_x1']['omega']:.1e} / {c['dcm_x1_vs_gem_x1']['i_a']:.1e} | "
                 f"{c['dcm_x100_vs_gem_x100']['omega']:.1e} / {c['dcm_x100_vs_gem_x100']['i_a']:.1e} | {'yes' if c['pass'] else '**no**'} | "
                 f"{c['info_dcm_x100_vs_gem_default']['omega']:.1e} / {c['info_dcm_x100_vs_gem_default']['i_a']:.1e} |")
    L += ["", "The last column is S-16's discretization error (one Euler step vs 100), not the converter model; it is reported, not decided on.", "",
          "## Cost (CPU s of one run)", "", "| Case | Default plant | DCM ×100 | DCM ×1000 |", "|---|---|---|---|"]
    L += [f"| {n} | {c['cpu_s']['gem_default']:.1f} | {c['cpu_s']['dcm_x100']:.1f} | {c['cpu_s']['dcm_x1000']:.1f} |" for n, c in out["cases"].items()]
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
