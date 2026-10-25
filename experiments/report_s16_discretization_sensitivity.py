"""R2 (round PLECS/OpenModelica, 2026-10-25): sensitivity of fault diagnosis to the native DC
plant's discretization (finding S-16).

The native plant ``dc_perm_ex_v1`` integrates one forward-Euler step per control period
TAU = 100 us; its armature time constant is ~1.19 ms. Every diagnosis dataset was generated with
it. This script asks whether its discretization error changes what the promoted classifiers see
and decide. Two stages, run separately:

``convergence`` (R2.1, decision G7): picks the study's reference integration. For three
operating scenarios (start-up, steady state, load-torque step), healthy and faulted, it compares
Euler with 1, 10, 100 and 1000 substeps per TAU and RK45 (rtol = atol = 1e-9) on i_a, omega and
torque. The reference is the coarsest Euler whose NRMS against the finest solution is < 1e-4 on
all three signals in every case. The controller acts every TAU in all of them; only the plant
integration changes (Scenario.dc_plant_solver / dc_plant_substeps).

``sensitivity`` (R2.2/R2.3): the pre-registered study (docs/audit/plecs_om/r2_preregistration.md).
NAT = the production plant, REF = the converged reference; same seeds, same configuration, paired
windows. Measures signal NRMS, fault-band spectral amplitudes, and top-1 agreement / accuracy /
macro-F1 / calibrated confidence / abstention of the three promoted dc_motor classifiers.

Usage:
  python experiments/report_s16_discretization_sensitivity.py convergence [--out docs/audit/plecs_om]
  python experiments/report_s16_discretization_sensitivity.py sensitivity [--speeds 100,150,200]
      [--severities 1.0,0.5] [--seeds 2] [--workers 8] [--cache sims.parquet] [--out docs/audit/plecs_om]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO_ROOT / "docs" / "audit" / "plecs_om"

# ----------------------------------------------------------------------------- convergence (R2.1)
#: integration name -> Scenario overrides
INTEGRATIONS = {
    "euler_x1": {},
    "euler_x10": {"dc_plant_substeps": 10},
    "euler_x100": {"dc_plant_substeps": 100},
    "euler_x1000": {"dc_plant_substeps": 1000},
    "rk45": {"dc_plant_solver": "rk45"},
}
EULER_ORDER = ("euler_x1", "euler_x10", "euler_x100", "euler_x1000")
FINEST = "rk45"
CONVERGENCE_TOL = 1e-4
SIGNALS = ("current_r", "rpm", "torque_nm")
SIGNAL_NAMES = {"current_r": "i_a", "rpm": "omega", "torque_nm": "torque"}
#: Load-torque step of the third scenario: 4 N·m (25 % of the motor's nominal 16 N·m) at 1.0 s.
LOAD_STEP_NM, LOAD_STEP_T_S = 4.0, 1.0
#: scenario -> (duration_s, extra load step?, time segment [t0, t1) the metric is taken over)
CONV_SCENARIOS = {
    "startup": (1.5, False, (0.0, 0.5)),
    "steady_state": (1.5, False, (0.5, 1.5)),
    "load_step": (1.5, True, (0.9, 1.5)),
}
CONV_SPEED = 150.0
#: Decision of the user (2026-10-25, R2 option a): the reference is RK45 (rtol = atol = 1e-9). Its
#: convergence (G7) is shown against the same RK45 at 1e-11 in every scenario and fault case.
RK45_TIGHT, RK45_TIGHT_TOL = "rk45_tol1e-11", 1e-11
CONV_FAULTS = (None, "outer_race")


def _conv_job(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen import runner

    scen, fault, integ = job
    duration, load_step, _seg = CONV_SCENARIOS[scen]
    if integ == RK45_TIGHT:  # same solver, 100x tighter tolerance (this process only)
        runner.DC_RK45_TOL = RK45_TIGHT_TOL
    original_load = runner.PolynomialStaticLoad
    if load_step:
        # The production plant has no load-step input. The validation benches' LoadTorqueStep is
        # PolynomialStaticLoad plus an external step; swapping it in (this process only) leaves
        # build_plant and everything else as run_scenario uses them.
        from driveflow.validation.benches import LoadTorqueStep

        runner.PolynomialStaticLoad = lambda load_parameter: LoadTorqueStep(load_parameter=load_parameter, step_nm=LOAD_STEP_NM, t_step_s=LOAD_STEP_T_S)
    sc = Scenario(scenario_id=f"r2conv_{scen}_{fault}_{integ}", fault_type=fault, seed=0, duration_s=duration, omega_ref_rad_s=CONV_SPEED,
                  **(INTEGRATIONS["rk45"] if integ == RK45_TIGHT else INTEGRATIONS[integ]))
    t0 = time.time()
    try:
        recs = run_scenario(sc)
    finally:
        runner.PolynomialStaticLoad = original_load
    arr = {s: np.array([r[s] for r in recs]) for s in SIGNALS}
    arr["t"] = np.array([r["timestamp_s"] for r in recs])
    return job, arr, time.time() - t0


def _nrms(a, b, scale):
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale) if scale > 0 else float("nan")


def convergence(args) -> dict:
    jobs = [(s, f, i) for s in CONV_SCENARIOS for f in CONV_FAULTS for i in INTEGRATIONS]
    # start-up and steady state share one simulation; only the segment differs
    jobs = [j for j in jobs if j[0] != "steady_state"]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        results = pool.map(_conv_job, jobs, chunksize=1)
    sims = {(s, f, i): arr for (s, f, i), arr, _w in results}
    wall = {f"{s}|{f}|{i}": round(w, 2) for (s, f, i), _a, w in results}
    cases = {}
    for scen, (_d, load_step, (t0, t1)) in CONV_SCENARIOS.items():
        src = "startup" if scen == "steady_state" else scen
        for fault in CONV_FAULTS:
            ref = sims[(src, fault, FINEST)]
            m = (ref["t"] > t0) & (ref["t"] <= t1)
            row = {}
            for integ in INTEGRATIONS:
                if integ == FINEST:
                    continue
                x = sims[(src, fault, integ)]
                row[integ] = {
                    # primary: normalised by the signal's range over the whole run of the finest
                    # solution (as B3 criterion 4); secondary: by the segment's own RMS
                    "nrms_full_range": {SIGNAL_NAMES[s]: _nrms(x[s][m], ref[s][m], float(np.ptp(ref[s]))) for s in SIGNALS},
                    "nrms_segment_rms": {SIGNAL_NAMES[s]: _nrms(x[s][m], ref[s][m], float(np.sqrt(np.mean(ref[s][m] ** 2)))) for s in SIGNALS},
                }
            cases[f"{scen}|{fault or 'healthy'}"] = row
    passing = {}
    for integ in EULER_ORDER:
        worst = max(v for c in cases.values() for v in c[integ]["nrms_full_range"].values())
        passing[integ] = {"worst_nrms_vs_finest": worst, "passes": worst < CONVERGENCE_TOL}
    # demonstrated convergence of the finest itself: Euler x1000 vs RK45 must be well below the tolerance
    chosen = next((i for i in EULER_ORDER if passing[i]["passes"]), None)
    return {
        "stage": "convergence", "finest": FINEST, "rk45_tol": 1e-9, "tolerance": CONVERGENCE_TOL,
        "rule": "reference = the coarsest Euler substep count whose NRMS vs the finest solution is < 1e-4 on i_a, omega and torque, in every scenario and fault case",
        "normalisation": "primary: RMS(diff over the segment) / range of the finest solution over the whole run; secondary (reported only): / RMS of the finest solution over the segment",
        "scenarios": {k: {"duration_s": v[0], "load_step": {"nm": LOAD_STEP_NM, "t_s": LOAD_STEP_T_S} if v[1] else None, "segment_s": v[2]} for k, v in CONV_SCENARIOS.items()},
        "speed_rad_s": CONV_SPEED, "faults": [f or "healthy" for f in CONV_FAULTS],
        "cases": cases, "per_integration": passing, "reference": chosen, "wall_s": wall,
    }


def _conv_md(r: dict) -> str:
    lines = ["## R2.1 Convergence of the reference (decision G7)", "",
             f"Finest solution: RK45, rtol = atol = {r['rk45_tol']:g}. Rule: {r['rule']}.",
             f"Normalisation: {r['normalisation']}.", "",
             f"Speed reference {r['speed_rad_s']:g} rad/s, PI controller, seed 0. Load step: {LOAD_STEP_NM:g} N·m at {LOAD_STEP_T_S:g} s (validation benches' `LoadTorqueStep`).", "",
             "| Scenario | Fault | Integration | i_a | omega | torque | i_a (segment-RMS norm.) |", "|---|---|---|---|---|---|---|"]
    for case, row in r["cases"].items():
        scen, fault = case.split("|")
        for integ, v in row.items():
            a = v["nrms_full_range"]
            lines.append(f"| {scen} | {fault} | {integ} | {a['i_a']:.2e} | {a['omega']:.2e} | {a['torque']:.2e} | {v['nrms_segment_rms']['i_a']:.2e} |")
    lines += ["", "| Integration | Worst NRMS vs RK45 | < 1e-4 |", "|---|---|---|"]
    for integ, v in r["per_integration"].items():
        lines.append(f"| {integ} | {v['worst_nrms_vs_finest']:.2e} | {'yes' if v['passes'] else 'no'} |")
    lines += ["", f"**Reference chosen: `{r['reference']}`.**" if r["reference"] else "**No Euler substep count reaches the tolerance: R2 stops here (R2.1).**", ""]
    return "\n".join(lines)


def rk45_convergence(args) -> dict:
    jobs = [(s, f, i) for s in CONV_SCENARIOS if s != "steady_state" for f in CONV_FAULTS for i in ("rk45", RK45_TIGHT, "euler_x1000")]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        results = pool.map(_conv_job, jobs, chunksize=1)
    sims = {(s, f, i): arr for (s, f, i), arr, _w in results}
    cases = {}
    for scen, (_d, _ls, (t0, t1)) in CONV_SCENARIOS.items():
        src = "startup" if scen == "steady_state" else scen
        for fault in CONV_FAULTS:
            fine, ref = sims[(src, fault, RK45_TIGHT)], sims[(src, fault, "rk45")]
            m = (fine["t"] > t0) & (fine["t"] <= t1)
            cases[f"{scen}|{fault or 'healthy'}"] = {
                "rk45_1e-9_vs_1e-11": {SIGNAL_NAMES[s]: _nrms(ref[s][m], fine[s][m], float(np.ptp(fine[s]))) for s in SIGNALS},
                "euler_x1000_vs_rk45_1e-11": {SIGNAL_NAMES[s]: _nrms(sims[(src, fault, "euler_x1000")][s][m], fine[s][m], float(np.ptp(fine[s]))) for s in SIGNALS}}
    worst = max(v for c in cases.values() for v in c["rk45_1e-9_vs_1e-11"].values())
    return {"stage": "rk45_convergence", "decision": "user, 2026-10-25: R2 reference = RK45 (option a)", "reference": "rk45, rtol = atol = 1e-9",
            "finer": "rk45, rtol = atol = 1e-11", "tolerance": CONVERGENCE_TOL, "worst_nrms_rk45_vs_finer": worst, "converged": worst < CONVERGENCE_TOL,
            "normalisation": "RMS(diff over the segment) / range of the 1e-11 solution over the whole run", "cases": cases}


def _rk45_md(r: dict) -> str:
    L = ["## R2.1 (continued) Convergence of the RK45 reference — decision of 2026-10-25 (option a)", "",
         f"Reference: {r['reference']}. Shown converged against {r['finer']} (G7); Euler × 1000 is listed for comparison.", "",
         "| Scenario | Fault | RK45 1e-9 vs 1e-11: i_a / omega / torque | Euler × 1000 vs RK45 1e-11: i_a / omega / torque |", "|---|---|---|---|"]
    for case, c in r["cases"].items():
        a, b = c["rk45_1e-9_vs_1e-11"], c["euler_x1000_vs_rk45_1e-11"]
        L.append(f"| {case.split('|')[0]} | {case.split('|')[1]} | {a['i_a']:.1e} / {a['omega']:.1e} / {a['torque']:.1e} | {b['i_a']:.1e} / {b['omega']:.1e} / {b['torque']:.1e} |")
    L += ["", f"Worst RK45 (1e-9) vs RK45 (1e-11): {r['worst_nrms_rk45_vs_finer']:.1e} (< {r['tolerance']:g}: {'yes' if r['converged'] else 'NO'}). "
          + ("**The RK45 reference is converged; R2 continues with it.**" if r["converged"] else "**Not converged: R2 stops.**"), ""]
    return "\n".join(L)


def _offset_job(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen import runner
    from driveflow.validation.benches import LoadTorqueStep

    t_step, overrides = job
    runner.PolynomialStaticLoad = lambda load_parameter: LoadTorqueStep(load_parameter=load_parameter, step_nm=LOAD_STEP_NM, t_step_s=t_step)
    recs = run_scenario(Scenario(scenario_id="r2_offset", seed=0, duration_s=1.1, omega_ref_rad_s=CONV_SPEED, **overrides))
    return np.array([r["current_r"] for r in recs])


def load_step_offset(args) -> dict:
    """Diagnostic of the load-step floor (finding S-17): Euler x1000 with the step at 1.0 s against
    RK45 with the step at 1.0 s, half a period earlier and one period earlier. Metric over
    t > 0.9 s, normalised by the RK45 range of the run."""
    shifts = (0.0, 0.5, 1.0)
    jobs = [(1.0, {"dc_plant_substeps": 1000})] + [(1.0 - k * 1e-4, {"dc_plant_solver": "rk45"}) for k in shifts]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        euler, *rk = pool.map(_offset_job, jobs, chunksize=1)
    m = slice(9000, None)
    return {"stage": "load_step_offset", "euler_x1000_step_at_s": 1.0,
            "nrms_euler_x1000_vs_rk45": {f"rk45 step {k:g} period(s) earlier": _nrms(euler[m], r[m], float(np.ptp(r))) for k, r in zip(shifts, rk)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("convergence", "rk45_convergence", "load_step_offset", "sensitivity"))
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 2) - 1)))
    ap.add_argument("--speeds", default="100,150,200")
    ap.add_argument("--severities", default="1.0,0.5")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"command": "python experiments/report_s16_discretization_sensitivity.py " + " ".join(argv if argv is not None else sys.argv[1:]),
            "host": platform.platform(), "python": platform.python_version()}
    if args.stage == "convergence":
        r = {**meta, **convergence(args)}
        r["total_wall_s"] = round(time.time() - t0, 1)
        (args.out / "r2_convergence.json").write_text(json.dumps(r, indent=2) + "\n")
        (args.out / "r2_convergence.md").write_text(_conv_md(r))
        print(_conv_md(r))
    elif args.stage == "rk45_convergence":
        r = {**meta, **rk45_convergence(args)}
        r["total_wall_s"] = round(time.time() - t0, 1)
        (args.out / "r2_rk45_convergence.json").write_text(json.dumps(r, indent=2) + "\n")
        print(_rk45_md(r))
    elif args.stage == "load_step_offset":
        r = {**meta, **load_step_offset(args)}
        (args.out / "r2_load_step_offset.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))
    else:
        # R2 stopped at R2.1 (S-17, r2_convergence.md): the pre-registered stage is written only
        # after the user picks the reference.
        raise SystemExit("sensitivity: not implemented; R2 stopped at R2.1 (see docs/audit/plecs_om/r2_convergence.md)")


if __name__ == "__main__":
    main()
