"""Recovery error of the VSC parameter identification (ai/pinn/identification.py) -- task E10b.

Simulates the closed-loop VSC/DPC plant (the same controller and reference as datagen/runner.py) with
KNOWN filter parameters -- nominal, a degraded capacitor, and L / R_f off nominal (sim/vsc_faults.py,
exact ZOH via the same augmented-matrix discretization as discretize_zoh) -- adds measurement noise
to the states, and estimates L, R_f and C with both estimators. Reports, per parameter and per noise
level, the relative error of the point estimate and the empirical coverage of the 5-95 % bootstrap
interval.

    python experiments/report_pinn_identification.py            # writes docs/audit/ai/pinn_identification.{md,json}
"""

from __future__ import annotations

import argparse
import multiprocessing
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from driveflow.ai.pinn import identification as idn
from driveflow.control.dpc.controller import DEFAULT_SCHEDULE_REGION, ScheduledDpcController
from driveflow.control.dpc.reference import RotatingReference
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH, _VSC_R_OHM
from driveflow.sim.vsc_faults import DegradedVscSystem
from driveflow.sim.vsc_system import TAU

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "ai"
#: (name, c_scale, l_scale, rf_scale)
CASES = [
    ("nominal", 1.0, 1.0, 1.0),
    ("C x0.8", 0.8, 1.0, 1.0),
    ("C x0.6", 0.6, 1.0, 1.0),
    ("C x0.5", 0.5, 1.0, 1.0),
    ("C x0.4", 0.4, 1.0, 1.0),  # the DPC loop diverges here: recorded, not identified
    ("L x0.8", 1.0, 0.8, 1.0),
    ("L x1.2", 1.0, 1.2, 1.0),
    ("R_f x0.5", 1.0, 1.0, 0.5),
    ("R_f x1.5", 1.0, 1.0, 1.5),
    ("C x0.7, L x1.1, R_f x1.2", 0.7, 1.1, 1.2),
]
NOISE_PCT = (0.0, 1.0, 2.0, 5.0)
WINDOW = (0.06, 0.02)  # start, length [s]: after the start-up transient, one 50 Hz period
DURATION_S = 0.1


def simulate(c_scale: float, l_scale: float, rf_scale: float, r_ohm: float = _VSC_R_OHM, duration_s: float = DURATION_S) -> pd.DataFrame:
    """runner._run_vsc_scenario's loop with a DegradedVscSystem that can also scale L and R_f."""
    system = DegradedVscSystem(r_ohm, c_scale=c_scale, esr_ohm=0.0, tau=TAU, l_scale=l_scale, rf_scale=rf_scale)
    controller = ScheduledDpcController(_DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH, r_ohm, DEFAULT_SCHEDULE_REGION)
    reference = RotatingReference(tau=TAU, phase0_rad=0.0)
    state = system.reset()
    controller.reset()
    rows = []
    for k in range(int(round(duration_s / TAU))):
        v_r, v_i = controller.control(state, reference, k)
        state = system.simulate(v_r, v_i)
        rows.append({"timestamp_s": (k + 1) * TAU, "vc_real": state.vc_real, "vc_imag": state.vc_imag, "i_f_real": state.i_f_real,
                     "i_f_imag": state.i_f_imag, "v_o_real": float(v_r), "v_o_imag": float(v_i)})
    return pd.DataFrame(rows)


def check_simulator_matches_runner() -> float:
    """The helper must be the runner's plant: max |difference| against run_scenario at C x0.6."""
    ours = simulate(0.6, 1.0, 1.0)
    ref = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="pinn_check", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=DURATION_S, vsc_capacitor_scale=0.6)))
    return float(max(np.max(np.abs(ours[c].to_numpy() - ref[c].to_numpy())) for c in idn.COLUMNS))


def _limit_threads(n: int) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = str(n)
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(n)


def run_case(case: tuple, repeats: int, pinn_bootstrap: int, ls_bootstrap: int) -> tuple[list, list]:
    """Every noise level and repetition of one case, both estimators: (rows, skipped)."""
    name, c_s, l_s, rf_s = case
    with np.errstate(all="ignore"):
        df = simulate(c_s, l_s, rf_s)
    truth = {"L": idn.NOMINAL["L"] * l_s, "R_f": idn.NOMINAL["R_f"] * rf_s, "C": idn.NOMINAL["C"] * c_s}
    try:
        clean = idn.from_dataframe(df, _VSC_R_OHM, *WINDOW)
    except ValueError as exc:
        bad = df[~np.isfinite(df[list(idn.COLUMNS)]).all(axis=1)]
        print(f"{name:28s} skipped: {exc}", flush=True)
        return [], [{"case": name, "reason": str(exc), "first_non_finite_s": float(bad["timestamp_s"].iloc[0]) if len(bad) else None}]
    rows = []
    for noise in NOISE_PCT:
        for rep in range(1 if noise == 0 else repeats):
            data = idn.add_noise(clean, noise, np.random.default_rng(1000 + rep))
            for est in idn.ESTIMATORS:
                res = idn.identify(data, est, n_bootstrap=pinn_bootstrap if est == "pinn" else ls_bootstrap, seed=rep)
                for p in idn.PARAMS:
                    r = res["params"][p]
                    rows.append({"case": name, "noise_pct": noise, "repeat": rep, "estimator": est, "param": p, "true": truth[p], "estimate": r["estimate"],
                                 "low": r["low"], "high": r["high"], "rel_error_pct": 100 * (r["estimate"] - truth[p]) / truth[p],
                                 "covered": bool(r["low"] <= truth[p] <= r["high"]), "rel_width_pct": 100 * (r["high"] - r["low"]) / truth[p],
                                 "elapsed_s": res["elapsed_s"]})
            print(f"{name:28s} noise {noise:3.0f}% rep {rep}: " + " | ".join(f"{e}: " + " ".join(f"{q['param']} {q['rel_error_pct']:+6.1f}%" for q in rows if q["noise_pct"] == noise and q["repeat"] == rep and q["estimator"] == e) for e in idn.ESTIMATORS), flush=True)
    return rows, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="noise realizations per noisy level")
    ap.add_argument("--pinn-bootstrap", type=int, default=5)
    ap.add_argument("--ls-bootstrap", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=5, help="cases run in parallel processes")
    ap.add_argument("--threads", type=int, default=2, help="TensorFlow threads per process")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t_start = time.perf_counter()
    sim_diff = check_simulator_matches_runner()
    assert sim_diff == 0.0, f"report simulator differs from datagen/runner.py by {sim_diff}"
    rows, skipped = [], []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx, initializer=_limit_threads, initargs=(args.threads,)) as pool:
        futures = [pool.submit(run_case, case, args.repeats, args.pinn_bootstrap, args.ls_bootstrap) for case in CASES]
        for fut in futures:
            case_rows, case_skipped = fut.result()
            rows += case_rows
            skipped += case_skipped
    table = pd.DataFrame(rows)
    summary = (table.assign(abs_err=table["rel_error_pct"].abs())
               .groupby(["estimator", "param", "noise_pct"])
               .agg(mean_abs_error_pct=("abs_err", "mean"), max_abs_error_pct=("abs_err", "max"), coverage=("covered", "mean"),
                    mean_rel_width_pct=("rel_width_pct", "mean"), n=("abs_err", "size"))
               .reset_index())
    timing = table.groupby("estimator")["elapsed_s"].mean().to_dict()
    meta = {"generated_by": "experiments/report_pinn_identification.py", "date": time.strftime("%Y-%m-%d"), "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "window_s": {"start": WINDOW[0], "length": WINDOW[1]}, "load_resistance_ohm": _VSC_R_OHM, "noise_pct": list(NOISE_PCT), "repeats": args.repeats,
            "pinn_bootstrap": args.pinn_bootstrap, "least_squares_bootstrap": args.ls_bootstrap, "jobs": args.jobs, "threads_per_job": args.threads, "uncertainty_method": idn.UNCERTAINTY_METHOD,
            "simulator_vs_runner_max_abs_diff": sim_diff, "skipped_cases": skipped, "mean_identify_time_s": timing, "total_runtime_s": time.perf_counter() - t_start,
            "cases": [{"name": n, "c_scale": c, "l_scale": l, "rf_scale": r} for n, c, l, r in CASES]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pinn_identification.json").write_text(json.dumps({"meta": meta, "summary": summary.to_dict(orient="records"), "runs": rows}, indent=1))
    (args.out / "pinn_identification.md").write_text(render_md(meta, summary, table))
    print(f"wrote {args.out / 'pinn_identification.md'} in {meta['total_runtime_s']:.0f} s")


def render_md(meta: dict, summary: pd.DataFrame, table: pd.DataFrame) -> str:
    lines = ["# VSC parameter identification: recovery error (E10b)", "",
             f"Generated by `{meta['generated_by']}` on {meta['date']} ({meta['machine']}). Raw values: [`pinn_identification.json`](pinn_identification.json).", "",
             f"Simulated closed-loop VSC/DPC runs with known L, R_f, C ({len(meta['cases'])} cases, below); window {meta['window_s']['start']}–{meta['window_s']['start'] + meta['window_s']['length']:.2f} s "
             f"({int(round(meta['window_s']['length'] / 1e-4))} samples, one 50 Hz period, after the start-up transient); load R = {meta['load_resistance_ohm']} Ω (given). "
             f"Gaussian noise on i_f and v_c with σ = the stated % of each signal's RMS; v_o exact. {meta['repeats']} noise realizations per noisy level.", "",
             f"Interval: {meta['uncertainty_method']}; {meta['pinn_bootstrap']} refits for the PINN, {meta['least_squares_bootstrap']} for least squares. "
             f"The report's simulator equals datagen/runner.py exactly (max |Δ| = {meta['simulator_vs_runner_max_abs_diff']:g} at C ×0.6).", "",
             "Every value is from simulation: the plant is the same model the estimators assume (no model mismatch, no ESR). Nothing here is measured.", ""]
    for sk in meta["skipped_cases"]:
        lines += [f"**Not identified: {sk['case']}.** The closed loop (trained DPC on this plant) diverges: first non-finite sample at t = {sk['first_non_finite_s']} s, so there is no window to identify. ({sk['reason']})", ""]
    for est in idn.ESTIMATORS:
        lines += [f"## {est}", "", f"Mean time per identification: {meta['mean_identify_time_s'][est]:.1f} s ({meta['jobs']} cases in parallel, {meta['threads_per_job']} threads each; slower than a single interactive run).", "",
                  "| Parameter | Noise | Mean abs. error | Max abs. error | Interval coverage | Mean interval width | n |", "|---|---|---|---|---|---|---|"]
        for _, r in summary[summary["estimator"] == est].iterrows():
            lines.append(f"| {r['param']} | {r['noise_pct']:g} % | {r['mean_abs_error_pct']:.2f} % | {r['max_abs_error_pct']:.2f} % | {r['coverage']:.0%} | {r['mean_rel_width_pct']:.2f} % | {r['n']} |")
        lines.append("")
    lines += ["## Per case (PINN, C estimate, mean over noise realizations)", "", "| Case | " + " | ".join(f"{n:g} %" for n in meta["noise_pct"]) + " |", "|---|" + "---|" * len(meta["noise_pct"])]
    pc = table[(table["estimator"] == "pinn") & (table["param"] == "C")].groupby(["case", "noise_pct"])["rel_error_pct"].mean()
    for c in meta["cases"]:
        if any(sk["case"] == c["name"] for sk in meta["skipped_cases"]):
            continue
        lines.append(f"| {c['name']} | " + " | ".join(f"{pc.get((c['name'], n), float('nan')):+.1f} %" for n in meta["noise_pct"]) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
