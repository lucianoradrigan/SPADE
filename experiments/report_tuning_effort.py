"""E7 with a control-effort term, a current constraint and a noisy evaluation grid, compared with C2.
Writes docs/audit/expansion/e7_tuning_effort.{md,json}.

Decided by the user on 2026-10-21, after the first E7 report pushed the speed-loop bandwidth to its
bound (no effort term). Everything below is fixed in optim/objectives.py ("J_effort") before running:
- PSO, Bayesian optimisation (on J_effort) and NSGA-II (overshoot, dip, effort; peak step current
  <= 97 A) run as real ``controller_tuning`` jobs on the TUNING grid, same budget and seed as the
  first report;
- every controller below is then simulated on EVALUATION_GRID_NOISY (0.6 rad/s speed-sensor noise
  on every condition, plant drift on two) and on the noise-free EVALUATION_GRID:
  * baseline PI (80 rad/s, zeta 0.9, b = 1);
  * C2: the same PI with setpoint weight b = 0.7 and b = 0.5 (TODO C2, closed 2026-10-20);
  * E7 v1: the best PSO candidate of e7_tuning_s04.json (objective without effort);
  * the best candidate of each method on J_effort.
Candidates only: no default is changed.

    python experiments/report_tuning_effort.py [--evaluations 96] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "expansion"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluations", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    os.environ["SPADE_WORKSPACE"] = tempfile.mkdtemp(prefix="spade_e7e_")

    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker
    from driveflow.optim import objectives as ob
    from driveflow.runs.schemas import ExperimentSpec

    s = JobService()
    jobs = {}
    for method in ("pso", "bayes", "nsga2"):
        spec = ExperimentSpec.create("controller_tuning", plant_profile_id="native:dc_perm_ex_v1", algorithm=method, seed=args.seed,
                                     params={"bounds": [list(b) for b in ob.BOUNDS], "objective": "J_effort"}, budget={"evaluations": args.evaluations})
        jobs[method] = s.submit(spec).job_id
    t0 = time.perf_counter()
    Worker().serve(once=True, poll_s=0.05)
    wall = time.perf_counter() - t0

    v1 = json.loads((OUT / "e7_tuning_s04.json").read_text())["methods"]["pso"]["best"]["params"]
    controllers = {"baseline": dict(ob.BASELINE), "C2 b=0.7": dict(ob.BASELINE, setpoint_weight=0.7), "C2 b=0.5": dict(ob.BASELINE, setpoint_weight=0.5),
                   "E7 v1 (PSO, no effort)": v1}
    methods = {}
    for method, job_id in jobs.items():
        job = s.get(job_id)
        run = s.run_for(job)
        rep = json.loads(s.store.artifact_path(next(a for a in run.artifacts if a["path"].endswith("tuning_report.json"))).read_text())
        feasible = [c for c in rep["candidates"] if c["tuning"]["current_constraint_ok"]] or rep["candidates"]
        best = min(feasible, key=lambda c: c["tuning"]["J"])
        methods[method] = {"state": job.state.value, "n_evaluations": run.metrics["n_evaluations"], "n_candidates": len(rep["candidates"]),
                           "best_params": best["params"], "best_tuning_J_effort": best["tuning"]["J"],
                           "front": [{"params": c["params"], "objectives": c.get("objectives")} for c in rep["candidates"]] if method == "nsga2" else None}
        controllers[f"E7 J_effort ({method})"] = best["params"]

    rows = {}
    for name, kw in controllers.items():
        noisy, noisy_rows = ob.evaluate(kw, ob.EVALUATION_GRID_NOISY)
        clean, clean_rows = ob.evaluate(kw, ob.EVALUATION_GRID)
        tune, _ = ob.evaluate(kw, ob.TUNING_GRID)
        rows[name] = {"params": kw, "tuning": dict(tune, J_effort=ob.j_effort(tune)), "evaluation_noisy": dict(noisy, J_effort=ob.j_effort(noisy)), "evaluation_noisy_rows": noisy_rows,
                      "evaluation_clean": dict(clean, J_effort=ob.j_effort(clean)), "evaluation_clean_rows": clean_rows}
    out = {"meta": {"command": "python experiments/report_tuning_effort.py " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items() if k != "out"),
                    "date": time.strftime("%Y-%m-%d"), "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
                    "evaluations": args.evaluations, "seed": args.seed, "wall_s_all_three": round(wall, 1), "objective": ob.OBJECTIVES["J_effort"][3],
                    "i_nominal_a": ob.I_NOMINAL_A, "noise_std_rad_s": ob.NOISE_STD_RAD_S, "baseline_tuning_normalisers": ob.baseline_tuning_summary(),
                    "search_space": dict(zip(ob.PARAMS, ob.BOUNDS)), "evaluation_grid_noisy": [vars(c) for c in ob.EVALUATION_GRID_NOISY],
                    "grids_overlap": ob.grids_overlap(ob.TUNING_GRID, ob.EVALUATION_GRID_NOISY)},
           "methods": methods, "controllers": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "e7_tuning_effort.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out / "e7_tuning_effort.md").write_text(render(out))
    print(render(out))


def _f(x, spec=".2f"):
    return "—" if x is None else format(x, spec)


def render(out: dict) -> str:
    m = out["meta"]
    L = ["# E7 with a control-effort term, a current constraint and sensor noise, compared with C2", "",
         f"Generated by `{m['command']}` on {m['date']} ({m['machine']}); raw values in [`e7_tuning_effort.json`](e7_tuning_effort.json). "
         f"Three `controller_tuning` jobs (PSO, Bayesian optimisation, NSGA-II), {m['evaluations']} evaluations each, seed {m['seed']}, {m['wall_s_all_three']} s.", "",
         "Simulated closed loop only. Every row is a **candidate**: no default was changed (D1; the user's decision of 2026-10-21: no default for now).", "",
         f"- Objective (fixed before running): {m['objective']}.",
         f"- Rated current I_nom = {m['i_nominal_a']:g} A; speed-sensor noise σ = {m['noise_std_rad_s']:g} rad/s (0.2 % of the rated 300 rad/s) on every noisy evaluation condition.",
         f"- Tuning and evaluation grids overlap: {m['grids_overlap']}.", "",
         "## Evaluation grid with sensor noise and drift (the comparison)", "",
         "| Controller | Bandwidth | Damping | b | Overshoot % | Settling ms | Dip rad/s | Effort V | Peak step current A | ≤ 97 A | J_effort |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in out["controllers"].items():
        p, e = r["params"], r["evaluation_noisy"]
        L.append(f"| {name} | {p['speed_bandwidth_rad_s']:.1f} | {p['speed_damping']:.3f} | {p['setpoint_weight']:.3f} | {_f(e['mean_overshoot_pct'])} | {_f(e['mean_settling_ms'], '.1f')} | "
                 f"{_f(e['mean_dip_rad_s'])} | {_f(e['mean_effort_v'], '.3f')} | {_f(e['max_step_peak_i_a'], '.1f')} | {'yes' if e['current_constraint_ok'] else '**no**'} | {_f(e['J_effort'])} |")
    L += ["", "## Same controllers, noise-free evaluation grid", "",
          "| Controller | Overshoot % | Settling ms | Dip rad/s | Effort V | Peak step current A | J_effort |", "|---|---|---|---|---|---|---|"]
    for name, r in out["controllers"].items():
        e = r["evaluation_clean"]
        L.append(f"| {name} | {_f(e['mean_overshoot_pct'])} | {_f(e['mean_settling_ms'], '.1f')} | {_f(e['mean_dip_rad_s'])} | {_f(e['mean_effort_v'], '.3f')} | {_f(e['max_step_peak_i_a'], '.1f')} | {_f(e['J_effort'])} |")
    L += ["", "## Per condition, noisy evaluation grid", ""]
    names = list(out["controllers"])
    L += ["| Condition | Metric | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    first = out["controllers"][names[0]]["evaluation_noisy_rows"]
    for i, row in enumerate(first):
        keys = ("overshoot_pct", "settling_ms", "effort_v", "peak_i_a") if row["kind"] == "step" else ("dip_rad_s", "effort_v", "peak_i_a")
        for k in keys:
            L.append(f"| {row['condition']} | {k} | " + " | ".join(_f(out["controllers"][n]["evaluation_noisy_rows"][i].get(k)) for n in names) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
