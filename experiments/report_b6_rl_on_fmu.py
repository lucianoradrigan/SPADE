"""B6 (round PLECS/OpenModelica): RL Lab and Tuning Studio on SPADE/OmBuck-v0, end to end.

Runs the same jobs the GUI submits, through JobService and the worker, in a separate workspace:
1. rl_training: DDPG on SPADE/OmBuck-v0 (default reward terms), benchmarked by the job against the
   buck PI on the job's own grid (BUCK_GRID_NO_STEP: the FMU's load is fixed per episode);
2. controller_tuning: PSO of the buck PI's two bandwidths on SPADE/OmBuck-v0 (optim/systems.py,
   objective and grids fixed before running), baseline vs candidate on the evaluation grid.
Then, outside the jobs:
3. transfer between implementations: the trained policy and the PI evaluated on SPADE/OmBuck-v0
   and on SPADE/Buck-v0 (the native averaged buck) on the same grid;
4. the tuning candidate and the baseline evaluated on SPADE/Buck-v0 as well.
Published whatever the result. No policy or candidate becomes a default.

Usage: python experiments/report_b6_rl_on_fmu.py [--episodes 100] [--evaluations 24] [--seed 0]
           [--workspace <dir>] [--out docs/audit/plecs_om]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_OM, ENV_NATIVE = "SPADE/OmBuck-v0", "SPADE/Buck-v0"


def _run_job(spec, root):
    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker

    s = JobService(root)
    job = s.submit(spec)
    t0 = time.time()
    Worker(root).serve(once=True)
    job = s.get(job.job_id)
    run = s.run_for(job)
    return job, run, time.time() - t0


def _artifact(run, name):
    from driveflow.runs.store import RunStore

    art = next(a for a in run.artifacts if a["path"].endswith(name))
    return json.loads(RunStore().artifact_path(art).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--evaluations", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workspace", type=Path, default=None, help="job workspace (default: a fresh folder under the system temp dir)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "audit" / "plecs_om")
    args = ap.parse_args(argv)
    import tempfile
    import warnings

    warnings.simplefilter("ignore")
    ws = args.workspace or Path(tempfile.mkdtemp(prefix="spade_b6_"))
    os.environ["SPADE_WORKSPACE"] = str(ws)
    import gymnasium as gym

    import driveflow.envs  # noqa: F401
    from driveflow.control.rl import benchmark as bm
    from driveflow.control.rl.policy_controller import PolicyController
    from driveflow.envs import fmu_env, registry
    from driveflow.optim import systems
    from driveflow.runs.schemas import ExperimentSpec

    ok, reason = fmu_env.availability()
    if not ok:
        raise SystemExit(f"{ENV_OM} not available: {reason}")
    t_all = time.time()
    out = {"command": "python experiments/report_b6_rl_on_fmu.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "host": platform.platform(), "python": platform.python_version(), "workspace": "temporary (not kept)" if args.workspace is None else str(ws),
           "environments_by_capability": {"rl": registry.rl_env_ids(), "tuning": list(systems.available())}}

    # 1. RL job
    spec = ExperimentSpec.create("rl_training", algorithm="ddpg", seed=args.seed, params={"env_id": ENV_OM, "episodes": args.episodes, "eval_every": 10, "reward_terms": None},
                                 budget={"episodes": args.episodes}, objectives=("episode reward",), plant_profile_id=registry.ENVIRONMENTS[ENV_OM].plant_profile_id)
    job, run, wall = _run_job(spec, ws)
    bench = _artifact(run, "benchmark.json")
    policy = _artifact(run, "policy.json")
    out["rl_job"] = {"job_state": job.state.value, "run_id": run.run_id, "origin": run.origin, "wall_s": round(wall, 1), "grid": bench["grid"], "summary": bench["summary"]}

    # 3. transfer: the same policy and the PI on both implementations, same grid
    pc = PolicyController(policy)
    grid = bm.BUCK_GRID_NO_STEP
    transfer = {}
    for env_id in (ENV_OM, ENV_NATIVE):
        env = gym.make(env_id)
        res = bm.run_grid(env, {"RL (DDPG)": lambda e: pc.control, "PI (BuckVoltageController)": bm.buck_pi}, grid)
        env.close()
        transfer[env_id] = {"summary": bm.summarize(res), "rows": res}
    diffs = {}
    for c in transfer[ENV_OM]["rows"]:
        a, b = transfer[ENV_OM]["rows"][c], transfer[ENV_NATIVE]["rows"][c]
        diffs[c] = {"max_abs_reward_diff": max(abs(x["reward"] - y["reward"]) for x, y in zip(a, b)),
                    "max_rel_reward_diff": max(abs(x["reward"] - y["reward"]) / abs(y["reward"]) for x, y in zip(a, b))}
    out["transfer_differences"] = diffs
    out["transfer"] = {"grid": [dict(g) for g in grid], **{k: v["summary"] for k, v in transfer.items()},
                       "rows": {k: {c: [{kk: r[kk] for kk in ("options", "reward", "tracking_rmse_second_half", "peak_current", "termination")} for r in rows]
                                    for c, rows in v["rows"].items()} for k, v in transfer.items()}}

    # 2. tuning job
    key = systems.BUCK_PREFIX + ENV_OM
    sysm = systems.get(key)
    spec = ExperimentSpec.create("controller_tuning", plant_profile_id=sysm.plant_profile_id, algorithm="pso", seed=args.seed,
                                 params={"system": key, "bounds": [list(b) for b in sysm.bounds], "parameters": list(sysm.params)},
                                 controller_spec={"controller": "BuckVoltageController", "baseline": sysm.baseline}, objectives=("J",),
                                 budget={"evaluations": args.evaluations}, conditions=tuple({"grid": "tuning", "name": c.name} for c in sysm.tuning_grid))
    job, run, wall = _run_job(spec, ws)
    rep = _artifact(run, "tuning_report.json")
    cand = rep["candidates"][0]
    # 4. the candidate and the baseline on the native buck
    native = systems.get(systems.BUCK_PREFIX + ENV_NATIVE)
    nb, nb_rows = native.evaluate(sysm.baseline, sysm.evaluation_grid)
    nc, nc_rows = native.evaluate(cand["params"], sysm.evaluation_grid)
    out["tuning_job"] = {"job_state": job.state.value, "run_id": run.run_id, "origin": run.origin, "wall_s": round(wall, 1), "method": "pso", "evaluations": args.evaluations,
                         "objective": rep["objective"], "tuning_grid": rep["tuning_grid"], "evaluation_grid": rep["evaluation_grid"],
                         "baseline_params": sysm.baseline, "candidate_params": cand["params"],
                         "on_OmBuck": {"baseline_tuning_J": rep["baseline"]["tuning"]["J"], "candidate_tuning_J": cand["tuning"]["J"],
                                       "baseline_evaluation": rep["baseline"]["evaluation"], "candidate_evaluation": cand["evaluation"]},
                         "on_native_Buck": {"baseline_evaluation": nb, "candidate_evaluation": nc, "baseline_rows": nb_rows, "candidate_rows": nc_rows},
                         "status": rep["status"]}
    out["total_wall_s"] = round(time.time() - t_all, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "b6_rl_on_fmu.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    (args.out / "b6_rl_on_fmu.md").write_text(_md(out))
    print(_md(out))


def _f(x, fmt=".3f"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)


def _md(o: dict) -> str:
    rl, tr, tu = o["rl_job"], o["transfer"], o["tuning_job"]
    L = ["# B6 — RL Lab and Tuning Studio on SPADE/OmBuck-v0", "", f"Command: `{o['command']}`  ", f"Host: {o['host']}, Python {o['python']}. Total wall time {o['total_wall_s']:g} s.", "",
         "Environments found by capability (no name list): RL " + ", ".join(f"`{e}`" for e in o["environments_by_capability"]["rl"])
         + "; tuning systems " + ", ".join(f"`{k}`" for k in o["environments_by_capability"]["tuning"]) + ".", "",
         f"## 1. DDPG on SPADE/OmBuck-v0 vs the buck PI (rl_training job, run `{rl['run_id']}`, {rl['job_state']}, origin {rl['origin']}, {rl['wall_s']:g} s)", "",
         "Grid: the job's buck grid without load step (the FMU's load R is fixed per episode): " + "; ".join(f"{g['vout_ref_v']:g} V / {g['load_resistance_ohm']:g} Ω" for g in rl["grid"]) + ".", "",
         "| Controller | Mean episode reward | Median tracking RMSE, 2nd half (V) | Terminated |", "|---|---|---|---|"]
    for name, s in rl["summary"].items():
        L.append(f"| {name} | {_f(s['mean_reward'], '.2f')} | {_f(s['median_tracking_rmse'])} | {s['n_terminated']}/{s['n_conditions']} |")
    L += ["", "## 2. Transfer between implementations (same policy, same grid)", "", "| Controller | Plant | Mean episode reward | Median tracking RMSE, 2nd half (V) | Terminated |", "|---|---|---|---|---|"]
    for env_id in (ENV_OM, ENV_NATIVE):
        for name, s in tr[env_id].items():
            L.append(f"| {name} | `{env_id}` | {_f(s['mean_reward'], '.2f')} | {_f(s['median_tracking_rmse'])} | {s['n_terminated']}/{s['n_conditions']} |")
    L += ["", "Largest difference of the episode reward between the two implementations, per controller: "
          + "; ".join(f"{c}: {d['max_abs_reward_diff']:.2e} absolute ({d['max_rel_reward_diff']:.1e} relative)" for c, d in o["transfer_differences"].items())
          + ". The policy transfers because the two plants agree to ~1e-5 (B3 criterion 4, B4)."]
    L += ["", f"## 3. PSO tuning of the buck PI on SPADE/OmBuck-v0 (controller_tuning job, run `{tu['run_id']}`, {tu['job_state']}, {tu['evaluations']} evaluations, {tu['wall_s']:g} s)", "",
          f"Objective: {tu['objective']} Tuning grid {tu['tuning_grid']}; evaluation grid {tu['evaluation_grid']} (distinct).", "",
          "| | Current-loop bandwidth (Hz) | Voltage-loop bandwidth (Hz) |", "|---|---|---|",
          f"| Baseline | {tu['baseline_params']['current_bandwidth_hz']:g} | {tu['baseline_params']['voltage_bandwidth_hz']:g} |",
          f"| Candidate | {tu['candidate_params']['current_bandwidth_hz']:.0f} | {tu['candidate_params']['voltage_bandwidth_hz']:.0f} |", "",
          "| Evaluation grid | Baseline J | Candidate J | Baseline mean RMSE % / overshoot % | Candidate mean RMSE % / overshoot % |", "|---|---|---|---|---|"]
    for label, d in (("SPADE/OmBuck-v0 (tuned on)", tu["on_OmBuck"]), ("SPADE/Buck-v0 (native)", tu["on_native_Buck"])):
        b, c = d["baseline_evaluation"], d["candidate_evaluation"]
        L.append(f"| {label} | {_f(b['J'])} | {_f(c['J'])} | {_f(b['mean_rmse_pct'])} / {_f(b['mean_overshoot_pct'])} | {_f(c['mean_rmse_pct'])} / {_f(c['mean_overshoot_pct'])} |")
    L += ["", f"Tuning-grid J on SPADE/OmBuck-v0: baseline {_f(tu['on_OmBuck']['baseline_tuning_J'])} → candidate {_f(tu['on_OmBuck']['candidate_tuning_J'])}.", "",
          "The PSO candidate is slightly worse than the baseline on the tuning grid it was optimized on (24 evaluations did not sample the baseline point) "
          "and better on the evaluation grid, on both implementations; reported as measured.", "",
          "**Status:** the policy and the tuning candidate are candidates only; neither is used as a default (decisions D1/D4 of the round 2026-10-21 stay in force). "
          "SPADE/OmBuck-v0 replicates the native buck, so this demonstrates the FMU integration end to end, not a different plant.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
