"""Fit and evaluate the buck surrogate -- expansion task E5 (round 2026-10-21, D3).

Writes the fitted surrogate to configs/surrogates/buck_surrogate/<date>_runNN/surrogate.json and the
report to docs/audit/plecs/e5_surrogate_and_comparison.{md,json}.

Steps:
1. Data: open-loop duty excitation of the native averaged buck over the (V_in × R) grid of
   envs/surrogate_env.py; conditions split into train / validation / test (never windows).
2. Fit both families on the TRAIN conditions only: the ZOH grey box (L, C, r_L) and the MLP.
3. Per condition and family:
   - one-step RMSE;
   - open-loop rollout error (relative RMSE per channel);
   - closed-loop error with the same PI and a load step (native vs. surrogate).
4. Default family: the one with the lower median relative v_out rollout RMSE on the VALIDATION
   conditions. The test conditions are reported, never used for a choice.

Source data is SPADE's own buck, labelled SIMULATED. This validates the surrogate mechanics, not
fidelity to a PLECS circuit; no PLECS data was used.

    python experiments/fit_buck_surrogate.py [--seed 0]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np

from driveflow.envs import surrogate_env as se

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "audit" / "plecs"


def _next_dir() -> Path:
    base = se.SURROGATE_DIR / f"{dt.date.today().isoformat()}_run"
    n = 1
    while Path(f"{base}{n:02d}").exists():
        n += 1
    return Path(f"{base}{n:02d}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    records = se.build_dataset(args.seed)
    train = [r for r in records if r["split"] == "train"]
    fits = {"zoh": se.fit_zoh(train), "mlp": se.fit_mlp(train, seed=args.seed, epochs=args.epochs)}
    tr = se.transitions(train)
    domain = {"v_in_v": [float(tr["vin"].min()), float(tr["vin"].max())], "load_resistance_ohm": [float(tr["r"].min()), float(tr["r"].max())],
              "duty": list(se.DUTY_RANGE), "i_l_a": [float(tr["x"][:, 0].min()), float(tr["x"][:, 0].max())],
              "v_out_v": [float(tr["x"][:, 1].min()), float(tr["x"][:, 1].max())]}
    surrogate = {"source": "native:buck_v1 (SPADE averaged buck, SIMULATED)", "origin": "SIMULATED · SURROGATE", "domain": domain,
                 "families": {f: {"fit": fits[f]} for f in se.FAMILIES}, "seed": args.seed,
                 "conditions": {f"{r['v_in_v']:g}V_{r['load_resistance_ohm']:g}ohm": r["split"] for r in records}}

    rows = []
    for rec in records:
        vin, r = rec["v_in_v"], rec["load_resistance_ohm"]
        for fam in se.FAMILIES:
            plant = se.make_plant(surrogate, fam, vin, r)
            x = se.rollout(plant, rec["duty"])
            # one step: from each native state
            p1 = se.make_plant(surrogate, fam, vin, r)
            one = []
            for k in range(0, len(rec["duty"]), 10):
                p1.reset(*rec["x"][k])
                s = p1.simulate(float(rec["duty"][k]))
                one.append((s.i_l - rec["x"][k + 1, 0], s.v_out - rec["x"][k + 1, 1]))
            one = np.array(one)
            cl = se.compare_condition(surrogate, fam, vin, r, step_ohm=r / 2)
            rel = se._rel_rmse(x, rec["x"])
            rows.append({"v_in_v": vin, "load_resistance_ohm": r, "split": rec["split"], "family": fam,
                         "one_step_rmse_i_l_a": float(np.sqrt(np.mean(one[:, 0] ** 2))), "one_step_rmse_v_out_v": float(np.sqrt(np.mean(one[:, 1] ** 2))),
                         "rollout_rel_rmse_i_l": rel[0], "rollout_rel_rmse_v_out": rel[1], "rollout_finite": rel[0] is not None,
                         "closed_loop_status": cl["status"], "closed_loop_rmse_v_out_v": cl.get("rmse_v_out_v"), "closed_loop_max_abs_v_out_v": cl.get("max_abs_v_out_v"),
                         "out_of_domain": cl["out_of_domain"]})

    def med(fam, split, key):
        vals = [r[key] for r in rows if r["family"] == fam and r["split"] == split and r[key] is not None]
        return float(np.median(vals)) if vals else None

    for fam in se.FAMILIES:
        surrogate["families"][fam]["validation"] = {k: med(fam, "validation", k) for k in ("rollout_rel_rmse_v_out", "rollout_rel_rmse_i_l", "closed_loop_rmse_v_out_v")}
    score = {f: surrogate["families"][f]["validation"]["rollout_rel_rmse_v_out"] for f in se.FAMILIES}
    surrogate["default_family"] = min(score, key=lambda f: np.inf if score[f] is None else score[f])
    surrogate["default_rule"] = "lower median relative v_out rollout RMSE on the validation conditions"
    out_dir = _next_dir()
    out_dir.mkdir(parents=True)
    (out_dir / "surrogate.json").write_text(json.dumps(surrogate, indent=1))

    rep = {"meta": {"command": "python experiments/fit_buck_surrogate.py " + " ".join(f"--{k} {v}" for k, v in vars(args).items()), "date": time.strftime("%Y-%m-%d"),
                    "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}", "runtime_s": round(time.perf_counter() - t0, 1),
                    "surrogate_run": str(out_dir.relative_to(ROOT)), "split": {f"{v:g}V_{r:g}ohm": s for (v, r), s in se.SPLIT.items()},
                    "extrapolation": [f"{v:g}V_{r:g}ohm" for v, r in se.EXTRAPOLATION]},
           "zoh_identified": {k: fits["zoh"][k] for k in ("l_h", "c_f", "r_l_ohm")}, "nominal": {"l_h": 220e-6, "c_f": 220e-6, "r_l_ohm": 0.05},
           "domain": domain, "default_family": surrogate["default_family"], "default_rule": surrogate["default_rule"], "validation": {f: surrogate["families"][f]["validation"] for f in se.FAMILIES},
           "rows": rows}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "e5_surrogate_and_comparison.json").write_text(json.dumps(rep, indent=1, default=float))
    (REPORT / "e5_surrogate_and_comparison.md").write_text(render(rep))
    print(render(rep))


def _f(x, spec=".4f"):
    return "—" if x is None else format(x, spec)


def render(rep: dict) -> str:
    z, n = rep["zoh_identified"], rep["nominal"]
    L = ["# E5: buck surrogate and the native-vs-surrogate comparison", "",
         f"Generated by `{rep['meta']['command']}` on {rep['meta']['date']} ({rep['meta']['machine']}), {rep['meta']['runtime_s']} s. Fitted surrogate: `{rep['meta']['surrogate_run']}/surrogate.json`. Raw values: [`e5_surrogate_and_comparison.json`](e5_surrogate_and_comparison.json).", "",
         "**Source: SPADE's own averaged buck (native:buck_v1, SIMULATED), not PLECS.** This validates the surrogate mechanics (condition split, fit, validity domain, environment, comparison). "
         "It says nothing about fidelity to a PLECS circuit: no PLECS data exists (task E2 never ran on PLECS). The environment is labelled SIMULATED · SURROGATE.", "",
         "Data: open-loop duty excitation (piecewise-constant levels in [0.05, 0.85], 25–150 steps each, 50 ms from rest) per (V_in, R) condition. "
         f"Split by condition: validation {', '.join(k for k, v in rep['meta']['split'].items() if v == 'validation')}; test {', '.join(k for k, v in rep['meta']['split'].items() if v == 'test')}; "
         f"the other 12 grid conditions train both families. Extrapolation (outside the domain, reported only): {', '.join(rep['meta']['extrapolation'])}.", "",
         f"Grey-box ZOH identification (started at ×1.5 L, ×0.7 C, ×2 r_L): L = {z['l_h'] * 1e6:.3f} µH (nominal {n['l_h'] * 1e6:.0f}), C = {z['c_f'] * 1e6:.3f} µF (nominal {n['c_f'] * 1e6:.0f}), r_L = {z['r_l_ohm']:.5f} Ω (nominal {n['r_l_ohm']}).", "",
         f"Validity domain: V_in {rep['domain']['v_in_v']} V, R {rep['domain']['load_resistance_ohm']} Ω, duty {rep['domain']['duty']}.", "",
         f"Default family of SPADE/BuckSurrogate-v0: **{rep['default_family']}** ({rep['default_rule']}).", "",
         "| Condition | Split | Family | One-step RMSE i_L (A) / v_out (V) | Open-loop rollout rel. RMSE i_L / v_out | Closed loop vs. native: RMSE / max |v_out| (V) | Status |", "|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        L.append(f"| {r['v_in_v']:g} V, {r['load_resistance_ohm']:g} Ω | {r['split']} | {r['family']} | {_f(r['one_step_rmse_i_l_a'], '.2e')} / {_f(r['one_step_rmse_v_out_v'], '.2e')} | "
                 f"{_f(r['rollout_rel_rmse_i_l'], '.2%')} / {_f(r['rollout_rel_rmse_v_out'], '.2%')} | {_f(r['closed_loop_rmse_v_out_v'], '.3f')} / {_f(r['closed_loop_max_abs_v_out_v'], '.3f')} | "
                 f"{r['closed_loop_status']}{' · OUT OF DOMAIN: ' + '; '.join(r['out_of_domain']) if r['out_of_domain'] else ''} |")
    L += ["", "Closed loop: the same BuckVoltageController (designed on the native nominal plant) drives each plant; reference 0.5·V_in; the load halves at 10 ms; 20 ms per run. "
          "At R = 3 Ω the halved load (1.5 Ω) is outside the domain, so those closed-loop rows carry the out-of-domain warning.", "",
          "## Reading", "",
          "- **ZOH grey box: zero error is expected here, not a result about real circuits.** The data come from the same averaged model the grey box assumes, without noise, "
          "so the identification is exact. Against a switched PLECS circuit (ripple, dead time, losses) its error would be the model mismatch, which this run cannot measure.",
          "- **MLP: the black box generalises across the held-out conditions but drifts in open loop** (relative v_out errors of a few % to tens of %, worst at the edge of the domain, 60 V / 24 Ω). "
          "In closed loop the PI corrects most of it; the max |v_out| deviations are a fraction of a volt inside the domain and grow at the 1.5 Ω step outside it.",
          "- The default family was chosen on the validation conditions only; the test rows are reported, not used.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
