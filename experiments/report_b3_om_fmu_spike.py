"""B3 (round PLECS/OM): GO / NO-GO of the OpenModelica FMUs against SPADE's native plants, under the
criteria pre-registered in docs/audit/plecs_om/b3_preregistration.md (committed before this script ran).

    SPADE_FMU_DIR=... python experiments/report_b3_om_fmu_spike.py --out docs/audit/plecs_om

Writes b3_om_fmu_spike.json and b3_figures/*.svg. The .md is written by hand from the JSON."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from driveflow.datagen.runner import TAU as DC_TAU
from driveflow.datagen.runner import build_plant
from driveflow.datagen.scenario import Scenario
from driveflow.integrations.fmu.adapter import FmuAdapter, FmuError
from driveflow.integrations.plecs.excitation import sequence
from driveflow.sim.buck_system import TAU as BUCK_TAU
from driveflow.sim.buck_system import BuckSystem

REPO = Path(__file__).resolve().parents[1]
FMU_DIR = Path(os.environ.get("SPADE_FMU_DIR", REPO / "data" / "fmu"))
DC_SUPPLY_V = 65.0  # datagen/runner.py:141
SUBSTEPS = 4  # criterion 5: Ts/4, the sub-step that ran most often in B1 (S-15)

BUCK = {"tau": BUCK_TAU, "levels": [0.0, 0.3, 0.6, 0.45], "n_level": 250, "prbs": dict(prbs_seed=5, D_min=0.1, D_max=0.8, T_hold=1e-3), "n_prbs": 1000,
        "signals": ["v_out", "i_L"]}
DC = {"tau": DC_TAU, "levels": [0.4, 0.7, 0.55], "n_level": 1000, "prbs": dict(prbs_seed=9, D_min=0.3, D_max=0.8, T_hold=20e-3), "n_prbs": 3000,
      "signals": ["omega", "i_a"]}
PLANT_OF = {"BuckAveraged": BUCK, "BuckSwitching": BUCK, "DcMotorAveraged": DC, "DcMotorSwitching": DC}


def duties(plant, prbs_seed=None):
    lv = np.repeat(plant["levels"], plant["n_level"])
    prbs = dict(plant["prbs"], **({"prbs_seed": prbs_seed} if prbs_seed else {}))
    t = np.arange(plant["n_prbs"]) * plant["tau"]
    return np.concatenate([lv, sequence("prbs", prbs, t)])


# ------------------------------------------------------------------------------------- native plants
def native_buck(d):
    sys_ = BuckSystem()
    sys_.reset()
    out = np.empty((len(d), 2))
    for k, x in enumerate(d):
        s = sys_.simulate(x)
        out[k] = (s.v_out, s.i_l)
    return {"v_out": out[:, 0], "i_L": out[:, 1], "freewheel_steps": 0}


def native_dc(d, euler_substeps=1):
    """euler_substeps > 1 only for the post-hoc diagnostic: the same plant with a finer Euler step."""
    import driveflow.datagen.runner as runner
    from driveflow.sim import EulerSolver

    original = runner.EulerSolver
    runner.EulerSolver = lambda: EulerSolver(nsteps=euler_substeps)
    try:
        system = build_plant(Scenario(scenario_id="b3_native"))
    finally:
        runner.EulerSolver = original
    system.seed(np.random.SeedSequence(0))
    state = system.reset()
    ii = system.CURRENTS_IDX[0]
    out, neg = np.empty((len(d), 2)), 0
    for k, x in enumerate(d):
        neg += int((state * system.limits)[ii] < 0)
        state = system.simulate(np.array([x]))
        ph = state * system.limits
        out[k] = (ph[system.OMEGA_IDX], ph[ii])
    return {"omega": out[:, 0], "i_a": out[:, 1], "freewheel_steps": neg}


# ------------------------------------------------------------------------------------------ FMU runs
class Run:
    """One FMU episode with the B3 driver; the DC averaged FMU gets the native converter's rule."""

    def __init__(self, name, params=None, supply=DC_SUPPLY_V, two_quadrant=False):
        self.name, self.params, self.supply, self.two_q = name, params, supply, two_quadrant
        self.ad = FmuAdapter()
        self.ad.load(FMU_DIR / f"{name}.fmu")

    def _input(self, duty, i_a):
        if self.name == "DcMotorAveraged":
            if self.two_q:
                return {"v_a": self.supply * duty}
            return {"v_a": self.supply * duty if i_a >= 0 else self.supply}
        return {"duty": duty}

    def run(self, d, tau, substeps=1):
        """Returns per-control-step outputs (end of step); with substeps > 1 also the trapezoid cycle
        average and the max-min ripple of the (substeps + 1) samples of each step."""
        ad = self.ad
        ad.reset(params=self.params)
        sig = [v.name for v in ad.info.outputs]
        prev = ad.get_outputs(check_range=False)
        h = tau / substeps
        end, avg, rip = np.empty((len(d), len(sig))), np.empty((len(d), len(sig))), np.empty((len(d), len(sig)))
        n, neg, t_err = 0, 0, 0.0
        for k, x in enumerate(d):
            i_a = prev.get("i_a", 0.0)
            neg += int(self.name == "DcMotorAveraged" and not self.two_q and i_a < 0)
            ad.set_inputs(self._input(float(x), i_a))
            samples = [[prev[s] for s in sig]]
            for _ in range(substeps):
                ad.do_step(n * h, h)
                n += 1
                t_err = max(t_err, abs(ad.time - n * h) / (n * h))
                prev = ad.get_outputs(check_range=False)
                samples.append([prev[s] for s in sig])
            arr = np.asarray(samples)
            end[k] = arr[-1]
            avg[k] = np.trapezoid(arr, axis=0) / substeps if substeps > 1 else arr[-1]
            rip[k] = arr.max(axis=0) - arr.min(axis=0)
        res = {s: end[:, j] for j, s in enumerate(sig)}
        res["avg"] = {s: avg[:, j] for j, s in enumerate(sig)}
        res["ripple"] = {s: rip[:, j] for j, s in enumerate(sig)}
        res["freewheel_steps"], res["time_rel_err"], res["n_doStep"] = neg, t_err, n
        return res

    def close(self):
        self.ad.close()


def nrms(a, ref):
    rng = float(np.max(ref) - np.min(ref))
    return float(np.sqrt(np.mean((np.asarray(a) - ref) ** 2)) / rng) if rng > 0 else float("nan")


def episode(name, d, tau, substeps=1, **kw):
    r = Run(name, **kw)
    try:
        t0 = time.perf_counter()
        out = r.run(d, tau, substeps)
        out["wall_s"] = time.perf_counter() - t0
        out["stats"] = r.ad.stats()
        out["error"] = None
    except FmuError as exc:
        out = {"error": f"[{exc.code}] {exc}", "stats": r.ad.stats(), "wall_s": None}
    finally:
        r.close()
    return out


def evaluate(name):
    plant = PLANT_OF[name]
    tau, sig = plant["tau"], plant["signals"]
    switching = name.endswith("Switching")
    d1, d2 = duties(plant), duties(plant, prbs_seed=77)
    native = (native_buck if plant is BUCK else native_dc)(d1)
    rec = {"tau_s": tau, "n_steps": len(d1), "native_freewheel_steps": native["freewheel_steps"]}
    a, b, c = episode(name, d1, tau), episode(name, d1, tau), episode(name, d2, tau)
    errs = [e["error"] for e in (a, b, c) if e["error"]]
    rec["errors"] = errs
    if errs:
        rec["criteria"] = {str(i): {"pass": False, "note": "the FMU did not simulate: " + errs[0]} for i in (1, 2, 3)}
    else:
        same = all(np.array_equal(a[s], b[s]) for s in sig)
        maxrel = max(float(np.max(np.abs(a[s] - b[s]) / np.maximum(np.abs(a[s]), 1e-300))) for s in sig)
        differ = all(not np.array_equal(a[s], c[s]) for s in sig)
        rec["criteria"] = {
            "1": {"pass": bool(differ and same), "other_sequence_differs": differ, "same_sequence_identical": same},
            "2": {"pass": bool(a["time_rel_err"] <= 1e-12), "max_rel_time_error": a["time_rel_err"], "n_doStep": a["n_doStep"]},
            "3": {"pass": bool(same or maxrel <= 1e-9), "bit_identical": same, "max_rel_diff": maxrel},
        }
        rec["fmu_freewheel_steps"] = a["freewheel_steps"]
    if not switching:
        if not errs:
            m = {s: nrms(a[s], native[s]) for s in sig}
            rec["criteria"]["4"] = {"pass": bool(all(v <= 0.01 for v in m.values())), "nrms_vs_native": m, "target": 0.01}
    else:
        f = episode(name, d1, tau, substeps=SUBSTEPS)
        if f["error"]:
            rec["criteria"]["5"] = {"pass": False, "note": "the FMU did not simulate at Ts/4: " + f["error"]}
        else:
            m = {s: nrms(f["avg"][s], native[s]) for s in sig}
            rec["criteria"]["5"] = {"pass": bool(all(v <= 0.03 for v in m.values())), "nrms_cycle_avg_vs_native": m, "target": 0.03,
                                    "nrms_boundary_sample_vs_native": {s: nrms(f[s], native[s]) for s in sig},
                                    "ripple_p50": {s: float(np.median(f["ripple"][s])) for s in sig},
                                    "ripple_max": {s: float(np.max(f["ripple"][s])) for s in sig}, "substeps": SUBSTEPS}
            rec["_cycle_avg"] = f["avg"]
    st = a["stats"]
    if switching:
        rec["criteria"]["6"] = {"pass": bool(st["p50_ms"] is not None), "p50_ms": st["p50_ms"], "p99_ms": st["p99_ms"],
                                "evaluation_only_if_p50_le_ms": 20.0, "evaluation_only": bool(st["p50_ms"] is not None and st["p50_ms"] <= 20.0)}
    else:
        rec["criteria"]["6"] = {"pass": bool(st["p50_ms"] is not None and st["p50_ms"] <= 0.5), "p50_ms": st["p50_ms"], "p99_ms": st["p99_ms"], "target_ms": 0.5}
    rec["wall_s_per_episode"] = a.get("wall_s")
    # criterion 7: 10 episodes of 1 s simulated
    n1 = int(round(1.0 / tau))
    long = []
    for seed in range(11, 21):
        dl = sequence("prbs", dict(plant["prbs"], prbs_seed=seed), np.arange(n1) * tau)
        e = episode(name, dl, tau)
        finite = e["error"] is None and all(np.all(np.isfinite(e[s])) for s in sig)
        long.append({"prbs_seed": seed, "ok": bool(finite), "error": e["error"], "wall_s": e["wall_s"]})
    rec["criteria"]["7"] = {"pass": all(x["ok"] for x in long), "episodes": long}
    rec["_native"], rec["_fmu"] = native, (a if not errs else None)
    return rec


def verdict(name, rec):
    c = rec["criteria"]
    hard = [k for k in ("1", "2", "3", "4", "7") if k in c and not c[k]["pass"]]
    if name.endswith("Switching"):
        if "5" not in c or not c["5"]["pass"]:
            hard.append("5")
        if hard:
            return "NO-GO", hard
        return ("GO" if c["6"].get("p50_ms", 1e9) <= 0.5 else "GO for evaluation only" if c["6"]["evaluation_only"] else "NO-GO"), []
    if not c["6"]["pass"]:
        hard.append("6")
    return ("NO-GO" if hard else "GO"), hard


def v4_comparison():
    ref = np.load(REPO / "tests" / "data" / "ref_data_v4.npz")
    names = list(ref["dc_solve_ivp_state_names"])
    st = ref["dc_solve_ivp_states"] * ref["dc_solve_ivp_limits"]
    acts = ref["dc_solve_ivp_actions"][:, 0]
    e = episode("DcMotorAveraged", acts, float(ref["tau"]), params={"loadA": 0.05, "loadB": 0.02, "loadC": 0.001, "Jload": 0.01},
                supply=420.0, two_quadrant=True)
    if e["error"]:
        return {"error": e["error"]}
    omega, i = st[1:, names.index("omega")], st[1:, names.index("i")]
    out = {"nrms": {"omega": nrms(e["omega"], omega), "i_a": nrms(e["i_a"], i)}, "n_steps": int(len(acts)), "input": "v_a = 420 * action (as pre-registered)",
           "_series": {"omega": (omega, e["omega"]), "i_a": (i, e["i_a"])}, "tau_s": float(ref["tau"])}
    # Post-hoc (deviation, declared): GEM's ContTwoQuadrantConverter applies u = 0, not 420*action, for
    # action < 0 (the reference's own u state). Re-run with the voltage the reference applied.
    applied = st[1:, names.index("u")]
    e2 = Run("DcMotorAveraged", params={"loadA": 0.05, "loadB": 0.02, "loadC": 0.001, "Jload": 0.01}, supply=1.0, two_quadrant=True)
    try:
        r2 = e2.run(applied, float(ref["tau"]))
    finally:
        e2.close()
    out["post_hoc_applied_voltage"] = {"input": "v_a = the reference's applied u (state 'u')", "nrms": {"omega": nrms(r2["omega"], omega), "i_a": nrms(r2["i_a"], i)},
                                       "steps_with_action_lt_0": int(np.sum(acts < 0)), "u_applied_when_action_lt_0": sorted({float(x) for x in applied[acts < 0]})}
    out["_series_post_hoc"] = {"omega": (omega, r2["omega"]), "i_a": (i, r2["i_a"])}
    return out


def dc_euler_diagnostic(fmu_series):
    """Post-hoc, declared: is the criterion-4 difference of DcMotorAveraged the native's single Euler
    step per 100 us? Same plant and sequence, Euler with 100 substeps."""
    d = duties(DC)
    fine = native_dc(d, euler_substeps=100)
    coarse = native_dc(d)
    return {"nrms_fmu_vs_native_euler100": {s: nrms(fmu_series[s], fine[s]) for s in DC["signals"]},
            "nrms_native_euler1_vs_euler100": {s: nrms(coarse[s], fine[s]) for s in DC["signals"]},
            "freewheel_steps_euler100": fine["freewheel_steps"]}


def figure(path, title, tau, pairs, ref_label="native"):
    """A plain SVG (no plotting dependency): one panel per signal, reference solid, FMU dashed."""
    w, ph, ml, mr, mt, gap = 900, 170, 70, 20, 34, 34
    h = mt + len(pairs) * (ph + gap)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="{ml}" y="18" font-size="13" fill="#222">{title}</text>',
           f'<line x1="{w - 330}" y1="14" x2="{w - 300}" y2="14" stroke="#1f5fa8" stroke-width="2"/><text x="{w - 294}" y="18" fill="#222">{ref_label}</text>',
           f'<line x1="{w - 200}" y1="14" x2="{w - 170}" y2="14" stroke="#d9711c" stroke-width="2" stroke-dasharray="5,3"/><text x="{w - 164}" y="18" fill="#222">OpenModelica FMU</text>']
    for n, (label, ref, fmu) in enumerate(pairs):
        ref, fmu = np.asarray(ref, float), np.asarray(fmu, float)
        y0 = mt + n * (ph + gap)
        lo, hi = float(min(ref.min(), fmu.min())), float(max(ref.max(), fmu.max()))
        hi = hi if hi > lo else lo + 1.0
        t_end = len(ref) * tau
        step = max(1, len(ref) // 1500)

        def pts(y):
            k = np.arange(0, len(y), step)
            xs = ml + (k + 1) * tau / t_end * (w - ml - mr)
            ys = y0 + ph - (y[k] - lo) / (hi - lo) * ph
            return " ".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, ys))

        out += [f'<rect x="{ml}" y="{y0}" width="{w - ml - mr}" height="{ph}" fill="none" stroke="#ccc"/>',
                f'<text x="8" y="{y0 + ph / 2}" fill="#222">{label}</text>',
                f'<text x="{ml - 4}" y="{y0 + 10}" text-anchor="end" fill="#555">{hi:.3g}</text>',
                f'<text x="{ml - 4}" y="{y0 + ph}" text-anchor="end" fill="#555">{lo:.3g}</text>',
                f'<polyline points="{pts(ref)}" fill="none" stroke="#1f5fa8" stroke-width="2"/>',
                f'<polyline points="{pts(fmu)}" fill="none" stroke="#d9711c" stroke-width="1.5" stroke-dasharray="5,3"/>',
                f'<text x="{w - mr}" y="{y0 + ph + 14}" text-anchor="end" fill="#555">{t_end:.3g} s</text>']
    out.append("</svg>")
    Path(path).write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    figs = args.out / "b3_figures"
    figs.mkdir(parents=True, exist_ok=True)
    man = json.loads((FMU_DIR / "manifest.json").read_text())
    doc = {"preregistration": "docs/audit/plecs_om/b3_preregistration.md", "fmu_manifest": man, "host": {"platform": platform.platform(), "python": platform.python_version()},
           "models": {}}
    for name in PLANT_OF:
        print("evaluating", name, flush=True)
        rec = evaluate(name)
        rec["verdict"], rec["failed_criteria"] = verdict(name, rec)
        native, fmu, avg = rec.pop("_native"), rec.pop("_fmu"), rec.pop("_cycle_avg", None)
        if name == "DcMotorAveraged" and fmu is not None:
            rec["post_hoc_euler_diagnostic"] = dc_euler_diagnostic(fmu)
        plant = PLANT_OF[name]
        series = avg if avg is not None else (fmu if fmu is not None else None)
        if series is not None:
            figure(figs / f"{name}.svg", f"{name}: native vs FMU{' (cycle average)' if avg is not None else ''}", plant["tau"],
                   [(s, native[s], series[s]) for s in plant["signals"]])
        doc["models"][name] = rec
        print(name, rec["verdict"], rec["failed_criteria"], flush=True)
    v4 = v4_comparison()
    if "_series" in v4:
        s = v4.pop("_series")
        figure(figs / "DcMotorAveraged_vs_gem_v4.svg", "DcMotorAveraged vs gym-electric-motor 3.0.3 (V4, RK45)", v4["tau_s"],
               [(k, ref, fmu) for k, (ref, fmu) in s.items()], ref_label="GEM 3.0.3 RK45")
        s = v4.pop("_series_post_hoc")
        figure(figs / "DcMotorAveraged_vs_gem_v4_applied_voltage.svg", "DcMotorAveraged vs GEM V4, post hoc: FMU fed the voltage GEM applied", v4["tau_s"],
               [(k, ref, fmu) for k, (ref, fmu) in s.items()], ref_label="GEM 3.0.3 RK45")
    doc["gem_v4_reference"] = v4
    (args.out / "b3_om_fmu_spike.json").write_text(json.dumps(doc, indent=2, default=float) + "\n")
    print(json.dumps({m: (r["verdict"], r["failed_criteria"]) for m, r in doc["models"].items()}, indent=1))
    print("V4:", v4.get("nrms", v4))


if __name__ == "__main__":
    main()
