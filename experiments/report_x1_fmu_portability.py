"""X1 (2026-10-26): do SPADE's OpenModelica FMUs work on Windows and macOS? Run on each OS by
.github/workflows/fmu-portability.yml after compiling the Model Exchange FMUs from their C sources on
that OS (scripts/fmu/compile_fmu_from_source.py).

1. Adapter on this OS with real third-party binaries: Modelica Association Reference FMUs (Dahlquist
   vs exp(-t), Stair time events, BouncingBall state events vs fmpy CVode), FMI 3.0 and 2.0 where the
   archive has a binary for this OS.
2. B3 GO/NO-GO of BuckAveraged (Model Exchange, compiled here) with B3's pre-registered criteria,
   unchanged (docs/audit/plecs_om/b3_preregistration.md): 1 action affects output, 2 t = n·Ts,
   3 reproducible reset, 4 NRMS <= 1 % of the native range on v_out and i_L against the native buck
   (sim/buck_system: the exact zero-order-hold solution of the same averaged model, so a converged
   reference as G7 requires), 6 p50 <= 0.5 ms per step, 7 ten 1 s episodes without failure.
   Criterion 5 applies to switching models only.
3. BuckSwitching (Model Exchange, compiled here): runs 10 ms at Ts? converges in the substep count?
   Reported, no verdict (no switching model is distributed).

Usage: python experiments/report_x1_fmu_portability.py --buck-me <fmu> [--switching-me <fmu>] [--reference-dir data/fmu/reference] --out <json>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

TAU = 2e-5
LEVELS, N_LEVEL = [0.0, 0.3, 0.6, 0.45], 250


def _adapter(path, interface="me", sub=10):
    from driveflow.integrations.fmu.adapter import FmuAdapter

    a = FmuAdapter(interface=interface, me_substeps=sub)
    a.load(path)
    return a


def _run_duties(a, duties, check=False):
    a.reset(seed=0)
    out = []
    for k, d in enumerate(duties):
        a.set_inputs({"duty": float(d)})
        a.do_step(k * TAU, TAU)
        o = a.get_outputs(check_range=check)
        out.append((o["v_out"], o["i_L"]))
    return np.array(out)


def reference_checks(ref_dir: Path) -> list:
    from driveflow.integrations.fmu.adapter import FmuError, inspect_fmu, supported

    rows = []

    def drive(p, n, dt, iface, out):
        a = _adapter(p, iface)
        a.reset()
        y = []
        for k in range(n):
            a.do_step(k * dt, dt)
            y.append(a.get_outputs(check_range=False)[out])
        st = a.stats()
        a.close()
        return np.array(y), st

    for v in ("3.0", "2.0"):
        for name in ("Dahlquist", "Stair", "BouncingBall"):
            p = ref_dir / v / f"{name}.fmu"
            if not p.exists():
                continue
            ok, why = supported(inspect_fmu(p))
            if not ok:
                rows.append({"fmu": f"{v}/{name}", "status": "skipped", "reason": why})
                continue
            try:
                if name == "Dahlquist":
                    y, _ = drive(p, 100, 0.1, "me", "x")
                    err = float(np.max(np.abs(y - np.exp(-np.arange(1, 101) * 0.1))))
                    rows.append({"fmu": f"{v}/{name}", "interface": "ME", "check": "vs exp(-t)", "value": err, "status": "pass" if err < 1e-9 else "fail"})
                    y, _ = drive(p, 100, 0.1, "cs", "x")
                    err = float(np.max(np.abs(y - 0.9 ** np.arange(1, 101))))
                    rows.append({"fmu": f"{v}/{name}", "interface": "CS", "check": "vs its own explicit Euler", "value": err, "status": "pass" if err < 1e-12 else "fail"})
                elif name == "Stair":
                    y, _ = drive(p, 85, 0.1, "me", "counter")
                    vals = [int(c) for c in y[9::10]][:8]
                    rows.append({"fmu": f"{v}/{name}", "interface": "ME", "check": "time events", "value": vals, "status": "pass" if vals == [2, 3, 4, 5, 6, 7, 8, 9] else "fail"})
                else:
                    import fmpy

                    ref = fmpy.simulate_fmu(str(p), fmi_type="ModelExchange", stop_time=3.0, output_interval=0.01, solver="CVode", relative_tolerance=1e-8,
                                            output=["h"], record_events=False)
                    y, st = drive(p, 300, 0.01, "me", "h")
                    err = float(np.max(np.abs(y - np.interp(np.arange(1, 301) * 0.01, ref["time"], ref["h"]))))
                    rows.append({"fmu": f"{v}/{name}", "interface": "ME", "check": "state events vs fmpy CVode", "value": err, "status": "pass" if err < 1e-6 else "fail"})
            except (FmuError, Exception) as exc:
                rows.append({"fmu": f"{v}/{name}", "status": "fail", "reason": f"{type(exc).__name__}: {exc}"})
    return rows


def b3_buck_me(path: Path) -> dict:
    from driveflow.sim.buck_system import BuckSystem

    duties = np.repeat(LEVELS, N_LEVEL)
    nat = BuckSystem()
    nat.reset()
    native = np.array([(s.v_out, s.i_l) for s in (nat.simulate(float(d)) for d in duties)])
    a = _adapter(path)
    crit = {}
    r1 = _run_duties(a, duties)
    r1b = _run_duties(a, duties)
    alt = _run_duties(a, np.repeat([0.2, 0.5, 0.7, 0.35], N_LEVEL))
    crit["1"] = bool(np.array_equal(r1, r1b) and not np.allclose(r1, alt))
    a.reset(seed=0)
    terr = 0.0
    for k in range(200):
        a.set_inputs({"duty": 0.5})
        a.do_step(k * TAU, TAU)
        terr = max(terr, abs(a.time - (k + 1) * TAU) / ((k + 1) * TAU))
    crit["2"] = terr <= 1e-12
    crit["3"] = bool(np.array_equal(r1, r1b))
    nrms = {s: float(np.sqrt(np.mean((r1[:, j] - native[:, j]) ** 2)) / np.ptp(native[:, j])) for j, s in enumerate(("v_out", "i_L"))}
    crit["4"] = max(nrms.values()) <= 0.01
    a.reset(seed=0)
    w = []
    for k in range(1000):
        a.set_inputs({"duty": 0.5})
        t0 = time.perf_counter()
        a.do_step(k * TAU, TAU)
        w.append(time.perf_counter() - t0)
    p50 = float(np.percentile(w, 50) * 1e3)
    crit["6"] = p50 <= 0.5
    fails = 0
    for ep in range(10):
        try:
            rng = np.random.default_rng(11 + ep)
            d = np.repeat(rng.uniform(0.1, 0.8, 50), 1000)  # 50 levels x 20 ms = 1 s
            r = _run_duties(a, d)
            fails += int(not np.all(np.isfinite(r)))
        except Exception:
            fails += 1
    crit["7"] = fails == 0
    a.close()
    failed = [k for k, v in crit.items() if not v]
    return {"criteria": crit, "nrms_vs_native": nrms, "time_rel_err": terr, "p50_ms": p50, "episode_failures": fails,
            "verdict": "GO" if not failed else "NO-GO", "failed_criteria": failed}


def switching(path: Path) -> dict:
    out = {}
    ys = {}
    for sub in (4, 40):
        a = _adapter(path, sub=sub)
        a.reset()
        y, err = [], None
        t0 = time.perf_counter()
        try:
            for k in range(500):
                a.set_inputs({"duty": 0.5})
                a.do_step(k * TAU, TAU)
                y.append(a.get_outputs(check_range=False)["v_out"])
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        ys[sub] = np.array(y)
        out[f"x{sub}"] = {"steps": len(y), "error": err, "p50_ms": a.stats()["p50_ms"], "v_out_end": float(y[-1]) if y else None, "wall_s": round(time.perf_counter() - t0, 2)}
        a.close()
    if all(len(v) == 500 for v in ys.values()):
        out["max_abs_diff_x4_vs_x40_V"] = float(np.max(np.abs(ys[4] - ys[40])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buck-me", type=Path, required=True)
    ap.add_argument("--switching-me", type=Path, default=None)
    ap.add_argument("--reference-dir", type=Path, default=Path("data/fmu/reference"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import fmpy

    from driveflow.integrations.fmu.adapter import current_platform, current_platform_fmi3, inspect_fmu

    info = inspect_fmu(args.buck_me)
    res = {"command": "python experiments/report_x1_fmu_portability.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "os": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "fmpy": fmpy.__version__,
           "fmi2_platform": current_platform(), "fmi3_platform": current_platform_fmi3(),
           "buck_me": {"sha256": info.sha256, "platforms": list(info.platforms)},
           "reference_fmus": reference_checks(args.reference_dir), "b3_buck_averaged_me": b3_buck_me(args.buck_me)}
    if args.switching_me:
        res["buck_switching_me"] = switching(args.switching_me)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("os", "fmi2_platform", "fmi3_platform")}, indent=1))
    print("reference:", [(r["fmu"], r.get("interface"), r["status"]) for r in res["reference_fmus"]])
    b = res["b3_buck_averaged_me"]
    print("B3 BuckAveraged ME:", b["verdict"], b["failed_criteria"], {k: f"{v:.1e}" for k, v in b["nrms_vs_native"].items()}, f"p50 {b['p50_ms']:.3f} ms")
    if "buck_switching_me" in res:
        print("switching:", res["buck_switching_me"])
    return 0 if b["verdict"] == "GO" and all(r["status"] != "fail" for r in res["reference_fmus"]) else 1


if __name__ == "__main__":
    sys.exit(main())
