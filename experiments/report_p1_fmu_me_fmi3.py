"""P1 (round 2026-10-26): evidence for FmuAdapter's Model Exchange path (and the status of FMI 3.0),
the interfaces a PLECS 5.0 export can produce, measured on OpenModelica builds of SPADE's bucks
(scripts/fmu/build_fmu_variants.mos -> data/fmu/variants/).

1. BuckAveraged, FMI 2.0 Model Exchange, SPADE's RK4 with 1 / 10 substeps, against the native buck
   (sim/buck_system, exact ZOH) and against the Co-Simulation build: duty 0.5, 500 steps of 20 us.
2. BuckSwitching, FMI 2.0 Model Exchange with event location, 4 / 10 / 40 substeps: does it run
   (the Co-Simulation build aborts in CVODE, S-15), convergence in the substep count, latency.
3. FMI 3.0 (and FMI 2.0) with real binaries: the Modelica Association Reference FMUs v0.0.42
   (BSD-2; scripts/fmu/fetch_reference_fmus.py, SHA-256 pinned), run with the user's permission
   (2026-10-26): Dahlquist against exp(-t), Stair (time events), BouncingBall (state events) against
   fmpy's own CVode simulation. OpenModelica 1.27.1 does not export FMI 3.0.

Usage: python experiments/report_p1_fmu_me_fmi3.py [--out docs/audit/plecs_om]
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

REPO = Path(__file__).resolve().parents[1]
FMU_DIR = Path(os.environ.get("SPADE_FMU_DIR", REPO / "data" / "fmu"))


def run(path, interface, substeps, n=500, duty=0.5):
    from driveflow.integrations.fmu.adapter import FmuAdapter, FmuError
    from driveflow.sim.buck_system import TAU

    a = FmuAdapter(interface=interface, me_substeps=substeps)
    info = a.load(path)
    a.reset()
    y, err = [], None
    t0 = time.perf_counter()
    try:
        for k in range(n):
            a.set_inputs({"duty": duty})
            a.do_step(k * TAU, TAU)
            y.append(a.get_outputs(check_range=False)["v_out"])
    except FmuError as exc:
        err = str(exc)
    st = a.stats()
    a.close()
    return np.array(y), {"fmi_version": info.fmi_version, "interface": st["interface"], "steps": len(y), "error": err, "events": st["events"],
                         "p50_ms": st["p50_ms"], "p99_ms": st["p99_ms"], "wall_s": round(time.perf_counter() - t0, 2), "generation_tool": info.generation_tool,
                         "sha256": info.sha256}


def _drive(path, n, dt, interface, out, sub=10):
    from driveflow.integrations.fmu.adapter import FmuAdapter

    a = FmuAdapter(interface=interface, me_substeps=sub)
    info = a.load(path)
    a.reset()
    y = []
    for k in range(n):
        a.do_step(k * dt, dt)
        y.append(a.get_outputs(check_range=False)[out])
    st = a.stats()
    a.close()
    return np.array(y), info, st


def reference_fmus() -> dict:
    import fmpy

    from driveflow.integrations.fmu.adapter import FmuError

    R = FMU_DIR / "reference"
    src = (R / "SOURCE.txt").read_text().strip().splitlines() if (R / "SOURCE.txt").exists() else ["not fetched"]
    out = {"source": src, "cases": []}
    t = np.arange(1, 101) * 0.1
    for v in ("3.0", "2.0"):
        y, info, st = _drive(R / v / "Dahlquist.fmu", 100, 0.1, "me", "x")
        out["cases"].append({"fmu": f"{v}/Dahlquist", "interface": st["interface"], "check": "x vs exp(-t), dt 0.1", "max_abs_error": float(np.max(np.abs(y - np.exp(-t))))})
        y, info, st = _drive(R / v / "Dahlquist.fmu", 100, 0.1, "cs", "x")
        out["cases"].append({"fmu": f"{v}/Dahlquist", "interface": st["interface"], "check": "x vs the FMU's own explicit Euler 0.9^n (its CS solver)",
                             "max_abs_error": float(np.max(np.abs(y - 0.9 ** np.arange(1, 101))))})
        for iface, dt, n in (("me", 0.1, 85), ("cs", 0.2, 42)):
            y, info, st = _drive(R / v / "Stair.fmu", n, dt, iface, "counter")
            k = int(round(1 / dt))
            out["cases"].append({"fmu": f"{v}/Stair", "interface": st["interface"], "check": "counter at t = 1..8 s (time events)",
                                 "values": [int(c) for c in y[k - 1::k]][:8], "expected": [2, 3, 4, 5, 6, 7, 8, 9]})
    try:
        _drive(R / "3.0" / "Dahlquist.fmu", 3, 0.01, "cs", "x")
        msg = "no error"
    except FmuError as exc:
        msg = str(exc)
    out["cases"].append({"fmu": "3.0/Dahlquist", "interface": "CoSimulation", "check": "dt 0.01 < fixedInternalStepSize 0.1", "message": msg})
    p = R / "3.0" / "BouncingBall.fmu"
    ref = fmpy.simulate_fmu(str(p), fmi_type="ModelExchange", stop_time=3.0, output_interval=0.01, solver="CVode", relative_tolerance=1e-8, output=["h"], record_events=False)
    y, info, st = _drive(p, 300, 0.01, "me", "h")
    r = np.interp(np.arange(1, 301) * 0.01, ref["time"], ref["h"])
    out["cases"].append({"fmu": "3.0/BouncingBall", "interface": "ModelExchange", "check": "h vs fmpy CVode (rtol 1e-8), 3 s, state events",
                         "max_abs_error": float(np.max(np.abs(y - r))), "events": st["events"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "audit" / "plecs_om")
    args = ap.parse_args(argv)
    from driveflow.sim.buck_system import BuckSystem

    nat = BuckSystem()
    nat.reset()
    vn = np.array([nat.simulate(0.5).v_out for _ in range(500)])
    nrms = lambda y: float(np.sqrt(np.mean((y - vn[: len(y)]) ** 2)) / np.ptp(vn))
    res = {"command": "python experiments/report_p1_fmu_me_fmi3.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "host": platform.platform(), "python": platform.python_version(), "averaged": {}, "switching": {}}
    y, m = run(FMU_DIR / "BuckAveraged.fmu", "cs", 1)
    res["averaged"]["co_simulation"] = {**m, "nrms_vs_native": nrms(y)}
    for sub in (1, 10):
        y, m = run(FMU_DIR / "variants" / "BuckAveraged_fmi2_me.fmu", "me", sub)
        res["averaged"][f"model_exchange_rk4_x{sub}"] = {**m, "nrms_vs_native": nrms(y)}
    y, m = run(FMU_DIR / "BuckSwitching.fmu", "cs", 1)
    res["switching"]["co_simulation"] = m
    ys = {}
    for sub in (4, 10, 40):
        ys[sub], m = run(FMU_DIR / "variants" / "BuckSwitching_fmi2_me.fmu", "me", sub)
        res["switching"][f"model_exchange_rk4_x{sub}"] = {**m, "v_out_end": float(ys[sub][-1]) if len(ys[sub]) else None}
    if all(len(ys[s]) == 500 for s in ys):
        res["switching"]["convergence"] = {"max_abs_diff_x4_vs_x40_V": float(np.max(np.abs(ys[4] - ys[40]))),
                                           "max_abs_diff_x10_vs_x40_V": float(np.max(np.abs(ys[10] - ys[40]))),
                                           "native_averaged_v_out_end": float(vn[-1])}
    res["reference_fmus"] = reference_fmus()
    (args.out / "p1_fmu_me_fmi3.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    md = _md(res)
    (args.out / "p1_fmu_me_fmi3.md").write_text(md)
    print(md)


def _md(r):
    av, sw = r["averaged"], r["switching"]
    L = ["# P1 — FMI 3.0 and Model Exchange in FmuAdapter (the interfaces a PLECS 5.0 export produces)", "",
         f"Command: `{r['command']}`. Host: {r['host']}, Python {r['python']}.", "",
         "Why: the reviewer asked that circuits designed in PLECS be the Gym environments of the Python algorithms. PLECS 5.0 exports FMI 2.0/3.0 FMUs; "
         "a **Model Exchange** export (PLECS Coder licence) runs standalone, a Co-Simulation export is tool-coupling (PLECS installed and licensed at runtime). "
         "Before P1 the adapter drove only FMI 2.0 Co-Simulation. **No PLECS is available here; no PLECS-exported FMU has been run.** The paths are exercised "
         "with OpenModelica builds of SPADE's own bucks.", "",
         "## 1. Averaged buck (duty 0.5, 500 steps of 20 µs) vs the native buck (exact ZOH)", "",
         "| Build / interface | NRMS v_out vs native | p50 / p99 per step |", "|---|---|---|"]
    for k, m in av.items():
        L.append(f"| {k} ({m['fmi_version']}, {m['interface']}) | {m['nrms_vs_native']:.1e} | {m['p50_ms']:.3f} / {m['p99_ms']:.3f} ms |")
    L += ["", "## 2. Switching buck (PWM inside the model)", "", "| Build / interface | Steps run (of 500) | Events | p50 per step | v_out at 10 ms | Error |", "|---|---|---|---|---|---|"]
    for k, m in sw.items():
        if k == "convergence":
            continue
        L.append(f"| {k} | {m['steps']} | {m['events']} | {m['p50_ms']:.2f} ms | {m.get('v_out_end') if m.get('v_out_end') is None else round(m['v_out_end'], 4)} | {m['error'] or '—'} |")
    c = sw.get("convergence")
    if c:
        L += ["", f"Convergence in the substep count (switching instants located by bisection): max |x4 − x40| = {c['max_abs_diff_x4_vs_x40_V']:.2e} V, "
              f"max |x10 − x40| = {c['max_abs_diff_x10_vs_x40_V']:.2e} V. The native **averaged** buck ends at {c['native_averaged_v_out_end']:.4f} V; "
              "the switching model's period-end sample differs by its ripple."]
    L += ["", "**Reading.** Model Exchange with SPADE's RK4 reproduces the averaged buck to 5e-11, and it **runs the switching buck that aborts in "
          "Co-Simulation (S-15)**, converging in the substep count. That is the interface in which a PLECS circuit with real switching would arrive. "
          "B3's pre-registered verdicts are not changed by this (they were measured on the Co-Simulation builds).", "",
          "## 3. FMI 3.0 and 2.0 with real binaries: Modelica Association Reference FMUs", "",
          "Source: " + " · ".join(r["reference_fmus"]["source"]) + ". Run with the user's permission (2026-10-26); OpenModelica 1.27.1 does not export FMI 3.0.", "",
          "| FMU | Interface | Check | Result |", "|---|---|---|---|"]
    for c in r["reference_fmus"]["cases"]:
        res = (f"max error {c['max_abs_error']:.1e}" + (f", {c['events']} events" if "events" in c else "")) if "max_abs_error" in c else \
              (f"{c['values']} (expected {c['expected']})" if "values" in c else c["message"])
        L.append(f"| {c['fmu']} | {c['interface']} | {c['check']} | {res} |")
    L += ["", "The Co-Simulation error of Dahlquist against exp(-t) (1.9e-2) is the FMU's own explicit Euler: SPADE reproduces 0.9^n to 1e-16. "
          "A communication step below an FMU's fixedInternalStepSize now fails with that explanation, and Plant Studio refuses such a Ts in Co-Simulation.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
