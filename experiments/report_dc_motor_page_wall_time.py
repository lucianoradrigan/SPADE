"""Decision E3 (Luciano, 2026-11-08): wall time of a typical run of the DC Motor page with the v1 plant
(GEM's one-quadrant converter, frozen) and the v2 plant (converter in discontinuous conduction, the GUI's
default since E3).

The call measured is the one the page's Run button makes, ``viz.adapters.simulation_adapter.run_dc``, with
the page's default controls (PI, speed mode, 300 rad/s, 1.0 s, seed 0, no noise, the stock motor's
parameters from the Parameters tab) and, as a second case, the same with an outer_race fault at the page's
default severities. One warm-up call, then REPEATS timed calls per plant and case, alternating v1 / v2;
time.perf_counter around the call, one process, machine otherwise idle.

    python experiments/report_dc_motor_page_wall_time.py
Writes docs/audit/assistant/dc_motor_page_wall_time.{md,json} (simulated).
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "audit" / "assistant" / "dc_motor_page_wall_time"
REPEATS = 5
PLANTS = ("dc_perm_ex_v1", "dc_perm_ex_v2")


def _params(fault: str) -> dict:
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
    from driveflow.viz.adapters import simulation_adapter as sim

    motor = sim.dc_defaults()["params"]
    return {"controller": "PI", "control_mode": "speed", "setpoint": 300.0, "fault_label": fault, "duration_s": 1.0, "seed": 0,
            "electrical_severity": 8.0, "mechanical_severity": None if fault == "healthy" else float(CALIBRATED_MECHANICAL_SEVERITY[fault]),
            "motor": {k: motor[k] for k in ("r_a", "l_a", "psi_e", "j_rotor") if k in motor}, "electrical_noise_pct": 0.0, "mechanical_noise_pct": 0.0,
            "pi_setpoint_weight": 1.0}


def main() -> None:
    from driveflow.viz.adapters import simulation_adapter as sim

    cases = {"healthy_300rad_s_1s": "healthy", "outer_race_300rad_s_1s": "outer_race"}
    sim.run_dc(_params("healthy"), {})  # warm-up (imports, first build)
    res = {"command": "python experiments/report_dc_motor_page_wall_time.py", "simulated": True, "host": platform.platform(), "python": platform.python_version(),
           "repeats": REPEATS, "call": "viz.adapters.simulation_adapter.run_dc (the DC Motor page's Run button)", "cases": {}}
    orig = sim.DC_PLANT
    try:
        for name, fault in cases.items():
            times = {p: [] for p in PLANTS}
            finals = {}
            for _ in range(REPEATS):
                for plant in PLANTS:
                    sim.DC_PLANT = plant
                    t0 = time.perf_counter()
                    df = sim.run_dc(_params(fault), {})
                    times[plant].append(time.perf_counter() - t0)
                    finals[plant] = (float(df["rpm"].iloc[-1]) * 3.141592653589793 / 30.0, float(df["current_r"].min()))
            res["cases"][name] = {p: {"median_s": statistics.median(times[p]), "min_s": min(times[p]), "max_s": max(times[p]), "all_s": times[p],
                                      "final_speed_rad_s": finals[p][0], "min_current_a": finals[p][1]} for p in PLANTS}
            res["cases"][name]["ratio_v2_over_v1"] = res["cases"][name]["dc_perm_ex_v2"]["median_s"] / res["cases"][name]["dc_perm_ex_v1"]["median_s"]
    finally:
        sim.DC_PLANT = orig
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")
    L = ["# DC Motor page: wall time of a typical run, v1 vs v2 (decision E3)", "",
         f"**Simulated.** Command: `{res['command']}`. Raw: `dc_motor_page_wall_time.json`. Host: {res['host']}, Python {res['python']}.",
         f"Call: {res['call']}. Page defaults (PI, speed mode, 300 rad/s, 1.0 s, seed 0, no noise, stock motor). 1 warm-up, then {REPEATS} timed calls per plant, alternating.", "",
         "| Case | v1 median (min–max) | v2 median (min–max) | v2 / v1 | Final speed v1 / v2 (rad/s) | Min current v1 / v2 (A) |", "|---|---|---|---|---|---|"]
    for name, c in res["cases"].items():
        a, b = c["dc_perm_ex_v1"], c["dc_perm_ex_v2"]
        L.append(f"| {name} | {a['median_s']:.2f} s ({a['min_s']:.2f}–{a['max_s']:.2f}) | {b['median_s']:.2f} s ({b['min_s']:.2f}–{b['max_s']:.2f}) | "
                 f"{c['ratio_v2_over_v1']:.2f} | {a['final_speed_rad_s']:.2f} / {b['final_speed_rad_s']:.2f} | {a['min_current_a']:.1f} / {b['min_current_a']:.1f} |")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
