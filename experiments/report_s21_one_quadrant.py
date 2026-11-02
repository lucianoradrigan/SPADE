"""Finding S-21 (found in round Assistant, phase 1, task F1.6): non-physical speed rise of the DC drive
when the speed PI asks the ONE-QUADRANT converter to brake with little load (SIMULATED).

Observed: with the 0b-D1 friction (2 % of rated torque) the speed PI at 120 rad/s with no extra load
overshoots on start-up and the speed then RISES (to 179 rad/s on dc_pmg132_v1, 197 rad/s on the
stock motor with that friction) while the current swings negative every other period. A one-quadrant
chopper cannot brake, but it cannot accelerate the motor when the controller asks for zero voltage
either: the speed should only coast down.

Hypothesis (not proven here): ``ContOneQuadrantConverter._convert`` returns duty 1 (full supply,
the diode) whenever the current at the START of the control period is negative, and the converter
output is held for the whole period (0.1 ms). With L_a = 19 uH the current then swings by hundreds of
amperes per period and the held full voltage injects energy. Plant substeps and RK45 hold the
converter output too, so they give the same result: a finer plant integration is not a converged
reference for this case.

    python experiments/report_s21_one_quadrant.py
Writes docs/audit/assistant/s21_one_quadrant.{md,json}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import dc_default_load_parameter
from driveflow.validation.benches import run_dc_closed_loop

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "s21_one_quadrant"
STOCK_MP = {"r_a": 0.016, "l_a": 1.9e-05, "psi_e": 0.165, "j_rotor": 0.025}
STOCK_NOMINAL = {"omega": 300.0, "torque": 16.0, "i": 97.0, "u": 60.0}
RPM = 2 * math.pi / 60


def bench(load):
    r = run_dc_closed_loop("PI", 120.0, duration_s=0.7, motor_parameter=STOCK_MP, load_parameter=load)
    w, i, u, T = r["omega"], r["i"], r["u"], r["torque"]
    late = slice(int(0.2 / 1e-4), None)
    return {"load": load or "stock (a = 0.01, b = 0.05)", "peak_speed_rad_s": float(w.max()), "final_speed_rad_s": float(w[-1]),
            "min_current_a": float(i.min()), "mean_current_after_0_2s_a": float(i[late].mean()), "mean_torque_after_0_2s_nm": float(T[late].mean()),
            "mean_power_u_i_after_0_2s_w": float((u[late] * i[late]).mean()), "fraction_steps_full_voltage_after_0_2s": float((u[late] > 64.0).mean())}


def scenario(**kw):
    recs = run_scenario(Scenario("s21", plant_config_id="dc_pmg132_v1", omega_ref_rad_s=120.0, duration_s=0.7, **kw))
    w = np.array([r["rpm"] for r in recs]) * RPM
    i = np.array([r["current_r"] for r in recs])
    return {"integration": kw or "Euler, one step per period (default)", "peak_speed_rad_s": float(w.max()), "final_speed_rad_s": float(w[-1]), "min_current_a": float(i.min())}


def main() -> None:
    d1 = dc_default_load_parameter(STOCK_MP, STOCK_NOMINAL)
    benches = [bench(None), bench(d1), bench({"a": 1.0, "b": 0.01})]
    scen = [scenario(), scenario(dc_plant_substeps=10), scenario(dc_plant_substeps=100), scenario(dc_plant_solver="rk45")]
    loads = [{"load_torque_nm": lt, **scenario(load_torque_nm=lt)} for lt in (0.0, 1.0, 5.0)]
    result = {"command": "python experiments/report_s21_one_quadrant.py", "simulated": True, "bench_stock_motor_120rad_s": benches,
              "dc_pmg132_v1_120rad_s_integration": scen, "dc_pmg132_v1_120rad_s_load": loads}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    lines = [
        "# S-21 — non-physical speed rise when a one-quadrant DC drive is asked to brake",
        "",
        "**Simulated evidence. Found in round Assistant, phase 1 (F1.6). Not fixed: needs Luciano's decision.**",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`s21_one_quadrant.json`](s21_one_quadrant.json).",
        "- **Converter:** `sim/converters.ContOneQuadrantConverter._convert` returns duty 1 (full supply) when the current at the start of the period is negative; the converter output is held for the whole 0.1 ms period.",
        "",
        "## Stock motor, speed PI at 120 rad/s, no extra load, three frictions (tuning bench `run_dc_closed_loop`)",
        "",
        "| Friction | Peak speed | Final speed | Min current | Mean current after 0.2 s | Mean torque after 0.2 s | Mean u·i after 0.2 s | Periods at full voltage after 0.2 s |",
        "|---|---|---|---|---|---|---|---|",
    ] + [f"| {b['load']} | {b['peak_speed_rad_s']:.1f} rad/s | {b['final_speed_rad_s']:.1f} rad/s | {b['min_current_a']:.1f} A | {b['mean_current_after_0_2s_a']:.1f} A | "
         f"{b['mean_torque_after_0_2s_nm']:.2f} N·m | {b['mean_power_u_i_after_0_2s_w']:.0f} W | {b['fraction_steps_full_voltage_after_0_2s']:.0%} |" for b in benches] + [
        "",
        "With the stock friction (6 N·m at 120 rad/s) the start-up overshoot is braked by friction and the current never goes negative. With the 0b-D1 friction "
        f"(a = {d1['a']:.3f} N·m, b = {d1['b']:.6f} N·m·s/rad) the speed keeps rising after the overshoot: the current swings between about -160 A and positive values every period, "
        "and the periods held at full voltage (the converter's response to a negative current) leave a positive mean current, torque and electrical power while the controller asks for braking.",
        "",
        "## `dc_pmg132_v1`, PI at 120 rad/s, no extra load: plant integration",
        "",
        "| Integration | Peak speed | Final speed | Min current |",
        "|---|---|---|---|",
    ] + [f"| {s['integration']} | {s['peak_speed_rad_s']:.2f} rad/s | {s['final_speed_rad_s']:.2f} rad/s | {s['min_current_a']:.1f} A |" for s in scen] + [
        "",
        "The same result with 10 or 100 Euler substeps and with RK45: the plant integration converges, but the converter decision is still taken once per period, so this is **not** a converged reference for the switching behaviour (lesson G7 applies).",
        "",
        "## `dc_pmg132_v1` at 120 rad/s with a little load",
        "",
        "| Extra load | Peak speed | Final speed |",
        "|---|---|---|",
    ] + [f"| {x['load_torque_nm']:g} N·m | {x['peak_speed_rad_s']:.1f} rad/s | {x['final_speed_rad_s']:.1f} rad/s |" for x in loads] + [
        "",
        "## What it affects",
        "",
        "- Any DC run where the speed PI asks for braking (overshoot, a lower reference, a load removed) and friction plus load are small: `dc_pmg132_v1` and every changed motor with known ratings (0b-D1 friction) at light load.",
        "- Tuning on a changed motor (0b-D2): the evaluation numbers of such motors include this effect (F1.6, case B: the baseline \"overshoot\" of 2321 %).",
        "- **S-19 is probably the same mechanism**, recorded on 2026-10-25 as physics (\"the mean speed rises up to +72 %\"): a one-quadrant drive cannot raise the speed above the reference by braking. To be re-checked.",
        "- The stock motor with the stock friction in the shipped grids: not affected while the current stays positive (its 94 % friction brakes the overshoot).",
        "",
        "## Not executed",
        "",
        "- A reference with the converter decision re-evaluated inside the period (event-based diode conduction / discontinuous conduction): no ejecutado. Needed to prove the hypothesis and to size the error.",
        "- Re-measurement of S-19 and of the 0b-D1 / 0b-D3 evidence: no ejecutado (waits for the decision).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
