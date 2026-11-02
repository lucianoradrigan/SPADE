"""Round Assistant 0b, D1: default load scaled to the motor (SIMULATED data only).

User decision 2026-11-02: the stock DC motor keeps its load (a = 0.01 N*m, b = 0.05 N*m*s/rad);
any other DC motor whose rated torque and speed are known gets a friction of 2 % of its rated
torque at rated speed, half constant and half viscous. This report shows why (the stock load is
94 % of the stock rated torque at rated speed) and the 220 V motor of case B at rated load, before
(stock friction, from docs/audit/assistant/f0_2_dc_supply_limits.json) and after.

    python experiments/report_0b_1_load_default.py
Writes docs/audit/assistant/f0b_1_load_default.{md,json}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import SCALED_FRICTION_FRACTION, STOCK_DC_LOAD_PARAMETER, dc_default_load_parameter

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant"
OUT = AUDIT / "f0b_1_load_default"
OMEGA_B = 1500 * 2 * math.pi / 60
T_B = 1.146 * 25.0
CASE_B = dict(motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05), dc_supply_v=220.0,
              dc_limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
              dc_nominal_overrides=dict(u=220.0, i=25.0, omega=OMEGA_B, torque=T_B))


def run(controller: str) -> dict:
    recs = run_scenario(Scenario(f"b_{controller}", controller_type=controller, omega_ref_rad_s=OMEGA_B, load_torque_nm=T_B, duration_s=2.0, **CASE_B))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    tail = slice(-2000, None)
    return {"controller": controller, "final_speed_rad_s": float(omega[-1]), "steady_current_a": float(np.mean([r["current_r"] for r in recs[-2000:]])),
            "steady_voltage_v": float(np.mean([r["voltage_v"] for r in recs[-2000:]])), "speed_below_rated_pct": float((OMEGA_B - omega[tail].mean()) / OMEGA_B * 100)}


def main() -> None:
    stock = STOCK_DC_LOAD_PARAMETER
    stock_fraction = (stock["a"] + stock["b"] * 300.0) / 16.0
    scaled = dc_default_load_parameter(CASE_B["motor_parameter_overrides"], CASE_B["dc_nominal_overrides"])
    before = {r["controller"]: r for r in json.loads((AUDIT / "f0_2_dc_supply_limits.json").read_text())["runs"] if r["load_torque_nm"] > 0}
    after = [run(c) for c in ("PI", "MPC")]
    result = {"command": "python experiments/report_0b_1_load_default.py", "simulated": True, "rule_fraction": SCALED_FRICTION_FRACTION,
              "stock_load": stock, "stock_friction_fraction_of_rated_torque": stock_fraction, "case_b_scaled_load": scaled,
              "case_b_rated_load_before": before, "case_b_rated_load_after": after}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    fric_b_before = stock["a"] + stock["b"] * OMEGA_B
    fric_b_after = scaled["a"] + scaled["b"] * OMEGA_B
    lines = [
        "# 0b D1 — default load scaled to the motor",
        "",
        "**Simulated evidence only.** User decision of 2026-11-02 (rule chosen after the proportional option was shown to keep the 94 % friction below).",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0b_1_load_default.json`](f0b_1_load_default.json).",
        "- **Rule** (`datagen/runner.dc_default_load_parameter`, also used by the tuning through `BaseMotor.bench_kwargs`):",
        "  - stock motor: the stock load, a = 0.01 N·m, b = 0.05 N·m·s/rad, unchanged bit for bit;",
        f"  - another motor (`motor_parameter_overrides` set) whose rated torque and speed are known (`dc_nominal_overrides` 'torque' and 'omega'): a + b·ω_rated = {SCALED_FRICTION_FRACTION:.0%} of the rated torque, half constant (a) and half viscous (b·ω); j_load stays 0.0025 kg·m²;",
        "  - another motor without those ratings keeps the stock load (as before; no rating is invented).",
        f"- **Why not proportional:** the stock friction at the stock rated speed is {stock['a'] + stock['b'] * 300:.2f} N·m, **{stock_fraction:.0%} of the 16 N·m rated torque**. Scaling it would give every other motor a friction close to its rated torque.",
        "",
        "## Case B (220 V motor) at rated load torque, reference = rated speed",
        "",
        f"Friction at 157 rad/s: {fric_b_before:.2f} N·m with the stock load, {fric_b_after:.3f} N·m with the rule (a = {scaled['a']:.4f} N·m, b = {scaled['b']:.6f} N·m·s/rad).",
        "",
        "| Controller | Final speed, stock friction (rad/s) | Final speed, scaled friction (rad/s) | Below rated | Steady current (A) | Steady voltage (V) |",
        "|---|---|---|---|---|---|",
    ] + [f"| {r['controller']} | {before[r['controller']]['final_speed_rad_s']:.2f} | {r['final_speed_rad_s']:.2f} | {r['speed_below_rated_pct']:.2f} % | {r['steady_current_a']:.1f} | {r['steady_voltage_v']:.1f} |" for r in after] + [
        "",
        f"The motor now carries rated load within 0.5 % of rated speed. It does not reach it exactly: ψ_e·ω_rated + R_a·I = 180.0 + {1.6 * (fric_b_after + T_B) / 1.146:.1f} V exceeds the 220 V supply by a little, "
        "because ψ_e = 1.146 was derived from P/(ω·I) for case B rather than from the back-EMF. That is the motor data, not the controller.",
        "",
        "Effect on earlier evidence: `f0_2_dc_supply_limits.md` shows case B with the stock friction (its rated-load rows); those numbers describe the behaviour before this decision.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
