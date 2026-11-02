"""Round Assistant 0b, D3: plant dc_pmg132_v1 with the PMG 132 datasheet values (SIMULATED data only).

User decision 2026-11-02: a separate plant, so dc_perm_ex_v1 (GEM's port, the plant of every
trained model and dataset) stays unchanged; psi_e = 0.181 V*s/rad (P/(omega*I) at 60 V).
Reports the plant's values against the datasheet and GEM, the design check, and the speed PI at
the rated speed with 0-100 % of the rated torque (plus the MPC at rated load).

    python experiments/report_0b_3_pmg132_plant.py
Writes docs/audit/assistant/f0b_3_pmg132_plant.{md,json}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from driveflow.control.design_check import controller_design_check, mpc_design_check
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import build_plant

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f0b_3_pmg132_plant"
OMEGA_RATED = 2870 * 2 * math.pi / 60


def run(controller: str, load_nm: float) -> dict:
    recs = run_scenario(Scenario("p", plant_config_id="dc_pmg132_v1", controller_type=controller, omega_ref_rad_s=OMEGA_RATED, load_torque_nm=load_nm, duration_s=2.0))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    tail = recs[-2000:]
    return {"controller": controller, "load_torque_nm": load_nm, "final_speed_rad_s": float(omega[-1]), "steady_error_rad_s": float(OMEGA_RATED - omega[-2000:].mean()),
            "steady_current_a": float(np.mean([r["current_r"] for r in tail])), "steady_voltage_v": float(np.mean([r["voltage_v"] for r in tail]))}


def main() -> None:
    p = build_plant(Scenario("p", plant_config_id="dc_pmg132_v1"))
    m, load = p.electrical_motor, p.mechanical_load
    checks = {"PI": controller_design_check(m.motor_parameter, load.j_total, p.tau), "MPC": mpc_design_check(m.motor_parameter, load.j_total, p.tau)}
    runs = [run("PI", f * 20.0) for f in (0.0, 0.25, 0.5, 0.75, 1.0)] + [run("MPC", 20.0)]
    result = {"command": "python experiments/report_0b_3_pmg132_plant.py", "simulated": True, "motor_parameter": dict(m.motor_parameter), "nominal": dict(m.nominal_values),
              "limits": dict(m.limits), "load": dict(load.base_load.load_parameter), "supply_v": float(p.supply.u_nominal), "design_checks": checks, "runs": runs}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 0b D3 — plant `dc_pmg132_v1` (PMG 132 datasheet at 60 V)",
        "",
        "**Simulated evidence only.** User decision of 2026-11-02, applied as a **separate plant** so `dc_perm_ex_v1` (GEM's port, the plant of every trained model, dataset and regression reference) stays unchanged.",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0b_3_pmg132_plant.json`](f0b_3_pmg132_plant.json). Datasheet reading: [`f0_7_s20_pmg132.md`](f0_7_s20_pmg132.md).",
        "- **Same plant structure as dc_perm_ex_v1:** GEM `DcPermanentlyExcitedMotor`, one-quadrant converter, 65 V supply, PI or MPC, the same bearing-fault and vibration paths. Every DC-only `Scenario` field applies; overrides merge over the datasheet values.",
        "- **Out of scope:** the diagnosis datasets and classifiers, the Gym environments and the GUI do not use it (decision A7: no GUI changes).",
        "",
        "| Quantity | dc_pmg132_v1 | Datasheet (60 V) | dc_perm_ex_v1 (GEM) |",
        "|---|---|---|---|",
        f"| r_a, l_a, J | {m.motor_parameter['r_a'] * 1e3:g} mΩ, {m.motor_parameter['l_a'] * 1e3:g} mH, {m.motor_parameter['j_rotor']:g} kg·m² | 16 mΩ, 0.019 mH, 0.025 kg·m² | same |",
        f"| ψ_e | {m.motor_parameter['psi_e']:g} V·s/rad | 0.181 (P/(ω·I)) or 0.194 ((U − I·R)/ω) | 0.165 |",
        f"| Rated speed / torque / current / voltage | {m.nominal_values['omega']:.1f} rad/s, {m.nominal_values['torque']:g} N·m, {m.nominal_values['i']:g} A, {m.nominal_values['u']:g} V | 2870 min⁻¹, 20 N·m, 110 A, 60 V (S1) | 300 rad/s, 16 N·m, 97 A, 60 V |",
        f"| Limits current / torque / speed | {m.limits['i']:g} A, {m.limits['torque']:g} N·m, {m.limits['omega']:g} rad/s | 200 A, 38 N·m (S2 10 min); no speed limit given | 210 A, 38 N·m, 400 rad/s |",
        f"| Default load at rated speed | {load.base_load.load_parameter['a'] + load.base_load.load_parameter['b'] * OMEGA_RATED:.2f} N·m (2 % of rated torque, D1) | — | 15.01 N·m (94 % of rated torque) |",
        "",
        f"**Design check:** PI **{checks['PI']['status']}** (τ_m/τ_a = {checks['PI']['tau_m_over_tau_a']:.1f}; ψ_e is larger than GEM's, so τ_m is shorter); MPC **{checks['MPC']['status']}** (horizon/τ_a {checks['MPC']['horizon_over_tau_a']:.2f}).",
        "",
        "## Rated speed (2870 min⁻¹), load 0–100 % of rated torque",
        "",
        "| Controller | Load (N·m) | Final speed (rad/s) | Steady error (rad/s) | Steady current (A) | Steady voltage (V) |",
        "|---|---|---|---|---|---|",
    ] + [f"| {r['controller']} | {r['load_torque_nm']:g} | {r['final_speed_rad_s']:.2f} | {r['steady_error_rad_s']:.3f} | {r['steady_current_a']:.1f} | {r['steady_voltage_v']:.1f} |" for r in runs] + [
        "",
        "At rated load the motor holds rated speed with ≈ 113 A (110 A rating plus the 2 % friction) and 56 V, inside the 160 A current cap (0.8 × 200 A) and the 65 V supply. "
        "The same load on dc_perm_ex_v1 cannot be held at 300 rad/s (F0.1: 188 A needed).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
