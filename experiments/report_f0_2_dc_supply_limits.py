"""Round Assistant, task F0.2: evidence for dc_supply_v / dc_limit_overrides / dc_nominal_overrides
and the controller-design check (SIMULATED data only).

1. How GEM's DcPermanentlyExcitedMotor treats limits (point 1 of the task).
2. controller_design_check / mpc_design_check on the stock motor, case B and a constructed case.
3. Case B (220 V, R_a 1.6 ohm, L_a 16 mH, psi_e 1.146 V*s, J 0.05 kg*m^2) with PI and MPC, no
   extra load and at rated load torque (psi_e x 25 A), reference = rated speed 1500 rpm.

    python experiments/report_f0_2_dc_supply_limits.py
Writes docs/audit/assistant/f0_2_dc_supply_limits.{md,json}.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np

from driveflow.control.design_check import controller_design_check, mpc_design_check
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import build_plant
from driveflow.sim import DcPermanentlyExcitedMotor

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f0_2_dc_supply_limits"
OMEGA_RATED = 1500 * 2 * math.pi / 60
CASE_B = dict(
    motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05),
    dc_supply_v=220.0,
    dc_limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
    dc_nominal_overrides=dict(u=220.0, i=25.0, omega=OMEGA_RATED, torque=1.146 * 25.0),
)
DURATION_S = 2.0


def gem_limits() -> list[dict]:
    cases = [("stock", {}), ("limit_values={'i': 150}", dict(limit_values={"i": 150})), ("motor_parameter={'r_a': 1.6}", dict(motor_parameter={"r_a": 1.6})),
             ("limit_values={'i': 0}", dict(limit_values={"i": 0}))]
    return [{"case": name, "i_limit_a": float(DcPermanentlyExcitedMotor(**kw).limits["i"])} for name, kw in cases]


def checks() -> dict:
    out = {}
    for name, sc in (("stock", Scenario("s")), ("case_b", Scenario("b", **CASE_B))):
        p = build_plant(sc)
        mp, j = p.electrical_motor.motor_parameter, p.mechanical_load.j_total
        out[name] = {"pi": controller_design_check(mp, j, p.tau), "mpc": mpc_design_check(mp, j, p.tau)}
    p = build_plant(Scenario("s"))
    mp = dict(p.electrical_motor.motor_parameter)
    out["constructed_l_a_x2"] = {"pi": controller_design_check({**mp, "l_a": 2 * mp["l_a"]}, p.mechanical_load.j_total, p.tau)}
    return out


def run(controller: str, load_nm: float | None) -> dict:
    recs = run_scenario(Scenario(f"b_{controller}", controller_type=controller, omega_ref_rad_s=OMEGA_RATED, duration_s=DURATION_S, load_torque_nm=load_nm, **CASE_B))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    i = np.array([r["current_r"] for r in recs])
    u = np.array([r["voltage_v"] for r in recs])
    tail = slice(-2000, None)
    reach = np.argmax(omega >= 0.95 * OMEGA_RATED) if np.any(omega >= 0.95 * OMEGA_RATED) else None
    return {"controller": controller, "load_torque_nm": load_nm or 0.0, "finite": bool(np.all(np.isfinite(omega))), "final_speed_rad_s": float(omega[-1]),
            "steady_error_rad_s": float(OMEGA_RATED - omega[tail].mean()), "overshoot_pct": float(max(0.0, (omega.max() - OMEGA_RATED) / OMEGA_RATED * 100)),
            "t_95_s": None if reach is None else float((reach + 1) * 1e-4), "peak_current_a": float(i.max()), "steady_current_a": float(i[tail].mean()),
            "steady_voltage_v": float(u[tail].mean()), "voltage_at_supply": bool(u[tail].mean() >= 220.0 - 0.05)}


def main() -> None:
    runs = [run(c, load) for load in (None, 1.146 * 25.0) for c in ("PI", "MPC")]
    result = {"command": "python experiments/report_f0_2_dc_supply_limits.py", "simulated": True, "machine": platform.platform(), "python": platform.python_version(),
              "case_b": {k: v for k, v in CASE_B.items()}, "omega_rated_rad_s": OMEGA_RATED, "gem_limits": gem_limits(), "design_checks": checks(), "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    c = result["design_checks"]
    lines = [
        "# F0.2 — DC supply voltage, motor limits and nominal values; controller-design check",
        "",
        "**Simulated evidence only.**",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0_2_dc_supply_limits.json`](f0_2_dc_supply_limits.json).",
        "- **New `Scenario` fields (dc_perm_ex_v1 only):** `dc_supply_v` (IdealVoltageSupply, default 65 V), `dc_limit_overrides` and `dc_nominal_overrides` "
        "(keys `omega`, `torque`, `i`, `u`). All three `None` takes the original code path; `tests/test_assistant_f0_dc_supply_limits.py` shows the explicit stock values (65 V, stock limits and nominals) "
        "are bit-identical to the default with PI and MPC.",
        "",
        "## 1. Does GEM overwrite an explicit current limit?",
        "",
        "`ElectricMotor._update_limits` fills a limit only when it is missing or 0; `DcPermanentlyExcitedMotor._update_limits` proposes `i = u / r_a` for that case. Measured:",
        "",
        "| Constructor arguments | i limit (A) |",
        "|---|---|",
    ] + [f"| `{g['case']}` | {g['i_limit_a']:g} |" for g in result["gem_limits"]] + [
        "",
        "- An explicit `i` is **kept** as given; nothing had to change to respect it.",
        "- `u / r_a` is used only for `i = 0` (3750 A here).",
        "- Changing `r_a` (or any motor parameter) without limits keeps the stock 210 A: GEM does not derive the limit from the new motor. A caller with another motor must pass `dc_limit_overrides`; the future assistant will have to fill it from the datasheet. Documented in the `Scenario` docstring.",
        "- Limits only normalise the state and set the controllers' current cap (PI 0.8 × i, MPC 0.85 × i); the simulator does not clip states with them.",
        "",
        "## 2. Controller-design check (`control/design_check.py`)",
        "",
        "Thresholds, declared in the module before any run: τ_m/τ_a ≥ 10; current-loop / speed-loop bandwidth ≥ 10; τ_a ≥ 2 control periods (Euler stability); MPC horizon ≥ τ_a.",
        "",
        "| Motor | τ_a (ms) | τ_m (ms) | τ_m/τ_a | Bandwidth ratio | PI | MPC horizon / τ_a | MPC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, label in (("stock", "stock (PMG 132 / GEM)"), ("case_b", "case B, 220 V")):
        pi, mpc = c[name]["pi"], c[name]["mpc"]
        lines.append(f"| {label} | {pi['tau_a_s'] * 1e3:.3g} | {pi['tau_m_s'] * 1e3:.3g} | {pi['tau_m_over_tau_a']:.1f} | {pi['bandwidth_ratio']:.1f} | **{pi['status']}** | {mpc['horizon_over_tau_a']:.2f} | **{mpc['status']}** |")
    pi = c["constructed_l_a_x2"]["pi"]
    lines.append(f"| constructed: stock with l_a × 2 | {pi['tau_a_s'] * 1e3:.3g} | {pi['tau_m_s'] * 1e3:.3g} | {pi['tau_m_over_tau_a']:.1f} | {pi['bandwidth_ratio']:.1f} | **{pi['status']}** | — | — |")
    lines += [
        "",
        f"- The stock motor passes with little margin (τ_m/τ_a = {c['stock']['pi']['tau_m_over_tau_a']:.1f}), consistent with the PI module's \"~100x\" comment being optimistic: it is about 14×.",
        "- **The MPC depends on the motor parameters**: its prediction model is built from r_a, l_a, psi_e and j_total (`control/mpc/controller.py`), with a fixed 20-step (2 ms) horizon sized for the stock τ_a.",
        "",
        "## 3. Case B: 220 V motor, reference = rated speed 1500 rpm",
        "",
        "| Controller | Load torque (N·m) | Final speed (rad/s) | Steady error (rad/s) | Overshoot | t to 95 % (s) | Peak / steady current (A) | Steady voltage (V) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(f"| {r['controller']} | {r['load_torque_nm']:.2f} | {r['final_speed_rad_s']:.2f} | {r['steady_error_rad_s']:.3f} | {r['overshoot_pct']:.2f} % | "
                     f"{r['t_95_s'] if r['t_95_s'] is None else round(r['t_95_s'], 3)} | {r['peak_current_a']:.1f} / {r['steady_current_a']:.1f} | {r['steady_voltage_v']:.1f}{' (= supply)' if r['voltage_at_supply'] else ''} |")
    no_load = [r for r in runs if r["load_torque_nm"] == 0.0]
    loaded = [r for r in runs if r["load_torque_nm"] > 0.0]
    needed_v = 1.146 * OMEGA_RATED + 1.6 * (0.01 + 0.05 * OMEGA_RATED + 1.146 * 25.0) / 1.146
    lines += [
        "",
        "## Reading",
        "",
        f"- **Without extra load, both controllers reach rated speed** despite the `warning`: PI error {no_load[0]['steady_error_rad_s']:.3f} rad/s and {no_load[0]['overshoot_pct']:.2f} % overshoot; "
        f"MPC {no_load[1]['steady_error_rad_s']:.3f} rad/s (its known offset, no integrator). The check is conservative here; it flags that the design assumptions do not hold, not that the run fails. "
        "So the instruction's S-20 condition (the 220 V motor not reaching its reference) does not occur.",
        f"- **At rated load torque ({1.146 * 25:.2f} N·m) neither reaches rated speed** "
        f"(PI {loaded[0]['final_speed_rad_s']:.1f}, MPC {loaded[1]['final_speed_rad_s']:.1f} rad/s; voltage at the 220 V supply). Holding 1500 rpm would need about {needed_v:.0f} V: "
        f"the stock load's friction (a + b·ω = {0.01 + 0.05 * OMEGA_RATED:.1f} N·m at this speed) adds to the rated torque, and back-EMF plus R_a·I exceeds the supply. "
        "This is the plant doing what its numbers say (the stock friction model was sized for the 60 V PMG 132, not for this motor), not a controller defect. "
        "Recorded as an observation for the assistant: with another motor, the default load friction is not neutral.",
        "",
        "## Not executed",
        "",
        "- Changing either controller: no ejecutado (out of scope; the check is diagnostic only).",
        "- A friction model scaled to the motor: no ejecutado; it would change a default (§7 of the round instructions).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
