"""Round Assistant, task F0.1: evidence for Scenario.load_torque_nm (SIMULATED data only).

Five load levels -- 0, 25, 50, 75 and 100 % of the stock motor's rated torque (16 N*m,
DcPermanentlyExcitedMotor nominal_values) -- at 150 and 300 rad/s, speed PI, healthy bearing,
3 s per run. Per run: final speed, steady-state speed error and mean current (last 0.2 s), the
current the load needs in steady state, (a + b*omega_ref + load) / psi_e, and whether the current
sits at the PI's reference cap (0.8 x the 210 A limit = 168 A) in steady state.

Also records the PolynomialStaticLoad check of the round instructions (F0.1 point 2): the speed
below which the constant term is smoothed linearly, a * tau_decay / j_total.

    python experiments/report_f0_1_load_torque.py
Writes docs/audit/assistant/f0_1_load_torque.{md,json}.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import build_plant
from driveflow.sim.loads import PolynomialStaticLoad

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f0_1_load_torque"
RATED_TORQUE_NM = 16.0
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SPEEDS = (150.0, 300.0)
DURATION_S = 3.0
STEADY_SAMPLES = 2000  # 0.2 s at 10 kHz
A, B, PSI_E = 0.01, 0.05, 0.165
I_CAP_A = 0.8 * 210.0


def run(omega_ref: float, load_nm: float) -> dict:
    recs = run_scenario(Scenario(f"f01_{omega_ref:g}_{load_nm:g}", omega_ref_rad_s=omega_ref, load_torque_nm=load_nm, duration_s=DURATION_S))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    i = np.array([r["current_r"] for r in recs])
    tail = slice(-STEADY_SAMPLES, None)
    needed = (A + B * omega_ref + load_nm) / PSI_E
    i_ss = float(i[tail].mean())
    return {
        "omega_ref_rad_s": omega_ref, "load_torque_nm": load_nm, "load_pct_of_rated": 100 * load_nm / RATED_TORQUE_NM,
        "final_speed_rad_s": float(omega[-1]), "steady_error_rad_s": float(omega_ref - omega[tail].mean()),
        "steady_mean_current_a": i_ss, "current_needed_a": float(needed),
        "at_current_cap_in_steady_state": bool(i_ss >= I_CAP_A - 0.01), "start_up_reaches_cap": bool(i.max() >= I_CAP_A - 0.01),
        "speed_still_changing_rad_s_per_s": float((omega[-1] - omega[-STEADY_SAMPLES]) / (STEADY_SAMPLES * 1e-4)),
    }


def smoothing_band(load_nm: float) -> float:
    plant = build_plant(Scenario("band", load_torque_nm=load_nm))
    base: PolynomialStaticLoad = plant.mechanical_load.base_load
    return float(base._omega_lim)


def main() -> None:
    rows = [run(w, lvl * RATED_TORQUE_NM) for w in SPEEDS for lvl in LEVELS]
    band = {f"{lvl * RATED_TORQUE_NM:g}": smoothing_band(lvl * RATED_TORQUE_NM) for lvl in LEVELS}
    result = {"command": "python experiments/report_f0_1_load_torque.py", "simulated": True, "machine": platform.platform(), "python": platform.python_version(),
              "rated_torque_nm": RATED_TORQUE_NM, "duration_s": DURATION_S, "pi_current_cap_a": I_CAP_A, "smoothing_band_rad_s": band, "runs": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# F0.1 — configurable DC-motor load (`Scenario.load_torque_nm`)",
        "",
        "**Simulated evidence only.** No measurement of a real motor is involved.",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0_1_load_torque.json`](f0_1_load_torque.json).",
        f"- **Setup:** stock `dc_perm_ex_v1` (GEM PMG 132 parameters), speed PI, healthy bearing, {DURATION_S:g} s per run; steady state = last 0.2 s. "
        f"Rated torque {RATED_TORQUE_NM:g} N·m (GEM nominal). The PI caps its current reference at 0.8 × 210 A = {I_CAP_A:g} A.",
        "",
        "## Is `PolynomialStaticLoad`'s `a` a passive load torque? (point 2)",
        "",
        "Read in `sim/loads.py`: the static load is `T = sign(ω)·c·ω² + b·ω + a_s`, with `a_s = sign(ω)·a` for |ω| > ω_lim and `a_s = (j_total/τ_decay)·ω` below it, "
        "ω_lim = a·τ_decay/j_total, τ_decay = 1 ms. So:",
        "",
        "- it always opposes the rotation and is zero at standstill, so it never drives the shaft (passive, Coulomb-type);",
        "- the linear band is continuous at ω_lim (both branches give `a` there) and keeps the Euler step stable: its time constant is τ_decay = 1 ms = 10 control periods;",
        "- the band grows with `a`, but stays small up to 100 % load: "
        + ", ".join(f"{k} N·m → {v:.3f} rad/s" for k, v in band.items()) + ";",
        "- `BearingFaultLoad` delegates `set_j_rotor` to the wrapped load, so the band is recomputed with the real inertia.",
        "",
        "**Conclusion:** adequate for loads up to 100 % of rated torque; implemented as `a = 0.01 + load_torque_nm`. It is not an active load (a hoist or a "
        "regenerating load would need a different model), and it does not ramp within a run (A1).",
        "",
        "## Load levels",
        "",
        "| ω_ref (rad/s) | Load (N·m, % rated) | Final speed (rad/s) | Steady error (rad/s) | Steady mean current (A) | Current the load needs (A) | At the 168 A cap in steady state |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['omega_ref_rad_s']:g} | {r['load_torque_nm']:g} ({r['load_pct_of_rated']:.0f} %) | {r['final_speed_rad_s']:.2f} | {r['steady_error_rad_s']:.3f} | "
                     f"{r['steady_mean_current_a']:.1f} | {r['current_needed_a']:.1f} | {'**yes**' if r['at_current_cap_in_steady_state'] else 'no'} |")
    capped = [r for r in rows if r["at_current_cap_in_steady_state"]]
    lines += [
        "",
        "## Reading",
        "",
        "- Wherever the needed current is below the cap, the speed PI holds the reference and the steady current equals (a + b·ω + load)/ψ_e (also asserted in `tests/test_assistant_f0_load_torque.py`, 0.1 % tolerance).",
        "- Every run saturates the current reference during the start-up from standstill; that is the PI's normal large-step behaviour, not the load.",
    ]
    for r in capped:
        w_eq = (PSI_E * I_CAP_A - A - r["load_torque_nm"]) / B
        lines.append(f"- **{r['load_pct_of_rated']:.0f} % at {r['omega_ref_rad_s']:g} rad/s:** the load needs {r['current_needed_a']:.0f} A, above the {I_CAP_A:g} A cap. "
                     f"The motor cannot hold the reference: with the current at the cap the speed tends to (ψ_e·{I_CAP_A:g} − a − load)/b = {w_eq:.0f} rad/s "
                     f"(final speed {r['final_speed_rad_s']:.1f} rad/s, still changing at {r['speed_still_changing_rad_s_per_s']:.1f} rad/s²). "
                     f"Cause: the stock load's viscous term b·ω already takes {B * r['omega_ref_rad_s']:.0f} N·m at this speed, so rated torque on top of it needs about twice the 97 A rated current. "
                     "It is the current limit at work (210 A limit, PI margin 0.8), not a defect.")
    lines += ["", "## Not executed", "", "- A load ramp within a run (`load_profile`): no ejecutado, Phase 3 (A1).", "- Active (overhauling) loads: no ejecutado, outside the model."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
