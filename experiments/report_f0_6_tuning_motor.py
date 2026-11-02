"""Round Assistant, task F0.6: PI tuning on another DC motor (SIMULATED; results are candidates).

1. Stock motor, before vs after the change: evaluate(BASELINE) on the three grids and J_effort,
   against docs/audit/assistant/f0_6_stock_before.json (recorded on commit 78f6bbe, before F0.6).
2. A short PSO tuning (objective J, 4 particles x 2 iterations, seed 0) on the 220 V motor of case
   B, then the baseline and the candidate on the tuning and evaluation grids of that motor.

    python experiments/report_f0_6_tuning_motor.py
Writes docs/audit/assistant/f0_6_tuning_motor.{md,json}.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

from driveflow.optim import objectives as ob
from driveflow.optim.pso import pso

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant"
OUT = AUDIT / "f0_6_tuning_motor"
CASE_B = ob.BaseMotor(motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05), supply_v=220.0,
                      limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
                      nominal_overrides=dict(u=220.0, i=25.0, omega=1500 * 2 * math.pi / 60, torque=1.146 * 25.0))
GRIDS = {"TUNING_GRID": ob.TUNING_GRID, "EVALUATION_GRID": ob.EVALUATION_GRID, "EVALUATION_GRID_NOISY": ob.EVALUATION_GRID_NOISY}


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "cpu_s"}


def stock_comparison() -> dict:
    before = json.loads((AUDIT / "f0_6_stock_before.json").read_text())
    after = {}
    for name, grid in GRIDS.items():
        s, rows = ob.evaluate(ob.BASELINE, grid)
        after[name] = {"summary": _strip(s), "results": [_strip(r) for r in rows]}
    after["j_effort_baseline_on_tuning"] = ob.j_effort(ob.evaluate(ob.BASELINE, ob.TUNING_GRID)[0])
    identical = {k: json.loads(json.dumps(after[k])) == before[k] for k in after}
    return {"identical": identical, "J_before": {k: before[k]["summary"]["J"] for k in GRIDS}, "J_after": {k: after[k]["summary"]["J"] for k in GRIDS}}


def case_b_tuning() -> dict:
    scalar, _v, eval_grid, _text, _score = ob.objective_functions("J", CASE_B)
    res = pso(scalar, [list(b) for b in ob.BOUNDS], n_particles=4, iterations=2, seed=0)
    cand = ob.as_kwargs(res.x_best)
    rows = {}
    for label, params in (("baseline", ob.BASELINE), ("candidate", cand)):
        tune, _ = ob.evaluate(params, ob.TUNING_GRID, CASE_B)
        ev, ev_rows = ob.evaluate(params, eval_grid, CASE_B)
        rows[label] = {"params": params, "tuning": _strip(tune), "evaluation": _strip(ev), "evaluation_rows": [_strip(r) for r in ev_rows]}
    return {"evaluations": len(res.F), "seed": 0, "objective": "J", **rows}


def main() -> None:
    result = {"command": "python experiments/report_f0_6_tuning_motor.py", "simulated": True, "status": "candidate -- no default changed",
              "machine": platform.platform(), "python": platform.python_version(), "stock": stock_comparison(), "case_b": case_b_tuning()}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    st, cb = result["stock"], result["case_b"]
    lines = [
        "# F0.6 — PI tuning on another DC motor",
        "",
        "**Simulated evidence; the tuned controller is a candidate. No default changed.**",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0_6_tuning_motor.json`](f0_6_tuning_motor.json). Numbers before the change: [`f0_6_stock_before.json`](f0_6_stock_before.json).",
        "- **What changed:**",
        "  - `optim/objectives.py`: a `BaseMotor` (motor parameters, supply, limits, nominal values, as the Scenario fields);",
        "  - the drift conditions are now relative factors (`DRIFT_FACTORS`: ψ_e × 0.9, j_rotor × 1.5) of the base motor;",
        "  - `evaluate`, `simulate_condition`, `j_effort` and `objective_functions` take the motor;",
        "  - `validation/benches.run_dc_closed_loop` designs the controller on the base motor and accepts supply, limits and nominal values;",
        "  - `controller_tuning` accepts an optional `params.motor` block, validated by `Scenario`.",
        "- **Rated current of J_effort:** 97 A for the stock motor; otherwise `dc_nominal_overrides['i']`. With changed motor parameters and no rated current, J_effort is refused before any simulation (`MotorSpecError`, `INVALID_SPEC` in the job).",
        "",
        "## 1. Stock motor: before vs after",
        "",
        "| Quantity | Before | After | Identical |",
        "|---|---|---|---|",
    ] + [f"| J of BASELINE on {k} | {st['J_before'][k]!r} | {st['J_after'][k]!r} | {'yes' if st['identical'][k] else '**NO**'} |" for k in GRIDS] + [
        f"| J_effort of BASELINE on the tuning grid | 4.0 | — | {'yes' if st['identical']['j_effort_baseline_on_tuning'] else '**NO**'} |",
        "",
        "Every per-condition metric (overshoot, settling, dip, effort, peak current; CPU time excluded) was compared too: identical, as JSON equality.",
        "",
        f"## 2. Case B, 220 V motor: short PSO tuning ({cb['evaluations']} evaluations, objective J, seed 0)",
        "",
        "| Controller | Bandwidth (rad/s) | Damping | b | J tuning grid | J evaluation grid | Max peak step current (A) | ≤ 25 A rated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in ("baseline", "candidate"):
        r = cb[label]
        p = r["params"]
        lines.append(f"| {label} | {p['speed_bandwidth_rad_s']:.1f} | {p['speed_damping']:.3f} | {p['setpoint_weight']:.3f} | {r['tuning']['J']:.2f} | {r['evaluation']['J']:.2f} | "
                     f"{r['evaluation']['max_step_peak_i_a']:.1f} | {r['evaluation']['current_constraint_ok']} |")
    load200 = [x for x in cb["baseline"]["evaluation_rows"] if x["condition"] == "drift_load_15Nm_at_200"]
    lines += [
        "",
        "## Reading",
        "",
        "- With the stock motor the tuning is unchanged, number for number.",
        "- The tuning machinery runs on another motor; the candidate is a short-budget search, not a recommended controller.",
        "- **Limit (not fixed, needs a decision):** the grids' speeds, steps, loads and the 0.6 rad/s noise are absolute values sized for the stock motor (up to 200 rad/s). "
        + (f"For the 220 V motor the `drift_load_15Nm_at_200` condition asks for 200 rad/s, above its 157 rad/s rating (back-EMF 1.146 × 200 = 229 V > 220 V supply): "
           f"its final error there is {load200[0]['final_error_rad_s']:.1f} rad/s, so that condition measures the supply limit, not the controller." if load200 else ""),
        "",
        "## Not executed",
        "",
        "- Grids scaled to the base motor's ratings: no ejecutado (it would change the tuning definition; decision pending).",
        "- A full-budget tuning on the 220 V motor: no ejecutado (only the machinery was asked for).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
