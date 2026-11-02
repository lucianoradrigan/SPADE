"""Round Assistant 0b, D2: tuning grids scaled to the base motor's ratings (SIMULATED; candidates).

1. Stock motor: evaluate(BASELINE) on the three grids, against the numbers recorded before F0.6
   (docs/audit/assistant/f0_6_stock_before.json).
2. The grids of the 220 V motor of case B (rated 157.1 rad/s, 28.65 N*m).
3. The same short PSO as F0.6 (objective J, 4 particles x 2 iterations, seed 0) on the scaled grids.

    python experiments/report_0b_2_tuning_grids.py
Writes docs/audit/assistant/f0b_2_tuning_grids.{md,json}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from driveflow.optim import objectives as ob
from driveflow.optim.pso import pso

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant"
OUT = AUDIT / "f0b_2_tuning_grids"
CASE_B = ob.BaseMotor(motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05), supply_v=220.0,
                      limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
                      nominal_overrides=dict(u=220.0, i=25.0, omega=1500 * 2 * math.pi / 60, torque=1.146 * 25.0))


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "cpu_s"}


def main() -> None:
    before = json.loads((AUDIT / "f0_6_stock_before.json").read_text())
    names = ("TUNING_GRID", "EVALUATION_GRID", "EVALUATION_GRID_NOISY")
    stock = {}
    for name, grid in zip(names, ob.grids_for(ob.STOCK)):
        s, rows = ob.evaluate(ob.BASELINE, grid)
        stock[name] = json.loads(json.dumps({"summary": _strip(s), "results": [_strip(r) for r in rows]})) == before[name]
    tuning, evaluation, noisy = ob.grids_for(CASE_B)
    grid_rows = [{"grid": g, "name": c.name, "kind": c.kind, "omega_rad_s": c.omega_rad_s, "step_rad_s": c.step_rad_s, "load_nm": c.load_nm, "noise_rad_s": c.noise_std_rad_s,
                  "drift": bool(c.drift)} for g, cs in (("tuning", tuning), ("evaluation", evaluation)) for c in cs]
    scalar, _v, eval_grid, text, _score = ob.objective_functions("J", CASE_B)
    res = pso(scalar, [list(b) for b in ob.BOUNDS], n_particles=4, iterations=2, seed=0)
    runs = {}
    for label, params in (("baseline", ob.BASELINE), ("candidate", ob.as_kwargs(res.x_best))):
        t, _ = ob.evaluate(params, tuning, CASE_B)
        e, er = ob.evaluate(params, eval_grid, CASE_B)
        runs[label] = {"params": params, "tuning": _strip(t), "evaluation": _strip(e), "evaluation_rows": [_strip(r) for r in er]}
    result = {"command": "python experiments/report_0b_2_tuning_grids.py", "simulated": True, "status": "candidate -- no default changed",
              "stock_identical_to_before_f0_6": stock, "case_b_grids": grid_rows, "case_b_tuning": {"evaluations": len(res.F), **runs}}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    lines = [
        "# 0b D2 — tuning grids scaled to the base motor's ratings",
        "",
        "**Simulated evidence; the tuned controller is a candidate. No default changed.** User decision of 2026-11-02.",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0b_2_tuning_grids.json`](f0b_2_tuning_grids.json).",
        "- **Rule** (`optim/objectives.grids_for`):",
        "  - speeds, speed steps and the speed-sensor noise × ω_rated / 300 rad/s;",
        "  - load steps × T_rated / 16 N·m (300 rad/s and 16 N·m are the stock ratings);",
        "  - the stock motor uses the module's grids unchanged (same objects);",
        "  - a changed motor without rated speed and torque cannot be tuned (`MotorSpecError`, `INVALID_SPEC` in the job).",
        "- **Stock motor, same numbers as before F0.6:** " + ", ".join(f"{k} {'yes' if v else '**NO**'}" for k, v in stock.items()) + ".",
        "",
        "## Grids of the 220 V motor (rated 157.1 rad/s, 28.65 N·m)",
        "",
        "| Grid | Condition | Kind | Speed (rad/s) | Step (rad/s) | Load (N·m) | Drift |",
        "|---|---|---|---|---|---|---|",
    ] + [f"| {r['grid']} | `{r['name']}` | {r['kind']} | {r['omega_rad_s']:.1f} | {r['step_rad_s']:.2f} | {r['load_nm']:.2f} | {'yes' if r['drift'] else 'no'} |" for r in grid_rows] + [
        "",
        f"The fastest condition is now {max(r['omega_rad_s'] for r in grid_rows):.1f} rad/s (2/3 of 157 = the stock grid's 200 of 300 rad/s), within the 220 V motor's range; the noisy grid uses "
        f"{ob.NOISE_STD_RAD_S * CASE_B.nominal_overrides['omega'] / 300:.3f} rad/s of sensor noise.",
        "",
        f"## Short PSO on the scaled grids ({len(res.F)} evaluations, objective J, seed 0)",
        "",
        "| Controller | Bandwidth (rad/s) | Damping | b | J tuning grid | J evaluation grid | Every evaluation run finite |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, r in runs.items():
        p = r["params"]
        lines.append(f"| {label} | {p['speed_bandwidth_rad_s']:.1f} | {p['speed_damping']:.3f} | {p['setpoint_weight']:.3f} | {r['tuning']['J']:.2f} | {r['evaluation']['J']:.2f} | {r['evaluation']['finite']} |")
    load_rows = [x for x in runs["baseline"]["evaluation_rows"] if x["kind"] == "load"]
    lines += [
        "",
        "Baseline load-step conditions on the evaluation grid: " + "; ".join(f"`{x['condition']}` dip {x['dip_rad_s']:.2f} rad/s, final error {x['final_error_rad_s']:.3f} rad/s" for x in load_rows) + ".",
        "With the absolute grids (F0.6) the 200 rad/s load condition ended 22.6 rad/s off because it exceeded the supply; on the scaled grid every condition is inside the motor's range.",
        "",
        "## Not executed",
        "",
        "- A full-budget tuning on the 220 V motor: no ejecutado (machinery only).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
