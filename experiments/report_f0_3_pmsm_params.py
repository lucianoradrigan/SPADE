"""Round Assistant, task F0.3: evidence for Scenario.pmsm_motor_parameters (SIMULATED data only).

1. Propagation: which functions receive the merged dict (read in control/classical/pmsm_foc.py).
2. Explicit MOTOR_PARAMS vs the default: bit-identical records.
3. Non-salient machine (l_d = l_q): MTPA vs naive torque.
4. Stock (salient) machine at several currents: MTPA vs naive torque, for contrast.

    python experiments/report_f0_3_pmsm_params.py
Writes docs/audit/assistant/f0_3_pmsm_params.{md,json}.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS
from driveflow.datagen import Scenario, run_scenario

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f0_3_pmsm_params"
NONSALIENT = {"l_d": 0.8e-3, "l_q": 0.8e-3}
KEYS = ("i_d", "i_q", "torque_nm", "rpm", "u_d", "u_q")


def run(controller: str, i_s: float, params: dict | None, duration_s: float = 0.03) -> list:
    return run_scenario(Scenario("p", controller_type=controller, plant_config_id="pmsm_v1", pmsm_current_ref_a=i_s, duration_s=duration_s, pmsm_motor_parameters=params))


def tail_torque(recs) -> float:
    return float(np.mean([r["torque_nm"] for r in recs[-50:]]))


def main() -> None:
    identical = {}
    for controller in ("FOC_MTPA", "FOC_naive"):
        a, b = run(controller, 150.0, None), run(controller, 150.0, dict(MOTOR_PARAMS))
        identical[controller] = all(np.array_equal(np.array([r[k] for r in a]), np.array([r[k] for r in b])) for k in KEYS)
    rows = []
    for label, params in (("non-salient (l_d = l_q = 0.8 mH)", NONSALIENT), ("stock (l_d 0.37 mH, l_q 1.2 mH)", None)):
        for i_s in (100.0, 200.0, 350.0):
            t_mtpa, t_naive = tail_torque(run("FOC_MTPA", i_s, params)), tail_torque(run("FOC_naive", i_s, params))
            rows.append({"machine": label, "i_s_a": i_s, "torque_mtpa_nm": t_mtpa, "torque_naive_nm": t_naive, "mtpa_over_naive": t_mtpa / t_naive})
    result = {"command": "python experiments/report_f0_3_pmsm_params.py", "simulated": True, "machine": platform.platform(), "python": platform.python_version(),
              "bit_identical_explicit_stock": identical, "torque_comparison": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# F0.3 — PMSM parameters (`Scenario.pmsm_motor_parameters`)",
        "",
        "**Simulated evidence only.**",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f0_3_pmsm_params.json`](f0_3_pmsm_params.json).",
        "- **Field:** `pmsm_motor_parameters` (pmsm_v1 only), partial override merged over `pmsm_foc.MOTOR_PARAMS`; keys `p`, `l_d`, `l_q`, `r_s`, `psi_p`, and also `j_rotor` "
        "(GEM's `PermanentMagnetSynchronousMotor` accepts it, default 0.03883 kg·m²) and `u_supply` (`build_pmsm_system` accepts it, default 330 V).",
        "",
        "## 1. Propagation (read in `control/classical/pmsm_foc.py`)",
        "",
        "| Function | Receives the merged dict | Uses the global `MOTOR_PARAMS` inside? |",
        "|---|---|---|",
        "| `build_pmsm_system(mp, u_supply)` | yes (`motor_parameter=mp`) | no, only as the default argument |",
        "| `DqCurrentController(system, mp)` | yes (gains and decoupling terms) | no |",
        "| `mtpa_id_iq(i_s, mp)` | yes | no |",
        "| `torque_of(i_d, i_q, mp)` (the record's `torque_nm`) | yes; it used to receive `MOTOR_PARAMS` explicitly in `runner._run_pmsm_scenario` | no |",
        "",
        "Other globals (`U_SUPPLY`, `I_LIM`, `LOAD_INERTIA`, `TAU`) are defaults or constants of the bench; `u_supply` is overridable, the |i_s| ≤ 400 A bound of `pmsm_current_ref_a` stays.",
        "",
        "**Change outside `Scenario`:** `mtpa_id_iq` divided by `l_d − l_q`, so a non-salient machine raised a division by zero. It now returns i_d = 0 for `l_d == l_q`, the formula's limit. The stock machine (l_d ≠ l_q) takes the original path.",
        "",
        "## 2. Default vs explicit `MOTOR_PARAMS`",
        "",
        "Bit-identical records (i_d, i_q, torque, rpm, u_d, u_q): " + ", ".join(f"{k} {'yes' if v else '**NO**'}" for k, v in identical.items()) + ".",
        "",
        "## 3–4. MTPA vs naive torque (mean of the last 5 ms of a 30 ms current step)",
        "",
        "| Machine | i_s (A) | MTPA torque (N·m) | Naive torque (N·m) | MTPA / naive |",
        "|---|---|---|---|---|",
    ] + [f"| {r['machine']} | {r['i_s_a']:g} | {r['torque_mtpa_nm']:.2f} | {r['torque_naive_nm']:.2f} | {r['mtpa_over_naive']:.4f} |" for r in rows] + [
        "",
        "- Non-salient: MTPA and the naive policy give the same torque (ratio 1 to rounding; the test tolerance is 0.5 %).",
        "- Stock salient machine: MTPA gains torque from the reluctance term, more at higher current.",
        "- `l_d > l_q` is accepted with a `UserWarning` (MTPA assumes l_q ≥ l_d); see `tests/test_assistant_f0_pmsm_params.py`.",
        "",
        "## Not executed",
        "",
        "- A PMSM from a real datasheet (e.g. EMRAX 228): no ejecutado. Its flux and Ke conventions are unresolved (docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md §3); that is Phase 2 work.",
        "- Field weakening or a speed loop: no ejecutado; the PMSM plant is a current-step bench near standstill.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
