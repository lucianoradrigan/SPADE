"""What Tuning Studio can tune, found by capability -- task B6 (round PLECS/OpenModelica).

A tuning system bundles the controller parameters, their bounds and baseline, the two fixed grids
(tuning, evaluation), the objective and the per-condition simulation. Two families:

- ``dc_motor_pi``: the DC-motor PI speed loop of E7 (optim/objectives.py, unchanged).
- ``buck_pi:<env_id>``: the buck's BuckVoltageController on any registered, available Gymnasium
  environment whose reference controller is that PI (envs.registry.rl_kind == "buck", without a
  surrogate's out-of-domain caveat being hidden). SPADE/Buck-v0 and SPADE/OmBuck-v0 qualify.

Buck PI system, fixed here before any run (B6):
- parameters: the PI's ``current_bandwidth_hz`` (baseline 1 / (10·Ts) = 5000 Hz at Ts = 20 µs) and
  ``voltage_bandwidth_hz`` (baseline current/5 = 1000 Hz);
- each condition is one 20 ms episode from rest (start-up included) at a fixed reference and load,
  without a load step (an FMU's load is set at reset);
- per condition: tracking RMSE over the whole episode as % of v_ref, overshoot % of v_ref, peak
  |i_L|; a terminated episode makes the evaluation non-finite (failed);
- J = mean RMSE % + mean overshoot %, lower is better;
- tuning grid: (24 V, 6 Ω), (12 V, 12 Ω); evaluation grid, distinct: (30 V, 8 Ω), (18 V, 20 Ω),
  (36 V, 4 Ω), (8 V, 3 Ω).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable

import numpy as np

DC_MOTOR = "dc_motor_pi"
BUCK_PREFIX = "buck_pi:"


@dataclasses.dataclass(frozen=True)
class BuckCondition:
    name: str
    vout_ref_v: float
    load_resistance_ohm: float


BUCK_PARAMS = ("current_bandwidth_hz", "voltage_bandwidth_hz")
BUCK_BOUNDS = ((1000.0, 10000.0), (200.0, 2500.0))
BUCK_BASELINE = {"current_bandwidth_hz": 5000.0, "voltage_bandwidth_hz": 1000.0}
BUCK_PARAM_LABELS = {"current_bandwidth_hz": ("Current-loop bandwidth", "Hz"), "voltage_bandwidth_hz": ("Voltage-loop bandwidth", "Hz")}
BUCK_TUNING_GRID = (BuckCondition("24V_6ohm", 24.0, 6.0), BuckCondition("12V_12ohm", 12.0, 12.0))
BUCK_EVALUATION_GRID = (BuckCondition("30V_8ohm", 30.0, 8.0), BuckCondition("18V_20ohm", 18.0, 20.0), BuckCondition("36V_4ohm", 36.0, 4.0),
                        BuckCondition("8V_3ohm", 8.0, 3.0))
BUCK_OBJECTIVE_TEXT = "J = mean tracking RMSE (% of v_ref, whole 20 ms episode from rest) + mean overshoot (% of v_ref). Lower is better."


@dataclasses.dataclass(frozen=True)
class TuningSystem:
    key: str
    plant_label: str
    controller_label: str
    plant_profile_id: str
    params: tuple
    bounds: tuple
    baseline: dict
    param_labels: dict
    tuning_grid: tuple
    evaluation_grid: tuple
    methods: tuple
    objective_text: str
    evaluate: Callable  # (x, grid) -> (summary with "J", rows)
    scalar_objective: Callable
    subject: str
    note: str = ""

    def grids_overlap(self) -> bool:
        return bool({c.name for c in self.tuning_grid} & {c.name for c in self.evaluation_grid})


def _buck_episode(env_id: str, params: dict, c: BuckCondition) -> dict:
    from driveflow.control.classical.buck_pi import BuckVoltageController
    from driveflow.envs import registry

    t0 = time.perf_counter()
    env = registry.make(env_id)
    try:
        u = env.unwrapped
        env.reset(seed=0, options={"vout_ref_v": c.vout_ref_v, "load_resistance_ohm": c.load_resistance_ohm})
        ctrl = BuckVoltageController(u.plant, current_bandwidth_hz=params["current_bandwidth_hz"], voltage_bandwidth_hz=params["voltage_bandwidth_hz"])
        v, i, term = [], [], None
        for _ in range(u.n_steps):
            _, _, terminated, truncated, info = env.step(np.array([ctrl.control(u.plant_state, u.vout_ref_v)]))
            v.append(info["v_out_v"])
            i.append(abs(info["i_l_a"]))
            if terminated:
                term = info.get("termination")
                break
            if truncated:
                break
    finally:
        env.close()
    v = np.asarray(v)
    ok = term is None and np.all(np.isfinite(v))
    return {"condition": c.name, "vout_ref_v": c.vout_ref_v, "load_resistance_ohm": c.load_resistance_ohm, "termination": term,
            "rmse_pct": float(100 * np.sqrt(np.mean((v - c.vout_ref_v) ** 2)) / c.vout_ref_v) if ok else float("nan"),
            "overshoot_pct": float(100 * max(0.0, np.max(v) - c.vout_ref_v) / c.vout_ref_v) if ok else float("nan"),
            "peak_current_a": float(np.max(i)) if len(i) else float("nan"), "final_v_out_v": float(v[-1]) if len(v) else float("nan"),
            "wall_s": time.perf_counter() - t0}


def _buck_evaluate(env_id: str):
    def evaluate(x, grid=BUCK_TUNING_GRID):
        params = x if isinstance(x, dict) else dict(zip(BUCK_PARAMS, map(float, x)))
        rows = [_buck_episode(env_id, params, c) for c in grid]
        rm, ov = [r["rmse_pct"] for r in rows], [r["overshoot_pct"] for r in rows]
        J = float(np.mean(rm) + np.mean(ov)) if all(math.isfinite(a) and math.isfinite(b) for a, b in zip(rm, ov)) else float("inf")
        return {"J": J, "mean_rmse_pct": float(np.mean(rm)), "mean_overshoot_pct": float(np.mean(ov)), "n_terminated": sum(r["termination"] is not None for r in rows)}, rows

    return evaluate


def buck_system(env_id: str) -> TuningSystem:
    from driveflow.envs import registry

    spec = registry.ENVIRONMENTS[env_id]
    ev = _buck_evaluate(env_id)
    return TuningSystem(key=BUCK_PREFIX + env_id, plant_label=f"{spec.display_name} ({env_id}, {spec.source})", controller_label="Buck PI cascade (BuckVoltageController)",
                        plant_profile_id=spec.plant_profile_id, params=BUCK_PARAMS, bounds=BUCK_BOUNDS, baseline=BUCK_BASELINE, param_labels=BUCK_PARAM_LABELS,
                        tuning_grid=BUCK_TUNING_GRID, evaluation_grid=BUCK_EVALUATION_GRID, methods=("pso", "bayes"), objective_text=BUCK_OBJECTIVE_TEXT,
                        evaluate=ev, scalar_objective=lambda x: ev(x)[0]["J"], subject=f"buck PI (BuckVoltageController) on {env_id}", note=spec.note or "")


def dc_motor_system() -> TuningSystem:
    from driveflow.optim import objectives as ob

    return TuningSystem(key=DC_MOTOR, plant_label="DC motor (native:dc_perm_ex_v1)", controller_label="PI cascade — speed loop (PICascadeController)",
                        plant_profile_id="native:dc_perm_ex_v1", params=ob.PARAMS, bounds=ob.BOUNDS, baseline=ob.BASELINE, param_labels={},
                        tuning_grid=ob.TUNING_GRID, evaluation_grid=ob.EVALUATION_GRID, methods=("pso", "bayes", "nsga2"),
                        objective_text="J = mean overshoot % + 0.1 × mean settling time (ms) + 2 × mean load-step dip (rad/s)",
                        evaluate=ob.evaluate, scalar_objective=ob.scalar_objective, subject="dc_motor PI speed loop (PICascadeController)")


def available() -> dict:
    """{key: label} of every system that can be tuned here, the DC motor first."""
    from driveflow.envs import registry

    out = {DC_MOTOR: "DC motor PI speed loop (native:dc_perm_ex_v1)"}
    for env_id in registry.rl_env_ids():
        if registry.rl_kind(env_id) == "buck":
            out[BUCK_PREFIX + env_id] = f"Buck PI on {env_id} ({registry.ENVIRONMENTS[env_id].source})"
    return out


def get(key: str) -> TuningSystem:
    if key == DC_MOTOR:
        return dc_motor_system()
    if key.startswith(BUCK_PREFIX):
        from driveflow.envs import registry

        env_id = key[len(BUCK_PREFIX):]
        registry.register_all()
        if registry.rl_kind(env_id) != "buck" or not registry.is_registered(env_id):
            raise KeyError(f"{env_id} is not a registered, available buck environment")
        return buck_system(env_id)
    raise KeyError(f"unknown tuning system {key!r}")
