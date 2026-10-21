"""Benchmark RL policies against SPADE's controllers on the SAME environment and condition grid --
expansion task E8 (spec Sec. 5 step 4).

Every controller runs through the same Gymnasium environment: the same plant, reset options,
limits, timestep, reward terms and metrics. The classical controllers read ``env.plant_state``
(and the reference the environment holds); a policy reads the observation.
- Buck (SPADE/Buck-v0): BuckVoltageController (the PI cascade of E6). Grid ``BUCK_GRID``:
  references and loads with a load step at half the episode.
- VSC (SPADE/Vsc-v0): DPC v3 (``dpc_trained_v3_closed_loop``) and v4 (``dpc_trained_v4_r_range``)
  alone, at 50 V, on the stable loads of experiments/measure_dpc_rmse_by_r_range.R_GRID (R >= 4 ohm;
  below R* every controller diverges, S-08).

Metrics per condition:
- the episode reward under the environment's reward terms;
- tracking RMSE in physical units over the second half (buck v_out − v_ref in V; VSC |v_c − v_ref| in V);
- the peak current;
- the termination reason (a divergence is a result, not an error).
"""

from __future__ import annotations

import math

import numpy as np

BUCK_GRID = (
    {"vout_ref_v": 24.0, "load_resistance_ohm": 6.0, "load_step_ohm": 3.0},
    {"vout_ref_v": 12.0, "load_resistance_ohm": 12.0, "load_step_ohm": 6.0},
    {"vout_ref_v": 30.0, "load_resistance_ohm": 8.0, "load_step_ohm": 16.0},
    {"vout_ref_v": 18.0, "load_resistance_ohm": 20.0, "load_step_ohm": 10.0},
)
VSC_GRID = tuple({"reference_magnitude_v": 50.0, "load_resistance_ohm": r, "phase0_rad": 0.0} for r in (4.0, 5.0, 8.0064, 10.0, 15.0, 20.0))


def buck_pi(env):
    from driveflow.control.classical.buck_pi import BuckVoltageController

    ctrl = BuckVoltageController(env.plant)
    return lambda obs: np.array([ctrl.control(env.plant_state, env.vout_ref_v)])


def vsc_dpc(env, which: str):
    from driveflow.control.dpc.controller import DpcController
    from driveflow.datagen.runner import _DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH

    ctrl = DpcController(_DPC_NOMINAL_WEIGHTS_PATH if which == "v3" else _DPC_WEIGHTS_PATH, env.load_resistance_ohm)
    ctrl.reset()
    return lambda obs: np.array(ctrl.control(env.plant_state, env.reference, env._k))


def episode(env, make_policy, options: dict, seed: int = 0) -> dict:
    """make_policy(env) is called after reset (so classical controllers see the episode's plant)."""
    obs, info = env.reset(seed=seed, options=options)
    act = make_policy(env.unwrapped)
    rewards, errs, currents, reason = [], [], [], None
    n = env.unwrapped.n_steps
    while True:
        obs, r, term, trunc, info = env.step(act(obs))
        rewards.append(float(r))
        if "v_out_v" in info:
            errs.append(info["v_out_v"] - info["v_ref_v"])
            currents.append(abs(info["i_l_a"]))
        else:
            errs.append(math.hypot(info["vc_real"] - info["v_ref_real"], info["vc_imag"] - info["v_ref_imag"]))
            currents.append(math.hypot(info["i_f_real"], info["i_f_imag"]))
        if term or trunc:
            reason = info.get("termination") if term else None
            break
    e = np.asarray(errs[len(errs) // 2:], dtype=float)
    return {"options": options, "reward": float(np.sum(rewards)), "steps": len(rewards), "complete": len(rewards) == n,
            "tracking_rmse_second_half": float(np.sqrt(np.mean(e ** 2))) if len(e) and np.all(np.isfinite(e)) else None,
            "peak_current": float(np.nanmax(currents)) if currents else None, "termination": reason}


def run_grid(env, controllers: dict, grid, seed: int = 0) -> dict:
    """{controller name: [episode result per condition]} -- the same grid for every controller."""
    return {name: [episode(env, mk, dict(opts), seed) for opts in grid] for name, mk in controllers.items()}


def summarize(results: dict) -> dict:
    out = {}
    for name, rows in results.items():
        rm = [r["tracking_rmse_second_half"] for r in rows if r["tracking_rmse_second_half"] is not None and r["termination"] is None]
        out[name] = {"mean_reward": float(np.mean([r["reward"] for r in rows])), "n_conditions": len(rows),
                     "n_terminated": sum(r["termination"] is not None for r in rows), "median_tracking_rmse": float(np.median(rm)) if rm else None}
    return out
