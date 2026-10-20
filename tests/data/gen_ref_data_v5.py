"""One-off script that records GEM 3.0.3's own classical controllers as reference data
(tests/data/ref_data_v5.npz) -- TODO_CLAUDE_CODE.md V5. Same pattern as gen_ref_data.py: not part of
driveflow, not run by the test suite; tests/control/test_gem_controller_comparison.py only reads the
.npz. GEM and gem_controllers come from the unmodified clone (../../gym-electric-motor/src).

1. DC speed loop: GemController.make(env, "Cont-SC-PermExDc-v0") -- GEM's cascaded PI (symmetric
   optimum, a = 4, EMF feedforward) -- on the same plant as driveflow.validation.build_dc_system:
   DcPermanentlyExcitedMotor defaults, one-quadrant converter, PolynomialStaticLoad(a=0.01, b=0.05,
   c=0, j_load=0.0025), 65 V supply, Euler, tau = 1e-4. Two references: the 2 rad/s step at
   150 rad/s used for S-04 (150 -> 152 at 0.4 s, 0.8 s) and a start-up step 0 -> 150 rad/s (0.5 s).
2. PMSM MTPA: GEM's own maximum-torque-per-current lookup table
   (PMSMOperationPointSelection._get_mtpc_lookup_table, 250 torques, 500 i_d candidates) for the
   PMSM parameters of control/classical/pmsm_foc.py, evaluated on a stand-in object that carries
   exactly the attributes that method reads.

Run with (gem_controllers 3.0.3 needs NumPy < 2: with NumPy 2 its DC operation-point stage gets
empty arrays and raises "truth value of an empty array is ambiguous" even on GEM's own default
"Cont-SC-PermExDc-v0" environment -- measured 2026-10-20):

    uv run --python 3.11 --with matplotlib --with "gymnasium>=0.29.1" --with "numpy<2" --with scipy \\
        --with control --with control-block-diagram --with ipython tests/data/gen_ref_data_v5.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gym-electric-motor" / "src"))

import numpy as np
import gym_electric_motor as gem
import gem_controllers as gc
from gem_controllers.stages.operation_point_selection.pmsm_ops import PMSMOperationPointSelection
from gym_electric_motor.physical_systems import converters as cv
from gym_electric_motor.physical_systems import mechanical_loads as ml
from gym_electric_motor.physical_systems import solvers as sv
from gym_electric_motor.physical_systems import voltage_supplies as vs

TAU = 1e-4
DC_LOAD = dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)
DC_SUPPLY_V = 65.0
PMSM_PARAMS = dict(p=3, l_d=0.37e-3, l_q=1.2e-3, r_s=18e-3, psi_p=66e-3)
PMSM_I_LIMIT = 400.0


def dc_run(reference_fn, duration_s):
    env_id = "Cont-SC-PermExDc-v0"
    env = gem.make(env_id, converter=cv.ContOneQuadrantConverter(), load=ml.PolynomialStaticLoad(load_parameter=DC_LOAD),
                   supply=vs.IdealVoltageSupply(u_nominal=DC_SUPPLY_V), ode_solver=sv.EulerSolver(), tau=TAU)
    controller = gc.GemController.make(env, env_id, block_diagram=False, should_plot=False, plot_references=False)
    ps = env.unwrapped.physical_system
    omega_idx, i_idx = ps.state_names.index("omega"), ps.state_names.index("i")
    omega_lim, i_lim = ps.limits[omega_idx], ps.limits[i_idx]
    (state, _ref), _ = env.reset(seed=0)
    controller.reset()
    n = int(round(duration_s / TAU))
    t = (np.arange(n) + 1) * TAU
    omega, i, ref = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n):
        ref[k] = reference_fn(k * TAU)
        action = controller.control(state, np.array([ref[k] / omega_lim]))
        (state, _r), _rew, term, _trunc, _info = env.step(action)
        omega[k], i[k] = state[omega_idx] * omega_lim, state[i_idx] * i_lim
        if term:
            raise RuntimeError(f"GEM terminated the episode at step {k} (limit violation)")
    return t, omega, i, ref


def gem_mtpc_table():
    ops = SimpleNamespace(p=PMSM_PARAMS["p"], l_d=PMSM_PARAMS["l_d"], l_q=PMSM_PARAMS["l_q"], psi_p=PMSM_PARAMS["psi_p"],
                          limit=np.array([PMSM_I_LIMIT, PMSM_I_LIMIT, 0.0]), i_sd_idx=0, i_sq_idx=1, torque_idx=2,
                          i_sd_limit=PMSM_I_LIMIT, i_sq_limit=PMSM_I_LIMIT, t_count=250, i_count=500)
    return PMSMOperationPointSelection._get_mtpc_lookup_table(ops)  # columns: torque, i_d, i_q, psi


def main():
    out = {"tau": TAU, "dc_load": str(DC_LOAD), "dc_supply_v": DC_SUPPLY_V}
    for name, fn, dur in (("small_step", lambda t: 150.0 if t < 0.4 else 152.0, 0.8), ("startup", lambda t: 150.0, 0.5)):
        t, omega, i, ref = dc_run(fn, dur)
        out[f"dc_{name}_t"], out[f"dc_{name}_omega"], out[f"dc_{name}_i"], out[f"dc_{name}_ref"] = t, omega, i, ref
        print(name, "final omega", omega[-1], "max i", i.max())
    out["pmsm_mtpc_table"] = gem_mtpc_table()
    out["pmsm_params"] = str(PMSM_PARAMS)
    path = Path(__file__).parent / "ref_data_v5.npz"
    np.savez(path, **out)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
