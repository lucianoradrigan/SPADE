"""B3 (2026-11-16): physics invariants of the native plants (simulated). Criteria and tolerances were committed before any
measurement (docs/audit/physics/invariants_criteria.md); results in docs/audit/physics/invariants.md.

A violation in a frozen v1 plant is a strict xfail citing S-21 (GEM's one-quadrant converter injects energy when braking at
light load). No v2 plant violates them."""

import numpy as np
import pandas as pd
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import PMG132_MOTOR_PARAMETER, PMG132_NOMINAL_VALUES, TAU, scaled_dc_load_parameter
from driveflow.sim.motors.dc_permanently_excited_motor import DcPermanentlyExcitedMotor

pytestmark = pytest.mark.simulation

STOCK_MP = dict(DcPermanentlyExcitedMotor().motor_parameter)
STOCK_LOAD = dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)  # runner.build_plant, dc_perm_ex_v1 stock load
PMG132_LOAD = scaled_dc_load_parameter(PMG132_NOMINAL_VALUES["torque"], PMG132_NOMINAL_VALUES["omega"])
PARAMS = {"dc_perm_ex_v1": (STOCK_MP, STOCK_LOAD), "dc_perm_ex_v2": (STOCK_MP, STOCK_LOAD),
          "dc_pmg132_v1": (PMG132_MOTOR_PARAMETER, PMG132_LOAD), "dc_pmg132_v2": (PMG132_MOTOR_PARAMETER, PMG132_LOAD)}
S21 = pytest.mark.xfail(strict=True, reason="S-21: GEM's one-quadrant converter (frozen v1 plant) applies the full supply when the current is negative and injects energy at light load")
PLANTS = ["dc_perm_ex_v1", "dc_perm_ex_v2", pytest.param("dc_pmg132_v1", marks=S21), "dc_pmg132_v2"]


def _run(plant, omega_ref, duration):
    df = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="b3_invariants", plant_config_id=plant, controller_type="PI", omega_ref_rad_s=omega_ref,
                                                         duration_s=duration, seed=0, electrical_noise_pct=0.0, mechanical_noise_pct=0.0)))
    return df["current_r"].to_numpy(float), df["voltage_v"].to_numpy(float), df["rpm"].to_numpy(float) * 2 * np.pi / 60


def _t_load(w, ld):
    return np.sign(w) * (ld["c"] * w ** 2 + ld["b"] * np.abs(w) + ld["a"])


@pytest.mark.parametrize("plant", PLANTS)
def test_i1_energy_balance_at_start_up(plant):
    """∫u·i dt = ∫R·i² dt + ∫T_L·ω dt + ½J·ω² + ½L·i², start-up 0 → 150 rad/s in 0.5 s: residual ≤ 2 %."""
    mp, ld = PARAMS[plant]
    i, u, w = _run(plant, 150.0, 0.5)
    e_in = np.sum(u * i) * TAU
    e_out = np.sum(mp["r_a"] * i ** 2) * TAU + np.sum(_t_load(w, ld) * w) * TAU + 0.5 * (mp["j_rotor"] + ld["j_load"]) * w[-1] ** 2 + 0.5 * mp["l_a"] * i[-1] ** 2
    assert abs(e_in - e_out) / e_in <= 0.02, (e_in, e_out)


@pytest.mark.parametrize("plant", ["dc_perm_ex_v1", "dc_perm_ex_v2", "dc_pmg132_v1", "dc_pmg132_v2"])
def test_i2_dc_steady_state(plant):
    """Last 20 % of a 1.0 s run at 150 rad/s: u = R·i + ψ·ω and ψ·i = T_L(ω), each within 1 %."""
    mp, ld = PARAMS[plant]
    i, u, w = _run(plant, 150.0, 1.0)
    n = len(i) // 5
    iw, uw, ww = i[-n:].mean(), u[-n:].mean(), w[-n:].mean()
    assert abs(uw - (mp["r_a"] * iw + mp["psi_e"] * ww)) <= 0.01 * abs(uw)
    assert abs(mp["psi_e"] * iw - _t_load(ww, ld)) <= 0.01 * abs(_t_load(ww, ld))


@pytest.mark.parametrize("plant", PLANTS)
def test_i3_no_energy_created_when_braking_at_light_load(plant):
    """The S-21 case: 120 rad/s, 0.7 s, no extra load. The speed must not end above 1.02 × the reference, and a
    one-quadrant converter cannot drive the armature current below −1 A."""
    i, u, w = _run(plant, 120.0, 0.7)
    assert w[-1] <= 1.02 * 120.0 and i.min() >= -1.0, (w[-1], i.min())


def test_i4_pmsm_torque_matches_the_dq_formula():
    """The plant's own torque state equals 1.5·p·(ψ_p·i_q + (L_d − L_q)·i_d·i_q) at every step (relative error ≤ 1e-6)."""
    from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS as mp, DqCurrentController, build_pmsm_system, mtpa_id_iq

    system = build_pmsm_system()
    system.seed(np.random.SeedSequence(0))
    state = system.reset()
    ctl = DqCurrentController(system)
    ctl.reset()
    worst, n = 0.0, 0
    for i_s in (50.0, 150.0, 300.0):
        i_d_ref, i_q_ref = mtpa_id_iq(i_s)
        for _ in range(400):
            action, _ = ctl.control(state, system, i_d_ref, i_q_ref)
            state = system.simulate(action)
            ph = state * system.limits
            i_d, i_q, t = float(ph[system.CURRENTS_IDX[3]]), float(ph[system.CURRENTS_IDX[4]]), float(ph[system.TORQUE_IDX])
            if abs(t) > 1.0:
                worst = max(worst, abs(t - 1.5 * mp["p"] * (mp["psi_p"] * i_q + (mp["l_d"] - mp["l_q"]) * i_d * i_q)) / abs(t))
                n += 1
    assert n > 1000 and worst <= 1e-6, (n, worst)
