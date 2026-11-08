"""dc_perm_ex_v2 step 1, variant B (``dc_converter_model="dcm_exact"``): unit tests required by the
pre-registration (docs/audit/assistant/dc_perm_ex_v2_convergence_preregistration.md §5 item 1, amendment 1).
The convergence itself is measured by experiments/report_dc_perm_ex_v2_convergence.py, not here."""

import numpy as np
import pytest

from driveflow.datagen.runner import build_plant
from driveflow.datagen.scenario import Scenario
from driveflow.sim.dc_dcm import DcmExactDcMotorSystem


def _plant(**kw):
    return build_plant(Scenario("dcm_exact_test", **kw))


def test_scenario_accepts_dcm_exact_and_rejects_substeps_or_solver():
    assert isinstance(_plant(dc_converter_model="dcm_exact"), DcmExactDcMotorSystem)
    with pytest.raises(ValueError, match="dcm_exact"):
        Scenario("x", dc_converter_model="dcm_exact", dc_plant_substeps=10)
    with pytest.raises(ValueError, match="dcm_exact"):
        Scenario("x", dc_converter_model="dcm_exact", dc_plant_solver="rk45")


@pytest.mark.parametrize("i0, u, e0, e_dot", [(50.0, 30.0, 20.0, 0.0), (50.0, 30.0, 20.0, 150.0), (10.0, 5.0, 25.0, -80.0), (0.0, 40.0, 20.0, 300.0)])
def test_closed_form_current_and_charge_match_a_fine_integration(i0, u, e0, e_dot):
    p = _plant(dc_converter_model="dcm_exact")
    r, l = p._r, p._l
    n = 200_000
    h = p.tau / n
    i, q = i0, 0.0
    for k in range(n):  # midpoint rule on L di/dt = u - R i - (e0 + e_dot t), fine enough for 1e-6 relative
        t = (k + 0.5) * h
        im = i + 0.5 * h * (u - r * i - (e0 + e_dot * k * h)) / l
        i_new = i + h * (u - r * im - (e0 + e_dot * t)) / l
        q += 0.5 * h * (i + i_new)
        i = i_new
    assert p._current(p.tau, i0, u, e0, e_dot) == pytest.approx(i, rel=1e-6, abs=1e-6)
    assert p._charge(p.tau, i0, u, e0, e_dot) == pytest.approx(q, rel=1e-6, abs=1e-9)


def test_first_zero_is_the_analytic_crossing_with_a_constant_back_emf():
    p = _plant(dc_converter_model="dcm_exact")
    tau_e = p._l / p._r
    i0, u, e0 = 20.0, 0.0, 20.0  # i_ss = -e0/R < 0: crosses at t* = tau_e ln((i0 - i_ss)/(-i_ss))
    i_ss = (u - e0) / p._r
    t_star = tau_e * np.log((i0 - i_ss) / (-i_ss))
    assert t_star < p.tau
    assert p._first_zero(p.tau, i0, u, e0, 0.0) == pytest.approx(t_star, rel=1e-9)
    assert p._first_zero(p.tau, 500.0, 60.0, 20.0, 0.0) is None  # stays positive


def test_current_never_negative_at_duty_zero_and_the_motor_coasts():
    """The S-21 mechanism (GEM: negative current, the motor accelerates at duty 0) must not appear."""
    p = _plant(dc_converter_model="dcm_exact", plant_config_id="dc_pmg132_v1")
    p.reset()
    w_idx = p.state_names.index("omega")
    state = np.array(p._ode_solver.y, dtype=float)
    state[w_idx] = 150.0
    p._ode_solver.set_initial_value(state, p._t)
    i_idx = p.state_names.index("i")
    omegas, currents = [], []
    for _ in range(2000):
        s = p.simulate(np.array([0.0]))
        omegas.append(s[w_idx] * p.limits[w_idx])
        currents.append(s[i_idx] * p.limits[i_idx])
    assert min(currents) >= 0.0
    assert omegas[-1] <= 150.0 + 1e-9  # friction only: it can only slow down


def test_gem_and_dcm_paths_are_untouched():
    """Byte identity of None vs "gem" and of "dcm" is covered by tests/test_s21_dcm_converter.py; here:
    adding the value did not change what None / "gem" / "dcm" build."""
    from driveflow.sim.dc_dcm import DcmDcMotorSystem
    from driveflow.sim.scml_system import DcMotorSystem

    assert type(_plant()) is DcMotorSystem and type(_plant(dc_converter_model="gem")) is DcMotorSystem
    assert type(_plant(dc_converter_model="dcm")) is DcmDcMotorSystem
