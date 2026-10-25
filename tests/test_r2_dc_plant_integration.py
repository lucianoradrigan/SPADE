"""R2.1 (round PLECS/OpenModelica, decision G3): optional integration of the native DC plant.

The default must stay the production plant bit for bit; the option is validated like every other
Scenario field (S-03); and finding S-17 (Euler substeps evaluate the right-hand side one control
period late in time) is pinned as a strict xfail.
"""

import numpy as np
import pandas as pd
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.export_parquet import records_to_dataframe
from driveflow.sim import EulerSolver


def _df(**kw):
    return records_to_dataframe(run_scenario(Scenario(scenario_id="r2", fault_type="outer_race", seed=3, duration_s=0.05, **kw)))


def test_default_and_one_substep_are_the_production_plant_bit_for_bit():
    base = _df()
    pd.testing.assert_frame_equal(base, _df(dc_plant_substeps=1), check_exact=True)
    pd.testing.assert_frame_equal(base, _df(dc_plant_solver="euler"), check_exact=True)


def test_finer_integrations_move_toward_rk45():
    ref = _df(dc_plant_solver="rk45")["current_r"].to_numpy()
    err = [np.sqrt(np.mean((_df(**kw)["current_r"].to_numpy() - ref) ** 2)) for kw in ({}, {"dc_plant_substeps": 10}, {"dc_plant_substeps": 100})]
    assert err[0] > err[1] > err[2] > 0


@pytest.mark.parametrize("kw, match", [
    ({"dc_plant_solver": "rk4"}, "dc_plant_solver must be one of"),
    ({"dc_plant_substeps": 0}, "dc_plant_substeps must be an int"),
    ({"dc_plant_substeps": 2.0}, "dc_plant_substeps must be an int"),
    ({"dc_plant_substeps": True}, "dc_plant_substeps must be an int"),
    ({"dc_plant_substeps": 10_001}, "dc_plant_substeps must be an int"),
    ({"dc_plant_solver": "rk45", "dc_plant_substeps": 10}, "applies only to dc_plant_solver='euler'"),
])
def test_invalid_integration_options_are_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        Scenario(scenario_id="bad", **kw)


def test_integration_options_apply_only_to_the_dc_plant():
    with pytest.raises(ValueError, match="only to plant_config_id='dc_perm_ex_v1'"):
        Scenario(scenario_id="bad", plant_config_id="buck_v1", dc_plant_substeps=10)


@pytest.mark.xfail(strict=True, reason="S-17: EulerSolver with nsteps > 1 evaluates the right-hand side at t_end + k*h/n instead of inside [t_start, t_end] (inherited from GEM)")
def test_s17_euler_substeps_evaluate_inside_the_step():
    seen = []

    def f(t, y):
        seen.append(t)
        return np.zeros_like(y)

    solver = EulerSolver(nsteps=4)
    solver.set_system_equation(f)
    solver.set_f_params()
    solver.set_initial_value(np.zeros(1), t=1.0)
    solver.integrate(1.1)
    assert all(1.0 - 1e-12 <= t <= 1.1 + 1e-12 for t in seen), seen
