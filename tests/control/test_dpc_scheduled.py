"""ScheduledDpcController (control/dpc/controller.py) -- TODO_CLAUDE_CODE.md C1, option 1;
docs/audit/stability/dpc_operating_point_selection.md."""

from pathlib import Path

import numpy as np
import pytest

from driveflow.control.dpc.controller import ScheduledDpcController
from driveflow.control.dpc.reference import RotatingReference
from driveflow.sim.vsc_system import VscSystem
from experiments.dpc_operating_point_selection import REGION, V3, V4

pytestmark = pytest.mark.control


def _sched(r=8.0064):
    return ScheduledDpcController(V3, V4, r, REGION)


def _pick(controller, magnitude_v):
    """Which checkpoint control() uses for one step at this setpoint."""
    state = VscSystem(load_resistance_ohm=controller.r).reset()
    controller.control(state, RotatingReference(magnitude_v=magnitude_v), 0)
    return "v3" if controller.using_nominal else "v4"


@pytest.mark.parametrize("r, mag, expected", [(8.0064, 50.0, "v3"), (8.0064, 10.0, "v3"), (8.0064, 75.0, "v4"),
                                               (4.0, 50.0, "v4"), (20.0, 50.0, "v4"), (11.0, 10.0, "v3"), (1000.0, 10.0, "v4")])
def test_region_selects_the_expected_checkpoint(r, mag, expected):
    assert _pick(_sched(r), mag) == expected


def test_hysteresis_keeps_the_choice_just_outside_the_boundary():
    c = _sched(8.0)
    assert _pick(c, 50.0) == "v3"
    c.set_load(REGION["r_max_ohm"] * 1.03)  # outside the band, inside the 5 % hysteresis
    assert _pick(c, 50.0) == "v3"
    c.set_load(20.0)
    assert _pick(c, 50.0) == "v4"
    c.set_load(REGION["r_min_ohm"] * 1.02)  # inside the band but within 5 % of its edge: stay on v4
    assert _pick(c, 50.0) == "v4"
    c.set_load(8.0064)
    assert _pick(c, 50.0) == "v3"
    assert c.switches == 2


def test_weights_exist():
    assert Path(V3).exists() and Path(V4).exists()


@pytest.mark.slow
def test_scheduled_controller_meets_the_c1_targets():
    # Measured 2026-10-20: 1.18 V at 8 ohm / 50 V, 1.75 V at 8 ohm / 10 V, 3.10 / 2.89 / 1.90 V at
    # 4 / 20 / 1000 ohm and 50 V (v4 alone: 2.52 and 4.24 V at the first two).
    def settled(r, mag):
        system, reference, c = VscSystem(load_resistance_ohm=r), RotatingReference(magnitude_v=mag), _sched(r)
        state, err = system.reset(), []
        for k in range(2000):
            state = system.simulate(*c.control(state, reference, k))
            vr = reference.at_step(k)
            err.append(np.hypot(state.vc_real - vr[0], state.vc_imag - vr[1]))
        return float(np.sqrt(np.mean(np.square(err[50:]))))

    assert settled(8.0064, 50.0) <= 1.3
    assert settled(8.0064, 10.0) <= 2.0
    for r in (4.0, 20.0, 1000.0):
        assert settled(r, 50.0) <= 3.5, r
