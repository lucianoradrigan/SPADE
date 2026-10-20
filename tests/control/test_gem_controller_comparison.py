"""SPADE's classical controllers against gym-electric-motor 3.0.3's own (gem_controllers) --
TODO_CLAUDE_CODE.md V5, docs/audit/upstream/gem_controller_comparison.md. Reference data:
tests/data/ref_data_v5.npz from tests/data/gen_ref_data_v5.py (GEM is never imported here)."""

from pathlib import Path

import numpy as np
import pytest

from driveflow.control.classical.pmsm_foc import mtpa_id_iq, torque_of
from driveflow.validation import run_dc_closed_loop, step_response_metrics

pytestmark = pytest.mark.control

REF = Path(__file__).resolve().parents[1] / "data" / "ref_data_v5.npz"
#: GEM builds its MTPC table on 500 i_d candidates over [-2.5 * 400 A, 0]: a 2.004 A grid.
GEM_GRID_HALF_STEP_A = 0.5 * 2.5 * 400.0 / 499


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip(f"{REF} not found -- regenerate with tests/data/gen_ref_data_v5.py")
    return np.load(REF)


def _spade_mtpa_for_torque(torque):
    lo, hi = 0.0, 1200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if torque_of(*mtpa_id_iq(mid)) < torque else (lo, mid)
    return mtpa_id_iq(hi)


def test_closed_form_mtpa_matches_gems_table_over_the_whole_torque_range(ref):
    # Measured 2026-10-20 over 2.9-716 N m: max |delta i_d| 1.00 A, max |delta i_q| 0.91 A, i.e.
    # within GEM's own grid resolution. Replaces the single-point check at nominal current.
    table = ref["pmsm_mtpc_table"]
    positive = table[table[:, 0] > 1.0]
    assert len(positive) >= 100
    for torque, i_d_gem, i_q_gem, _psi in positive:
        i_d, i_q = _spade_mtpa_for_torque(torque)
        assert abs(i_d - i_d_gem) <= GEM_GRID_HALF_STEP_A + 1e-6, torque
        assert abs(i_q - i_q_gem) <= GEM_GRID_HALF_STEP_A, torque


def test_gem_pi_reference_run_is_sane(ref):
    assert ref["dc_small_step_omega"][-1] == pytest.approx(152.0, abs=1e-6)
    assert ref["dc_startup_omega"][-1] == pytest.approx(150.0, abs=1e-6)


def test_spade_pi_overshoots_less_than_gems_on_the_small_step(ref):
    """Different tunings, not an equality check: GEM's symmetric optimum (a = 4) overshoots 33.6 %
    on the 2 rad/s step, SPADE's pole-placement PI 15.1 % (b = 1). Both reach zero steady-state
    error."""
    gem = step_response_metrics(ref["dc_small_step_t"], ref["dc_small_step_omega"], 152.0, 150.0, t_step=0.4)
    r = run_dc_closed_loop("PI", lambda t: 150.0 if t < 0.4 else 152.0, duration_s=0.8)
    spade = step_response_metrics(r["t"], r["omega"], 152.0, 150.0, t_step=0.4)
    assert gem.overshoot_pct == pytest.approx(33.6, abs=0.5)
    assert spade.overshoot_pct < gem.overshoot_pct
    assert gem.steady_state_error < 1e-6 and spade.steady_state_error < 1e-6
