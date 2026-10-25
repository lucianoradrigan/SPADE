"""B4 (round PLECS/OpenModelica): FmuGymEnv and SPADE/OmBuck-v0.

Tests marked fmu need fmpy and data/fmu/BuckAveraged.fmu (or $SPADE_FMU_DIR) and are skipped
without them. The last tests have no marker: without the FMU the environment must not be offered.
"""

import numpy as np
import pytest

from driveflow.control.rl import benchmark as bm
from driveflow.envs import fmu_env, registry

OM = "SPADE/OmBuck-v0"


def _need_fmu():
    pytest.importorskip("fmpy")
    ok, reason = fmu_env.availability()
    if not ok:
        pytest.skip(reason)


def _pi_trajectory(env_id, actions=None, seed=0):
    """(actions, [v_out, i_L, reward] per step); actions=None runs the native PI."""
    env = registry.make(env_id)
    obs, _ = env.reset(seed=seed)
    act = bm.buck_pi(env.unwrapped) if actions is None else None
    out, used = [], []
    for k in range(env.unwrapped.n_steps):
        a = act(obs) if actions is None else actions[k]
        used.append(a)
        obs, r, term, trunc, info = env.step(a)
        out.append((info["v_out_v"], info["i_l_a"], r))
        if term or trunc:
            break
    env.close()
    return used, np.array(out), term, trunc


@pytest.mark.fmu
def test_check_env():
    _need_fmu()
    from gymnasium.utils.env_checker import check_env

    env = registry.make(OM)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.fmu
def test_native_pi_actions_stay_within_b3_criterion_4():
    """Criterion 4 of B3: RMS(FMU - native) <= 1 % of the native range, on v_out and i_L."""
    _need_fmu()
    actions, native, _, _ = _pi_trajectory("SPADE/Buck-v0")
    _, om, term, trunc = _pi_trajectory(OM, actions)
    assert not term and trunc and len(om) == len(native)
    for j in (0, 1):
        nrms = np.sqrt(np.mean((om[:, j] - native[:, j]) ** 2)) / np.ptp(native[:, j])
        assert nrms <= 0.01, (j, nrms)


@pytest.mark.fmu
def test_reset_is_reproducible_bit_for_bit():
    _need_fmu()
    env = registry.make(OM, vout_ref_v=None, load_resistance_ohm=None)
    traces = []
    for _ in range(2):
        env.action_space.seed(5)
        obs, info = env.reset(seed=11)
        trace = [obs, np.array([info["v_ref_v"], info["load_resistance_ohm"]])]
        for _ in range(200):
            obs, r, *_ = env.step(env.action_space.sample())
            trace.append(np.append(obs, r))
        traces.append(np.concatenate(trace))
    env.close()
    assert np.array_equal(traces[0], traces[1])


@pytest.mark.fmu
def test_solver_failure_terminates_without_success_reward(monkeypatch):
    _need_fmu()
    env = registry.make(OM)
    u = env.unwrapped
    u.reset(seed=0)
    _, r_ok, *_ = u.step([0.5])

    def boom(**_kw):
        raise RuntimeError("simulated CVODE failure")

    monkeypatch.setattr(u.adapter._slave, "doStep", boom)
    _, r, term, trunc, info = u.step([0.5])
    assert term and not trunc
    assert info["termination"] == "solver_failure" and "simulated CVODE failure" in info["solver_error"] and info["solver_error_code"] == "solver"
    assert r == -u.solver_failure_penalty and r < r_ok
    env.close()


@pytest.mark.fmu
def test_out_of_range_action_policy():
    _need_fmu()
    env = registry.make(OM, action_out_of_range="reject")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="outside"):
        env.step([1.5])
    env.close()
    env = registry.make(OM)
    env.reset(seed=0)
    *_, info = env.step([1.5])
    assert info["action_clipped"] and info["action_applied"][0] == 1.0
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan])
    env.close()


def test_without_the_fmu_the_environment_is_not_offered(tmp_path, monkeypatch):
    from driveflow.viz.adapters import environments_adapter as ea
    from driveflow.viz.views.environments import table_rows

    monkeypatch.setenv("SPADE_FMU_DIR", str(tmp_path))
    try:
        ok, reason = registry.availability(OM)
        assert not ok and ("not built" in reason or "fmpy is not installed" in reason)
        registry.register_all()
        assert not registry.is_registered(OM)
        with pytest.raises(RuntimeError, match="not available"):
            registry.make(OM)
        c = ea.contract(OM)
        assert c["unavailable"] == reason and ea.snippet(OM) is None
        row = next(r for r in table_rows([c], []) if r["Environment ID"] == OM)
        assert row["Status"] == f"Not available: {reason}" and row["Source"] == "SIMULATED · OPENMODELICA"
    finally:
        monkeypatch.undo()
        registry.register_all()


def test_only_the_buck_fmu_environment_is_registered():
    """G1/G2: no SPADE/OmDcMotor-v0 nor the switching models."""
    om = {e for e, s in registry.ENVIRONMENTS.items() if s.source == "SIMULATED · OPENMODELICA"}
    assert om == {OM}
    assert "replicates the native averaged buck" in registry.ENVIRONMENTS[OM].note.lower()
