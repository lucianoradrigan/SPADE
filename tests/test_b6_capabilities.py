"""B6 (round PLECS/OM): RL Lab and Tuning Studio list environments by capability, not by name; the
buck PI can be tuned on any available buck environment; the RL grid has no load step on an FMU
environment."""

import pytest

from driveflow.control.rl import benchmark as bm
from driveflow.envs import fmu_env, registry
from driveflow.jobs.handlers import _validate_tuning
from driveflow.optim import systems
from driveflow.runs.schemas import ExperimentSpec, SpadeError


def test_rl_environments_come_from_capabilities():
    ids = registry.rl_env_ids()
    assert {"SPADE/Buck-v0", "SPADE/BuckSurrogate-v0", "SPADE/Vsc-v0"} <= set(ids)
    assert "SPADE/DcMotor-v0" not in ids and "SPADE/Pmsm-v0" not in ids  # no supported reference controller
    assert all(registry.rl_kind(e) in ("buck", "vsc") for e in ids)
    assert ("SPADE/OmBuck-v0" in ids) == fmu_env.availability()[0]


def test_without_the_fmu_neither_rl_nor_tuning_offers_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_FMU_DIR", str(tmp_path))
    try:
        assert "SPADE/OmBuck-v0" not in registry.rl_env_ids()
        assert systems.BUCK_PREFIX + "SPADE/OmBuck-v0" not in systems.available()
        with pytest.raises(KeyError):
            systems.get(systems.BUCK_PREFIX + "SPADE/OmBuck-v0")
    finally:
        monkeypatch.undo()
        registry.register_all()


def test_buck_grid_drops_the_load_step_only_where_unsupported():
    import gymnasium as gym

    env = gym.make("SPADE/Buck-v0")
    assert bm.buck_grid(env) is bm.BUCK_GRID
    env.close()
    assert all("load_step_ohm" not in c for c in bm.BUCK_GRID_NO_STEP) and len(bm.BUCK_GRID_NO_STEP) == len(bm.BUCK_GRID)


def test_buck_tuning_spec_is_validated_against_the_system():
    key = systems.BUCK_PREFIX + "SPADE/Buck-v0"
    good = dict(algorithm="pso", seed=0, params={"system": key, "bounds": [[1000.0, 10000.0], [200.0, 2500.0]]}, budget={"evaluations": 8})
    _validate_tuning(ExperimentSpec.create("controller_tuning", **good))
    for bad in ({"algorithm": "nsga2"}, {"params": {"system": key, "bounds": [[10.0, 20.0], [200.0, 2500.0]]}}, {"params": {"system": "buck_pi:SPADE/Nope-v0", "bounds": []}}):
        with pytest.raises(SpadeError):
            _validate_tuning(ExperimentSpec.create("controller_tuning", **{**good, **bad}))


def test_buck_tuning_job_on_the_native_buck_end_to_end(tmp_path):
    """A small PSO through the worker on SPADE/Buck-v0 (no FMU needed): a candidate, never a default."""
    import json

    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import RunStore

    key = systems.BUCK_PREFIX + "SPADE/Buck-v0"
    s = JobService()
    job = s.submit(ExperimentSpec.create("controller_tuning", algorithm="pso", seed=0, params={"system": key, "bounds": [[1000.0, 10000.0], [200.0, 2500.0]]},
                                         budget={"evaluations": 4}))
    Worker(s.root).serve(once=True)
    run = s.run_for(s.get(job.job_id))
    assert run.status == "completed", run.failure_reason
    art = next(a for a in run.artifacts if a["path"].endswith("tuning_report.json"))
    rep = json.loads(RunStore().artifact_path(art).read_text())
    assert rep["system"] == key and rep["status"].startswith("candidate") and len(rep["candidates"]) == 1
    assert rep["baseline"]["evaluation"]["J"] > 0 and len(rep["baseline"]["evaluation_rows"]) == len(systems.BUCK_EVALUATION_GRID)
    assert not rep["evaluation_reused_tuning_conditions"]


@pytest.mark.fmu
def test_tuning_studio_lists_the_fmu_buck_when_available():
    if not fmu_env.availability()[0]:
        pytest.skip("BuckAveraged FMU not available")
    assert systems.BUCK_PREFIX + "SPADE/OmBuck-v0" in systems.available()
    sysm = systems.get(systems.BUCK_PREFIX + "SPADE/OmBuck-v0")
    summary, rows = sysm.evaluate(sysm.baseline, sysm.tuning_grid)
    native, nrows = systems.get(systems.BUCK_PREFIX + "SPADE/Buck-v0").evaluate(sysm.baseline, sysm.tuning_grid)
    assert abs(summary["J"] - native["J"]) <= 0.01 * native["J"]  # the FMU replicates the native buck
