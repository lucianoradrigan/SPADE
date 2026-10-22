"""Task A1 (round PLECS/OM, route a): PLECS sweep → dataset → surrogate → SPADE/<Profile>Surrogate-v0,
end to end against the fake PLECS server in its excited-buck TEST DOUBLE mode. This checks the
mechanics, not fidelity to PLECS."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from driveflow.envs import registry, surrogate_plecs
from driveflow.integrations.plecs import batch
from driveflow.integrations.plecs import excitation as ex
from driveflow.integrations.plecs import manifest as mf
from driveflow.jobs.service import JobService
from driveflow.jobs.worker import Worker
from driveflow.runs.schemas import ExperimentSpec
from driveflow.runs.store import RunStore
from driveflow.sim.buck_system import TAU
from fake_plecs import FakePlecs

MAN = mf.from_dict(mf.EXAMPLE_BUCK_SURROGATE)
SPAN = 0.02


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_PLECS_DATA", str(tmp_path / "plecs_data"))
    monkeypatch.setenv("SPADE_SURROGATE_DATA", str(tmp_path / "surrogates"))
    yield
    for env_id in [k for k, v in registry.ENVIRONMENTS.items() if v.plant_profile_id.startswith("surrogate:plecs:")]:
        registry.ENVIRONMENTS.pop(env_id)
        gym.registry.pop(env_id, None)


@pytest.fixture
def circuit(tmp_path):
    p = tmp_path / "buck_excited.plecs"
    p.write_text("Plecs { Name \"buck_excited\" }\n")
    return p


def _sweep(root, circuit, fake, fail_if=None):
    profile = RunStore(root).save_profile(batch.create_profile("buck_excited", circuit, MAN, rpc_url=fake.url))
    conds = batch.design_conditions(MAN, {"Vin": (24, 48, 3), "Rload": (3, 12, 3), "prbs_seed": (5, 77, 2)}, "grid",
                                    fixed={"D_min": 0.1, "D_max": 0.8, "T_hold": 0.001})
    s = JobService(root)
    spec = ExperimentSpec.create("plecs_sweep", plant_profile_id=profile.profile_id, conditions=conds, seed=0, algorithm="grid",
                                 params={"time_span_s": SPAN, "time_limit_s": 30.0, "n_output_points": int(round(SPAN / TAU)) + 1})
    job = s.submit(spec)
    Worker(root).serve(once=True, poll_s=0.05)
    return s, s.run_for(s.get(job.job_id))


# ----------------------------------------------------------------------------- excitation
def test_prbs_reference_is_deterministic_and_two_level():
    t = np.arange(0, 0.05, TAU)
    a = ex.sequence("prbs", {"prbs_seed": 5, "D_min": 0.1, "D_max": 0.8, "T_hold": 0.001}, t)
    b = ex.sequence("prbs", {"prbs_seed": 5, "D_min": 0.1, "D_max": 0.8, "T_hold": 0.001}, t)
    c = ex.sequence("prbs", {"prbs_seed": 6, "D_min": 0.1, "D_max": 0.8, "T_hold": 0.001}, t)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and set(np.unique(a)) == {0.1, 0.8}
    # constant within a hold
    k = np.floor(t / 0.001 + 1e-9).astype(int)
    assert all(len(set(a[k == j])) == 1 for j in range(10))
    with pytest.raises(ex.ExcitationError):
        ex.prbs7_bits(0, 3)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.pop("excitation"), "needs an 'excitation' block"),
    (lambda d: d["excitation"].update(action_output="nope"), "action_output"),
    (lambda d: d["excitation"]["params"].update(T_hold="Thold_missing"), "not declared"),
    (lambda d: d["surrogate"]["states"].update(i_l="current"), "surrogate.states"),
])
def test_manifest_blocks_are_validated(mutate, msg):
    d = json.loads(json.dumps(mf.EXAMPLE_BUCK_SURROGATE))
    mutate(d)
    with pytest.raises(mf.ManifestError, match=msg):
        mf.from_dict(d)


# ----------------------------------------------------------------------------- end to end
def test_sweep_to_surrogate_to_gym_env(tmp_path, circuit):
    root = tmp_path / "spade_workspace"
    with FakePlecs() as fake:
        s, sweep = _sweep(root, circuit, fake)
    assert sweep.status == "completed" and len(sweep.conditions) == 18
    d = batch.data_root() / "buck_excited" / sweep.run_id
    df = pd.read_parquet(d / sweep.conditions[0]["file"])
    assert {"duty", "v_out", "i_L", "prbs_seed"} <= set(df.columns)  # the applied action is a column
    meta, recs = surrogate_plecs.load_dataset(d)
    assert all(r["consistency"]["max_abs_error"] == 0.0 for r in recs)

    spec = ExperimentSpec.create("surrogate_fit", seed=0, params={"profile_id": "buck_excited", "sweep_run_id": sweep.run_id, "epochs": 3})
    job = s.submit(spec)
    Worker(root).serve(once=True, poll_s=0.05)
    run = s.run_for(s.get(job.job_id))
    assert run.origin == "SIMULATED · SURROGATE" and run.metrics["check_env_passed"], run.metrics
    env_id = run.metrics["env_id"]
    assert env_id == "SPADE/BuckExcitedSurrogate-v0"
    prov = run.metrics["provenance"]
    assert prov["sweep_run_id"] == sweep.run_id and prov["circuit_sha256"] == meta["circuit_sha256"] and prov["plecs_profile"] == "buck_excited"
    # split by operating point: both seeds of a point share a split; every split is populated
    assert env_id in surrogate_plecs.published()
    sur = json.loads((Path(run.metrics["surrogate_dir"]) / "surrogate.json").read_text())
    by_point = {}
    for r in sur["rows"]:
        by_point.setdefault((r["v_in_v"], r["load_resistance_ohm"]), set()).add(r["split"])
    assert all(len(v) == 1 for v in by_point.values()) and {s for v in by_point.values() for s in v} == {"train", "validation", "test"}
    # the ZOH grey box recovers the averaged buck the double simulates (same model, no noise)
    z = sur["families"]["zoh"]["fit"]
    assert z["l_h"] == pytest.approx(220e-6, rel=1e-3) and z["c_f"] == pytest.approx(220e-6, rel=1e-3)
    # a later process sees the published environment
    registry.ENVIRONMENTS.pop(env_id, None)
    del gym.registry[env_id]
    registry.register_all()
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.unwrapped.metadata["origin"] == "SIMULATED · SURROGATE" and env.unwrapped.metadata["surrogate_provenance"]["sweep_run_id"] == sweep.run_id
    env.close()


def test_a_sweep_without_excitation_cannot_feed_a_surrogate(tmp_path):
    p = tmp_path / "plain.plecs"
    p.write_text("x")
    root = tmp_path / "spade_workspace"
    plain = mf.from_dict(mf.EXAMPLE_BUCK)
    with FakePlecs() as fake:
        profile = RunStore(root).save_profile(batch.create_profile("plain", p, plain, rpc_url=fake.url))
        s = JobService(root)
        sweep_job = s.submit(ExperimentSpec.create("plecs_sweep", plant_profile_id="plain", conditions=[{"Vin": 24.0, "Rload": 6.0, "Vref": 12.0}] * 3, seed=0,
                                                   algorithm="grid", params={"time_span_s": 0.01, "time_limit_s": 5.0}))
        Worker(root).serve(once=True, poll_s=0.05)
    sweep = s.run_for(s.get(sweep_job.job_id))
    job = s.submit(ExperimentSpec.create("surrogate_fit", seed=0, params={"profile_id": "plain", "sweep_run_id": sweep.run_id}))
    Worker(root).serve(once=True, poll_s=0.05)
    run = s.run_for(s.get(job.job_id))
    assert run.status == "failed" and "excitation" in run.conditions[0]["error"]


def test_wrong_sample_spacing_is_refused_not_resampled(tmp_path, circuit):
    root = tmp_path / "spade_workspace"
    with FakePlecs() as fake:
        profile = RunStore(root).save_profile(batch.create_profile("buck_excited", circuit, MAN, rpc_url=fake.url))
        s = JobService(root)
        conds = batch.design_conditions(MAN, {"Vin": (24, 48, 3)}, "grid", fixed={"prbs_seed": 5.0})
        job = s.submit(ExperimentSpec.create("plecs_sweep", plant_profile_id=profile.profile_id, conditions=conds, seed=0, algorithm="grid",
                                             params={"time_span_s": 0.01, "time_limit_s": 5.0}))
        Worker(root).serve(once=True, poll_s=0.05)
    run = s.run_for(s.get(job.job_id))
    with pytest.raises(surrogate_plecs.SurrogateSourceError, match="exactly every"):
        surrogate_plecs.load_dataset(batch.data_root() / "buck_excited" / run.run_id)
