"""PLECS batch route (expansion task E2): manifest, profile, sweep design, the plecs_sweep job in the
worker and the Scenario "plecs:<profile_id>" path -- all against the fake PLECS server."""

import json

import numpy as np
import pandas as pd
import pytest

from driveflow.datagen.runner import run_scenario
from driveflow.datagen.scenario import Scenario
from driveflow.integrations.plecs import batch
from driveflow.integrations.plecs import manifest as mf
from driveflow.jobs.service import JobService
from driveflow.jobs.worker import Worker
from driveflow.runs.schemas import ExperimentSpec, SpadeError
from driveflow.runs.store import RunStore
from fake_plecs import TAU_S, FakePlecs


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_PLECS_DATA", str(tmp_path / "plecs_data"))


@pytest.fixture
def circuit(tmp_path):
    p = tmp_path / "buck_demo.plecs"
    p.write_text("Plecs { Name \"buck_demo\" }\n")
    return p


MANIFEST = mf.from_dict(mf.EXAMPLE_BUCK)


# ----------------------------------------------------------------------------- manifest
@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.update(outputs=[]), "at least one output"),
    (lambda d: d["model_vars"][0].update(unit=""), "unit is required"),
    (lambda d: d["model_vars"][0].update(minimum=200.0), "minimum must be below maximum"),
    (lambda d: d["model_vars"][0].update(default=1.0), "outside"),
    (lambda d: d["model_vars"].append(dict(d["model_vars"][0])), "duplicate"),
    (lambda d: d["outputs"].append({"name": "timestamp_s", "unit": "s"}), "reserved"),
    (lambda d: d["outputs"].append({"name": "Vin", "unit": "V"}), "both as ModelVar and output"),
    (lambda d: d["model_vars"][0].update(name="2bad"), "not a valid variable name"),
])
def test_manifest_validation(mutate, msg):
    d = json.loads(json.dumps(mf.EXAMPLE_BUCK))
    mutate(d)
    with pytest.raises(mf.ManifestError, match=msg):
        mf.from_dict(d)


def test_manifest_round_trip_and_value_checks():
    m = mf.from_json(MANIFEST.to_json())
    assert m == MANIFEST and m.output_names == ["v_out", "i_L"]
    assert m.check_values({"Vin": 30}) == {"Vin": 30.0}
    with pytest.raises(mf.ManifestError, match="outside its declared range"):
        m.check_values({"Vin": 500.0})
    with pytest.raises(mf.ManifestError, match="not a ModelVar"):
        m.check_values({"Lf": 1e-3})


# ----------------------------------------------------------------------------- profile and design
def test_profile_records_hash_and_is_stale_when_the_circuit_changes(circuit):
    from driveflow.runs.profiles import is_stale

    p = batch.create_profile("buck_demo", circuit, MANIFEST, rpc_url="http://localhost:1080")
    assert p.kind == "plecs_rpc_batch" and p.origin == "SIMULATED · PLECS" and len(p.file_hash) == 64
    assert not p.capabilities.can_simulate_batch and not p.capabilities.can_step_interactively and p.status == "imported"
    assert not is_stale(p, circuit)
    circuit.write_text("Plecs { Name \"buck_demo\" } // edited\n")
    assert is_stale(p, circuit)


@pytest.mark.parametrize("pid", ["Buck Demo", "x", "-lead", "a" * 70])
def test_profile_id_is_checked(circuit, pid):
    with pytest.raises(ValueError, match="profile id"):
        batch.create_profile(pid, circuit, MANIFEST)


def test_grid_and_lhs_designs():
    grid = batch.design_conditions(MANIFEST, {"Vin": (24, 48, 3), "Rload": (2, 10, 3)}, "grid", fixed={"Vref": 12.0})
    assert len(grid) == 9 and {c["Vref"] for c in grid} == {12.0} and sorted({c["Vin"] for c in grid}) == [24.0, 36.0, 48.0]
    a = batch.design_conditions(MANIFEST, {"Vin": (24, 48), "Rload": (2, 10)}, "lhs", n_samples=10, seed=7)
    b = batch.design_conditions(MANIFEST, {"Vin": (24, 48), "Rload": (2, 10)}, "lhs", n_samples=10, seed=7)
    c = batch.design_conditions(MANIFEST, {"Vin": (24, 48), "Rload": (2, 10)}, "lhs", n_samples=10, seed=8)
    assert a == b and a != c and len(a) == 10
    # Latin hypercube: exactly one sample per stratum of each variable
    strata = sorted(int((x["Vin"] - 24) / 24 * 10) for x in a)
    assert strata == list(range(10))
    with pytest.raises(mf.ManifestError, match="outside"):
        batch.design_conditions(MANIFEST, {"Vin": (5, 48, 2)})
    with pytest.raises(ValueError, match="budget"):
        batch.design_conditions(MANIFEST, {"Vin": (12, 100, 50), "Rload": (1, 50, 50)})


# ----------------------------------------------------------------------------- the job
def _submit(store_root, profile, conditions, **params):
    s = JobService(store_root)
    spec = ExperimentSpec.create("plecs_sweep", plant_profile_id=profile.profile_id, conditions=conditions, seed=0, algorithm="grid",
                                 params={"time_span_s": 0.01, "time_limit_s": 5.0, **params})
    job = s.submit(spec)
    Worker(store_root).serve(once=True, poll_s=0.05)
    return s, s.get(job.job_id)


def test_sweep_in_the_worker_records_failures_and_each_row_carries_its_own_parameters(tmp_path, circuit, monkeypatch):
    root = tmp_path / "spade_workspace"
    with FakePlecs(fail_if=lambda mv: "Solver did not converge" if mv["Rload"] < 2 else None) as fake:
        profile = RunStore(root).save_profile(batch.create_profile("buck_demo", circuit, MANIFEST, rpc_url=fake.url))
        conds = batch.design_conditions(MANIFEST, {"Vref": (6, 18, 3), "Rload": (1.5, 9, 2)}, "grid")
        s, job = _submit(root, profile, conds)
        assert ("close", "buck_demo") in fake.calls and fake.calls[0][0] == "load"
    run = s.run_for(job)
    assert run.status == "partially_completed" and run.origin == "SIMULATED · PLECS"
    failed = [c for c in run.conditions if c["status"] == "failed"]
    assert len(failed) == 3 and all("[simulation_failed]" in c["error"] and c["condition"]["Rload"] == 1.5 for c in failed)
    out = batch.data_root() / "buck_demo" / run.run_id
    table = pd.read_csv(out / "conditions.csv")
    assert list(table["status"]).count("failed") == 3 and table.loc[table.status == "failed", "file"].isna().all()  # never interpolated
    man = json.loads((out / "manifest.json").read_text())
    assert man["circuit_sha256"] == profile.file_hash and man["origin"] == "SIMULATED · PLECS" and "does not provide an interactive" in man["note"]
    # Regression for the Fund. notebook defect: every row has its own combination's parameters.
    for c in run.conditions:
        if c["status"] != "ok":
            continue
        df = pd.read_parquet(out / c["file"])
        for k, v in c["condition"].items():
            assert (df[k] == v).all()
        expected = c["condition"]["Vref"] * (1 - np.exp(-df["timestamp_s"] / TAU_S))
        np.testing.assert_allclose(df["v_out"], expected, rtol=1e-12)
    params = {(c["condition"]["Vref"], c["condition"]["Rload"]) for c in run.conditions if c["status"] == "ok"}
    assert len(params) == 3
    newest = RunStore(root).get_profile("buck_demo")
    assert newest.version == 2 and newest.capabilities.can_simulate_batch and not newest.capabilities.can_step_interactively
    assert run.run_id in newest.capabilities.diagnostic_messages[0]


def test_timeouts_and_a_dropped_connection_are_recorded_per_condition(tmp_path, circuit):
    root = tmp_path / "spade_workspace"
    with FakePlecs(delay_s=lambda mv: 1.0 if mv["Vref"] == 10 else 0.0, drop_after=3) as fake:
        profile = RunStore(root).save_profile(batch.create_profile("buck_demo", circuit, MANIFEST, rpc_url=fake.url))
        conds = [{"Vref": v} for v in (5, 10, 15, 20, 25)]
        s, job = _submit(root, profile, conds, time_limit_s=0.3)
    run = s.run_for(job)
    by_v = {c["condition"]["Vref"]: c for c in run.conditions}
    assert by_v[5]["status"] == "ok" and "[timeout]" in by_v[10]["error"] and by_v[15]["status"] == "ok"
    assert all(by_v[v]["status"] == "failed" for v in (20, 25))
    assert any(code in by_v[25]["error"] for code in ("[unreachable]", "[connection_lost]"))
    assert run.status == "partially_completed"


def test_stale_profile_refuses_to_sweep(tmp_path, circuit):
    root = tmp_path / "spade_workspace"
    with FakePlecs() as fake:
        profile = RunStore(root).save_profile(batch.create_profile("buck_demo", circuit, MANIFEST, rpc_url=fake.url))
        circuit.write_text("changed")
        s, job = _submit(root, profile, [{"Vref": 5.0}])
        assert not any(c[0] == "simulate" for c in fake.calls)
    run = s.run_for(job)
    assert run.status == "failed" and "Re-run verification" in run.conditions[0]["error"]


def test_spec_validation(tmp_path, circuit):
    root = tmp_path / "spade_workspace"
    profile = RunStore(root).save_profile(batch.create_profile("buck_demo", circuit, MANIFEST))
    s = JobService(root)
    bad = [({"conditions": [{"Vin": 500.0}], "params": {"time_span_s": 0.01}}, "outside"),
           ({"conditions": [{"Vin": 30.0}], "params": {"time_span_s": -1}}, "simulated time"),
           ({"conditions": [], "params": {"time_span_s": 0.01}}, "at least one condition")]
    for fields, msg in bad:
        with pytest.raises(SpadeError, match=msg):
            s.submit(ExperimentSpec.create("plecs_sweep", plant_profile_id=profile.profile_id, **fields))


# ----------------------------------------------------------------------------- Scenario integration
def test_scenario_plecs_plant_runs_through_run_scenario(tmp_path, circuit):
    root = tmp_path / "spade_workspace"
    with FakePlecs() as fake:
        RunStore(root).save_profile(batch.create_profile("buck_demo", circuit, MANIFEST, rpc_url=fake.url))
        recs = run_scenario(Scenario(scenario_id="p1", plant_config_id="plecs:buck_demo", controller_type="circuit", duration_s=0.005,
                                     plecs_model_vars={"Vref": 9.0}, plecs_output_points=11))
    df = pd.DataFrame(recs)
    assert len(df) == 11 and (df["Vref"] == 9.0).all() and (df["Vin"] == 48.0).all()  # unset ModelVars use the manifest default
    assert (df["plant_config_id"] == "plecs:buck_demo").all() and df["timestamp_s"].iloc[-1] == pytest.approx(0.005)


@pytest.mark.parametrize("kwargs, err", [
    (dict(plant_config_id="plecs:buck_demo", controller_type="PI"), NotImplementedError),
    (dict(controller_type="circuit"), NotImplementedError),
    (dict(plant_config_id="plecs:Bad Id", controller_type="circuit"), ValueError),
    (dict(plecs_model_vars={"Vin": 1.0}), ValueError),
    (dict(plant_config_id="plecs:buck_demo", controller_type="circuit", plecs_model_vars={"Vin": float("inf")}), ValueError),
    (dict(plant_config_id="plecs:buck_demo", controller_type="circuit", plecs_output_points=1), ValueError),
])
def test_scenario_validation(kwargs, err):
    with pytest.raises(err):
        Scenario(scenario_id="x", **kwargs)
