"""Run/profile/approval contracts (driveflow.runs.schemas) -- expansion task E0."""

import pytest

from driveflow.runs.schemas import (ORIGINS, SCHEMA_VERSION, ApprovalRecord, EnvironmentCapability, ErrorCode, ExperimentSpec, PlantProfile, Run,
                                    SignalSeries, SpadeError)


def _run(**kw):
    base = dict(run_id="run-1", status="completed", origin="SIMULATED", kind="scenario_sweep", conditions=[{"index": 0, "condition": {"a": 1}, "status": "ok"}])
    return Run(**{**base, **kw})


@pytest.mark.parametrize("entity", [
    lambda: _run(metrics={"x": 1.5}, artifacts=[{"path": "runs/run-1/a.txt", "bytes": 3, "sha256": "0" * 64}]),
    lambda: ExperimentSpec.create("scenario_sweep", conditions=[{"duration_s": 0.1}], seed=3, params={"k": [1, 2]}),
    lambda: PlantProfile(profile_id="p", kind="plecs_rpc_batch", engine="plecs", display_name="Buck", file_hash="ab" * 32,
                         capabilities=EnvironmentCapability(can_simulate_batch=True, diagnostic_messages=["ok"])),
    lambda: SignalSeries(name="v_out", unit="V", run_id="run-1", origin="SIMULATED · PLECS", samples_ref="runs/run-1/a.parquet"),
    lambda: ApprovalRecord(approval_id="a1", subject="dpc default", baseline_run="run-0", candidate_run="run-1", gate_report={"blocking": []},
                           approver="Luciano", user_confirmed_at="2026-10-20T00:00:00+00:00", decision="approved"),
])
def test_round_trip_preserves_the_entity_and_records_the_version(entity):
    e = entity()
    d = e.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION and d["entity"] == type(e).__name__
    assert type(e).from_dict(d) == e


def test_newer_schema_or_wrong_entity_is_refused():
    d = _run().to_dict()
    with pytest.raises(SpadeError) as exc:
        Run.from_dict({**d, "schema_version": SCHEMA_VERSION + 1})
    assert exc.value.code == ErrorCode.SCHEMA_VERSION
    with pytest.raises(SpadeError):
        ExperimentSpec.from_dict(d)
    with pytest.raises(SpadeError):
        Run.from_dict({**d, "surprise": 1})


@pytest.mark.parametrize("bad", [dict(status="done"), dict(origin="LIVE"), dict(status="failed"),
                                 dict(conditions=[{"index": 0, "condition": {}, "status": "maybe"}]), dict(metrics={"x": object()})])
def test_invalid_runs_are_rejected(bad):
    with pytest.raises(SpadeError) as exc:
        _run(**bad)
    assert exc.value.code == ErrorCode.INVALID_SPEC


def test_live_is_not_a_provenance_label():
    assert not any("LIVE" in o for o in ORIGINS)


def test_capabilities_default_off_and_rl_needs_stepping():
    cap = EnvironmentCapability()
    assert not (cap.can_simulate_batch or cap.can_step_interactively or cap.can_reset_episode or cap.can_train_rl)
    with pytest.raises(SpadeError):
        EnvironmentCapability(can_train_rl=True, can_step_interactively=False)


def test_file_backed_profiles_need_a_hash():
    with pytest.raises(SpadeError):
        PlantProfile(profile_id="p", kind="plecs_rpc_batch", engine="plecs", display_name="x")


def test_config_hash_ignores_id_and_time_but_not_content():
    a = ExperimentSpec.create("scenario_sweep", conditions=[{"duration_s": 0.1}], seed=1)
    b = ExperimentSpec.create("scenario_sweep", conditions=[{"duration_s": 0.1}], seed=1)
    c = ExperimentSpec.create("scenario_sweep", conditions=[{"duration_s": 0.1}], seed=2)
    assert a.experiment_id != b.experiment_id and a.config_hash() == b.config_hash() != c.config_hash()
    assert len(a.config_hash()) == 64


@pytest.mark.parametrize("bad", [dict(approver=" "), dict(decision="maybe"), dict(gate_report={})])
def test_approval_needs_a_named_person_a_decision_and_a_gate_report(bad):
    base = dict(approval_id="a", subject="s", baseline_run=None, candidate_run="r", gate_report={"blocking": []}, approver="L",
                user_confirmed_at="t", decision="approved")
    with pytest.raises(SpadeError):
        ApprovalRecord(**{**base, **bad})
