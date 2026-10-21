"""Durable run store (driveflow.runs.store) -- expansion task E0."""

import threading

import pytest

from driveflow.runs.compare import compare_runs
from driveflow.runs.profiles import NATIVE_PROFILES, is_stale, mark_stale
from driveflow.runs.schemas import ApprovalRecord, ErrorCode, ExperimentSpec, PlantProfile, Run, SpadeError
from driveflow.runs.store import RunStore, environment_info, reproducible_command, safe_command, sha256_file, workspace_root


def _run(run_id, status="completed", conds=None, kind="scenario_sweep"):
    conds = conds if conds is not None else [{"index": 0, "condition": {"w": 1}, "status": "ok", "metrics": {"e": 1.0}}]
    return Run(run_id=run_id, status=status, origin="SIMULATED", kind=kind, conditions=conds, failure_reason="x" if status == "failed" else None)


def test_workspace_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_WORKSPACE", str(tmp_path / "ws"))
    assert workspace_root() == tmp_path / "ws" and (tmp_path / "ws").is_dir()


def test_runs_experiments_and_artifacts_persist_across_store_instances(tmp_path):
    s = RunStore(tmp_path)
    spec = s.save_experiment(ExperimentSpec.create("scenario_sweep", conditions=[{"duration_s": 0.1}]))
    art = s.write_artifact("run-a", "sub/data.bin", b"abc")
    s.save_run(_run("run-a"))
    t = RunStore(tmp_path)
    assert t.get_experiment(spec.experiment_id) == spec
    assert t.get_run("run-a") == _run("run-a")
    assert art == {"path": "runs/run-a/sub/data.bin", "bytes": 3, "sha256": sha256_file(tmp_path / "runs" / "run-a" / "sub" / "data.bin")}
    assert t.artifact_path(art).read_bytes() == b"abc"


def test_runs_are_immutable_and_missing_ones_are_typed_errors(tmp_path):
    s = RunStore(tmp_path)
    s.save_run(_run("run-a"))
    with pytest.raises(SpadeError):
        s.save_run(_run("run-a"))
    with pytest.raises(SpadeError) as exc:
        s.get_run("nope")
    assert exc.value.code == ErrorCode.NOT_FOUND


@pytest.mark.parametrize("name", ["../escape.txt", "/etc/passwd", ""])
def test_artifact_names_cannot_leave_the_run_directory(tmp_path, name):
    with pytest.raises(SpadeError):
        RunStore(tmp_path).write_artifact("run-a", name, b"x")


def test_concurrent_writers_do_not_lose_runs(tmp_path):
    RunStore(tmp_path)
    errors = []

    def writer(k):
        try:
            store = RunStore(tmp_path)
            for i in range(20):
                store.save_run(_run(f"run-{k}-{i}"))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(RunStore(tmp_path).list_runs(limit=1000)) == 120


def test_commands_carry_no_absolute_paths_or_credentials():
    cmd = reproducible_command("run-x")
    assert cmd == "python -m driveflow.jobs.worker run runs/run-x/spec.json"
    for bad in ["python x.py /home/me/data", "python x.py C:\\Users\\me", "curl http://u:p@host/x", "python ~/x.py"]:
        with pytest.raises(SpadeError):
            safe_command(bad)


def test_environment_info_records_versions_without_importing_tensorflow():
    env = environment_info()
    assert env["python"] and "numpy" in env["packages"] and "commit" in env


def test_approval_is_enforced_in_the_backend(tmp_path):
    s = RunStore(tmp_path)
    s.save_run(_run("base"))
    s.save_run(_run("cand"))
    rec = dict(subject="vsc dpc default", baseline_run="base", candidate_run="cand", approver="Luciano", user_confirmed_at="2026-10-20T00:00:00+00:00")
    with pytest.raises(SpadeError) as exc:
        s.record_approval(ApprovalRecord(approval_id="a1", gate_report={"blocking": ["worse overshoot at 4 ohm"]}, decision="approved", **rec))
    assert exc.value.code == ErrorCode.APPROVAL_BLOCKED and s.list_approvals() == []
    # A rejection of a blocked candidate is a valid decision to record.
    s.record_approval(ApprovalRecord(approval_id="a2", gate_report={"blocking": ["worse"]}, decision="rejected", **rec))
    s.record_approval(ApprovalRecord(approval_id="a3", gate_report={"blocking": []}, decision="approved", **rec))
    with pytest.raises(SpadeError):  # append-only
        s.record_approval(ApprovalRecord(approval_id="a3", gate_report={"blocking": []}, decision="approved", **rec))
    with pytest.raises(SpadeError):  # the candidate must be a recorded run
        s.record_approval(ApprovalRecord(approval_id="a4", gate_report={"blocking": []}, decision="approved", **{**rec, "candidate_run": "ghost"}))
    assert [a.approval_id for a in s.list_approvals("vsc dpc default")] == ["a2", "a3"]


def test_profiles_versioned_and_stale_when_the_file_changes(tmp_path):
    f = tmp_path / "buck.plecs"
    f.write_text("v1")
    p = PlantProfile(profile_id="buck", kind="plecs_rpc_batch", engine="plecs", display_name="Buck", origin="SIMULATED · PLECS", file_hash=sha256_file(f))
    s = RunStore(tmp_path)
    s.save_profile(p)
    assert not is_stale(p, f)
    f.write_text("v2")
    assert is_stale(p, f) and is_stale(p, tmp_path / "gone.plecs")
    s.save_profile(mark_stale(p))
    latest = s.get_profile("buck")
    assert latest.version == 2 and latest.status == "stale" and not latest.capabilities.can_simulate_batch
    assert s.get_profile("buck", 1) == p
    stepping = {k for k, p in NATIVE_PROFILES.items() if p.capabilities.can_step_interactively}
    assert stepping == {"native:dc_perm_ex_v1", "native:vsc_dpc_v1", "native:buck_v1"}  # E1/E6 wrappers; the PMSM has none
    assert not any(p.capabilities.can_train_rl for p in NATIVE_PROFILES.values())


class TestCompare:
    def _c(self, w, status="ok", e=None):
        return {"index": w, "condition": {"w": w}, "status": status, "metrics": {} if e is None else {"e": e}}

    def test_partial_overlap_compares_the_intersection_and_counts_the_rest(self):
        base = _run("b", conds=[self._c(1, e=2.0), self._c(2, e=2.0), self._c(3, e=2.0)])
        cand = _run("c", conds=[self._c(1, e=1.0), self._c(2, e=1.5), self._c(4, e=0.1)])
        r = compare_runs([base, cand], "e")
        assert r["comparable"] and r["n_shared"] == 2 and r["n_excluded"] == 2
        assert r["verdicts"][0]["improved"] and r["verdicts"][0]["better"] == 2

    def test_failures_are_shown_not_imputed_and_block_an_improvement(self):
        base = _run("b", conds=[self._c(1, e=2.0), self._c(2, e=2.0)])
        cand = _run("c", status="partially_completed", conds=[self._c(1, e=0.5), {**self._c(2), "status": "failed", "error": "diverged"}])
        r = compare_runs([base, cand], "e")
        assert r["rows"][1]["c"] == "Failed"
        assert not r["verdicts"][0]["improved"] and r["verdicts"][0]["new_failures"] == 1

    def test_disjoint_grids_or_kinds_are_not_comparable(self):
        assert not compare_runs([_run("b", conds=[self._c(1, e=1)]), _run("c", conds=[self._c(2, e=1)])], "e")["comparable"]
        assert not compare_runs([_run("b"), _run("c", kind="selftest.steps")], "e")["comparable"]
        assert compare_runs([_run("b")], "e")["reason"] == "Select at least two runs."

    def test_worse_candidate_is_no_improvement(self):
        r = compare_runs([_run("b", conds=[self._c(1, e=1.0)]), _run("c", conds=[self._c(1, e=3.0)])], "e")
        assert r["comparable"] and not r["verdicts"][0]["improved"] and r["verdicts"][0]["worse"] == 1
