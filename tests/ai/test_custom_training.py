"""Custom classifier training (AI Diagnostics → Train a model): request validation, the condition
grid, simulation, training and the classifier_training job (driveflow.ai.custom_training)."""

import json

import numpy as np
import pytest

from driveflow.ai import custom_training as ct
from driveflow.ai.registry import load_run_model


def _tiny_dc(**model):
    r = ct.default_request("dc_motor")
    r["operating"].update(setpoints=[150.0], severity_scales=[1.0], runs_per_cell=3, duration_s=0.15, skip_start_s=0.05, drift={}, noise_max_pct=0.0)
    r["model"].update(preset="esp32_dscnn", input_window=256, channels=["current_r", "acc_x"], epochs=1, cap_per_class=60, keep_rpm_level=False, **model)
    return r


class TestRequest:
    def test_defaults_are_valid_for_both_domains(self):
        for domain in ct.DOMAINS:
            s = ct.summary(ct.default_request(domain))
            assert s["n_runs"] >= 3 * len(s["classes"])

    @pytest.mark.parametrize("change, message", [
        (lambda r: r.update(classes=r["classes"][:1]), "at least two classes"),
        (lambda r: r["operating"].update(setpoints=list(range(10, 340, 10)), runs_per_cell=5), "the limit is"),
        (lambda r: (r["model"].update(input_window=8192), r["operating"].update(duration_s=0.8)), "one window is"),
        (lambda r: r["model"].update(channels=["vc_real"]), "not available for dc_motor"),
        (lambda r: r["classes"].append({"label": "gear", "fault_type": "gear", "mechanical_severity": 0.02}), "characteristic order"),
        (lambda r: r["operating"].update(controllers=["DPC"]), "PI and/or MPC"),
        (lambda r: r.update(name="bad name!"), "name"),
    ])
    def test_invalid_requests_say_what_to_change(self, change, message):
        r = ct.default_request("dc_motor")
        change(r)
        with pytest.raises(ct.RequestError, match=message):
            ct.validate_request(r)

    def test_too_few_runs_per_class_is_refused(self):
        r = ct.default_request("dc_motor")
        r["operating"].update(setpoints=[150.0], severity_scales=[1.0], runs_per_cell=2)
        with pytest.raises(ct.RequestError, match="at least 3"):
            ct.validate_request(r)


class TestConditions:
    def test_classes_are_balanced_and_the_grid_is_reproducible(self):
        r = ct.default_request("dc_motor")
        r["operating"].update(controllers=["PI", "MPC"])
        a, b = ct.build_conditions(r), ct.build_conditions(r)
        assert a == b
        counts = {c["label"]: sum(x["label"] == c["label"] for x in a) for c in r["classes"]}
        assert len(set(counts.values())) == 1
        assert {x["scenario"]["controller_type"] for x in a} == {"PI", "MPC"}
        assert len({x["scenario"]["seed"] for x in a}) == len(a)

    def test_noise_and_drift_stay_in_their_ranges(self):
        r = ct.default_request("dc_motor")
        conds = ct.build_conditions(r)
        psi = np.array([c["scenario"]["motor_parameter_overrides"]["psi_e"] for c in conds])
        assert psi.min() >= 0.165 * 0.9 - 1e-9 and psi.max() <= 0.165 * 1.1 + 1e-9
        noise = np.array([c["scenario"]["electrical_noise_pct"] for c in conds])
        assert noise.max() <= 50.0 and (noise == 0).mean() > 0.1

    def test_torque_mode_and_custom_fault_reach_the_scenario(self):
        r = ct.default_request("dc_motor")
        r["operating"].update(control_mode="torque", setpoints=[16.0])
        r["classes"].append(ct.dc_class("unbalance", "unbalance", 1.0, [], 0.03))
        conds = ct.build_conditions(r)
        assert all(c["scenario"]["torque_ref_nm"] == 16.0 and "omega_ref_rad_s" not in c["scenario"] for c in conds)
        unb = [c for c in conds if c["label"] == "unbalance"]
        assert unb and all(c["scenario"]["fault_order_override"] == 1.0 for c in unb)
        assert {c["scenario"]["mechanical_severity"] for c in unb} == {0.03 * s for s in r["operating"]["severity_scales"]}

    def test_vsc_degraded_class_draws_inside_the_range_and_healthy_stays_nominal(self):
        r = ct.default_request("vsc_dpc")
        r["operating"]["voltage_limit"] = True
        conds = ct.build_conditions(r)
        deg = [c["scenario"] for c in conds if c["label"] == ct.VSC_DEGRADED]
        assert all(0.5 <= d["vsc_capacitor_scale"] <= 0.8 for d in deg)
        assert all("vsc_capacitor_scale" not in c["scenario"] for c in conds if c["label"] == ct.HEALTHY)
        assert all(c["scenario"]["vsc_voltage_limit_v"] == 134.0 for c in conds)


class TestSimulateAndTrain:
    def test_end_to_end_writes_a_loadable_candidate(self, tmp_path):
        r = _tiny_dc()
        df, outcomes = ct.simulate(ct.build_conditions(r))
        assert all(o["status"] == "ok" for o in outcomes)
        assert set(df["label"]) == {"normal", "outer_race", "inner_race"}
        run_dir = ct.train(r, df, root=tmp_path / "models", dataset_root=tmp_path / "data")
        m = json.loads((run_dir / "metrics.json").read_text())
        assert m["classes"] == ["inner_race", "normal", "outer_race"] and m["channels"] == ["current_r", "acc_x"]
        assert np.array(m["test_confusion_matrix"]).shape == (3, 3)
        assert "ood" in m and m["status"].startswith("candidate")
        assert (run_dir / "training_request.json").exists() and json.loads((run_dir / "run_metadata.json").read_text())["dataset_sha256"]
        model, config, _ = load_run_model(run_dir, "classifier")
        assert model.output_shape[-1] == 3 and config.input_window == 256
        listed = ct.list_model_runs("dc_motor", root=tmp_path / "models")
        assert [x["run_dir"] for x in listed] == [run_dir] and listed[0]["origin"] == "script"

    def test_cancel_marks_the_remaining_runs_skipped(self):
        r = _tiny_dc()
        conds = ct.build_conditions(r)
        calls = []
        df, outcomes = ct.simulate(conds, cancelled=lambda: len(calls) > 0, progress=lambda *a: calls.append(a))
        assert [o["status"] for o in outcomes] == ["ok"] + ["skipped"] * (len(conds) - 1)
        assert df["source_file"].nunique() == 1

    def test_parallel_simulation_keeps_condition_order(self):
        r = _tiny_dc()
        conds = ct.build_conditions(r)[:3]
        _, outcomes = ct.simulate(conds, workers=2)
        assert [o["index"] for o in outcomes] == [0, 1, 2] and all(o["status"] == "ok" for o in outcomes)


class TestJob:
    def test_classifier_training_job_runs_in_the_worker(self, tmp_path, monkeypatch):
        from driveflow.jobs.worker import Worker
        from driveflow.runs.store import RunStore, workspace_root
        from driveflow.viz.adapters import training_adapter as tra

        monkeypatch.setattr(ct, "CUSTOM_ROOT", tmp_path / "models")
        monkeypatch.setattr(ct, "DATASET_ROOT", tmp_path / "data")
        job = tra.submit(_tiny_dc(), workers=1)
        assert tra.active_jobs()[0]["job_id"] == job.job_id
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
        run = tra.runs("dc_motor")[0]
        assert run.status == "completed", run.failure_reason
        assert run.conditions[-1]["condition"]["step"] == "training" and run.conditions[-1]["status"] == "ok"
        run_dir = tra.run_dir_of(run)
        assert run_dir is not None and run_dir.is_relative_to(tmp_path / "models")
        assert any(a["path"].endswith("metrics.json") for a in RunStore().get_run(run.run_id).artifacts)

    def test_invalid_request_is_refused_at_submit(self):
        from driveflow.runs.schemas import SpadeError
        from driveflow.viz.adapters import training_adapter as tra

        r = _tiny_dc()
        r["classes"] = r["classes"][:1]
        with pytest.raises((ct.RequestError, SpadeError)):
            tra.submit(r, workers=1)
