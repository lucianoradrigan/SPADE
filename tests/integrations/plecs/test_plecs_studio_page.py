"""Simulation / PLECS Plant Studio (expansion task E2): navigable without PLECS, the Dataset route end
to end against the fake PLECS server, and nothing that suggests a Gymnasium plant."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.jobs.worker import Worker
from driveflow.runs.store import RunStore, workspace_root
from fake_plecs import FakePlecs, closed_port_url

APP_PATH = str(Path(__file__).resolve().parents[3] / "src" / "driveflow" / "viz" / "app.py")


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_PLECS_DATA", str(tmp_path / "plecs_data"))


def _open():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.switch_page("pages/plecs_plant_studio.py").run()
    assert not at.exception
    return at


def _text(at):
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind)]
    parts += [str(getattr(b, "label", "")) for b in at.button]
    return " ".join(parts)


def test_navigable_without_plecs_and_no_gym_promise():
    at = _open()
    assert [t.label for t in at.tabs][:4] == ["My Plants", "Create Plant", "Compare with SPADE", "Connection & Diagnostics"]
    text = _text(at)
    assert "No PLECS plant profiles yet. Create your first plant to get started." in text
    assert "What would you like to do with your PLECS circuit?" in text
    assert "Interactive stepping is not available with the detected backend." in text
    # B5 (round PLECS/OM): the Gym card opens the OpenModelica FMU route; PLECS RPC still gives no stepping
    assert not at.button(key="plecs_goal_gym").disabled and at.button(key="plecs_goal_surrogate").disabled and not at.button(key="plecs_goal_dataset").disabled
    assert "No comparison has been generated yet." in text
    for banned in ("Gym ready", "Publish to Environments", "LIVE", "Connected"):
        assert banned not in text


def test_connection_test_reports_unreachable_and_remote_hosts():
    at = _open()
    at.text_input(key="plecs_url").set_value(closed_port_url()).run()
    at.button(key="plecs_test").click().run()
    assert not at.exception and any("Cannot reach RPC endpoint" in e.value for e in at.error)
    at.text_input(key="plecs_url").set_value("http://10.1.2.3:1080").run()
    at.button(key="plecs_test").click().run()
    assert any("Backend running on a different host" in e.value for e in at.error)


def test_connection_test_against_a_running_server():
    with FakePlecs() as fake:
        at = _open()
        at.text_input(key="plecs_url").set_value(fake.url).run()
        at.button(key="plecs_test").click().run()
    assert any("PLECS available" in s.value for s in at.success)
    assert "Not derived from RPC connectivity" in _text(at)


def test_dataset_route_end_to_end(tmp_path):
    circuit = tmp_path / "buck_demo.plecs"
    circuit.write_text("Plecs { Name \"buck_demo\" }\n")
    with FakePlecs(fail_if=lambda mv: "Solver did not converge" if mv["Vref"] > 90 else None) as fake:
        at = _open()
        at.text_input(key="plecs_url").set_value(fake.url).run()
        at.button(key="plecs_goal_dataset").click().run()
        at.text_input(key="plecs_new_id").set_value("buck_demo").run()
        at.radio(key="plecs_new_how").set_value("File on this computer").run()
        at.text_input(key="plecs_new_path").set_value(str(circuit)).run()
        at.button(key="plecs_import").click().run()
        assert not at.exception and any("Imported" in s.value for s in at.success), [e.value for e in at.error]
        assert "No simulation has been executed for this configuration." in _text(at)
        at.multiselect(key="plecs_swept_buck_demo").set_value(["Vref"]).run()
        at.number_input(key="plecs_lo_buck_demo_Vref").set_value(10.0).run()
        at.number_input(key="plecs_hi_buck_demo_Vref").set_value(95.0).run()
        at.number_input(key="plecs_lv_buck_demo_Vref").set_value(3).run()
        at.button(key="plecs_start").click().run()
        assert any("Queued as job" in s.value for s in at.success)
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
        at.run()
    assert not at.exception
    text = _text(at)
    assert "2 completed / 1 failed" in text and "never interpolated" in text and "does not provide an interactive Gymnasium plant" in text
    table = next(d.value for d in at.dataframe if "Status" in getattr(d.value, "columns", []))
    assert list(table["Status"]) == ["ok", "ok", "failed"] and list(table["Vref"]) == [10.0, 52.5, 95.0]
    p = RunStore().get_profile("buck_demo")
    assert p.capabilities.can_simulate_batch and not p.capabilities.can_step_interactively
    at.switch_page("pages/plecs_plant_studio.py").run()
    plants = next(d.value for d in at.dataframe if "Readiness" in getattr(d.value, "columns", []))
    assert plants.iloc[0]["Batch sweep"] == "Shown" and plants.iloc[0]["Interactive stepping"] == "Not supported"


def test_invalid_manifest_is_reported_and_nothing_is_imported(tmp_path):
    circuit = tmp_path / "c.plecs"
    circuit.write_text("x")
    at = _open()
    at.button(key="plecs_goal_dataset").click().run()
    at.text_input(key="plecs_new_id").set_value("bad_manifest").run()
    at.radio(key="plecs_new_how").set_value("File on this computer").run()
    at.text_input(key="plecs_new_path").set_value(str(circuit)).run()
    at.text_area(key="plecs_new_manifest").set_value('{"model_vars": [], "outputs": []}').run()
    at.button(key="plecs_import").click().run()
    assert any("at least one output" in e.value for e in at.error)
    assert RunStore().list_profiles() == []


def test_partial_sweep_is_visible_with_counts_and_causes(tmp_path):
    """Task A1: a sweep where 2 of 9 conditions fail shows as Partially completed in My Plants and in
    Experiment Runs, with the counts, the cause of each failure and a link to the log."""
    from driveflow.integrations.plecs import batch
    from driveflow.integrations.plecs import manifest as mf
    from driveflow.jobs.service import JobService
    from driveflow.runs.schemas import ExperimentSpec

    circuit = tmp_path / "buck_demo.plecs"
    circuit.write_text("Plecs { Name \"buck_demo\" }\n")
    man = mf.from_dict(mf.EXAMPLE_BUCK)
    fail = {(6.0, 2.0): "Solver did not converge at t = 1.2 ms", (18.0, 10.0): "Solver did not converge at t = 4.0 ms"}
    with FakePlecs(fail_if=lambda mv: fail.get((mv["Vref"], mv["Rload"]))) as fake:
        RunStore().save_profile(batch.create_profile("buck_demo", circuit, man, rpc_url=fake.url))
        conds = batch.design_conditions(man, {"Vref": (6, 18, 3), "Rload": (2, 10, 3)}, "grid")
        s = JobService()
        job = s.submit(ExperimentSpec.create("plecs_sweep", plant_profile_id="buck_demo", conditions=conds, seed=0, algorithm="grid",
                                             params={"time_span_s": 0.01, "time_limit_s": 5.0}))
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
    at = _open()
    text = _text(at)
    assert "Partially completed" in text and "7 of 9 conditions completed, 2 failed" in text
    tables = [d.value for d in at.dataframe]
    sweeps = next(t for t in tables if "Completed" in t.columns and "Run" in t.columns)
    assert list(sweeps["Completed"]) == ["7 / 9"] and list(sweeps["Outcome"]) == ["Partially completed"]
    failures = next(t for t in tables if "Cause" in t.columns)
    assert len(failures) == 2 and set(failures["Cause"]) == {"Solver error / no convergence"}
    run_id = sweeps["Run"].iloc[0]
    at.button(key=f"plecs_log_{run_id}").click().run()
    assert not at.exception
    # Experiment Runs: the same job, its failures with causes, never averaged
    assert at.session_state["runs_selected_job"] == job.job_id
    texts = " ".join(str(e.value) for kind in ("markdown", "caption", "info", "warning", "error", "success", "code") for e in at.get(kind))
    assert "condition 3/9 failed" in texts or "failed" in texts
    assert any("Cause" in getattr(d.value, "columns", []) and (d.value["Cause"] == "Solver error / no convergence").sum() == 2 for d in at.dataframe)


def test_guide_names_the_routes_of_create_plant():
    """The diagram sends the user to Create Plant, so its routes must be that tab's goal cards."""
    from driveflow.viz.views import plecs_plant_studio as studio

    source = Path(studio.__file__).read_text(encoding="utf-8")
    for route in studio.ROUTES:
        assert f'st.markdown("**{route["title"]}**")' in source

