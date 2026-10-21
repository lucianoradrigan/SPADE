"""Experiment Runs page and the E0 components (expansion task E0; spec Sec. 6, 9, 13 Phase 0).

The SPADE workspace is a fresh temporary directory per test (tests/conftest.py)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.jobs.worker import Worker
from driveflow.runs.schemas import ExperimentSpec, Run
from driveflow.viz.components.approval_gate import approval_view
from driveflow.viz.components.connection_status import connection_view
from driveflow.viz.components.failure_panel import failure_view
from driveflow.viz.components.job_card import job_view
from driveflow.viz.components.provenance_footer import provenance_view
from driveflow.viz.components.run_comparison import comparison_view
from driveflow.viz.components.source_badge import source_badge_view
from driveflow.viz.components.state_panel import STATES

APP = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page("pages/experiment_runs.py").run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "success", "code") for e in at.get(kind) if isinstance(getattr(e, "value", None), str)]
    return " ".join(parts + [str(getattr(h, "body", "")) for h in at.get("html")])


def test_empty_workspace_shows_empty_states_and_no_fabricated_rows():
    at = _page()
    text = _text(at)
    assert "No jobs yet" in text and "No worker is running" in text and "Select at least two runs" in text
    assert not at.dataframe  # no job table without jobs


def test_job_queued_from_the_page_survives_navigation_and_reload():
    at = _page()
    at.button(key="runs_submit").click().run()
    assert not at.exception, at.exception
    jobs = JobService().list()
    assert len(jobs) == 1 and jobs[0].state == JobState.QUEUED and jobs[0].kind == "scenario_sweep"
    at.switch_page("pages/overview.py").run()
    at.switch_page("pages/experiment_runs.py").run()
    assert jobs[0].job_id in _text(at) or at.selectbox(key="runs_job_select").value == jobs[0].job_id
    fresh = _page()  # a new browser session: state comes from the store, not st.session_state
    assert fresh.selectbox(key="runs_job_select").value == jobs[0].job_id
    assert "Job queued" in _text(fresh)


def test_cancel_from_the_page_is_recorded():
    job = JobService().submit(ExperimentSpec.create("selftest.steps", params={"steps": 2, "step_s": 0.0}))
    at = _page()
    at.button(key=f"runs_card_{job.job_id}_cancel").click().run()
    assert JobService().get(job.job_id).state == JobState.CANCELLED


def test_finished_job_shows_run_details_failures_and_command():
    s = JobService()
    job = s.submit(ExperimentSpec.create("selftest.steps", params={"steps": 3, "step_s": 0.0, "fail_steps": [1]}))
    Worker(s.root).serve(once=True)
    at = _page()
    text = _text(at)
    run = s.run_for(s.get(job.job_id))
    assert "Completed with unsuccessful conditions" in text and "1 failed · 0 skipped of 3 conditions" in text
    assert run.command in text and run.config_hash in text
    assert "Resume unavailable" in text


def test_sidebar_counts_jobs():
    JobService().submit(ExperimentSpec.create("selftest.steps", params={"steps": 1, "step_s": 0.0}))
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert any("Jobs: 0 running · 1 queued" in c.value for c in at.sidebar.caption)


def test_interactive_page_runs_are_mirrored_as_legacy_records():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page("pages/dc_motor.py").run()
    at.slider(key="dc_duration_slider").set_value(0.1).run()
    at.button(key="dc_run_button").click().run()
    runs = [r for r in JobService().store.list_runs() if r.status == "legacy_session"]
    assert len(runs) == 1 and runs[0].kind == "session:dc" and runs[0].params["duration_s"] == 0.1 and runs[0].command is None


# ------------------------------------------------------------------ components: five states each
def _run(status="completed", **kw):
    return Run(run_id="run-1", status=status, origin="SIMULATED", kind="k", command="python -m driveflow.jobs.worker run runs/run-1/spec.json",
               failure_reason="boom" if status == "failed" else None, **kw)


class _Job:
    def __init__(self, state, **kw):
        self.state, self.phase, self.error = JobState(state), kw.get("phase"), kw.get("error")


_CMP_OK = {"comparable": True, "n_shared": 2, "n_excluded": 0, "verdicts": [{"improved": True}]}
CASES = {
    "SourceBadge": [source_badge_view(None), source_badge_view("SIMULATED", loading=True), source_badge_view("LIVE"),
                    source_badge_view("SIMULATED · PLECS", run_id="r"), source_badge_view("SIMULATED", run_failed=True)],
    "ProvenanceFooter": [provenance_view(None), provenance_view(None, loading=True), provenance_view(None, error="io"),
                         provenance_view(_run()), provenance_view(_run("failed"))],
    "ConnectionStatus": [connection_view("PLECS", h) for h in ("not_configured", "checking", "unavailable", "available", "capability_missing")],
    "JobCard": [job_view(None), job_view(_Job("running")), job_view(_Job("failed", error="x")), job_view(_Job("completed")),
                job_view(_Job("partially_completed"))],
    "RunComparison": [comparison_view(None), comparison_view(None, loading=True), comparison_view({"comparable": False, "reason": "These runs do not share the same evaluation grid."}),
                      comparison_view(_CMP_OK), comparison_view({**_CMP_OK, "verdicts": [{"improved": False}]})],
    "ApprovalGate": [approval_view(None, None), approval_view("r", None, loading=True), approval_view("r", {"blocking": []}, {"error": "disk full"}),
                     approval_view("r", {"blocking": []}), approval_view("r", {"blocking": ["worse overshoot"]})],
    "FailurePanel": [failure_view([]), failure_view(None, loading=True), failure_view(None, error="io"),
                     failure_view([{"status": "ok"}]), failure_view([{"status": "ok"}, {"status": "failed"}])],
}


@pytest.mark.parametrize("name", list(CASES))
def test_each_component_reaches_all_five_states(name):
    assert [v.state for v in CASES[name]] == list(STATES)


def test_spec_microcopy_is_used():
    msgs = {v.message for views in CASES.values() for v in views}
    for s in ("No data source selected", "Source cannot be verified", "No run generated", "Reproducibility manifest unavailable", "PLECS not configured",
              "Testing connection…", "PLECS RPC unavailable", "No jobs yet", "Select at least two runs", "Runs are not comparable",
              "Candidate did not improve", "No candidate selected", "Approval record not saved", "All conditions completed", "No failures recorded"):
        assert s in msgs, s


def test_components_render_in_streamlit_without_errors():
    def app():
        from driveflow.jobs.schemas import Job, JobState
        from driveflow.runs.schemas import Run
        from driveflow.viz.components.approval_gate import approval_gate
        from driveflow.viz.components.connection_status import connection_status
        from driveflow.viz.components.failure_panel import failure_panel
        from driveflow.viz.components.job_card import job_card
        from driveflow.viz.components.provenance_footer import provenance_footer
        from driveflow.viz.components.run_comparison import run_comparison
        from driveflow.viz.components.source_badge import source_badge

        run = Run(run_id="run-1", status="partially_completed", origin="SIMULATED", kind="k", failure_reason="1 of 2 failed",
                  conditions=[{"index": 0, "condition": {"w": 1}, "status": "ok"}, {"index": 1, "condition": {"w": 2}, "status": "failed", "error": "diverged"}],
                  command="python -m driveflow.jobs.worker run runs/run-1/spec.json", environment={"commit": "abc", "python": "3.12"})
        job = Job(job_id="job-1", experiment_id="e", kind="k", engine="native", state=JobState.RUNNING, queued_at="t", progress_done=1, progress_total=4)
        source_badge("SIMULATED", run_id="run-1")
        source_badge(None)
        provenance_footer(run)
        connection_status("PLECS", "not_configured")
        job_card(job, None, on_cancel=lambda: None)
        failure_panel(list(run.conditions))
        run_comparison({"comparable": True, "n_shared": 1, "n_excluded": 1, "metric": "e", "lower_is_better": True,
                        "rows": [{"condition": {"w": 1}, "a": 1.0, "b": "Failed"}],
                        "verdicts": [{"candidate": "b", "baseline": "a", "n_paired": 0, "better": 0, "worse": 0, "new_failures": 1,
                                      "baseline_mean": None, "candidate_mean": None, "improved": False}]})
        approval_gate("s", "run-1", {"blocking": ["worse overshoot"]}, on_submit=lambda a, d: {"error": "not called"})

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "ETA unavailable until enough evaluations are completed." in text and "Approval blocked. No default has been changed." in text
    assert at.checkbox[0].disabled  # blocked: the confirmation is disabled (the backend refuses anyway)
