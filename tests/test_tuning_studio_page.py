"""Design & Control / Tuning Studio (expansion task E7): empty state, a real tuning job through the
worker for each method, the candidate review, and the blocked "Use as default" (decision D1)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.jobs.worker import Worker
from driveflow.optim import objectives as ob
from driveflow.runs.store import RunStore, workspace_root
from driveflow.viz.adapters import tuning_adapter as ta

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/tuning_studio.py").run()
    assert not at.exception
    return at


def _text(at):
    return " ".join(e.value for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind))


def test_empty_state_and_fixed_grids():
    at = _open()
    text = _text(at)
    assert "No simulation has been executed for this configuration." in text
    assert "Evaluation grid (fixed before tuning, never optimized on)" in text
    assert not ob.grids_overlap() and ob.REUSED_WARNING not in text


@pytest.mark.parametrize("method", ["pso", "nsga2"])
def test_tuning_job_end_to_end(method):
    at = _open()
    at.radio(key="tun_method").set_value(method).run()
    at.slider(key="tun_evals").set_value(8).run()
    at.button(key="tun_start").click().run()
    assert any("Queued as job" in s.value for s in at.success)
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    at.button(key="tun_refresh").click().run()
    assert not at.exception
    text = _text(at)
    assert "Evaluation-grid J: baseline 27.83" in text
    assert ta.NOT_AUTHORIZED in " ".join(str(h.body) for h in at.get("html")) or ta.NOT_AUTHORIZED in text
    table = next(d.value for d in at.dataframe if "Metric" in getattr(d.value, "columns", []))
    assert set(table["Evaluation condition"]) >= {c.name for c in ob.EVALUATION_GRID}
    # Use as default is blocked: approval impossible even with a name and the box ticked
    run_id = ta.runs()[0].run_id
    assert at.button(key=f"tun_gate_{run_id}_approve").disabled
    at.button(key=f"tun_save_{run_id}_0").click().run()
    assert any("No default was changed" in s.value for s in at.success)
    expected = "saved_candidate_best.json" if method == "pso" else "saved_candidate_pareto_0.json"
    saved = RunStore().run_dir(run_id) / expected
    assert saved.exists() and '"status": "candidate -- not a default"' in saved.read_text()


def test_blocked_approval_is_refused_by_the_backend():
    rep = {"baseline": {"evaluation": {"J": 10.0}}, "evaluation_reused_tuning_conditions": False}
    gate = ta.gate_report(rep, {"evaluation": {"J": 5.0}, "params": {}})
    assert gate["blocking"] == [ta.NOT_AUTHORIZED]
    from driveflow.runs.schemas import new_id

    class _Run:
        run_id = new_id("run")

    out = ta.record_decision(_Run(), {}, gate, "Tester", "approved")
    assert "error" in out and "Approval blocked" in out["error"]


def test_spec_validation():
    from driveflow.jobs.service import JobService
    from driveflow.runs.schemas import ExperimentSpec, SpadeError

    s = JobService()
    for fields, msg in [(dict(algorithm="cmaes", seed=0, params={"bounds": [list(b) for b in ob.BOUNDS]}, budget={"evaluations": 8}), "algorithm"),
                        (dict(algorithm="pso", seed=0, params={"bounds": [[10, 500], [0.6, 1.2], [0.3, 1.0]]}, budget={"evaluations": 8}), "inside"),
                        (dict(algorithm="pso", seed=0, params={"bounds": [list(b) for b in ob.BOUNDS]}, budget={"evaluations": 2}), "budget")]:
        with pytest.raises(SpadeError, match=msg):
            s.submit(ExperimentSpec.create("controller_tuning", **fields))
