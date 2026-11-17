"""AI-N42 (2026-11-16): the AI Diagnostics "Classifier" selector reset to the promoted model on every rerun.

`_model_choice` did `requested = st.session_state.pop("ia_model_request", None)` and then `if requested in options`; with no
request that is None, and None is the "Promoted model (registry)" option, so the user's choice was overwritten before the
widget rendered. Present since 2026-10-21 (v0.2.0 and v0.2.1): in a real browser only the promoted model could be evaluated.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from driveflow.ai.custom_training import list_classifier_runs

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")
RUN = ROOT / "configs" / "classifiers" / "pc_server" / "2026-11-04_run01"


def _classifier(at):
    return [s for s in at.selectbox if s.label == "Classifier"][0]


def test_the_chosen_classifier_survives_reruns_and_is_the_one_evaluated():
    value = next(str(r["run_dir"]) for r in list_classifier_runs("dc_motor") if Path(r["run_dir"]).resolve() == RUN.resolve())
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    _classifier(at).set_value(value).run()
    assert _classifier(at).value == value
    at.slider(key="ia_omega_ref_slider").set_value(120.0).run()  # an unrelated widget: a rerun must keep the choice
    assert _classifier(at).value == value
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception and _classifier(at).value == value
    shown = " ".join(str(getattr(h, "body", "")) for h in at.get("html"))
    assert "pc_server/2026-11-04_run01" in shown, "the badge does not name the selected run"


def test_an_explicit_request_still_selects_a_run():
    value = next(str(r["run_dir"]) for r in list_classifier_runs("dc_motor") if Path(r["run_dir"]).resolve() == RUN.resolve())
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    at.session_state["ia_model_request"] = value
    at.run()
    assert _classifier(at).value == value
    at.run()
    assert _classifier(at).value == value
