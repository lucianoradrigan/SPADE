"""AI-N40 corrected (decision D1, 2026-11-16): every promoted classifier declares its trained envelope (speed range of its
training grid) and every page that shows a prediction applies a deterministic rule: outside the envelope there is no
prediction (no class, no bars, no confidence), only 'Outside the trained range (X–Y rad/s): no diagnosis'. D2: AI
Diagnostics defaults to 150 rad/s. D3: the classifiers selectable in the GUI are the PC-tier runs offered by AI
Diagnostics (the promoted one and every other PC-tier run on the machine; the RPi5 and ESP32 runs are not selectable,
they are edge exports). Here: the promoted run (declared envelope), the B4 5-class candidate and a 3-class run (envelopes
deduced from their datasets); the release gate G6 covers every selectable run."""

from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from driveflow.ai import envelope

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")
ENTRIES = yaml.safe_load((ROOT / "configs" / "registry.yaml").read_text(encoding="utf-8"))["entries"]
CLASSIFIERS = {k: v for k, v in ENTRIES.items() if k.endswith("/classifier")}
OUTSIDE = "Outside the trained range"


@pytest.mark.parametrize("key", sorted(CLASSIFIERS))
def test_every_promoted_classifier_declares_its_trained_envelope(key):
    env = envelope.declared_envelope(ROOT / CLASSIFIERS[key])
    assert env is not None, f"{key}: no trained_envelope in audit_findings.json"
    lo, hi = env["speed_rad_s"]
    assert 0 < lo < hi and env["source"]


def test_the_rule_is_deterministic_on_the_boundaries():
    run = ROOT / CLASSIFIERS["dc_motor/pc/classifier"]
    lo, hi = envelope.declared_envelope(run)["speed_rad_s"]
    assert envelope.check(run, lo)["status"] == envelope.check(run, hi)["status"] == "inside"
    assert envelope.check(run, hi + 1)["status"] == envelope.check(run, lo - 1)["status"] == "outside"
    assert envelope.check(run, hi + 1)["message"] == f"Outside the trained range ({lo:.0f}–{hi:.0f} rad/s): no diagnosis"
    assert envelope.check(run, None)["status"] == "unknown_speed"


def test_ai_diagnostics_defaults_to_150_rad_s():
    from driveflow.viz.views import ai_diagnostics

    assert ai_diagnostics.DC_DEFAULT_OMEGA_REF == 150.0


def _sample(run: str | None, speed: float) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    at.slider(key="ia_omega_ref_slider").set_value(speed).run()
    gen = [b for b in at.button if "Generate" in b.label][0]
    if run is not None:
        # AppTest cannot drive this selectbox (format_func: it resends the label, which the page drops), so the page's
        # own request key is set right before the run that evaluates; the badge check makes sure the chosen run is used
        from driveflow.ai.custom_training import list_classifier_runs

        at.session_state["ia_model_request"] = next(str(r["run_dir"]) for r in list_classifier_runs("dc_motor") if Path(r["run_dir"]).resolve() == (ROOT / run).resolve())
    gen.click().run()
    assert not at.exception
    if run is not None:
        shown = " ".join(str(getattr(h, "body", "")) for h in at.get("html"))
        assert "/".join(Path(run).parts[-2:]) in shown, f"the selected run {run} is not the one evaluated"
    return at


#: selectable classifier -> the run to select in AI Diagnostics (None: the promoted PC run, the page's default model)
TIERS = {"promoted": None, "b4_candidate_5class": "configs/classifiers/pc_server_5class_w2048/2026-11-10_run01",
         "pc_3class": "configs/classifiers/pc_server/2026-11-04_run01"}


@pytest.mark.parametrize("tier", sorted(TIERS))
@pytest.mark.parametrize("speed", [80.0, 150.0, 220.0])
def test_inside_the_envelope_there_is_a_prediction_and_healthy_is_normal_or_abstains(tier, speed):
    at = _sample(TIERS[tier], speed)
    assert not any(OUTSIDE in e.value for e in at.error)
    shown = [m.value for m in at.metric]
    assert shown, f"{tier} at {speed}: no prediction"
    assert any(v in ("normal", "No class suggested") for v in shown), (tier, speed, shown)


@pytest.mark.parametrize("tier", sorted(TIERS))
@pytest.mark.parametrize("speed", [300.0, 350.0])
def test_outside_the_envelope_there_is_no_prediction(tier, speed):
    at = _sample(TIERS[tier], speed)
    assert any(OUTSIDE in e.value and "no diagnosis" in e.value for e in at.error), (tier, speed)
    assert not at.metric and not at.get("plotly_chart"), f"{tier} at {speed}: a prediction is still shown"
