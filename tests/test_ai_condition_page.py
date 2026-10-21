"""AI Diagnostics condition states in the GUI (expansion task E10a, spec Sec. 8.2): Known condition,
Uncertain, Unknown condition, Input invalid and "OOD detection unavailable".

The promoted model is replaced by a stub whose probabilities and OOD score are fixed, so each state
is reached deterministically; the decision itself is the real ai/condition.assess()."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import driveflow.viz.adapters.ai_adapter as da_ai
import driveflow.viz.views.ai_diagnostics as ai
from driveflow.ai.registry import RegistryError

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")
CLASSES = ["inner_race", "normal", "outer_race"]
OOD = {"method": "mahalanobis", "target_coverage": 0.95, "fitted_on": {"statistics": "train", "thresholds": "val"},
       "mahalanobis": {"threshold": 10.0, "layer": None}, "energy": {"threshold": -1.0}}
CAL = {"temperature": 1.0, "abstain_below": 0.6}


class _Model:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)

    def __call__(self, x, training=False):
        return SimpleNamespace(numpy=lambda: np.repeat(self.probs[np.newaxis], len(x), axis=0))


def _install(monkeypatch, tmp_path, probs, metrics, ood_score=None):
    run_dir = tmp_path / "stub_family" / "run01"
    run_dir.mkdir(parents=True)
    config = SimpleNamespace(tier="pc", input_window=64)
    full = {"classes": CLASSES, "channels": ["current_r", "rpm"], **metrics}
    monkeypatch.setattr(da_ai, "promoted_model", lambda d, t, b: (_Model(probs), config, full, run_dir))
    monkeypatch.setattr(da_ai, "window_score", lambda model, w, ood: ood_score)

    def _no_regressor(*a, **k):
        raise RegistryError("no regressor in this test")

    monkeypatch.setattr(ai, "promoted_model", _no_regressor)


def _run():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    at.button(key="ia_generate").click().run()
    assert not at.exception
    return at


def _boxes(at):
    return {kind: " ".join(e.value for e in at.get(kind)) for kind in ("success", "info", "warning", "error")}


def _captions(at):
    return " ".join(c.value for c in at.caption)


def test_known_condition(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, [0.05, 0.9, 0.05], {"ood": OOD, "calibration": CAL}, ood_score=3.0)
    at = _run()
    b = _boxes(at)
    assert "Known condition" in b["success"]
    assert "Unknown condition" not in b["error"] and "Uncertain" not in b["warning"]
    assert any(m.value == "normal" for m in at.metric)
    assert "OOD detection unavailable" not in _captions(at)


def test_uncertain(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, [0.4, 0.35, 0.25], {"ood": OOD, "calibration": CAL}, ood_score=3.0)
    at = _run()
    b = _boxes(at)
    assert "Uncertain — low confidence" in b["warning"] and "cannot distinguish reliably among its known classes" in b["warning"]
    assert not b["success"] and "Unknown condition" not in b["error"]
    assert any(m.value == "No class suggested" for m in at.metric)


def test_unknown_condition(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, [0.01, 0.98, 0.01], {"ood": OOD, "calibration": CAL}, ood_score=42.0)
    at = _run()
    b = _boxes(at)
    assert "Unknown condition — REQUIRES VERIFICATION" in b["error"] and "42.00" in b["error"] and "10.00" in b["error"]
    # a confident "normal" must not appear as a healthy result while the OOD flag is up
    assert not b["success"] and all(m.value != "normal" for m in at.metric)
    assert any(m.value == "Unknown condition" for m in at.metric)


def test_input_invalid(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, [0.05, 0.9, 0.05], {"ood": OOD, "calibration": CAL, "channels": ["current_r", "not_a_channel"]}, ood_score=3.0)
    at = _run()
    b = _boxes(at)
    assert "Input invalid" in b["error"] and "not_a_channel" in b["error"]
    assert not at.metric and not b["success"]


def test_detector_absent(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, [0.01, 0.98, 0.01], {"calibration": CAL}, ood_score=1e9)
    at = _run()
    b = _boxes(at)
    assert "OOD detection unavailable" in _captions(at)
    assert "Unknown condition" not in b["error"]
    # without an OOD check a confident answer is not shown in green
    assert not b["success"] and "confidence check only" in b["info"]
    assert "OOD score" not in " ".join(b.values())


@pytest.mark.parametrize("probs, score, state", [([0.05, 0.9, 0.05], 3.0, "known"), ([0.01, 0.98, 0.01], 42.0, "unknown")])
def test_evidence_records_the_state(monkeypatch, tmp_path, probs, score, state):
    _install(monkeypatch, tmp_path, probs, {"ood": OOD, "calibration": CAL}, ood_score=score)
    at = _run()
    assert any(e.proto.label == "Export evidence (JSON)" for e in at.get("download_button"))
    import pandas as pd

    df = pd.DataFrame({"current_r": np.linspace(0, 1, 100), "rpm": np.linspace(0, 1, 100)})
    a, _, info = da_ai.assess_last_window("dc_motor", df)
    ev = da_ai.condition_evidence(a, info["model_id"], {"Run ID": "r1"}, None)
    assert ev["state"] == state and ev["ood_available"] is True and ev["input"] == {"Run ID": "r1"}
    assert ev["ood_score"] == score and "secondary_agent_hypothesis" not in ev


def test_ood_validation_report_matches_run_and_threshold(tmp_path):
    import json

    run_rel = "configs/classifiers/stub/run01"
    report = {"run": run_rel, "eval": {"methods": {"mahalanobis": {"threshold": 10.0, "in_distribution_false_unknown_rate": 0.04,
                                                                   "ood_ball": {"n": 9, "unknown_rate": 0.1, "auroc_vs_id_test": 0.8}}}}}
    (tmp_path / "ood_detector_a.json").write_text(json.dumps(report))
    (tmp_path / "ood_detector_b.json").write_text(json.dumps({**report, "eval": {"methods": {"mahalanobis": {"threshold": 99.0}}}}))
    run_dir = da_ai.REPO_ROOT / run_rel
    rep = da_ai.ood_validation_report(run_dir, {"ood": OOD}, tmp_path)
    assert rep["path"].endswith("ood_detector_a.json") and rep["ood_sets"] == {"ball": {"n": 9, "unknown_rate": 0.1, "auroc": 0.8}}
    assert da_ai.ood_validation_report(run_dir, {"ood": {**OOD, "mahalanobis": {"threshold": 5.0}}}, tmp_path) is None
    assert da_ai.ood_validation_report(da_ai.REPO_ROOT / "configs/other", {"ood": OOD}, tmp_path) is None
    assert da_ai.ood_validation_report(run_dir, {}, tmp_path) is None


def test_real_a1_report_is_found_for_its_run():
    """The A1 report of configs/classifiers/pc_server/2026-10-18_run01 is matched by its threshold."""
    import json

    rep_json = json.loads((da_ai.OOD_REPORT_DIR / "ood_detector.json").read_text())
    thr = rep_json["eval"]["methods"]["mahalanobis"]["threshold"]
    rep = da_ai.ood_validation_report(da_ai.REPO_ROOT / rep_json["run"], {"ood": {**OOD, "mahalanobis": {"threshold": thr}}})
    assert rep is not None and "ball" in rep["ood_sets"] and 0 <= rep["ood_sets"]["ball"]["unknown_rate"] <= 1
