"""D1 (2026-10-22): transfer-learning acceptance floors -- macro-F1 >= 0.88 and recall >= 0.70 for
every class, always applied, raise-only, recorded with their origin in the validation and gate reports."""

import numpy as np
import pytest

from driveflow.ai.promotion import PromotionGate
from driveflow.ai.transfer.validator import (
    CLASSIFICATION_FLOORS,
    D1_ORIGIN,
    REVIEWER_ORIGIN,
    FloorError,
    ModelValidator,
    floor_checks,
    resolve_floors,
)


class _Fixed:
    """A 'model' that returns fixed probabilities (one row per test window)."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)

    def predict(self, X, verbose=0):
        return self.probs[: len(X)]


def _case(y_true, y_pred, k):
    return np.zeros((len(y_true), 4, 1), np.float32), np.asarray(y_true), np.eye(k)[np.asarray(y_pred)]


def _report(y_true, y_pred, k, thresholds=None):
    X, y, probs = _case(y_true, y_pred, k)
    rep = ModelValidator({}, thresholds or {}).evaluate(_Fixed(probs), {"X": X, "y": y}, "classification", base_comparable=False,
                                                        class_names=[f"c{i}" for i in range(k)], history={"loss": [1.0, 0.5]})
    return rep.to_dict()


def _gate(report, approved=False):
    return PromotionGate().evaluate(
        dataset_summary={"ok": True, "sha256": "x", "n_acquisitions": 9}, split_manifest={"disjoint": True, "n_groups": {"train": 3, "val": 3, "test": 3}},
        test_evaluation={"n_evaluations": 1, "used_for_selection": False, "same_test_for_base": False, "base_comparable": False}, validation_report=report,
        base_unchanged=True, lineage={"base_run_dir": "b", "base_weights_sha256": "s", "dataset_sha256": "d", "strategy": "new_head", "seed": 0, "split": {"a": 1}},
        human_approval={"approved": approved, "approved_by": "reviewer" if approved else None})


def _status(rep, name):
    return {c["name"]: c["status"] for c in rep["checks"]}[name]


def test_defaults_are_the_d1_values():
    assert CLASSIFICATION_FLOORS == {"min_macro_f1": 0.88, "min_per_class_recall": 0.70}
    assert resolve_floors(None) == {"min_macro_f1": {"value": 0.88, "origin": D1_ORIGIN}, "min_per_class_recall": {"value": 0.70, "origin": D1_ORIGIN}}


def test_macro_f1_090_with_one_class_at_recall_05_is_blocked():
    # 5 classes: classes 0-3 have 20 windows, all right; class 4 has 16, half right and half spread over the others.
    y_true = [k for k in range(4) for _ in range(20)] + [4] * 16
    y_pred = [k for k in range(4) for _ in range(20)] + [4] * 8 + [0, 0, 1, 1, 2, 2, 3, 3]
    rep = _report(y_true, y_pred, 5)
    assert rep["metrics"]["macro_f1"] == pytest.approx(0.895, abs=0.005)  # ≈ 0.90, above the macro floor
    assert _status(rep, "min_macro_f1") == "PASS" and _status(rep, "min_per_class_recall") == "FAIL"
    dec = _gate(rep, approved=True)
    assert not dec.approved and "validation:min_per_class_recall" in [i.name for i in dec.blocking]


def test_macro_f1_087_is_blocked():
    checks = {c.name: c.status.value for c in floor_checks(0.87, {"a": 0.8, "b": 0.9}, resolve_floors(None))}
    assert checks == {"min_macro_f1": "FAIL", "min_per_class_recall": "PASS"}


def test_090_with_every_class_above_070_passes_the_floors_but_still_needs_a_reviewer():
    # 5 classes, 20 windows each; 2 errors in each of classes 0-4 -> recall 0.9 everywhere, macro-F1 0.9
    y_true = [k for k in range(5) for _ in range(20)]
    y_pred = [k if i >= 2 else (k + 1) % 5 for k in range(5) for i in range(20)]
    rep = _report(y_true, y_pred, 5)
    assert rep["metrics"]["macro_f1"] == pytest.approx(0.90, abs=1e-6)
    assert _status(rep, "min_macro_f1") == "PASS" and _status(rep, "min_per_class_recall") == "PASS"
    dec = _gate(rep, approved=False)
    assert [i.name for i in dec.blocking] == ["human_approval"]
    assert dec.to_dict()["floors"]["min_macro_f1"] == {"value": 0.88, "origin": D1_ORIGIN}
    assert _gate(rep, approved=True).approved


@pytest.mark.parametrize("name, value", [("min_macro_f1", 0.80), ("min_per_class_recall", 0.5), ("min_class_recall", 0.6)])
def test_a_reviewer_cannot_lower_a_floor(name, value):
    with pytest.raises(FloorError, match="raise a floor, not lower it"):
        resolve_floors({name: value})
    y = [0] * 10 + [1] * 10
    with pytest.raises(FloorError):
        _report(y, y, 2, thresholds={name: value})


def test_a_reviewer_can_raise_a_floor_and_the_origin_is_recorded():
    floors = resolve_floors({"min_macro_f1": 0.95})
    assert floors["min_macro_f1"] == {"value": 0.95, "origin": REVIEWER_ORIGIN} and floors["min_per_class_recall"]["origin"] == D1_ORIGIN


def test_the_gate_rejects_an_edited_report_with_lower_floors():
    y_true = [k for k in range(5) for _ in range(20)]
    rep = _report(y_true, y_true, 5)
    rep["floors"]["min_macro_f1"]["value"] = 0.5  # tampered after the fact
    dec = _gate(rep, approved=True)
    assert not dec.approved and "acceptance_floors" in [i.name for i in dec.blocking]
