"""Inference-only access to promoted models.

Caching a keras.Model with st.cache_resource is correct HERE and only here: these instances are
used for prediction and are never trained (transfer learning loads its own private copy and
fine-tunes a clone -- AUDIT AI-04). RegistryError is not cached, so a later promotion is picked
up on the next call."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from driveflow.ai import condition
from driveflow.ai.calibration import UNCERTAIN, UNKNOWN, decide
from driveflow.ai.ood import window_score
from driveflow.ai.registry import RegistryError, load_promoted_model, resolve
from driveflow.models.common import normalize_window_like_training


REPO_ROOT = Path(__file__).resolve().parents[4]
OOD_REPORT_DIR = REPO_ROOT / "docs" / "audit" / "ai"


@st.cache_resource(show_spinner=False)
def cached_promoted_model(domain: str, tier: str, block: str, run_dir_key: str = ""):
    # run_dir_key is part of the cache key on purpose (no leading underscore: Streamlit does not
    # hash underscore-prefixed arguments).
    del run_dir_key
    return load_promoted_model(domain, tier, block)


def promoted_model(domain: str, tier: str, block: str):
    """(model, config, metrics, run_dir) -- keyed on the resolved run directory, so promoting or
    rolling back a model invalidates the cached instance instead of serving the old one."""
    run_dir = resolve(domain, tier, block)
    model, config, metrics = cached_promoted_model(domain, tier, block, str(run_dir))
    return model, config, metrics, run_dir


def normalize_window(window: np.ndarray, metrics: dict | None = None) -> tuple:
    """Per-window, per-channel normalisation, delegated to the training pipeline's own rule
    (windowing.normalize_window_like_training): z-score of a live channel with the window's own
    statistics, zeros for a dead/near-constant one. This used a different rule before (sigma
    floor 1e-6, no dead-channel zeroing), so a near-constant channel -- e.g. current_r/torque_nm
    of a healthy motor in steady state -- reached the model as amplified noise in the GUI but as
    zeros at training time (docs/AI_TECHNICAL_AUDIT.md, AI-N01).

    metrics: the run's metrics.json; its channels + level_scaling (AI-N07) select the same
    per-channel rule the run was trained with. None = the original rule for every channel."""
    if metrics is None:
        return normalize_window_like_training(window)
    return normalize_window_like_training(window, metrics["channels"], metrics.get("level_scaling"))


def classify_last_window(domain: str, df) -> dict | None:
    """AI suggestion on the last input_window rows of df, or None if no classifier is promoted /
    channels are missing / too few rows. Returns probabilities with the model's identity."""
    try:
        model, config, metrics, run_dir = promoted_model(domain, "pc", "classifier")
    except RegistryError:
        return None
    channels = metrics["channels"]
    if len(df) < config.input_window or any(c not in df.columns or df[c].isna().all() for c in channels):
        return None
    window = df[channels].iloc[-config.input_window :].to_numpy(dtype=np.float32)
    window_n, _, _ = normalize_window(window, metrics)
    raw = model(window_n[np.newaxis, ...], training=False).numpy()[0]
    ood_score = window_score(model, window_n, metrics["ood"]) if "ood" in metrics else None
    label, probs = decide(raw, list(metrics["classes"]), metrics, ood_score)
    k = int(np.argmax(probs))
    return {
        "classes": list(metrics["classes"]), "probs": [float(p) for p in probs], "predicted": metrics["classes"][k], "confidence": float(probs[k]),
        "decision": label, "abstained": label == UNCERTAIN, "unknown": label == UNKNOWN, "synthetic_only": label in (metrics.get("synthetic_only_classes") or []), "ood_checked": ood_score is not None, "calibrated": "calibration" in metrics,
        "model_id": f"{run_dir.parent.name}/{run_dir.name}", "test_macro_f1": metrics.get("test_macro_f1"), "n_test": metrics.get("n_test"),
    }


def run_metrics(run_dir) -> dict:
    with (run_dir / "metrics.json").open() as f:
        return json.load(f)


def assess_last_window(domain: str, df: pd.DataFrame, tier: str = "pc"):
    """Condition assessment (ai/condition.py, task E10a) of the promoted classifier on the last
    input_window rows of df: (assessment, window, model_info), or None if no classifier is promoted.
    An invalid input is assessed as such and never reaches the model."""
    try:
        model, config, metrics, run_dir = promoted_model(domain, tier, "classifier")
    except RegistryError:
        return None
    info = {"model_id": f"{run_dir.parent.name}/{run_dir.name}", "run_dir": run_dir, "tier": config.tier, "channels": list(metrics["channels"]),
            "input_window": config.input_window, "metrics": metrics}
    problems = condition.window_problems(df, info["channels"], config.input_window)
    if problems:
        return condition.invalid(problems, metrics), None, info
    window = df[info["channels"]].iloc[-config.input_window:].to_numpy(dtype=np.float32)
    window_n, _, _ = normalize_window(window, metrics)
    raw = model(window_n[np.newaxis, ...], training=False).numpy()[0]
    available, _ = condition.ood_availability(metrics)
    ood_score = window_score(model, window_n, metrics["ood"]) if available else None
    return condition.assess(raw, list(metrics["classes"]), metrics, ood_score), window, info


def ood_validation_report(run_dir: Path, metrics: dict, report_dir: Path = OOD_REPORT_DIR) -> dict | None:
    """The A1 evaluation (experiments/report_ood_detector.py) of THIS run's fitted detector: the
    report whose run path and threshold match the run's metrics.json, or None. Several reports can
    cover one run (different feature layers); the threshold picks the one actually in use."""
    ood = metrics.get("ood") or {}
    method = ood.get("method")
    if method not in ("mahalanobis", "energy"):
        return None
    threshold = ood[method].get("threshold")
    try:
        run_rel = str(Path(run_dir).resolve().relative_to(REPO_ROOT))
    except ValueError:
        run_rel = str(run_dir)
    for path in sorted(report_dir.glob("ood_detector*.json")):
        try:
            data = json.loads(path.read_text())
            res = data["eval"]["methods"][method]
        except (OSError, ValueError, KeyError):
            continue
        if data.get("run") != run_rel or threshold is None or not np.isclose(res.get("threshold", np.nan), threshold, rtol=1e-9):
            continue
        ood_sets = {k[len("ood_"):]: v for k, v in res.items() if k.startswith("ood_") and isinstance(v, dict)}
        return {"path": path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path), "method": method,
                "id_false_unknown_rate": res.get("in_distribution_false_unknown_rate"),
                "ood_sets": {k: {"n": v.get("n"), "unknown_rate": v.get("unknown_rate"), "auroc": v.get("auroc_vs_id_test")} for k, v in ood_sets.items()}}
    return None


def agent_hypothesis(domain: str, df: pd.DataFrame, operating_point: dict | None = None) -> dict:
    """Secondary hint from the simulation-based agent (driveflow.agents): the closest simulated
    hypothesis and its anomaly score for the whole input. It runs several physics simulations, so
    the page triggers it on demand. operating_point None = the agent estimates it from the data."""
    from driveflow.viz.adapters.monitoring_adapter import AGENT_CLASS_BY_DOMAIN

    agent = AGENT_CLASS_BY_DOMAIN[domain]()
    telemetry = {f: df[f].to_numpy(dtype=float) for f in agent.monitored_features if f in df.columns and df[f].notna().all()}
    if not telemetry:
        raise ValueError(f"None of the agent's features ({', '.join(agent.monitored_features)}) is present and complete.")
    score, closest = agent.detect_anomaly(telemetry, operating_point=operating_point)
    return {"score": score, "closest_hypothesis": closest, "features": sorted(telemetry), "operating_point": dict(agent.last_operating_point or {}),
            "hypotheses": sorted(agent.HYPOTHESES)}


def condition_evidence(a, model_id: str, provenance: dict | None, agent_result: dict | None) -> dict:
    """What "Export evidence" writes: the assessment, the model and the input's provenance."""
    ev = {"state": a.state, "label": a.label, "model": model_id, "problems": list(a.problems), "top_class": a.top_class, "confidence": a.confidence,
          "calibrated": a.calibrated, "abstain_below": a.abstain_below, "ood_available": a.ood_available, "ood_reason": a.ood_reason, "ood_method": a.ood_method,
          "ood_score": a.ood_score, "ood_threshold": a.ood_threshold, "classes": list(a.classes), "probs": list(a.probs), "input": dict(provenance or {})}
    if agent_result is not None:
        ev["secondary_agent_hypothesis"] = agent_result
    return ev
