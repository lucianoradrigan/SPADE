"""Inference-only access to promoted models.

Caching a keras.Model with st.cache_resource is correct HERE and only here: these instances are
used for prediction and are never trained (transfer learning loads its own private copy and
fine-tunes a clone -- AUDIT AI-04). RegistryError is not cached, so a later promotion is picked
up on the next call."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from driveflow.ai.calibration import UNCERTAIN, UNKNOWN, decide
from driveflow.ai.ood import window_score
from driveflow.ai.registry import RegistryError, load_promoted_model, resolve
from driveflow.models.common import normalize_window_like_training


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
