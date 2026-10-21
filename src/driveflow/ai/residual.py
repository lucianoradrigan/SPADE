"""Forecast-residual anomaly detector for the dc_motor forecasters -- expansion task E14.

A forecaster trained only on HEALTHY windows predicts the next ``horizon`` samples of current_r and
rpm. When a bearing fault adds torque ripple, the plant leaves the healthy dynamics the model learned
and the forecast error grows. That error is the anomaly score. It does not depend on the
classifier's fault classes.

Score of one window (decided before any faulted window was looked at):
1. Normalise the context exactly like training (windowing.normalize_window_like_training), predict,
   and map the forecast back to PHYSICAL units (``y * sigma + mu``). A residual in per-window
   z-score units would divide out the very ripple amplitude a fault adds.
2. Per target channel: RMSE of the physical residual over the horizon.
3. Divide each channel's RMSE by that channel's MEDIAN RMSE on healthy validation windows, then
   average over channels. A healthy window scores about 1.

``fit_threshold`` sets the threshold to the ``quantile`` (default 0.99) of the score on healthy
validation windows. It REFUSES any window that is not labelled "normal": the threshold must never
see a fault. The parameters live in a run's ``metrics.json["residual_detector"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driveflow.models.common import normalize_window_like_training, window_starts

HEALTHY = "normal"
DEFAULT_QUANTILE = 0.99


class ResidualDetectorError(ValueError):
    pass


def raw_windows(df: pd.DataFrame, channels: list, window: int, horizon: int, cap_per_group: int = 60, group_col: str = "source_file",
                time_col: str = "timestamp_s", start_s: float | None = None, end_s: float | None = None) -> tuple:
    """Physical (context + future) segments with the SAME placement as
    build_direct_forecast_windows: (segments (n, window+horizon, C), groups, labels, t_start).

    start_s / end_s: keep only segments that lie inside [start_s, end_s] of each run, e.g. the
    steady state after the start-up transient."""
    seg = window + horizon
    out, groups, labels, t0s = [], [], [], []
    for src, sub in df.groupby(group_col, sort=False):
        if len(sub) < seg or not all(c in sub.columns and sub[c].notna().all() for c in channels):
            continue
        t = sub[time_col].to_numpy(dtype=float) if time_col in sub.columns else None
        vals = sub[channels].to_numpy(dtype=np.float32)
        lab = str(sub["label"].iloc[0]) if "label" in sub.columns else HEALTHY
        kept = 0
        for s in window_starts(len(sub), seg, t):
            if t is not None and ((start_s is not None and t[s] < start_s) or (end_s is not None and t[s + seg - 1] > end_s)):
                continue
            out.append(vals[s:s + seg])
            groups.append(src)
            labels.append(lab)
            t0s.append(float(t[s]) if t is not None else float(s))
            kept += 1
            if kept >= cap_per_group:
                break
    if not out:
        return np.empty((0, seg, len(channels)), np.float32), np.array([]), np.array([]), np.array([])
    return np.stack(out), np.array(groups), np.array(labels), np.array(t0s)


def forecast(model, segments: np.ndarray, channels: list, window: int, level_scaling: dict | None = None, batch_size: int = 512) -> np.ndarray:
    """Physical-unit forecast of the horizon for each segment's context: (n, horizon, C)."""
    n = len(segments)
    if n == 0:
        return np.empty((0, segments.shape[1] - window, segments.shape[2]), np.float32)
    ctx = np.empty((n, window, segments.shape[2]), np.float32)
    mus = np.empty((n, segments.shape[2]), np.float32)
    sigmas = np.empty_like(mus)
    for i in range(n):
        ctx[i], mus[i], sigmas[i] = normalize_window_like_training(segments[i, :window], channels, level_scaling)
    pred = model.predict(ctx, batch_size=batch_size, verbose=0) if n > batch_size else np.asarray(model(ctx, training=False))
    return pred * sigmas[:, None, :] + mus[:, None, :]


def channel_rmse(segments: np.ndarray, pred: np.ndarray, channels: list, target_channels: list, window: int) -> np.ndarray:
    """(n, n_targets) RMSE of the physical residual over the horizon, per target channel."""
    idx = [channels.index(c) for c in target_channels]
    resid = segments[:, window:, idx] - pred[:, :, idx]
    return np.sqrt(np.mean(resid ** 2, axis=1))


def scores_from_rmse(rmse: np.ndarray, params: dict) -> np.ndarray:
    scale = np.asarray(params["channel_scale"], dtype=float)
    return np.mean(np.asarray(rmse, dtype=float) / scale, axis=1)


def fit_threshold(val_rmse: np.ndarray, val_labels, target_channels: list, quantile: float = DEFAULT_QUANTILE) -> dict:
    """Detector parameters from HEALTHY validation windows only. Raises ResidualDetectorError if
    any window is not labelled "normal", or if there are too few windows for the quantile."""
    labels = np.asarray(val_labels).astype(str)
    bad = sorted(set(labels) - {HEALTHY})
    if bad:
        raise ResidualDetectorError(f"fit_threshold accepts healthy ('{HEALTHY}') validation windows only; got labels {bad}")
    rmse = np.asarray(val_rmse, dtype=float)
    if rmse.ndim != 2 or rmse.shape[1] != len(target_channels):
        raise ResidualDetectorError(f"val_rmse must be (n, {len(target_channels)}), got {rmse.shape}")
    if not 0.5 < quantile < 1:
        raise ResidualDetectorError(f"quantile must be in (0.5, 1), got {quantile}")
    if len(rmse) < int(np.ceil(1 / (1 - quantile))):
        raise ResidualDetectorError(f"{len(rmse)} windows are too few for a {quantile:.0%} quantile")
    scale = np.median(rmse, axis=0)
    scale[scale <= 0] = 1.0
    params = {"target_channels": list(target_channels), "channel_scale": [float(x) for x in scale], "quantile": float(quantile),
              "fitted_on": {"windows": "validation", "labels": [HEALTHY], "n": int(len(rmse))},
              "score": "mean over target channels of RMSE(physical residual over the horizon) / median healthy-validation RMSE"}
    params["threshold"] = float(np.quantile(scores_from_rmse(rmse, params), quantile))
    return params


def score(model, segments: np.ndarray, metrics: dict) -> np.ndarray:
    """Scores of segments (n, window+horizon, C) in the run's channel order."""
    det = metrics["residual_detector"]
    window = int(metrics["input_window"])
    pred = forecast(model, segments, list(metrics["channels"]), window, metrics.get("level_scaling"))
    return scores_from_rmse(channel_rmse(segments, pred, list(metrics["channels"]), det["target_channels"], window), det)


def is_anomalous(scores, metrics: dict) -> np.ndarray:
    return np.asarray(scores) > metrics["residual_detector"]["threshold"]
