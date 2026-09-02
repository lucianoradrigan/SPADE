"""Fixed-length training windows from driveflow's diagnosis dataset (already domain-filtered by
dataset.py). Adapted from paper_federative's `repo/02_local_baselines_etapa1/train_baseline.py`
(`discover_channels`/`downsample_window`/`load_dataset`) and
`repo/03_forecast_regressors_etapa2/train_forecast_envelope_combined.py`
(`load_forecast_envelope`) -- simplified because driveflow's data comes from ONE simulation
engine at a single, known sample rate (`fs_hz = 1/TAU`), not five real datasets at five different
native rates that need per-file decimation/resampling to a common target. No `downsample_*`
function exists here for that reason -- there is nothing to resample.

WINDOW_S=0.5s matches the original's physical window duration; WINDOW_SAMPLES is computed from
driveflow's own fs_hz (5000 at TAU=1e-4s), not hardcoded to the original's 4000 (FS_TARGET=8000Hz)
-- see docs/macro_fase_C1_diagnosis.md for why this wasn't resampled to match instead.
"""

import math

import numpy as np
import pandas as pd

WINDOW_S = 0.5

#: driveflow's schema subset of paper_federative's CANDIDATE_CHANNELS (canonical order) --
#: acc_x2/y2/z2, audio, temperature_a/b don't exist in driveflow's schema at all (no second
#: sensor, no audio channel, no temperature simulated), so they're not listed as candidates here.
CANDIDATE_CHANNELS = [
    "acc_x", "acc_y", "acc_z",
    "current_r", "current_s", "current_t",
    "rpm", "torque_nm",
]


def window_samples_for(fs_hz: float, window_s: float = WINDOW_S) -> int:
    return int(round(fs_hz * window_s))


def is_live_window(w: np.ndarray) -> bool:
    """True if the window has real variation -- a near-constant window (e.g. current_s/t,
    always NaN and excluded upstream, or a channel with only quantization noise) would have its
    tiny sigma amplified to huge values if z-score normalized; it's zeroed instead. Same
    threshold as paper_federative's `_is_live_window`."""
    mu, sigma = float(w.mean()), float(w.std())
    return sigma > 1e-3 * (abs(mu) + 1e-6)


def discover_channels(
    df: pd.DataFrame, window_samples: int, min_frac: float = 0.5, min_live_frac: float = 0.5, group_col: str = "source_file"
) -> list:
    """Which of CANDIDATE_CHANNELS are populated AND genuinely vary within a window.

    Samples check-windows spread across every group (scenario run) present in df, one window per
    group, rather than the first few windows of the concatenated array. This matters specifically
    for driveflow's data in a way it didn't for the original's per-file discover_channels: the
    electrical/mechanical path here is fully deterministic (no process/measurement noise) --
    current_r/rpm/torque_nm are *exactly* constant once a healthy run settles, and only carry
    variation (the fault's torque ripple) in faulted runs. A sequential sample landing entirely
    on healthy-run windows would flag those channels as "dead" and discard exactly the signal
    A.5's own closing criterion validated as diagnostic (see docs/patch2_retiro_modulo_C.md /
    the current_r FFT peak check) -- found as a real bug generating the first Fase C dataset, not
    a hypothetical. Spreading across groups means a channel that's flat for one class (e.g.
    "normal") but live for others still gets a fair live_frac, instead of being judged only on
    whichever class happened to sort first in the concatenated DataFrame.
    """
    if len(df) < window_samples:
        return []
    groups = df[group_col].unique() if group_col in df.columns else [None]
    present = []
    for c in CANDIDATE_CHANNELS:
        if c not in df.columns or df[c].notna().mean() <= min_frac:
            continue
        live, checked = 0, 0
        for g in groups:
            sub_vals = df[c].to_numpy() if g is None else df.loc[df[group_col] == g, c].to_numpy()
            if len(sub_vals) < window_samples:
                continue
            # A window from the run's settled tail, not the start: the initial transient (e.g.
            # a speed reference step) is generically "live" for every class -- sampling it would
            # make every channel look live for a spurious reason, defeating the point of this
            # check. Taking the LAST full window instead reads steady-state behavior, which is
            # exactly where the diagnostic signal this function exists to detect (or reject) is.
            start = len(sub_vals) - (len(sub_vals) % window_samples) - window_samples
            w = sub_vals[start : start + window_samples].astype(np.float32)
            checked += 1
            if is_live_window(w):
                live += 1
        if checked > 0 and live / checked >= min_live_frac:
            present.append(c)
    return present


def build_classification_windows(
    df: pd.DataFrame,
    classes: list,
    channels: list,
    window_samples: int,
    cap_per_class: int = 500,
    group_col: str = "source_file",
    label_col: str = "label",
) -> tuple:
    """Adapted from load_dataset(): windows a labeled DataFrame into (X, y_str, groups).
    `groups` = group_col's value for each window (a driveflow scenario run is already
    single-label for its whole duration, unlike the original's per-file mixed-label case, so
    grouping is simpler here -- no need to filter df by label within a group first).

    Returns (None, None, None) if no windows could be built at all.
    """
    all_X, all_y, all_groups = [], [], []

    for cls in classes:
        cls_df = df[df[label_col] == cls]
        if cls_df.empty:
            continue
        groups_present = cls_df[group_col].unique()
        win_per_group = max(20, math.ceil(cap_per_class / len(groups_present)))
        collected = 0

        for src in groups_present:
            if collected >= cap_per_class:
                break
            sub = cls_df[cls_df[group_col] == src]
            n_rows = len(sub)
            if n_rows < window_samples:
                continue
            n_wins = min(win_per_group, n_rows // window_samples, cap_per_class - collected)

            col_values = {c: sub[c].to_numpy() for c in channels if c in sub.columns and sub[c].notna().all()}

            for i in range(n_wins):
                sl = slice(i * window_samples, (i + 1) * window_samples)
                chans = []
                for c in channels:
                    if c in col_values:
                        w = col_values[c][sl].astype(np.float32)
                        if is_live_window(w):
                            mu, sigma = w.mean(), w.std()
                            w = (w - mu) / sigma
                        else:
                            w = np.zeros_like(w)
                    else:
                        w = np.zeros(window_samples, dtype=np.float32)
                    chans.append(w)
                all_X.append(np.stack(chans, axis=-1))
                all_y.append(cls)
                all_groups.append(src)
                collected += 1

    if not all_X:
        return None, None, None
    return np.array(all_X, dtype=np.float32), np.array(all_y), np.array(all_groups)


def build_forecast_windows(
    df: pd.DataFrame,
    channels: list,
    window_samples: int,
    horizon_samples: int,
    n_bins: int,
    cap_per_group: int = 60,
    group_col: str = "source_file",
) -> tuple:
    """Adapted from load_forecast_envelope(): (context window, future RMS-per-bin envelope)
    pairs for the envelope forecaster. Unlike build_classification_windows, this ignores
    `label` -- the forecaster predicts signal magnitude, not fault class, and paper_federative's
    original trains it across all conditions pooled together the same way.

    Returns (X, Y, groups): X (n, window_samples, C), Y (n, n_bins, C).
    """
    bin_size = horizon_samples // n_bins
    seg_len = window_samples + n_bins * bin_size
    all_X, all_Y, all_groups = [], [], []

    for src in df[group_col].unique():
        sub = df[df[group_col] == src]
        n_rows = len(sub)
        if n_rows < seg_len:
            continue
        present_cols = [c for c in channels if c in sub.columns and sub[c].notna().all()]
        if len(present_cols) < len(channels):
            continue
        col_values = {c: sub[c].to_numpy() for c in present_cols}

        n_segments = min(cap_per_group, n_rows // seg_len)
        for i in range(n_segments):
            sl = slice(i * seg_len, i * seg_len + seg_len)
            ctx_chans, env_chans = [], []
            for c in channels:
                raw = col_values[c][sl].astype(np.float32)
                ctx_raw = raw[:window_samples]
                hor_raw = raw[window_samples : window_samples + n_bins * bin_size]
                if is_live_window(ctx_raw):
                    mu, sigma = ctx_raw.mean(), ctx_raw.std()
                    ctx_n = (ctx_raw - mu) / sigma
                    hor_n = (hor_raw - mu) / sigma
                else:
                    ctx_n = np.zeros_like(ctx_raw)
                    hor_n = np.zeros_like(hor_raw)
                env = np.sqrt(np.mean(hor_n.reshape(n_bins, bin_size) ** 2, axis=1))
                ctx_chans.append(ctx_n.astype(np.float32))
                env_chans.append(env.astype(np.float32))
            all_X.append(np.stack(ctx_chans, axis=-1))
            all_Y.append(np.stack(env_chans, axis=-1))
            all_groups.append(src)

    if not all_X:
        return None, None, None
    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32), np.array(all_groups)


def build_direct_forecast_windows(
    df: pd.DataFrame,
    channels: list,
    window_samples: int,
    horizon_samples: int,
    cap_per_group: int = 60,
    group_col: str = "source_file",
) -> tuple:
    """Like build_forecast_windows, but the target Y is the raw (normalized) future signal itself
    -- (n, horizon_samples, C) -- not an RMS-per-bin envelope. Used by
    models/regressors/builder.py's build_forecaster (the config-driven LSTM/GRU family,
    docs/design_ai_layer_transversal.md Sec. 6.3) -- a sibling of build_forecast_windows /
    envelope_forecaster.py's RMS-envelope convention, not a replacement for it: Fase D.1's
    dpc_tracking_forecaster still uses that pairing directly (INSTRUCTIONS.md Sec. 6).

    Returns (X, Y, groups): X (n, window_samples, C), Y (n, horizon_samples, C).
    """
    seg_len = window_samples + horizon_samples
    all_X, all_Y, all_groups = [], [], []

    for src in df[group_col].unique():
        sub = df[df[group_col] == src]
        n_rows = len(sub)
        if n_rows < seg_len:
            continue
        present_cols = [c for c in channels if c in sub.columns and sub[c].notna().all()]
        if len(present_cols) < len(channels):
            continue
        col_values = {c: sub[c].to_numpy() for c in present_cols}

        n_segments = min(cap_per_group, n_rows // seg_len)
        for i in range(n_segments):
            sl = slice(i * seg_len, i * seg_len + seg_len)
            ctx_chans, fut_chans = [], []
            for c in channels:
                raw = col_values[c][sl].astype(np.float32)
                ctx_raw = raw[:window_samples]
                fut_raw = raw[window_samples : window_samples + horizon_samples]
                if is_live_window(ctx_raw):
                    mu, sigma = ctx_raw.mean(), ctx_raw.std()
                    ctx_n = (ctx_raw - mu) / sigma
                    fut_n = (fut_raw - mu) / sigma
                else:
                    ctx_n = np.zeros_like(ctx_raw)
                    fut_n = np.zeros_like(fut_raw)
                ctx_chans.append(ctx_n.astype(np.float32))
                fut_chans.append(fut_n.astype(np.float32))
            all_X.append(np.stack(ctx_chans, axis=-1))
            all_Y.append(np.stack(fut_chans, axis=-1))
            all_groups.append(src)

    if not all_X:
        return None, None, None
    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32), np.array(all_groups)
