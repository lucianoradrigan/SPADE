"""Time-series and spectrum figures with units on every axis (guide Sec. 3.4, DC Motor page)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from driveflow.analysis import amplitude_spectrum, dominant_peak
from driveflow.viz.theme.plotly import series_color, style_figure


def time_series_figure(df, rows: list, ref_cols: list | None = None, role: str = "electrical", vlines: list | None = None) -> go.Figure:
    """rows: [(column, axis label with unit)]. ref_cols aligned with rows: None, a column name, a
    scalar (constant setpoint) or an array the length of df (NaN = no reference active).
    Same contract as the legacy dashboard's _plotly_lines, now theme-token driven."""
    ref_cols = ref_cols or [None] * len(rows)
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for i, ((col, ylabel), ref) in enumerate(zip(rows, ref_cols), start=1):
        fig.add_trace(go.Scatter(x=df["timestamp_s"], y=df[col], mode="lines", name=col, line=dict(width=1.5, color=series_color(role)), showlegend=False), row=i, col=1)
        if ref is not None:
            ref_y = df[ref] if isinstance(ref, str) else ([ref] * len(df) if np.isscalar(ref) else ref)
            fig.add_trace(go.Scatter(x=df["timestamp_s"], y=ref_y, mode="lines", name="reference", line=dict(width=1.2, color=series_color("reference"), dash="dash"), showlegend=False), row=i, col=1)
        fig.update_yaxes(title_text=ylabel, row=i, col=1, title_font=dict(size=12))
    for t in vlines or []:
        fig.add_vline(x=float(t), line_width=1, line_dash="dot", line_color=series_color("reference"))
    style_figure(fig, height=150 * len(rows) + 60, legend=False)
    fig.update_xaxes(title_text="Time (s)", row=len(rows), col=1)
    return fig


def spectrum_figure(signal, fs_hz: float, name: str, unit: str, role: str = "electrical", markers: dict | None = None, skip_fraction: float = 0.2, log_y: bool = True, fmax_hz: float | None = None) -> tuple:
    """Returns (figure, peak) where peak = (f_hz, amplitude) of the dominant component above 1 Hz,
    computed from THIS signal (annotated only because it was computed). `markers` are labelled
    vertical lines at physically meaningful frequencies (e.g. BPFO from the run's own columns)."""
    freqs, amp = amplitude_spectrum(signal, fs_hz, skip_fraction=skip_fraction)
    if fmax_hz is not None:
        keep = freqs <= fmax_hz
        freqs, amp = freqs[keep], amp[keep]
    peak = dominant_peak(freqs, amp)
    fig = go.Figure(go.Scatter(x=freqs, y=amp, mode="lines", name=name, line=dict(width=1.2, color=series_color(role))))
    for label, f in (markers or {}).items():
        if f is not None and np.isfinite(f) and f > 0 and (fmax_hz is None or f <= fmax_hz):
            fig.add_vline(x=float(f), line_width=1, line_dash="dot", line_color=series_color("reference"), annotation_text=label, annotation_position="top")
    if peak[0] is not None:
        fig.add_trace(go.Scatter(x=[peak[0]], y=[peak[1]], mode="markers", name=f"peak {peak[0]:.1f} Hz", marker=dict(size=9, color=series_color("secondary"), symbol="diamond")))
    style_figure(fig, height=320, x_title="Frequency (Hz)", y_title=f"Amplitude ({unit})")
    if log_y:
        fig.update_yaxes(type="log")
    return fig, peak
