"""Plotly conventions (guide Sec. 3.3): transparent backgrounds, consistent margins and legend,
axis titles WITH units, zoom/export controls left on. Fonts and grid colours are deliberately not
hard-coded -- st.plotly_chart's default theme="streamlit" adapts them to the active light/dark
theme, which a fixed hex value would break in one of the two.

Chart titles stay in Streamlit markdown above each chart (same lesson as the legacy dashboard:
Plotly's own title collides with a top legend)."""

import plotly.graph_objects as go

from driveflow.viz.theme.tokens import SERIES

#: Passed to every st.plotly_chart: keep zoom/pan and PNG export, drop the noisy extras.
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "toImageButtonOptions": {"format": "png", "scale": 2}}


def style_figure(fig: go.Figure, height: int = 320, x_title: str | None = None, y_title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=16, t=28 if legend else 12, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=12)),
        font=dict(size=13),
    )
    # Only set titles that were given: subplot figures carry per-row axis titles already.
    fig.update_xaxes(zeroline=False, tickfont=dict(size=12), **({"title_text": x_title, "title_font": dict(size=13)} if x_title else {}))
    fig.update_yaxes(zeroline=False, tickfont=dict(size=12), **({"title_text": y_title, "title_font": dict(size=13)} if y_title else {}))
    return fig


def series_color(role: str) -> str:
    return SERIES.get(role, SERIES["electrical"])
