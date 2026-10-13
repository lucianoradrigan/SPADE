"""Compatibility shim: the IA tab moved to driveflow.viz.views.ai_diagnostics (AI Diagnostics page)."""

from driveflow.viz.views.ai_diagnostics import channel_upload_template_df as _channel_upload_template_df  # noqa: F401
from driveflow.viz.views.ai_diagnostics import generate_sample_dataframe as _generate_sample_dataframe  # noqa: F401
from driveflow.viz.views.ai_diagnostics import render as _render_fase_ia  # noqa: F401
