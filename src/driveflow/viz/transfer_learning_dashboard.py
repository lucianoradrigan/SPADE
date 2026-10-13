"""Compatibility shim: the TL tab moved to driveflow.viz.views.transfer_learning (wizard UI) and
driveflow.ai.transfer.workflow (Streamlit-free orchestration). The legacy helpers that carried the
AUDIT AI-03/AI-06 defects (_prepare_windows with val == test, ungated _save_finetuned_artifacts)
were removed rather than kept as dead code; their replacements are workflow.prepare_dataset() and
workflow.save_candidate()."""

from driveflow.ai.transfer.workflow import next_run_dir as _next_run_dir  # noqa: F401
from driveflow.viz.views.transfer_learning import render as _render_fase_tl  # noqa: F401
