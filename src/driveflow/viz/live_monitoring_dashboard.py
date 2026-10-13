"""Compatibility shim: the LM tab moved to driveflow.viz.views.monitoring (Monitoring page) and
driveflow.viz.adapters.monitoring_adapter (agent orchestration)."""

from driveflow.viz.adapters.monitoring_adapter import alert_from_anomaly_score as _alert_from_anomaly_score  # noqa: F401
from driveflow.viz.views.monitoring import render as _render_fase_lm  # noqa: F401


def _record_classifier_confidence_if_available(domain, df, server_agent):
    """Legacy signature: returns the recorded confidence (float) or None."""
    from driveflow.viz.adapters.monitoring_adapter import record_classifier_confidence_if_available

    out = record_classifier_confidence_if_available(domain, df, server_agent)
    return None if out is None else out["confidence"]
