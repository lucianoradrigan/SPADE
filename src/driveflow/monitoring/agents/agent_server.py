"""PC-tier monitoring agent (docs/design_ai_layer_transversal.md Sec. 4.3, Sec. 8 step 9): the
"razonamiento contextual amplio" role -- unlike GatewayAgent (Raspberry Pi 5), it does not
evaluate rule conditions itself. It consumes the Alert events a GatewayAgent already produced
(one ServerAgent, many domains: A and B each keep their own GatewayAgent+ruleset upstream, but
share one ServerAgent, since correlating across them is exactly what this tier is for) and
classifier confidence scores, and reasons across domains and time:

- Aggregates an alert history per domain (Sec. 5.3's "historial de alertas" / dashboard detail
  view) and derives a current per-domain badge status ("ok"/"alert", Sec. 5.3's top-bar
  indicator) from it -- no rule evaluation of its own, so it can't drift out of sync with
  whatever the domain's GatewayAgent already decided.
- Tracks a rolling window of classifier confidence per domain and flags drift (a sustained drop
  vs. an older baseline window) as a suggestion to retrain -- Sec. 4.3's "detecta drift de
  confianza en los clasificadores, puede sugerir reentrenamiento". This is a heuristic
  trigger for a human to look into, not an automatic retraining action.

Correlating "between A and B" (Sec. 4.3) means aggregating each domain's own alerts/confidence
into one place that can see both -- NOT mixing their telemetry or models, which would violate the
domain isolation Sec. 1 requires.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceDriftReport:
    domain: str
    baseline_mean: float
    recent_mean: float
    drift: float
    suggest_retraining: bool


class ServerAgent:
    def __init__(self, confidence_window: int = 20, baseline_window: int = 100, drift_threshold: float = 0.15):
        """confidence_window: how many of the most recent predictions count as "recent".
        baseline_window: how many predictions of history to retain in total (older ones are
        dropped); the baseline mean is computed over whatever of that history isn't in the
        recent window. drift_threshold: baseline_mean - recent_mean at or above this triggers
        suggest_retraining."""
        if confidence_window >= baseline_window:
            raise ValueError(f"confidence_window ({confidence_window}) must be < baseline_window ({baseline_window}), or there's no older history left to compare against")
        self._confidence_window = confidence_window
        self._baseline_window = baseline_window
        self._drift_threshold = drift_threshold
        self._confidence_history: dict = {}
        self._alert_log: list = []
        self._domain_status: dict = {}

    def record_alert(self, domain: str, alert) -> None:
        """Ingest one Alert emitted by `domain`'s GatewayAgent.step(). Updates that domain's
        badge status (Sec. 5.3) and appends to the alert log -- `alert` needs only rule_name/
        severity/action/timestamp/resolved attributes, so this also accepts an
        agent_gateway.Alert or any duck-typed equivalent (e.g. a future ESP32 watchdog event)."""
        self._alert_log.append(
            {
                "domain": domain,
                "rule_name": alert.rule_name,
                "severity": alert.severity,
                "action": alert.action,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
            }
        )
        self._domain_status[domain] = "ok" if alert.resolved else "alert"

    def domain_status(self, domain: str) -> str:
        """"ok" until the first unresolved alert for `domain` is recorded; "alert" until it is
        resolved. A domain with no alerts recorded yet is "ok"."""
        return self._domain_status.get(domain, "ok")

    def alert_history(self, domain: str | None = None) -> list:
        """Full log in recording order (Sec. 5.3's bitácora), optionally filtered to one domain."""
        if domain is None:
            return list(self._alert_log)
        return [entry for entry in self._alert_log if entry["domain"] == domain]

    def record_classifier_confidence(self, domain: str, confidence: float) -> None:
        """Sec. 4.3's confidence-drift detection input: one classifier prediction's confidence
        score for `domain`, appended to its rolling history (oldest dropped past
        baseline_window)."""
        history = self._confidence_history.setdefault(domain, deque(maxlen=self._baseline_window))
        history.append(confidence)

    def check_confidence_drift(self, domain: str) -> ConfidenceDriftReport | None:
        """None if `domain` doesn't have enough recorded confidence history yet to compare a
        baseline against a recent window (needs more than confidence_window points total).
        Otherwise compares the mean of the most recent `confidence_window` predictions against
        the mean of whatever older history is retained."""
        history = self._confidence_history.get(domain)
        if history is None or len(history) <= self._confidence_window:
            return None
        values = list(history)
        recent = values[-self._confidence_window :]
        baseline = values[: -self._confidence_window]
        baseline_mean = sum(baseline) / len(baseline)
        recent_mean = sum(recent) / len(recent)
        drift = baseline_mean - recent_mean
        return ConfidenceDriftReport(
            domain=domain,
            baseline_mean=baseline_mean,
            recent_mean=recent_mean,
            drift=drift,
            suggest_retraining=drift >= self._drift_threshold,
        )
