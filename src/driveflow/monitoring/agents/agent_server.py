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

Concurrent alerts (AUDIT AI-08): the badge used to be `"ok" if alert.resolved else "alert"` of the
LAST event received, so resolving alert A while B was still firing showed the domain as healthy.
Active alerts are now tracked individually, keyed by (domain, rule_name, source_id), and the
domain's status is derived from ALL of them: domain_health() = the highest severity among active
alerts; HEALTHY only when the domain has actually reported (record_evaluation / heartbeat) and no
alert is active; UNKNOWN when nothing has been reported; OFFLINE when a connected source's
heartbeat is older than its timeout. Acknowledging an alert (operator has seen it) does NOT change
health; only a resolution does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

#: Rule/agent severity vocabulary -> health level. "low" maps to WARNING on purpose: an active,
#: unresolved alert of any severity is never HEALTHY. Configurable per ServerAgent.
DEFAULT_SEVERITY_LEVELS = {"critical": "CRITICAL", "high": "CRITICAL", "medium": "WARNING", "warning": "WARNING", "low": "WARNING", "info": "WARNING"}
_HEALTH_RANK = {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass
class ActiveAlert:
    domain: str
    rule_name: str
    source_id: str
    severity: str
    level: str
    first_seen: float
    last_seen: float
    action: str
    acknowledged_by: str | None = None
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceDriftReport:
    domain: str
    baseline_mean: float
    recent_mean: float
    drift: float
    suggest_retraining: bool


class ServerAgent:
    def __init__(self, confidence_window: int = 20, baseline_window: int = 100, drift_threshold: float = 0.15, severity_levels: dict | None = None):
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
        self._severity_levels = {**DEFAULT_SEVERITY_LEVELS, **(severity_levels or {})}
        self._active: dict = {}  # (domain, rule_name, source_id) -> ActiveAlert
        self._last_report: dict = {}  # domain -> (timestamp, source_mode)
        self._heartbeat: dict = {}  # domain -> (timestamp, timeout_s)

    def _level(self, severity: str) -> str:
        return self._severity_levels.get(str(severity).lower(), "WARNING")

    def record_alert(self, domain: str, alert, source_id: str = "default") -> None:
        """Ingest one Alert emitted by `domain`'s GatewayAgent.step(). Updates the set of active
        alerts and appends to the alert log -- `alert` needs only rule_name/severity/action/
        timestamp/resolved attributes, so this also accepts an agent_gateway.Alert or any
        duck-typed equivalent (e.g. a future ESP32 watchdog event). A resolution only clears the
        alert with the SAME (rule_name, source_id); every other active alert stays active."""
        key = (domain, alert.rule_name, source_id)
        self._alert_log.append(
            {
                "domain": domain,
                "rule_name": alert.rule_name,
                "source_id": source_id,
                "severity": alert.severity,
                "action": alert.action,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
            }
        )
        self._last_report[domain] = (alert.timestamp, self._last_report.get(domain, (None, None))[1])
        if alert.resolved:
            self._active.pop(key, None)
            return
        existing = self._active.get(key)
        if existing is None:
            self._active[key] = ActiveAlert(domain, alert.rule_name, source_id, alert.severity, self._level(alert.severity), alert.timestamp, alert.timestamp, alert.action)
        else:
            existing.last_seen = alert.timestamp
            existing.severity, existing.level = alert.severity, self._level(alert.severity)

    def record_evaluation(self, domain: str, timestamp: float, source_mode: str = "simulated") -> None:
        """Marks that `domain` WAS evaluated at `timestamp` (e.g. one monitoring pass ran), even if
        it raised no alert -- required before the domain can be reported HEALTHY."""
        self._last_report[domain] = (timestamp, source_mode)

    def record_heartbeat(self, domain: str, timestamp: float, timeout_s: float) -> None:
        """For a CONNECTED telemetry source: the last time it delivered data, and how long silence
        is tolerated before the domain is reported OFFLINE."""
        self._heartbeat[domain] = (timestamp, timeout_s)
        self.record_evaluation(domain, timestamp, source_mode="connected")

    def active_alerts(self, domain: str | None = None) -> list:
        return [a for a in self._active.values() if domain is None or a.domain == domain]

    def acknowledge(self, domain: str, rule_name: str, by: str, source_id: str = "default", note: str | None = None) -> bool:
        """Operator has SEEN the alert. Does not resolve it and does not change health."""
        alert = self._active.get((domain, rule_name, source_id))
        if alert is None:
            return False
        alert.acknowledged_by = by
        if note:
            alert.notes.append(note)
        self._alert_log.append({"domain": domain, "rule_name": rule_name, "source_id": source_id, "severity": alert.severity, "action": "acknowledged", "timestamp": alert.last_seen, "resolved": False, "by": by})
        return True

    def domain_health(self, domain: str, now: float | None = None) -> str:
        """HEALTHY / WARNING / CRITICAL / OFFLINE / UNKNOWN -- see module docstring."""
        hb = self._heartbeat.get(domain)
        if hb is not None and now is not None and now - hb[0] > hb[1]:
            return "OFFLINE"
        active = self.active_alerts(domain)
        if active:
            return max((a.level for a in active), key=lambda lvl: _HEALTH_RANK.get(lvl, 1))
        if domain not in self._last_report or self._last_report[domain][0] is None:
            return "UNKNOWN"
        return "HEALTHY"

    def last_report(self, domain: str) -> tuple:
        """(timestamp, source_mode) of the last evaluation/alert for `domain`, or (None, None)."""
        return self._last_report.get(domain, (None, None))

    def domain_status(self, domain: str) -> str:
        """Legacy two-state badge kept for existing callers: "alert" while ANY alert of `domain`
        is active, else "ok". Unlike domain_health() it cannot express UNKNOWN -- new code (the
        dashboard, the motor status view) must use domain_health()."""
        return "alert" if self.active_alerts(domain) else "ok"

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
