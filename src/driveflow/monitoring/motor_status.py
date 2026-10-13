"""Motor status view-model: the ONLY input the motor illustration is allowed to react to.

The motor picture in the dashboard is a metaphor, not a sensor. Its colour/label must therefore be
derived from backend evidence -- the ServerAgent's active alerts and whether the domain has been
evaluated at all -- never from the image, a demo loop, or the fault a simulation was configured
with. derive_motor_view() is that derivation (AUDIT AI-08, guide Sec. 4.1):

- CRITICAL: at least one active alert whose severity maps to CRITICAL.
- WARNING: active alert(s), none critical.
- HEALTHY: the domain was actually evaluated (a monitoring pass / heartbeat exists), no alert is
  active, and the source is not stale/offline.
- OFFLINE: a connected source missed its heartbeat timeout.
- UNKNOWN: nothing evaluated yet, or a demo / unavailable source -- never HEALTHY by default.

A classifier prediction is carried separately (`prediction`) as an "AI suggestion": it is shown
with its model version and confidence but does not by itself raise an alert or change status.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STATUSES = ("healthy", "warning", "critical", "offline", "unknown")
SOURCE_MODES = ("simulated", "connected", "demo", "unavailable")


@dataclass(frozen=True)
class MotorViewModel:
    motor_id: str
    domain: str
    source_mode: str
    status: str
    rpm: float | None = None
    rpm_target: float | None = None
    temperature_c: float | None = None
    vibration_rms_g: float | None = None
    timestamp_utc: str | None = None
    active_alerts: tuple = field(default_factory=tuple)
    model_id: str | None = None
    confidence: float | None = None
    prediction: str | None = None
    status_reason: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.source_mode not in SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {SOURCE_MODES}, got {self.source_mode!r}")

    @property
    def animate(self) -> bool:
        """Rotation cue only with a valid, positive speed from a non-offline source."""
        return self.rpm is not None and self.rpm > 0 and self.status != "offline" and self.source_mode != "unavailable"


def derive_motor_view(
    server_agent,
    domain: str,
    *,
    motor_id: str,
    source_mode: str,
    rpm: float | None = None,
    rpm_target: float | None = None,
    vibration_rms_g: float | None = None,
    timestamp_utc: str | None = None,
    now: float | None = None,
    prediction: str | None = None,
    confidence: float | None = None,
    model_id: str | None = None,
) -> MotorViewModel:
    """Builds the view-model from a ServerAgent (or None when no monitoring exists)."""
    if server_agent is None or source_mode in ("demo", "unavailable"):
        status, reason, alerts = "unknown", "no monitoring evidence for this source" if source_mode != "demo" else "demo source -- not verified", ()
    else:
        health = server_agent.domain_health(domain, now=now)
        alerts = tuple(f"{a.rule_name} ({a.severity})" for a in server_agent.active_alerts(domain))
        status = health.lower()
        reason = {
            "critical": "critical alert active",
            "warning": "warning alert active",
            "healthy": "evaluated, no active alert",
            "offline": "connected source missed its heartbeat",
            "unknown": "domain not evaluated yet",
        }[status]
    return MotorViewModel(
        motor_id=motor_id, domain=domain, source_mode=source_mode, status=status, rpm=rpm, rpm_target=rpm_target,
        vibration_rms_g=vibration_rms_g, timestamp_utc=timestamp_utc, active_alerts=alerts, model_id=model_id,
        confidence=confidence, prediction=prediction, status_reason=reason,
    )
