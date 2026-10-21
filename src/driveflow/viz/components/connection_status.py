"""ConnectionStatus (spec Sec. 9, 3.14.5): whether an external tool answers. "Available" means the
software responds -- it is never a telemetry or LIVE indicator."""

from __future__ import annotations

from driveflow.viz.components.state_panel import ComponentView, render_state

HEALTH = ("not_configured", "checking", "unavailable", "available", "capability_missing")


def connection_view(provider: str, health: str, host: str | None = None, capabilities: tuple = (), last_checked: str | None = None,
                    missing_capability: str | None = None) -> ComponentView:
    if health not in HEALTH:
        raise ValueError(f"health must be one of {HEALTH}, got {health!r}")
    where = " · ".join(p for p in (f"host {host}" if host else None, f"checked {last_checked} UTC" if last_checked else None) if p) or None
    if health == "not_configured":
        return ComponentView("empty", f"{provider} not configured", where)
    if health == "checking":
        return ComponentView("loading", "Testing connection…", where)
    if health == "unavailable":
        return ComponentView("error", f"{provider} RPC unavailable", where)
    if health == "capability_missing":
        return ComponentView("negative", f"Connected; required capability unavailable: {missing_capability or 'unspecified'}", where)
    caps = ", ".join(capabilities) if capabilities else "no capability verified yet"
    return ComponentView("success", f"{provider} available (RPC)", f"{caps}" + (f" · {where}" if where else ""))


def connection_status(provider: str, health: str, **kwargs) -> None:
    render_state(connection_view(provider, health, **kwargs))
