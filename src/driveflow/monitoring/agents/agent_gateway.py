"""Raspberry Pi 5 tier monitoring agent (docs/design_ai_layer_transversal.md Sec. 4.3, Sec. 8
step 9): evaluates one domain's RuleSet (monitoring.rules.schema) against a stream of telemetry
readings. Bounded, stateful rule engine -- per-rule hysteresis (a condition must hold continuously
for `hysteresis_seconds` before it fires) and debounce (a firing rule doesn't re-fire on every
subsequent tick while it stays true; only the true->alerting and alerting->cleared transitions
produce an Alert). No dependency on the PC tier's ServerAgent -- matches Sec. 4.3's "puede operar
sin conexión al PC"; feeding its Alert events to a ServerAgent (agent_server.py) is the caller's
choice, not something GatewayAgent does itself.

Rule conditions were already syntax-validated against a whitelist of AST node types at
RuleSet-load time (schema.py's module docstring) -- no Call/Attribute/Subscript nodes are
possible, so compiling and eval()'ing them here against a telemetry namespace with
`__builtins__` stripped is safe (same reasoning as, and consistent with, that validation).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from driveflow.monitoring.rules.schema import RuleSet


class TelemetryFieldMissing(KeyError):
    """A rule's condition referenced a telemetry field absent from the reading passed to step()."""


def _compile_condition(condition: str):
    return compile(ast.parse(condition, mode="eval"), "<rule-condition>", "eval")


@dataclass(frozen=True)
class Alert:
    """One state transition for a rule: fired (resolved=False) when its condition has held for
    at least `hysteresis_seconds`, or cleared (resolved=True) when the condition goes false again
    after having fired. Nothing is emitted for every tick a condition stays true or false --
    that's the debounce Sec. 4.3 asks for."""

    rule_name: str
    severity: str
    action: str
    timestamp: float
    resolved: bool = False


class _RuleState:
    __slots__ = ("condition_since", "active")

    def __init__(self):
        self.condition_since = None
        self.active = False


class GatewayAgent:
    """One instance per (domain, live telemetry stream) -- see module docstring."""

    def __init__(self, ruleset: RuleSet):
        self.ruleset = ruleset
        self._compiled = {rule.name: _compile_condition(rule.condition) for rule in ruleset.rules}
        self._state = {rule.name: _RuleState() for rule in ruleset.rules}

    def step(self, telemetry: dict, timestamp: float) -> list:
        """Evaluates every rule once against `telemetry` at `timestamp` (a monotonically
        increasing clock the caller controls -- e.g. simulation time, not wall time, so this is
        deterministic and testable). Returns the Alerts (fired or cleared) produced on this
        tick, in rule-declaration order; usually empty."""
        events = []
        for rule in self.ruleset.rules:
            try:
                truthy = bool(eval(self._compiled[rule.name], {"__builtins__": {}}, dict(telemetry)))
            except NameError as exc:
                raise TelemetryFieldMissing(
                    f"rule {rule.name!r} condition {rule.condition!r} needs a telemetry field this reading doesn't have: {exc}"
                ) from exc
            state = self._state[rule.name]
            if truthy:
                if state.condition_since is None:
                    state.condition_since = timestamp
                if not state.active and (timestamp - state.condition_since) >= rule.hysteresis_seconds:
                    state.active = True
                    events.append(Alert(rule.name, rule.severity, rule.action, timestamp))
            else:
                if state.active:
                    events.append(Alert(rule.name, rule.severity, rule.action, timestamp, resolved=True))
                state.condition_since = None
                state.active = False
        return events

    def active_alerts(self) -> list:
        """Rule names currently in the alerting state (survives across step() calls -- e.g. for a
        dashboard badge that needs "what's wrong right now", not just this tick's transitions)."""
        return [name for name, state in self._state.items() if state.active]
