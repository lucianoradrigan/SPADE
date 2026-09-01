"""Rule schema for monitoring/rules/*.yaml (docs/design_ai_layer_transversal.md Sec. 6.5): the
only validated entry point for loading a domain's rule set, so a malformed rule fails at load
time (import/test time) rather than once it's deployed on an agent tier (PC/Raspberry Pi 5/ESP32,
Sec. 4.3 of that document).

`condition` is a boolean expression string, evaluated later by an agent (monitoring/agents/*.py,
not implemented yet -- see the design doc's Sec. 8 implementation order) against a namespace of
telemetry field names. It is NOT eval()'d here or anywhere in this module: only its *syntax* is
checked (via ast.parse plus a whitelist of allowed node types), so a typo'd condition fails at
rule-load time instead of at whatever later point an agent first evaluates it -- and this module
never becomes a place a hand-edited (or future LLM-generated) rules YAML could execute arbitrary
code from.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Graded severity vocabulary -- closed set so a typo (e.g. "hihg") fails at load time instead of
#: silently falling through whatever downstream code branches on severity.
ALLOWED_SEVERITIES = frozenset({"low", "medium", "high", "critical"})

#: AST node types a condition expression may use: comparisons, boolean/arithmetic combinators,
#: names (telemetry fields), and literal constants. Deliberately excludes Call, Attribute,
#: Subscript, comprehensions, lambdas, etc. -- a monitoring rule condition has no legitimate need
#: for a function call or attribute access, and allowing them would make this the one place a
#: rules YAML file could execute arbitrary code.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.Name,
    ast.Load,
    ast.Constant,
)


class RuleValidationError(ValueError):
    """A rule (or rule set) failed schema validation -- raised at load time, see module docstring."""


def _validate_condition_syntax(condition: str, rule_name: str) -> None:
    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise RuleValidationError(f"rule {rule_name!r}: condition {condition!r} is not a valid expression: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleValidationError(
                f"rule {rule_name!r}: condition {condition!r} uses disallowed syntax ({type(node).__name__}) -- "
                "only comparisons/boolean/arithmetic expressions over telemetry field names and literals are "
                "permitted, see monitoring/rules/schema.py's module docstring."
            )


@dataclass(frozen=True)
class Rule:
    """One monitoring rule (docs/design_ai_layer_transversal.md Sec. 6.5).

    Args:
        name: Unique identifier within its RuleSet.
        condition: Boolean expression over telemetry field names, e.g.
            "load_resistance_ohm >= 1.0 and load_resistance_ohm <= 3.0". Syntax-checked here (see
            module docstring); evaluated later by an agent, not by this module.
        severity: One of ALLOWED_SEVERITIES.
        hysteresis_seconds: How long the condition must hold before an agent acts on it (debounce
            -- Sec. 4.3's "histéresis, debounce" for the Raspberry Pi 5 tier). Must be >= 0.
        action: Free-form label naming what an agent should do when the rule fires (e.g. "alert").
            Deliberately not a closed enum here: the set of actions an agent can take is a
            property of the agent implementation (not built yet, Sec. 8 order step 9), not of the
            rule schema.
    """

    name: str
    condition: str
    severity: str
    hysteresis_seconds: float
    action: str

    def __post_init__(self):
        if not self.name:
            raise RuleValidationError("rule name must be non-empty")
        if self.severity not in ALLOWED_SEVERITIES:
            raise RuleValidationError(f"rule {self.name!r}: severity {self.severity!r} not in {sorted(ALLOWED_SEVERITIES)}")
        if self.hysteresis_seconds < 0:
            raise RuleValidationError(f"rule {self.name!r}: hysteresis_seconds must be >= 0, got {self.hysteresis_seconds}")
        if not self.action:
            raise RuleValidationError(f"rule {self.name!r}: action must be non-empty")
        _validate_condition_syntax(self.condition, self.name)


@dataclass(frozen=True)
class RuleSet:
    """All rules for one domain ("dc_motor" or "vsc_dpc", matching the domains DIAGNOSIS_PLANT_
    CONFIG_IDS/Scenario.plant_config_id distinguish -- see models/common/dataset.py), loaded from
    one YAML file."""

    domain: str
    rules: tuple

    def __post_init__(self):
        if not self.domain:
            raise RuleValidationError("RuleSet.domain must be non-empty")
        names = [r.name for r in self.rules]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise RuleValidationError(f"domain {self.domain!r}: duplicate rule name(s) {dupes}")


def load_ruleset(path) -> RuleSet:
    """Loads and validates one rules YAML file. Raises RuleValidationError on any schema
    violation -- malformed structure, unknown severity, invalid condition syntax, duplicate rule
    names -- so a bad rule fails here, not once an agent tries to act on it."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "domain" not in raw or "rules" not in raw:
        raise RuleValidationError(f"{path}: expected a mapping with 'domain' and 'rules' keys, got {raw!r}")
    try:
        rules = tuple(Rule(**r) for r in raw["rules"])
    except TypeError as exc:
        raise RuleValidationError(f"{path}: malformed rule entry: {exc}") from exc
    return RuleSet(domain=raw["domain"], rules=rules)


def load_all_rulesets(rules_dir) -> dict:
    """Loads every *.yaml file in rules_dir, keyed by domain. Raises RuleValidationError if two
    files declare the same domain (ambiguous which one an agent should use)."""
    rules_dir = Path(rules_dir)
    result = {}
    for yaml_path in sorted(rules_dir.glob("*.yaml")):
        ruleset = load_ruleset(yaml_path)
        if ruleset.domain in result:
            raise RuleValidationError(f"domain {ruleset.domain!r} declared in both {yaml_path} and a previous file")
        result[ruleset.domain] = ruleset
    return result
