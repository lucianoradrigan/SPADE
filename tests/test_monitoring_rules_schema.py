"""Regression test for the rule-schema safeguard (docs/design_ai_layer_transversal.md Sec. 6.5,
Sec. 8 step 2): every YAML file under monitoring/rules/ must load and validate cleanly, and a
malformed rule must fail at load time -- same pattern as
tests/test_diagnosis_dataset_filter.py for the Fase C domain filter.
"""

from pathlib import Path

import pytest

from driveflow.monitoring.rules import Rule, RuleSet, RuleValidationError, load_all_rulesets, load_ruleset
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM

RULES_DIR = Path(__file__).resolve().parents[1] / "src" / "driveflow" / "monitoring" / "rules"


class TestShippedRuleFiles:
    def test_all_shipped_yaml_files_load(self):
        rulesets = load_all_rulesets(RULES_DIR)
        assert "vsc_dpc" in rulesets

    def test_vsc_dpc_has_the_load_resistance_divergence_rule(self):
        ruleset = load_ruleset(RULES_DIR / "vsc_dpc.yaml")
        rule = next(r for r in ruleset.rules if r.name == "dpc_load_resistance_divergence")
        assert rule.severity == "high"
        assert rule.action == "alert"
        assert "load_resistance_ohm" in rule.condition

    def test_divergence_rule_upper_bound_stays_below_the_dashboard_floor(self):
        """Cross-check against the analytic/empirical source of truth (Patch 9): the rule's
        divergence region must stay strictly below MIN_STABLE_LOAD_RESISTANCE_OHM, so this rule
        can't silently drift out of sync with sim.vsc_system's own stability floor."""
        ruleset = load_ruleset(RULES_DIR / "vsc_dpc.yaml")
        rule = next(r for r in ruleset.rules if r.name == "dpc_load_resistance_divergence")
        upper_bound = float(rule.condition.split("<=")[-1].strip())
        assert upper_bound < MIN_STABLE_LOAD_RESISTANCE_OHM


class TestLoadAllRulesetsRejectsDuplicateDomains:
    def test_duplicate_domain_across_files_raises(self, tmp_path):
        (tmp_path / "a.yaml").write_text(
            "domain: vsc_dpc\nrules:\n  - name: r1\n    condition: 'x > 0'\n    severity: low\n"
            "    hysteresis_seconds: 0\n    action: log\n"
        )
        (tmp_path / "b.yaml").write_text(
            "domain: vsc_dpc\nrules:\n  - name: r2\n    condition: 'x > 0'\n    severity: low\n"
            "    hysteresis_seconds: 0\n    action: log\n"
        )
        with pytest.raises(RuleValidationError, match="declared in both"):
            load_all_rulesets(tmp_path)


class TestRuleValidation:
    def _base_kwargs(self):
        return dict(name="r1", condition="x > 0", severity="low", hysteresis_seconds=0.0, action="alert")

    def test_valid_rule_constructs(self):
        Rule(**self._base_kwargs())

    def test_unknown_severity_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["severity"] = "hihg"
        with pytest.raises(RuleValidationError, match="severity"):
            Rule(**kwargs)

    def test_negative_hysteresis_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["hysteresis_seconds"] = -1.0
        with pytest.raises(RuleValidationError, match="hysteresis_seconds"):
            Rule(**kwargs)

    def test_empty_action_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["action"] = ""
        with pytest.raises(RuleValidationError, match="action"):
            Rule(**kwargs)

    def test_malformed_condition_syntax_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["condition"] = "x >"
        with pytest.raises(RuleValidationError, match="not a valid expression"):
            Rule(**kwargs)

    @pytest.mark.parametrize(
        "condition",
        [
            "__import__('os').system('echo pwned')",
            "x.__class__",
            "[i for i in range(3)]",
            "(lambda: 1)()",
            "open('/etc/passwd')",
        ],
    )
    def test_disallowed_syntax_rejected(self, condition):
        kwargs = self._base_kwargs()
        kwargs["condition"] = condition
        with pytest.raises(RuleValidationError, match="disallowed syntax"):
            Rule(**kwargs)

    def test_duplicate_rule_names_in_a_ruleset_rejected(self):
        rule = Rule(**self._base_kwargs())
        with pytest.raises(RuleValidationError, match="duplicate rule name"):
            RuleSet(domain="vsc_dpc", rules=(rule, rule))


class TestLoadRulesetRejectsMalformedFiles:
    def test_missing_domain_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("rules: []\n")
        with pytest.raises(RuleValidationError, match="domain"):
            load_ruleset(path)

    def test_missing_rules_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("domain: vsc_dpc\n")
        with pytest.raises(RuleValidationError, match="rules"):
            load_ruleset(path)

    def test_unknown_field_in_rule_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "domain: vsc_dpc\nrules:\n  - name: r1\n    condition: 'x > 0'\n    severity: low\n"
            "    hysteresis_seconds: 0\n    action: alert\n    extra_unknown_field: true\n"
        )
        with pytest.raises(RuleValidationError, match="malformed rule entry"):
            load_ruleset(path)
