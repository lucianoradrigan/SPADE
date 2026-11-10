"""Round C (Luciano, 2026-11-09): tests.yml cancels superseded runs on dev and feat/* only, never on main."""

import re
from pathlib import Path

import yaml

WF = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml").read_text())


def _cancels(ref: str) -> bool:
    """Evaluate the cancel-in-progress expression for a ref (it only uses ==, ||, startsWith)."""
    expr = WF["concurrency"]["cancel-in-progress"].strip()[3:-2].strip()
    expr = re.sub(r"startsWith\(github\.ref, '([^']*)'\)", lambda m: str(ref.startswith(m.group(1))), expr)
    expr = re.sub(r"github\.ref == '([^']*)'", lambda m: str(ref == m.group(1)), expr)
    return eval(expr.replace("||", " or "), {"__builtins__": {}})  # noqa: S307 -- only True/False/or remain


def test_the_group_is_per_branch():
    assert "github.ref" in WF["concurrency"]["group"]


def test_cancel_only_on_dev_and_feature_branches():
    assert _cancels("refs/heads/dev") and _cancels("refs/heads/feat/S-25-x")
    assert not _cancels("refs/heads/main")
    assert not _cancels("refs/heads/release/v0.2.0") and not _cancels("refs/pull/7/merge")
