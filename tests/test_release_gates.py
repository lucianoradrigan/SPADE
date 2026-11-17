"""B1 (2026-11-16): scripts/release_gates.py. The cheap gates run for real here; the heavy ones are checked through their
pieces: a gate that cannot run is 'manual', never 'pass'."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_gates", ROOT / "scripts" / "release_gates.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def _args(tmp_path, **kw):
    import argparse

    a = argparse.Namespace(label="test", commit=rg.sh(["git", "rev-parse", "HEAD"])[1].strip(), branch="dev", previous_tag=None,
                           browser_python="/nonexistent/python", out=tmp_path / "out", work=tmp_path, started=rg.now(), argv=["--label", "test"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_every_gate_has_a_runner_and_the_docs_name_them():
    assert set(rg.RUNNERS) == set(rg.GATES) == {f"G{i}" for i in range(1, 9)}
    text = (ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")
    assert all(g in text for g in rg.GATES) and "scripts/release_gates.py" in text and "manual" in text


def test_g7_and_g8_run_on_this_checkout(tmp_path):
    tag = rg.previous_tag(rg.sh(["git", "rev-parse", "HEAD"])[1].strip())
    if tag is None:
        pytest.skip("no v* tag in this clone")
    r7 = rg.g7(_args(tmp_path, previous_tag=tag))
    assert r7["status"] == "pass", r7  # the byte-for-byte tests are frozen (CLAUDE.md)
    r8 = rg.g8(_args(tmp_path))
    assert r8["status"] == "pass", r8
    assert set(r8["allowed"]) <= set(rg.HOME_PATH_ALLOWED)


def test_g7_without_a_baseline_is_manual_not_pass(tmp_path):
    assert rg.g7(_args(tmp_path))["status"] == "manual"


def test_browser_gates_without_playwright_are_manual_not_pass(tmp_path):
    assert rg.g4(_args(tmp_path))["status"] == "manual"


@pytest.mark.parametrize("row,ok", [
    ({"expected": "inside", "outside_box": False, "shown": ["normal"], "exception": False, "evaluated_is_selected": True}, True),
    ({"expected": "inside", "outside_box": False, "shown": ["No class suggested"], "exception": False, "evaluated_is_selected": True}, True),
    ({"expected": "inside", "outside_box": False, "shown": ["cage"], "exception": False, "evaluated_is_selected": True}, False),
    ({"expected": "inside", "outside_box": False, "shown": ["normal"], "exception": False, "evaluated_is_selected": False}, False),  # AI-N42
    ({"expected": "outside", "outside_box": True, "shown": [], "exception": False, "evaluated_is_selected": True}, True),
    ({"expected": "outside", "outside_box": True, "shown": ["ball"], "exception": False, "evaluated_is_selected": True}, False),
    ({"expected": "outside", "outside_box": False, "shown": ["normal"], "exception": False, "evaluated_is_selected": True}, False),  # AI-N40
    ({"expected": "unknown_envelope", "outside_box": False, "shown": ["normal"], "exception": False, "evaluated_is_selected": True}, False),
])
def test_g6_case_rule(row, ok):
    assert rg.g6_ok(row) is ok


def test_report_says_manual_is_not_a_pass(tmp_path):
    a = _args(tmp_path)
    results = [rg.gate(g, "pass", "cmd", "ok") for g in rg.GATES[:-1]] + [rg.gate("G8", "manual", "cmd", "not run here")]
    jp, mp = rg.write_report(a, results)
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["all_pass"] is False and data["simulated"] is True
    assert "not all gates pass" in mp.read_text(encoding="utf-8")
    with pytest.raises(AssertionError):
        rg.gate("G1", "passed", "cmd", "x")


def test_unknown_gate_is_rejected():
    with pytest.raises(SystemExit):
        rg.main(["--label", "x", "--only", "G9"])
