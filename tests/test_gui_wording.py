"""No visible GUI text may say "live": SPADE has no real telemetry, and LIVE / CONNECTED is reserved
for it (Documentation page, theme tokens). The scan reads every string literal in
src/driveflow/viz/ that is not a docstring -- what a page can render -- and fails on the word
"live" outside the documented exceptions below."""

import ast
import re
from pathlib import Path

VIZ = Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz"
LIVE = re.compile(r"\blive\b", re.IGNORECASE)

#: (file, text fragment) pairs allowed to contain "live", each with its reason.
EXCEPTIONS = {
    # HTML accessibility attribute, not text.
    ("components/state_panel.py", "aria-live"),
    # The Documentation page states that LIVE is reserved and not implemented.
    ("views/documentation.py", "CONNECTED / LIVE"),
}


def _docstring_nodes(tree) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _visible_strings():
    for path in sorted(VIZ.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_nodes(tree)
        rel = path.relative_to(VIZ).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                yield rel, node.value


def test_no_visible_text_says_live():
    offending = []
    for rel, text in _visible_strings():
        if LIVE.search(text) and not any(rel == f and frag in text for f, frag in EXCEPTIONS):
            offending.append(f"{rel}: {text[:120]!r}")
    assert not offending, "visible GUI text says 'live':\n" + "\n".join(offending)


def test_the_scan_sees_the_motor_status_note():
    texts = [t for rel, t in _visible_strings() if rel == "components/motor_status.py"]
    assert any("simulation run or monitoring pass" in t for t in texts)
