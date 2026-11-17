"""D8 (2026-11-16): the Phase 2 held-out set (tests/assistant/golden/holdout.yaml) is only read with --final-eval.

- the list was fixed by experiments/select_golden_holdout.py (seed, method) and is one third of the golden set;
- the only Phase 2 code that reads expected files (experiments/score_golden_extraction.py) refuses held-out IDs without
  --final-eval, and by default scores only the development IDs;
- no other development code (experiments/, src/) opens tests/assistant/golden/expected/ directly. The catalogue report
  (experiments/report_golden_set.py) only checks that the files exist and is allowed.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import score_golden_extraction as S  # noqa: E402

HOLD = yaml.safe_load((ROOT / "tests" / "assistant" / "golden" / "holdout.yaml").read_text(encoding="utf-8"))
#: the guarded scorer; the catalogue report (existence only); the selection script (metadata only); the downloader (its
#: docstring names the directory, it never reads it)
ALLOWED = {"experiments/score_golden_extraction.py", "experiments/report_golden_set.py", "experiments/select_golden_holdout.py",
           "experiments/fetch_golden_datasheets.py"}


def test_the_holdout_is_one_third_with_seed_and_method():
    n = len(list(S.EXPECTED_DIR.glob("*.json")))
    assert HOLD["seed"] and HOLD["method"] and HOLD["n_holdout"] == len(HOLD["ids"]) == round(n / 3)
    assert set(HOLD["ids"]) <= {p.stem for p in S.EXPECTED_DIR.glob("*.json")}


def test_the_selection_is_reproducible():
    import select_golden_holdout as H

    assert sorted(H.select()["ids"]) == sorted(HOLD["ids"])


def test_the_scorer_refuses_held_out_ids_and_skips_them_by_default(tmp_path):
    held = sorted(HOLD["ids"])[0]
    with pytest.raises(S.HoldoutError):
        S.load_expected(held)
    assert S.load_expected(held, final_eval=True)["id"] == held
    with pytest.raises(S.HoldoutError):
        S.main(["--outputs", str(tmp_path), "--ids", held])
    out = tmp_path / "r.json"
    S.main(["--outputs", str(tmp_path), "--json", str(out)])
    import json

    scored = {d["id"] for d in json.loads(out.read_text(encoding="utf-8"))["documents"]}
    assert scored and not scored & set(HOLD["ids"])


def test_no_other_development_code_reads_the_expected_files():
    offenders = []
    for base in ("experiments", "src"):
        for p in (ROOT / base).rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            if rel in ALLOWED:
                continue
            if re.search(r"golden[/\\\\\\]+expected|golden\\W+expected", p.read_text(encoding="utf-8", errors="replace")):
                offenders.append(rel)
    assert not offenders, offenders
