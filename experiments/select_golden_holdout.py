"""Phase 2 held-out set (decision D8, 2026-11-16): one third of the golden set, stratified by document style (A-F,
docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md §2) and by machine group, with a fixed seed. Uses only each
document's style and its motor/machine type (metadata), never its expected values. Run once; the list is committed
before any Phase 2 iteration and is not changed afterwards.

    python experiments/select_golden_holdout.py   # writes tests/assistant/golden/holdout.yaml
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EXPECTED = REPO / "tests" / "assistant" / "golden" / "expected"
OUT = REPO / "tests" / "assistant" / "golden" / "holdout.yaml"
SEED = 20261116
#: styles of the documents with expected values (docs/audit/assistant/golden_set.md); rating plates (W*, IG*) are style F
STYLE = {"D1": "A", "D2": "C", "D3": "D", "D5": "A", "P1": "C", "P2": "A", "P3": "C", "P5": "C", "P6": "C", "P7": "A",
         "P9": "B", "P10": "D", "I1": "E"}


def machine_group(e: dict) -> str:
    t = (e.get("motor_type") or e.get("machine_type") or "").lower()
    if "induction" in t or "asynchronous" in t:
        return "induction"
    if "generator" in t or "exciter" in t:
        return "generator"
    if any(k in t for k in ("pmsm", "bldc", "brushless", "synchronous", "servo")):
        return "pmsm_bldc"
    if "dc" in t:
        return "dc"
    return "other"


def select() -> dict:
    docs = {}
    for p in sorted(EXPECTED.glob("*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        docs[e["id"]] = (STYLE.get(e["id"], "F"), machine_group(e))
    strata = {}
    for i, key in docs.items():
        strata.setdefault(key, []).append(i)
    target = round(len(docs) / 3)
    rng = random.Random(SEED)
    quota = {k: math.floor(len(v) / 3) for k, v in strata.items()}
    rest = sorted(strata, key=lambda k: (-(len(strata[k]) / 3 - quota[k]), rng.random()))
    for k in rest[: target - sum(quota.values())]:
        quota[k] += 1
    ids = []
    for k in sorted(strata):
        members = sorted(strata[k])
        rng.shuffle(members)
        ids += members[: quota[k]]
    return {"decision": "D8 (Luciano, 2026-11-16)", "seed": SEED, "method": "stratified by (style A-F, machine group); per stratum floor(n/3), the remaining "
            "slots to the strata with the largest remainder (ties by the seeded RNG); members shuffled with the seeded RNG",
            "total_documents": len(docs), "n_holdout": len(ids), "ids": sorted(ids),
            "strata": {f"{s}/{g}": {"n": len(v), "held_out": quota[(s, g)]} for (s, g), v in sorted(strata.items())}}


if __name__ == "__main__":
    res = select()
    OUT.write_text("# Phase 2 held-out set: read only with experiments/score_golden_extraction.py --final-eval (D8).\n" + yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(res["n_holdout"], "of", res["total_documents"], res["strata"])
