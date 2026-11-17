"""B4 (2026-11-16): the OOD map's seed block is disjoint from every earlier seed block, so its burned, exploratory data
cannot coincide with any training or evaluation run."""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ood_map", ROOT / "experiments" / "ood_map_2026_11_16.py")
om = importlib.util.module_from_spec(spec)
spec.loader.exec_module(om)
BLOCK = set(om.SEED_BLOCK)


def test_one_distinct_seed_per_run():
    j = om.jobs()
    assert len(j) == len(om.SPEEDS_RAD_S) * len(om.LOADS_NM) * len(om.FAULTS) == 200
    assert sorted(x["seed"] for x in j) == sorted(BLOCK)


def test_disjoint_from_the_documented_blocks():
    b4c = yaml.safe_load((ROOT / "configs" / "b4c_plan.yaml").read_text(encoding="utf-8"))
    known = {"earlier grids": range(1001, 1001 + 675), "b4 grids": range(80001, 80001 + 675), "b4 off-grid": range(85001, 85001 + 60),
             "E4": range(90001, 90001 + 60), "E5": range(95001, 95001 + 30), "evaluation": range(200, 204), "AI-N27 held-out": range(60001, 60001 + 1000),
             "E4c": range(b4c["e4c_first_run_seed"], b4c["e4c_first_run_seed"] + 150)}
    for name, r in known.items():
        assert not BLOCK & set(r), name


def test_no_other_script_or_plan_uses_a_seed_of_the_block():
    lo, hi = min(BLOCK), max(BLOCK)
    for path in list((ROOT / "experiments").glob("*.py")) + list((ROOT / "configs").glob("*.yaml")):
        if path.name == "ood_map_2026_11_16.py":
            continue
        for n in re.findall(r"(?<![\w.])(\d{6})(?![\w.])", path.read_text(encoding="utf-8", errors="replace")):
            assert not lo <= int(n) <= hi, f"{path.name} mentions {n}"


def test_no_stored_dataset_contains_a_seed_of_the_block():
    data = ROOT / "data"
    files = [p for p in data.glob("*.parquet") if "ood_map_2026-11-16" not in p.name] if data.exists() else []
    if not files:
        pytest.skip("no data/ in this checkout")
    import pyarrow.parquet as pq

    for p in files:
        if "seed" not in pq.read_schema(p).names:
            continue
        seeds = set(pq.read_table(p, columns=["seed"]).column("seed").to_pylist())
        assert not seeds & BLOCK, p.name
