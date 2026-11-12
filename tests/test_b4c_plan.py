"""B4c (pre-registration docs/audit/b4/b4c_confirmatory_preregistration.md, approved 2026-11-12): the E4c set uses a seed
block disjoint from every earlier set (as Q3), matches E4's layout per speed, and no condition equals an off-grid healthy run."""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import generate_b4_sets as G  # noqa: E402

C = yaml.safe_load((REPO / "configs" / "b4c_plan.yaml").read_text())
EARLIER = {"grids_before_b4": range(1001, 1001 + 675), "b4_grids": range(80001, 80001 + 675), "b4_offgrid": range(85001, 85001 + 60),
           "b4_e4": range(90001, 90001 + 60), "b4_e5": range(95001, 95001 + 30), "ai_n27_heldout": range(60001, 60001 + 1000),
           "evaluation": range(200, 204)}


def test_e4c_seeds_are_disjoint_from_every_earlier_block():
    seeds = [j["seed"] for j in G.e4c_jobs()]
    assert len(seeds) == len(set(seeds)) == 150 and min(seeds) == 100001 and max(seeds) == 100150
    for name, block in EARLIER.items():
        assert not set(seeds) & set(block), name


def test_e4c_layout_matches_e4_per_speed():
    jobs = G.e4c_jobs()
    assert sorted({j["speed"] for j in jobs}) == [45.0, 50.0, 55.0, 65.0, 70.0]
    for s in {j["speed"] for j in jobs}:
        js = [j for j in jobs if j["speed"] == s]
        assert len(js) == len([j for j in G.e4_jobs("5class") if j["speed"] == 50.0]) == 30
        assert {j["fault"] for j in js} == {None, "outer_race", "inner_race", "ball", "cage"}
        assert {j["severity_scale"] for j in js if j["fault"]} == {1.0, 1.5}
    assert all(j["plant_config_id"] == "dc_perm_ex_v2" and j["duration_s"] == 2.5 for j in jobs)


def test_e4c_conditions_are_not_offgrid_healthy_runs():
    off = {float(c["omega_ref_rad_s"]) for c in G.PLAN["esp32_offgrid_healthy"]}
    assert not off & {j["speed"] for j in G.e4c_jobs()}


def test_e4c_does_not_change_e4():
    assert [j["seed"] for j in G.e4_jobs("5class")] == list(range(90001, 90061))
