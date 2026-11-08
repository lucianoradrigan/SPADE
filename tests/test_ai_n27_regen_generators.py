"""AI-N27 option 2 (docs/audit/assistant/ai_n27_regen_preregistration.md): the generators' converter
options. Default output is unchanged (no extra column); "dcm" is passed to the Scenario and tagged;
the held-out set E2a has the pre-registered size, cells and seeds."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import generate_ai_n27_heldout as h  # noqa: E402
import generate_diagnosis_grid_dataset as g  # noqa: E402
import report_classifier_evaluation as ce  # noqa: E402


def _job(**kw):
    job = g.build_jobs(1, 0.02, 7, (None, "outer_race"))[1]
    job.update(kw)
    return job


def test_grid_default_has_no_converter_columns_and_dcm_is_tagged():
    gem = g._simulate(_job())
    dcm = g._simulate(_job(dc_converter_model="dcm", dc_plant_substeps=4))
    assert "grid_dc_converter_model" not in gem.columns and "grid_dc_plant_substeps" not in gem.columns
    assert set(dcm["grid_dc_converter_model"]) == {"dcm"} and set(dcm["grid_dc_plant_substeps"]) == {4}
    assert gem["source_file"].equals(dcm["source_file"]) and len(gem) == len(dcm)
    assert np.array_equal(gem["timestamp_s"].to_numpy(), dcm["timestamp_s"].to_numpy())


def test_grid_cli_tags_every_job(monkeypatch, tmp_path):
    seen = []

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, f, jobs):
            seen.extend(jobs)
            raise SystemExit(0)

    monkeypatch.setattr(g.mp, "get_context", lambda _m: type("C", (), {"Pool": _Pool})())
    try:
        g.main(["--dc-converter-model", "dcm", "--dc-plant-substeps", "1000", "--seeds", "1", "--out", str(tmp_path / "x.parquet")])
    except SystemExit:
        pass
    assert seen and all(j["dc_converter_model"] == "dcm" and j["dc_plant_substeps"] == 1000 for j in seen)
    # same acquisitions, draws and names as without the options; since decision E3 (2026-11-08) the CLI's
    # default plant is dc_perm_ex_v2 (--plant dc_perm_ex_v1 reproduces the grids made before)
    base = g.build_jobs(1, 2.5, 7, (None, "outer_race", "inner_race"))
    assert all(j["plant_config_id"] == "dc_perm_ex_v2" for j in seen)
    assert [{k: v for k, v in j.items() if not k.startswith("dc_") and k != "plant_config_id"} for j in seen] == base


def test_eval_cache_simulation_takes_plant_options():
    job = ("nominal_steady_state", None, 200, {}, 1.0, 0.02)
    a = ce._simulate(job)
    b = ce._simulate(job + ({"dc_converter_model": "gem"},))
    assert a["current_r"].equals(b["current_r"]) and a["source_file"].equals(b["source_file"])


def test_heldout_set_is_the_preregistered_one():
    j3, j5 = h.build_jobs("3class"), h.build_jobs("5class")
    assert len(j3) == 45 and len(j5) == 90
    for jobs, (faults, n, cells) in ((j3, h.GRIDS["3class"]), (j5, h.GRIDS["5class"])):
        assert all(j["dc_converter_model"] == "dcm" and j["dc_plant_substeps"] == 1000 and j["duration_s"] == 2.5 for j in jobs)
        assert {(j["speed"], j["severity_scale"]) for j in jobs if j["fault"]} == set(cells)
        seeds = [j["seed"] for j in jobs]
        assert len(set(seeds)) == len(seeds) and min(seeds) == 60001
        for c in (None, *faults):
            assert sum(j["fault"] == c for j in jobs) == n * len(cells)
    grid_seeds = {j["seed"] for j in g.build_jobs(3, 2.5, 7, (None, "outer_race", "inner_race", "ball", "cage"))}
    assert not grid_seeds & {j["seed"] for j in j3 + j5} and not {200, 201, 202, 203} & {j["seed"] for j in j3 + j5}
    assert h.build_jobs("3class") == j3  # reproducible


def test_ood_cache_jobs_take_plant_options():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import experiments.report_ood_detector as od

    base, dcm = od._jobs("custom_orders"), od._jobs("custom_orders", {"dc_converter_model": "dcm", "dc_plant_substeps": 1000})
    assert len(base) == len(dcm) and all(not isinstance(j[-1], dict) for j in base)
    assert all(j[:-1] == b and j[-1] == {"dc_converter_model": "dcm", "dc_plant_substeps": 1000} for j, b in zip(dcm, base))
    short = ("custom", "c", "unknown_1p3x", 1.3, 11, 120.0, 1.0, {"dc_converter_model": "gem"})
    od.DURATION_S, saved = 0.02, od.DURATION_S
    try:
        a, b = od._sim(short[:-1]), od._sim(short)
    finally:
        od.DURATION_S = saved
    assert a["current_r"].equals(b["current_r"])
