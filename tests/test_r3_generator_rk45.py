"""R3 (round PLECS/OM): the dataset generator's --dc-plant-solver option. Euler (default) output is
unchanged (no extra column); rk45 integrates the plant with the R2 reference and says so."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import generate_diagnosis_grid_dataset as g  # noqa: E402


def _job(**kw):
    job = g.build_jobs(1, 0.02, 7, (None, "outer_race"))[1]
    job.update(kw)
    return job


def test_default_job_has_no_solver_column_and_rk45_is_tagged_and_differs():
    euler = g._simulate(_job())
    rk45 = g._simulate(_job(dc_plant_solver="rk45"))
    assert "grid_dc_plant_solver" not in euler.columns
    assert set(rk45["grid_dc_plant_solver"]) == {"rk45"}
    assert len(euler) == len(rk45) and not euler["current_r"].equals(rk45["current_r"])
    assert euler["source_file"].equals(rk45["source_file"])


def test_cli_rk45_flag_tags_every_job(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(g, "_simulate", lambda job: seen.append(job) or None)

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, f, jobs):
            for j in jobs:
                f(j)
            raise SystemExit(0)

    monkeypatch.setattr(g.mp, "get_context", lambda _m: type("C", (), {"Pool": _Pool})())
    try:
        g.main(["--dc-plant-solver", "rk45", "--seeds", "1", "--out", str(tmp_path / "x.parquet")])
    except SystemExit:
        pass
    assert seen and all(j["dc_plant_solver"] == "rk45" for j in seen)
