"""Task A2: the PLECS researcher kit end to end against the fake PLECS server (Buck with the excited
test double, DAB as a dataset), the evidence package and its ingestion. PLECS itself: not executed
(requires the researcher's machine)."""

import json
import zipfile
from pathlib import Path

import pytest

from driveflow.integrations.plecs import kit
from fake_plecs import FakePlecs, closed_port_url

TEMPLATES = Path(__file__).resolve().parents[3] / "configs" / "plecs" / "manifest_templates"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    # run_kit points these at its own work folder; monkeypatch restores them afterwards
    for k in ("SPADE_PLECS_DATA", "SPADE_SURROGATE_DATA", "SPADE_WORKSPACE"):
        monkeypatch.setenv(k, str(tmp_path / f"orig_{k}"))


def _circuit(tmp_path, name):
    p = tmp_path / name
    p.write_text("Plecs { Name \"x\" }\n")
    return p


def _read(zp):
    with zipfile.ZipFile(zp) as z:
        return json.loads(z.read("evidence.json")), set(z.namelist())


def test_templates_are_labelled_and_valid():
    for name in ("buck_example.yaml", "dab_example.yaml"):
        text = (TEMPLATES / name).read_text()
        assert text.startswith("# EXAMPLE CONFIGURATION — NOT EXECUTED")
        man, k, _ = kit.load_template(TEMPLATES / name)
        assert kit.design(man, k)
    man, k, _ = kit.load_template(TEMPLATES / "dab_example.yaml")
    conds = kit.design(man, k)
    assert len(conds) == 9 and all(c["Ro"] == pytest.approx(c["Vref"] ** 2 / c["P"]) for c in conds)


@pytest.mark.parametrize("expr", ["__import__('os').system('x')", "Vref.__class__", "open('f')", "Unknown * 2"])
def test_derived_expressions_are_arithmetic_only(expr):
    with pytest.raises(kit.KitError):
        kit.safe_eval(expr, {"Vref": 200.0, "P": 600.0})


def test_buck_kit_end_to_end_and_ingestion(tmp_path):
    with FakePlecs() as fake:
        zp = kit.run_kit(_circuit(tmp_path, "my_buck.plecs"), TEMPLATES / "buck_example.yaml", tmp_path / "results", fake.url, command="test")
    ev, names = _read(zp)
    assert ev["status"] == "completed", ev["failure"]
    assert ev["steps"]["sweep"]["n_ok"] == 18 and ev["steps"]["surrogate"]["check_env_passed"]
    assert ev["steps"]["compare_with_spade"]["rows"] and all(r["split"] in ("validation", "test") for r in ev["steps"]["compare_with_spade"]["rows"])
    assert {"evidence.json", "hashes.json", "log.txt", "template.yaml", "surrogate.json", "dataset/manifest.json", "dataset/conditions.csv"} <= names
    assert not any(n.endswith(".plecs") for n in names)  # the circuit itself is never shipped
    res = kit.ingest(zp, tmp_path / "docs", tmp_path / "data")
    assert res["status"] == "completed" and Path(res["data"], "conditions.csv").exists() and Path(res["docs"], "evidence.json").exists()
    with pytest.raises(kit.KitError, match="ingested before"):
        kit.ingest(zp, tmp_path / "docs", tmp_path / "data")


def test_dab_kit_returns_a_dataset(tmp_path):
    with FakePlecs() as fake:
        zp = kit.run_kit(_circuit(tmp_path, "DAB_sample1.plecs"), TEMPLATES / "dab_example.yaml", tmp_path / "results", fake.url)
    ev, names = _read(zp)
    assert ev["status"] == "completed" and ev["steps"]["sweep"]["n_ok"] == 9 and "skipped" in ev["steps"]["surrogate"]
    assert "surrogate.json" not in names and "dataset/conditions.csv" in names


def test_failure_still_writes_a_package_with_the_cause(tmp_path):
    zp = kit.run_kit(_circuit(tmp_path, "my_buck.plecs"), TEMPLATES / "buck_example.yaml", tmp_path / "results", closed_port_url())
    ev, names = _read(zp)
    assert ev["status"] == "failed" and "not reachable" in ev["failure"]["message"]
    assert {"evidence.json", "log.txt", "hashes.json"} <= names
    assert kit.ingest(zp, tmp_path / "docs", tmp_path / "data")["status"] == "failed"


def test_tampered_package_is_refused(tmp_path):
    with FakePlecs() as fake:
        zp = kit.run_kit(_circuit(tmp_path, "my_buck.plecs"), TEMPLATES / "buck_example.yaml", tmp_path / "results", fake.url)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(zp) as src, zipfile.ZipFile(bad, "w") as dst:
        for n in src.namelist():
            data = src.read(n)
            if n == "dataset/conditions.csv":
                data = data.replace(b"ok", b"ko", 1)
            dst.writestr(n, data)
    with pytest.raises(kit.KitError, match="SHA-256 mismatch"):
        kit.ingest(bad, tmp_path / "docs", tmp_path / "data")
    assert not (tmp_path / "docs").exists()
