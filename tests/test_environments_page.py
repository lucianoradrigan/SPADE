"""Environments page (expansion task E1; spec Sec. 8.5): the table is read from the registered
Gymnasium environments, snippets exist only for registered IDs, and the contract check runs the
real check_env."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from driveflow.envs import registry
from driveflow.viz.adapters import environments_adapter as ea
from driveflow.viz.views.environments import table_rows

APP = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page("pages/environments.py").run()
    assert not at.exception, at.exception
    return at


def test_table_rows_from_real_contracts():
    rows = table_rows([ea.contract(e) for e in ea.env_ids()], ea.planned())
    by_id = {r["Environment ID"]: r for r in rows}
    dc = by_id["SPADE/DcMotor-v0"]
    assert dc["Status"] == "Registered" and dc["Source"] == "SIMULATED"
    assert "omega [rad/s]" in dc["Observation"] and "duty_cycle ∈ [0, 1] 1" in dc["Action"]
    assert "current_a 210" in dc["Physical limits"]
    assert "v_o_real ∈ [-2000, 2000] V" in by_id["SPADE/Vsc-v0"]["Action"]
    for name in registry.PLANNED:
        assert by_id[name]["Status"].startswith("Not registered")


def test_snippets_only_for_registered_ids():
    assert all(ea.snippet(e) for e in ea.env_ids())
    assert ea.snippet("SPADE/Pmsm-v0") and ea.snippet("PLECS plant environments") is None


def test_page_renders_both_environments_and_planned_rows():
    at = _page()
    df = at.dataframe[0].value
    assert set(registry.ENVIRONMENTS) <= set(df["Environment ID"])
    assert set(registry.PLANNED) <= set(df["Environment ID"])
    codes = [c.value for c in at.code]
    assert len(codes) == len(registry.ENVIRONMENTS) and all('gym.make("SPADE/' in c for c in codes)
    assert not any("PLECS" in c for c in codes)


def test_contract_check_button_runs_check_env():
    at = _page()
    at.button(key="env_verify_SPADE/Vsc-v0").click().run()
    assert not at.exception
    text = " ".join(str(h.body) for h in at.get("html")) + " ".join(c.value for c in at.caption)
    assert "Contract check passed" in text and "check_env passed" in text
