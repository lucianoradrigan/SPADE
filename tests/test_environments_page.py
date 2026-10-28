"""Environment contracts (expansion task E1; spec Sec. 8.5): the table is read from the registered
Gymnasium environments and snippets exist only for registered IDs. The standalone Environments
page is no longer in the navigation; these helpers still back RL Lab."""

from driveflow.envs import registry
from driveflow.viz.adapters import environments_adapter as ea
from driveflow.viz.views.environments import table_rows

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
    # an FMU-backed environment without its built FMU is listed as not available, with no snippet (B4)
    assert all(ea.snippet(e) for e in ea.env_ids() if registry.availability(e)[0])
    assert ea.snippet("SPADE/Pmsm-v0") and ea.snippet("PLECS plant environments") is None
