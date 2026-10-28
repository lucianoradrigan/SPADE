"""PLECS Plant Studio → Compare with SPADE (expansion task E5): native buck vs. surrogate."""

from pathlib import Path

from streamlit.testing.v1 import AppTest
from tests.streamlit_compat import chart_keys

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/plecs_plant_studio.py").run()
    assert not at.exception
    return at


def _text(at):
    return " ".join(e.value for kind in ("caption", "info", "warning", "error", "markdown") for e in at.get(kind))


def test_empty_states_before_any_run():
    at = _open()
    assert "Native SPADE buck vs. surrogate" in _text(at) and _text(at).count("No comparison has been generated yet.") >= 2


def test_inside_domain_comparison_with_the_mlp():
    at = _open()
    at.selectbox(key="cmp_family").set_value("mlp").run()
    at.button(key="cmp_run").click().run()
    assert not at.exception
    assert "mlp: v_out RMSE" in _text(at) and "Surrogate outside validated operating range" not in _text(at)
    assert chart_keys(at) >= {"cmp_v_out", "cmp_i_l"}


def test_out_of_domain_is_warned_and_grid_lists_every_condition():
    at = _open()
    at.number_input(key="cmp_vin").set_value(72.0).run()
    at.button(key="cmp_run").click().run()
    assert "Surrogate outside validated operating range" in " ".join(w.value for w in at.warning) and "does not claim equivalence" in _text(at)
    at.button(key="cmp_grid").click().run()
    table = next(d.value for d in at.dataframe if "Split" in getattr(d.value, "columns", []))
    assert set(table["Split"]) == {"validation", "test", "extrapolation"} and set(table["Status"]) <= {"OK", "Failed"}
