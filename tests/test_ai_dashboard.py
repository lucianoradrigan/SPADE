"""AppTest-driven smoke tests for the IA phase (docs/design_ai_layer_transversal.md Sec. 5.1,
Sec. 8 step 7): the landing page exposes an "Enter Fase IA" card, and entering it + generating a
sample run renders without exceptions for both domains. Each panel's state (a real result vs. a
"not registered" info message) is checked against whatever driveflow.ai.registry actually has
promoted at test time -- not a fixed assumption about registry contents, since that is real repo
state that legitimately changes as more models get trained/promoted (see
docs/design_ai_layer_transversal.md's status note for what is registered as of a given commit).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.ai.registry import RegistryError, resolve
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM

#: AppTest.from_file resolves a relative path against the CALLER's directory, not cwd -- absolute
#: to avoid that surprise.
DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "dashboard.py")


def _is_registered(domain: str, block: str) -> bool:
    try:
        resolve(domain, "pc", block)
        return True
    except RegistryError:
        return False


class TestLandingPage:
    def test_ia_card_is_present(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        assert at.button(key="enter_phase_IA").label == "Enter Fase IA →"


class TestEnterIaPhase:
    def test_renders_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        assert not at.exception
        assert at.selectbox(key="ia_domain").value == "dc_motor"


class TestGenerateSampleRun:
    @pytest.mark.parametrize("domain", ["dc_motor", "vsc_dpc"])
    def test_panels_match_registry_state(self, domain):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.selectbox(key="ia_domain").set_value(domain).run()
        at.sidebar.button(key="ia_generate").click().run()
        assert not at.exception

        info_texts = " ".join(el.value for el in at.info)
        if _is_registered(domain, "classifier"):
            assert "No classifier is registered" not in info_texts
        else:
            assert "No classifier is registered" in info_texts
        if _is_registered(domain, "regressor"):
            assert "No regressor is registered" not in info_texts
        else:
            assert "No regressor is registered" in info_texts


class TestSampleRunControlsAreReal:
    """Regression test for the sample-run sidebar actually being parametrized (not a single
    hardcoded run) -- covers the "generar sample run muy simple, no deja modificar nada" gap:
    domain-specific widgets must exist, and changing one must change the generated data."""

    def _rows_caption_count(self, at) -> int:
        text = next(el.value for el in at.caption if "row(s) available for evaluation" in el.value)
        return int(text.split()[0])

    def test_dc_motor_exposes_fault_and_severity_widgets(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        assert not at.exception
        assert at.sidebar.selectbox(key="ia_fault_type").options == ["healthy", "outer_race", "inner_race", "ball", "cage"]
        # "healthy" is the default -- severity sliders only render for an actual fault.
        at.sidebar.selectbox(key="ia_fault_type").set_value("outer_race").run()
        assert not at.exception
        assert at.sidebar.slider(key="ia_elec_severity") is not None
        assert at.sidebar.slider(key="ia_mech_severity") is not None

    def test_vsc_dpc_exposes_load_resistance_and_reference_widgets(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.selectbox(key="ia_domain").set_value("vsc_dpc").run()
        assert not at.exception
        r_slider = at.sidebar.slider(key="ia_load_r")
        assert r_slider.min == pytest.approx(MIN_STABLE_LOAD_RESISTANCE_OHM, abs=1e-3)
        assert at.sidebar.slider(key="ia_ref_mag") is not None
        assert at.sidebar.slider(key="ia_ref_omega") is not None

    def test_changing_duration_changes_the_generated_row_count(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.sidebar.slider(key="ia_duration").set_value(0.1).run()
        at.sidebar.button(key="ia_generate").click().run()
        assert not at.exception
        short_rows = self._rows_caption_count(at)

        at.sidebar.slider(key="ia_duration").set_value(0.6).run()
        at.sidebar.button(key="ia_generate").click().run()
        assert not at.exception
        long_rows = self._rows_caption_count(at)

        assert long_rows > short_rows
