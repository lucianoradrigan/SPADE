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
