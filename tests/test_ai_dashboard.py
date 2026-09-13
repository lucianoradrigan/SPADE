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
from driveflow.viz.ai_dashboard import _channel_upload_template_df
from driveflow.viz.dpc_upload_validation import validate_dc_motor_upload, validate_vsc_dpc_forecast_upload

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

    def test_dc_motor_exposes_fault_speed_and_severity_widgets(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        assert not at.exception
        assert at.sidebar.selectbox(key="ia_fault_type").options == ["healthy", "outer_race", "inner_race", "ball", "cage"]
        # Sec. 4's default speed setpoint (Fase A's own default, GEM's real nominal ω) --
        # NOT Scenario's own dataclass default (150.0), see ai_dashboard.py's
        # _DC_MOTOR_DEFAULT_OMEGA_REF_RAD_S comment.
        assert at.sidebar.slider(key="ia_omega_ref_slider").value == 300.0
        # "healthy" is the default -- severity sliders only render for an actual fault.
        at.sidebar.selectbox(key="ia_fault_type").set_value("outer_race").run()
        assert not at.exception
        assert at.sidebar.slider(key="ia_elec_severity_slider") is not None
        assert at.sidebar.slider(key="ia_mech_severity_slider") is not None

    def test_vsc_dpc_exposes_load_resistance_and_reference_widgets(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.selectbox(key="ia_domain").set_value("vsc_dpc").run()
        assert not at.exception
        r_slider = at.sidebar.slider(key="ia_load_r_slider")
        assert r_slider.min == pytest.approx(MIN_STABLE_LOAD_RESISTANCE_OHM, abs=1e-3)
        assert at.sidebar.slider(key="ia_ref_mag_slider") is not None
        assert at.sidebar.slider(key="ia_ref_omega_slider") is not None

    def test_changing_duration_changes_the_generated_row_count(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.sidebar.slider(key="ia_duration_slider").set_value(0.1).run()
        at.sidebar.button(key="ia_generate").click().run()
        assert not at.exception
        short_rows = self._rows_caption_count(at)

        at.sidebar.slider(key="ia_duration_slider").set_value(0.6).run()
        at.sidebar.button(key="ia_generate").click().run()
        assert not at.exception
        long_rows = self._rows_caption_count(at)

        assert long_rows > short_rows

    def test_duration_custom_value_escape_hatch_goes_past_the_slider_max(self):
        """The slider alone tops out at 1.0s -- the "Custom value" checkbox must allow going
        past that, same as Fase A/B, so a value the slider itself could never reach still
        produces more rows than the slider's own max."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.sidebar.slider(key="ia_duration_slider").set_value(1.0).run()  # slider's own max
        at.sidebar.button(key="ia_generate").click().run()
        rows_at_slider_max = self._rows_caption_count(at)

        at.sidebar.checkbox(key="ia_duration_custom").set_value(True).run()
        at.sidebar.number_input(key="ia_duration_custom_input").set_value(1.5).run()
        at.sidebar.button(key="ia_generate").click().run()
        assert not at.exception
        assert self._rows_caption_count(at) > rows_at_slider_max


class TestUploadFormatHelp:
    """Regression test for the "Upload a file" data source not specifying its expected format
    or giving an example -- the gap flagged directly ("falta especificar el formato, o dar un
    ejemplo de formato")."""

    @pytest.mark.parametrize("domain", ["dc_motor", "vsc_dpc"])
    def test_shows_format_expander_and_example_downloads(self, domain):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.selectbox(key="ia_domain").set_value(domain).run()
        at.sidebar.radio(key="ia_source").set_value("Upload a file").run()
        assert not at.exception
        expander_labels = [el.label for el in at.sidebar.expander]
        assert "Required format" in expander_labels
        assert at.sidebar.download_button(key=f"ia_template_csv_{domain}") is not None
        assert at.sidebar.download_button(key=f"ia_template_json_{domain}") is not None

    @pytest.mark.parametrize(
        "domain,validator",
        [("dc_motor", validate_dc_motor_upload), ("vsc_dpc", validate_vsc_dpc_forecast_upload)],
    )
    def test_example_template_round_trips_through_its_own_validator(self, domain, validator):
        """The example must itself be accepted by the exact validator the upload path runs --
        otherwise "here's the format" would be actively misleading. Guards specifically against
        including an always-NaN candidate channel (dc_motor's current_s/current_t, single-phase
        model) in the template: that would make _coerce_numeric drop every row."""
        template_df = _channel_upload_template_df(domain)
        assert not template_df.empty
        clean_df, _, messages = validator(template_df)
        assert clean_df is not None, messages
        assert len(clean_df) == len(template_df)
        assert not any(level == "error" for level, _ in messages)


def _has_tflite(domain: str, tier: str, block: str) -> bool:
    try:
        run_dir = resolve(domain, tier, block)
    except RegistryError:
        return False
    return (run_dir / "model.tflite").exists()


class TestEdgeDeploymentDownloads:
    """Regression test for the gap flagged directly: the real, distilled+exported .tflite
    artifacts (Sec. 8 step 8) existed on disk but were never reachable from the tab -- a user
    could evaluate the PC-tier model here but had no way to get the Raspberry Pi 5/ESP32 files
    onto real hardware. Checked against whatever the registry actually has (like
    TestGenerateSampleRun above), not a fixed assumption."""

    @pytest.mark.parametrize("domain", ["dc_motor", "vsc_dpc"])
    def test_renders_without_exceptions(self, domain):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.selectbox(key="ia_domain").set_value(domain).run()
        assert not at.exception

    @pytest.mark.parametrize(
        "domain,tier,block",
        [(d, t, b) for d in ["dc_motor", "vsc_dpc"] for t in ["rpi5", "esp32"] for b in ["classifier", "regressor"]],
    )
    def test_download_button_matches_real_tflite_availability(self, domain, tier, block):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_IA").click().run()
        at.selectbox(key="ia_domain").set_value(domain).run()
        assert not at.exception

        button_key = f"ia_edge_dl_{domain}_{tier}_{block}"
        buttons = {el.key: el for el in at.sidebar.download_button} | {el.key: el for el in at.download_button}
        if _has_tflite(domain, tier, block):
            assert button_key in buttons, f"expected a download button for {domain}/{tier}/{block}"
        else:
            assert button_key not in buttons
