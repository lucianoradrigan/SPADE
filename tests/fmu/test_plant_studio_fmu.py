"""B5 (round PLECS/OM): PLECS Plant Studio → Use as a Gymnasium plant, engine "OpenModelica FMU".

AppTest over the real page: without fmpy, without FMUs, BuckAveraged from import to publication
(verification run by the worker), a NO-GO FMU blocked, an FMU for another platform "Not
supported", and a changed FMU hash making the verification Stale. Tests that need the built FMU
are marked fmu and skipped without it.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.integrations.fmu import gym_profile as gp

APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "driveflow" / "viz" / "app.py")
BUILT = Path(os.environ.get("SPADE_FMU_DIR", Path(__file__).resolve().parents[2] / "data" / "fmu"))


def _need(name="BuckAveraged"):
    pytest.importorskip("fmpy")
    if not (BUILT / f"{name}.fmu").exists():
        pytest.skip(f"{name}.fmu not built")
    return BUILT / f"{name}.fmu"


def _open(stage=None):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/plecs_plant_studio.py").run()
    at.button(key="plecs_goal_gym").click().run()
    if stage:
        at.radio(key="fmu_stage").set_value(stage).run()
    assert not at.exception, at.exception
    return at


def _text(at):
    parts = [str(e.value) for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind)]
    parts += [str(h.body) for h in at.get("html")]
    return " ".join(parts)


def _copy(dst: Path, *names, platform_dir=None):
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = _need(n)
        if platform_dir is None:
            shutil.copy(src, dst / src.name)
            continue
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst / src.name, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                zout.writestr(item.filename.replace("binaries/linux64/", f"binaries/{platform_dir}/"), zin.read(item.filename))
    return dst


def _specified_profile(pid="om_buck"):
    p = gp.new_profile(pid, "OM buck", "BuckAveraged")
    m = gp.suggest_mapping(p)
    m["observations"] = [{"output": "i_L", "unit": "A", "scale": 15.0}, {"output": "v_out", "unit": "V", "scale": 48.0}]
    m["reference"], m["safety"] = {"output": "v_out", "value": 24.0}, {"output": "i_L", "rated": 15.0}
    p = gp.set_mapping(p, m)
    return gp.set_gym(p, {"tau_s": 2e-5, "duration_s": 0.004, "parameters": {"R": 6.0}, "reward_terms": None, "action_out_of_range": "clip"})


def test_without_fmpy_the_engine_is_not_supported(monkeypatch):
    monkeypatch.setitem(sys.modules, "fmpy", None)  # import fmpy -> ImportError
    at = _open("2. Engine")
    assert any("fmpy is not installed" in e.value for e in at.error)
    text = _text(at)
    assert "Engine: Not supported" in text
    at.radio(key="fmu_stage").set_value("3. Circuit").run()
    assert not at.exception


def test_without_fmus_the_circuit_stage_says_how_to_get_them(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_FMU_DIR", str(tmp_path / "empty"))
    at = _open("3. Circuit")
    assert any("No FMU found" in i.value and "build_fmus.sh" in i.value for i in at.info)
    assert "Publish to Environments" not in " ".join(str(b.label) for b in at.button)


@pytest.mark.fmu
def test_buck_averaged_from_import_to_publication(tmp_path, monkeypatch):
    from driveflow.envs import registry
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    monkeypatch.setenv("SPADE_FMU_DIR", str(_copy(tmp_path / "fmu", "BuckAveraged")))
    at = _open("3. Circuit")
    at.selectbox(key="fmu_pick").set_value("BuckAveraged").run()
    at.text_input(key="fmu_new_id").set_value("om_buck").run()
    at.button(key="fmu_import").click().run()
    assert any("Imported" in s.value for s in at.success), _text(at)
    at.radio(key="fmu_stage").set_value("4. Signals").run()
    at.button(key="fmu_suggest").click().run()
    for key, value in (("fmu_sc_om_buck_i_L", 15.0), ("fmu_sc_om_buck_v_out", 48.0), ("fmu_refv_om_buck", 24.0), ("fmu_safr_om_buck", 15.0)):
        at.number_input(key=key).set_value(value).run()
    at.button(key="fmu_save_mapping").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    at.radio(key="fmu_stage").set_value("5. Gym behavior").run()
    at.number_input(key="fmu_dur_om_buck").set_value(4.0).run()
    at.button(key="fmu_save_gym").click().run()
    assert not at.error, [e.value for e in at.error]
    at.radio(key="fmu_stage").set_value("6. Validate & publish").run()
    assert at.button(key="fmu_publish").disabled
    at.button(key="fmu_verify").click().run()
    assert "VERIFYING" in _text(at)
    assert Worker(workspace_root()).serve(once=True) == 1
    at.button(key="fmu_refresh").click().run()
    text = _text(at)
    assert "Validated" in text and "13/13 checks passed" in text, text
    assert not at.button(key="fmu_publish").disabled
    at.button(key="fmu_publish").click().run()
    assert "SPADE/FmuOmBuck-v0" in _text(at)
    registry.register_all()
    assert registry.is_registered("SPADE/FmuOmBuck-v0") and registry.ENVIRONMENTS["SPADE/FmuOmBuck-v0"].source == "SIMULATED · OPENMODELICA"


@pytest.mark.fmu
def test_a_no_go_fmu_is_shown_with_its_verdict_and_cannot_be_published(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_FMU_DIR", str(_copy(tmp_path / "fmu", "BuckAveraged", "DcMotorAveraged")))
    at = _open("3. Circuit")
    warn = " ".join(w.value for w in at.warning)
    assert "DcMotorAveraged: B3 verdict NO-GO" in warn and gp.B3_LINK in warn and "Reference use only (see B3 post-hoc analysis)" in warn
    p = gp.new_profile("om_dc", "DC", "DcMotorAveraged")
    assert any(b.startswith("B3 verdict NO-GO") for b in gp.publish_blockers(p))
    assert gp.readiness(p)["Publishability"][0] == "Not supported"
    with pytest.raises(ValueError, match="B3 verdict NO-GO"):
        gp.publish(p)


@pytest.mark.fmu
def test_an_fmu_for_another_platform_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_FMU_DIR", str(_copy(tmp_path / "fmu", "BuckAveraged", platform_dir="win64")))
    at = _open("3. Circuit")
    table = at.dataframe[0].value
    row = table[table["FMU"] == "BuckAveraged"].iloc[0]
    assert row["Status"] == "Not supported" and "win64" in row["Why"]
    assert at.button(key="fmu_import").disabled
    assert any("Not supported" in e.value for e in at.error)


@pytest.mark.fmu
def test_a_changed_fmu_hash_makes_the_verification_stale(tmp_path, monkeypatch):
    d = _copy(tmp_path / "fmu", "BuckAveraged")
    monkeypatch.setenv("SPADE_FMU_DIR", str(d))
    p = _specified_profile()
    p = gp.record_verification(p.profile_id, p.version, gp.verify(p, log=lambda _m: None), "run_test")
    assert gp.verification_current(p) and gp.publish_blockers(p) == []
    with zipfile.ZipFile(d / "BuckAveraged.fmu", "a") as z:  # a rebuild: same model, new bytes
        z.writestr("documentation/rebuilt.txt", "rebuilt")
    assert gp.is_stale(p) and gp.display_state(p) == "STALE"
    assert gp.readiness(p)["Validation"] == ("Needs attention", "Stale — Revalidate")
    assert any(b.startswith("Revalidation required") for b in gp.publish_blockers(p))
    at = _open()
    at.session_state["fmu_profile"] = p.profile_id
    at.radio(key="fmu_stage").set_value("6. Validate & publish").run()
    assert "Stale — Revalidate" in _text(at) and at.button(key="fmu_publish").disabled
