"""B8 (round PLECS/OM): `python -m driveflow.fmu fetch` against a local HTTP server standing in for
a GitHub Release -- download with SHA-256 check, a wrong hash rejected and deleted, an FMU built from
another model version not installed (unless --allow-mismatch, recorded), a platform without FMU, and
the Plant Studio button running it as a job."""

import functools
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from driveflow import fmu
from driveflow.fmu.__main__ import main

TAG = "fmu-vtest"
PAYLOAD = b"PK\x05\x06" + b"\x00" * 18  # an empty zip: fetch checks bytes and hashes, not FMU contents


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Serves <root>/<TAG>/{fmu_manifest.json, BuckAveraged-<plat>.fmu} and latest.json; returns a writer."""
    root = tmp_path / "srv"
    (root / TAG).mkdir(parents=True)
    (root / "latest.json").write_text(json.dumps({"tag_name": TAG}))

    def write(sha=None, mo=None, platform=None, payload=PAYLOAD):
        plat = platform or fmu.fmi_platform()
        name = f"BuckAveraged-{plat}.fmu"
        (root / TAG / name).write_bytes(payload)
        man = {"source_mo_sha256": mo if mo is not None else fmu.sha256_file(fmu.MODEL_SOURCE), "openmodelica": "1.27.1",
               "fmus": [{"model": "BuckAveraged", "platform": plat, "file": name, "sha256": sha or hashlib.sha256(payload).hexdigest(), "b3_verdict": "GO"}]}
        (root / TAG / "fmu_manifest.json").write_text(json.dumps(man))

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("SPADE_FMU_RELEASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setenv("SPADE_FMU_DIR", str(tmp_path / "fmu"))
    yield write
    srv.shutdown()


def test_download_verifies_the_hash_and_installs(release, tmp_path):
    release()
    assert main(["fetch", "--release", TAG]) == 0
    out = tmp_path / "fmu" / "BuckAveraged.fmu"
    assert out.read_bytes() == PAYLOAD
    log = json.loads((tmp_path / "fmu" / "fetch_log.json").read_text())
    assert log[-1]["release"] == TAG and log[-1]["records"][0]["status"] == "installed"


def test_latest_release_is_used_when_no_tag_is_given(release, tmp_path):
    release()
    assert fmu.fetch()[0]["status"] == "installed"


def test_a_wrong_hash_is_rejected_and_nothing_is_left(release, tmp_path, capsys):
    release(sha="0" * 64)
    assert main(["fetch", "--release", TAG]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not (tmp_path / "fmu" / "BuckAveraged.fmu").exists()


def test_another_model_version_warns_and_is_not_installed_unless_allowed(release, tmp_path, capsys):
    release(mo="f" * 64)
    assert main(["fetch", "--release", TAG]) == 1
    assert fmu.MISMATCH in capsys.readouterr().out
    assert not (tmp_path / "fmu" / "BuckAveraged.fmu").exists()
    assert main(["fetch", "--release", TAG, "--allow-mismatch"]) == 0
    assert (tmp_path / "fmu" / "BuckAveraged.fmu").exists()
    log = json.loads((tmp_path / "fmu" / "fetch_log.json").read_text())
    assert log[-1]["allow_mismatch"] is True and log[-1]["mo_mismatch"] is True


def test_a_platform_without_fmu_gets_the_right_message(release, capsys):
    release(platform="plan9_64")
    assert main(["fetch", "--release", TAG]) == 2
    err = capsys.readouterr().err
    assert f"no prebuilt FMU for this platform ({fmu.fmi_platform()}, {fmu.machine()})" in err and "BUILD_FMUS_LOCALLY.md" in err and "plan9_64" in err


def test_plant_studio_download_runs_as_a_job(release, tmp_path):
    from streamlit.testing.v1 import AppTest

    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker

    release()
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py"), default_timeout=300)
    at.run()
    at.switch_page("pages/plecs_plant_studio.py").run()
    at.button(key="plecs_goal_gym").click().run()
    at.radio(key="fmu_stage").set_value("2. Engine").run()
    at.text_input(key="fmu_dl_tag").set_value(TAG).run()
    at.button(key="fmu_download").click().run()
    assert not at.exception and any("Queued as job" in s.value for s in at.success)
    s = JobService()
    assert Worker(s.root).serve(once=True) == 1
    run = s.run_for(s.list(limit=1)[0])
    assert run.status == "completed" and (tmp_path / "fmu" / "BuckAveraged.fmu").exists()


def test_release_bundle_attaches_only_go_fmus(tmp_path):
    """scripts/fmu/make_release_bundle.py on a fake build: NO-GO and G2-excluded models stay out."""
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[1]
    d = tmp_path / "fmu"
    d.mkdir()
    shas = {}
    for m in ("BuckAveraged", "BuckSwitching", "DcMotorAveraged"):
        with zipfile.ZipFile(d / f"{m}.fmu", "w") as z:
            z.writestr("binaries/linux64/x.so", m)
        shas[m] = hashlib.sha256((d / f"{m}.fmu").read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps({"fmi_version": "2.0", "fmu_type": "CoSimulation", "solver": "CVODE", "openmodelica": "1.27.1", "openmodelica_image": "img@sha",
                                                 "msl": "4.1.0", "source_mo_sha256": "a" * 64, "commit": "c", "fmus": {m: {"sha256": h} for m, h in shas.items()}}))
    b3 = tmp_path / "b3.json"
    b3.write_text(json.dumps({"models": {"BuckAveraged": {"verdict": "GO"}, "BuckSwitching": {"verdict": "NO-GO", "failed_criteria": ["1"]},
                                         "DcMotorAveraged": {"verdict": "GO"}}}))
    out = tmp_path / "release"
    r = subprocess.run([sys.executable, str(root / "scripts" / "fmu" / "make_release_bundle.py"), "--fmu-dir", str(d), "--b3", str(b3), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    man = json.loads((out / "fmu_manifest.json").read_text())
    assert [f["file"] for f in man["fmus"]] == ["BuckAveraged-linux64.fmu"]
    assert set(man["not_attached"]) == {"BuckSwitching", "DcMotorAveraged"} and "G2" in man["not_attached"]["DcMotorAveraged"]
    assert "University of Chicago" in (out / "THIRD_PARTY_NOTICES.txt").read_text() and (out / "licenses.zip").exists()


def test_an_fmu_for_another_cpu_is_not_installed(release, tmp_path, capsys):
    """X1: the macOS FMU is compiled on arm64; a manifest entry with "machine" installs only on that CPU."""
    release()
    man_path = next((tmp_path / "srv" / TAG).glob("fmu_manifest.json"))
    man = json.loads(man_path.read_text())
    man["fmus"][0]["machine"] = "riscv64"
    man_path.write_text(json.dumps(man))
    assert main(["fetch", "--release", TAG]) == 2
    assert "riscv64" in capsys.readouterr().err and not (tmp_path / "fmu" / "BuckAveraged.fmu").exists()
    man["fmus"][0]["machine"] = fmu.machine()
    man_path.write_text(json.dumps(man))
    assert main(["fetch", "--release", TAG]) == 0


def test_release_bundle_adds_fmus_compiled_on_other_oses(tmp_path):
    """X1: --extra attaches a Model Exchange FMU compiled on another OS when B3's functional criteria
    passed there, records its full verdict (latency included) and CPU, and refuses a report from
    another platform."""
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[1]
    d = tmp_path / "fmu"
    d.mkdir()
    with zipfile.ZipFile(d / "BuckAveraged.fmu", "w") as z:
        z.writestr("binaries/linux64/x.so", "cs")
    (d / "manifest.json").write_text(json.dumps({"fmi_version": "2.0", "fmu_type": "CoSimulation", "solver": "CVODE", "openmodelica": "1.27.1", "openmodelica_image": "i",
                                                 "msl": "4.1.0", "source_mo_sha256": "a" * 64, "commit": "c",
                                                 "fmus": {"BuckAveraged": {"sha256": hashlib.sha256((d / "BuckAveraged.fmu").read_bytes()).hexdigest()}}}))
    b3 = tmp_path / "b3.json"
    b3.write_text(json.dumps({"models": {"BuckAveraged": {"verdict": "GO"}}}))

    def other(os_name, plat, machine, failed):
        f = tmp_path / f"{os_name}.fmu"
        with zipfile.ZipFile(f, "w") as z:
            z.writestr(f"binaries/{plat}/x", os_name)
        rep = tmp_path / f"{os_name}.json"
        rep.write_text(json.dumps({"os": os_name, "machine": machine, "cpu": "cpu", "fmi2_platform": plat, "reference_fmus": [{"fmu": "3.0/Dahlquist", "status": "pass"}],
                                   "b3_buck_averaged_me": {"verdict": "NO-GO" if failed else "GO", "failed_criteria": failed, "functional_ok": failed in ([], ["6"]),
                                                           "p50_ms": 0.7, "nrms_vs_native": {"v_out": 3e-11}}}))
        return ["--extra", "BuckAveraged", str(f), str(rep)]

    def bundle(*extra):
        out = tmp_path / "release"
        shutil.rmtree(out, ignore_errors=True)
        r = subprocess.run([sys.executable, str(root / "scripts" / "fmu" / "make_release_bundle.py"), "--fmu-dir", str(d), "--b3", str(b3), "--out", str(out), *extra],
                           capture_output=True, text=True)
        return r, (json.loads((out / "fmu_manifest.json").read_text()) if r.returncode == 0 else None)

    import shutil

    r, man = bundle(*other("Windows", "win64", "AMD64", ["6"]), *other("macOS", "darwin64", "arm64", []), *other("Bad", "win64", "AMD64", ["4"]))
    assert r.returncode == 0, r.stderr
    by = {f["file"]: f for f in man["fmus"]}
    assert set(by) == {"BuckAveraged-linux64.fmu", "BuckAveraged-win64.fmu", "BuckAveraged-darwin64.fmu"}
    assert by["BuckAveraged-win64.fmu"]["b3_verdict"] == "NO-GO" and "criterion 6" in by["BuckAveraged-win64.fmu"]["b3_note"]
    assert by["BuckAveraged-win64.fmu"]["machine"] == "x86_64" and by["BuckAveraged-darwin64.fmu"]["machine"] == "arm64"
    assert by["BuckAveraged-darwin64.fmu"]["interface"] == "ModelExchange" and by["BuckAveraged-linux64.fmu"]["interface"] == "CoSimulation"
    assert "functional criteria failed" in man["not_attached"]["BuckAveraged-win64"]
    r, _ = bundle("--extra", "BuckAveraged", str(tmp_path / "macOS.fmu"), str(tmp_path / "Windows.json"))
    assert r.returncode != 0 and "measured on win64" in r.stderr
