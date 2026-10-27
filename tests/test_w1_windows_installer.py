"""W1: the Windows installer's pieces that can be checked off Windows. The installer itself is built
and installed on a Windows runner by .github/workflows/windows-installer.yml."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "packaging" / "windows"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, WIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_runs_the_app_and_worker_from_the_source_tree():
    launch = _load("launch")
    assert launch.REPO == ROOT and (launch.REPO / "src" / "driveflow" / "viz" / "app.py").exists()
    assert (ROOT / "src" / "driveflow" / "jobs" / "worker.py").exists()


def test_inno_script_points_at_the_launcher_and_installs_per_user():
    iss = (WIN / "spade.iss").read_text()
    assert "PrivilegesRequired=lowest" in iss and "{localappdata}\\SPADE" in iss
    assert set(re.findall(r"\{app\}\\SPADE\\(packaging\\windows\\launch\.py)", iss)) == {"packaging\\windows\\launch.py"}
    assert (ROOT / "LICENSE").exists()


def test_embedded_python_path_file(tmp_path, monkeypatch):
    """build_installer enables site and puts ..\\SPADE\\src on the embedded Python's path."""
    b = _load("build_installer")
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "python312._pth").write_text("python312.zip\n.\n\n# Uncomment to run site.main() automatically\n#import site\n")
    import zipfile

    def fake_urlopen(url, timeout):
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("python312._pth", (tmp_path / "python" / "python312._pth").read_text())
            z.writestr("python.exe", b"")

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R(buf.getvalue())

    monkeypatch.setattr(b.urllib.request, "urlopen", fake_urlopen)
    stage = tmp_path / "stage"
    stage.mkdir()
    info = b.embed_python(stage)
    lines = (stage / "python" / "python312._pth").read_text().splitlines()
    assert lines[-3:] == ["Lib\\site-packages", "..\\SPADE\\src", "import site"] and "#import site" not in lines
    assert len(info["sha256"]) == 64 and info["url"].endswith("-embed-amd64.zip")
