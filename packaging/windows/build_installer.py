"""Builds SPADE's double-click Windows installer (SPADE-Setup-<version>-<commit>.exe: the 7-character
commit it was built from is in the file name, the installed program's name and BUILD_INFO.json). Runs on Windows with
the Python that the installer will embed (same minor version) and Inno Setup 6
(.github/workflows/windows-installer.yml).

Steps, into --stage:
1. python\\  the official embeddable CPython of this exact version (python.org), SHA-256 recorded;
   python3XX._pth enables site and adds ..\\SPADE\\src, so SPADE runs from its source tree and the
   repository-relative paths (configs/, data/, workspace/) resolve as in a clone.
2. python\\Lib\\site-packages  SPADE's dependencies with the viz and fmu extras (pip --target, from
   this machine's Python of the same version); SPADE itself is not installed there. TensorFlow's
   C++ headers (tensorflow/include, compile-time only, the longest paths) are removed.
3. The Microsoft C++ runtime DLLs TensorFlow needs (msvcp140*.dll), copied next to python.exe, so
   no separate VC++ redistributable install is needed.
4. SPADE\\  `git archive` of the commit being built.
5. SPADE\\data\\fmu  the prebuilt OpenModelica FMUs for win64 (python -m driveflow.fmu fetch
   --release <tag>, SHA-256 checked), when --fmu-release is given.
6. BUILD_INFO.json, then ISCC packaging/windows/spade.iss.

Usage: python packaging/windows/build_installer.py --stage build/win --out dist [--fmu-release fmu-v2] [--iscc <ISCC.exe>]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ISCC_DEFAULT = r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
VC_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def embed_python(stage: Path) -> dict:
    ver = platform.python_version()
    url = f"https://www.python.org/ftp/python/{ver}/python-{ver}-embed-amd64.zip"
    zpath = stage / "python-embed.zip"
    with urllib.request.urlopen(url, timeout=300) as r:
        zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(stage / "python")
    info = {"version": ver, "url": url, "sha256": sha256(zpath)}
    zpath.unlink()
    pth = next((stage / "python").glob("python3*._pth"))
    lines = [l for l in pth.read_text().splitlines() if l.strip() and l.strip() != "#import site"]
    pth.write_text("\n".join(lines + ["Lib\\site-packages", "..\\SPADE\\src", "import site", ""]))
    return info


def dependencies(stage: Path) -> None:
    site = stage / "python" / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "--no-compile", "--target", str(site), f"{REPO}[viz,fmu]"])
    for p in list(site.glob("driveflow")) + list(site.glob("driveflow-*.dist-info")) + list(site.glob("__editable__*")):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.rmtree(site / "tensorflow" / "include", ignore_errors=True)
    shutil.rmtree(site / "bin", ignore_errors=True)


def vc_runtime(stage: Path) -> list[str]:
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    got = []
    for name in VC_DLLS:
        dst = stage / "python" / name
        if not dst.exists() and (sys32 / name).exists():
            shutil.copy(sys32 / name, dst)
            got.append(name)
    return got


def source_tree(stage: Path) -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    tar = subprocess.run(["git", "archive", "--format=tar", commit], cwd=REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(stage / "SPADE", filter="data")
    return commit


def fmus(stage: Path, tag: str) -> list:
    env = dict(os.environ, SPADE_FMU_DIR=str(stage / "SPADE" / "data" / "fmu"))
    run([str(stage / "python" / "python.exe"), "-m", "driveflow.fmu", "fetch", "--release", tag], env=env, cwd=stage / "SPADE")
    return json.loads((stage / "SPADE" / "data" / "fmu" / "fetch_log.json").read_text())[-1]["records"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fmu-release", default=None)
    ap.add_argument("--iscc", default=ISCC_DEFAULT)
    ap.add_argument("--version", default=None, help="installer version (default: the package version)")
    args = ap.parse_args(argv)
    if platform.system() != "Windows":
        print("the Windows installer is built on Windows", file=sys.stderr)
        return 2
    if args.stage.exists():
        shutil.rmtree(args.stage)
    args.stage.mkdir(parents=True)
    info = {"python": embed_python(args.stage)}
    dependencies(args.stage)
    info["vc_runtime_dlls_copied"] = vc_runtime(args.stage)
    info["commit"] = source_tree(args.stage)
    info["fmus"] = fmus(args.stage, args.fmu_release) if args.fmu_release else "not included"
    site = args.stage / "python" / "Lib" / "site-packages"
    info["packages"] = sorted(p.name[: -len(".dist-info")] for p in site.glob("*.dist-info"))
    longest = max((len(str(p.relative_to(args.stage))) for p in args.stage.rglob("*")), default=0)
    info["longest_relative_path"] = longest
    version = args.version or next(l.split('"')[1] for l in (REPO / "pyproject.toml").read_text().splitlines() if l.startswith("version"))
    info["version"] = version
    (args.stage / "SPADE" / "packaging" / "windows" / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "packages"}, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    run([args.iscc, f"/DSTAGE={args.stage.resolve()}", f"/DAPPVERSION={version}", f"/DCOMMIT={info['commit'][:7]}", f"/DOUTDIR={args.out.resolve()}",
         str(args.stage / "SPADE" / "packaging" / "windows" / "spade.iss")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
