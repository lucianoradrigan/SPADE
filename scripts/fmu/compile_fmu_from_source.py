"""Compiles the platform binary of an OpenModelica source-code FMU on THIS machine and writes an FMU
that also carries it (task X1: FMUs for Windows and macOS; the pinned build produces Linux binaries).

OpenModelica FMUs ship their C sources with a CMakeLists.txt that installs the shared library into
binaries/<platform> for the host (win64, linux64, darwin64). This script runs that build with CMake
and the host C compiler, standard library only. Model Exchange FMUs need no CVODE (SPADE integrates
them); a Co-Simulation FMU built with CVODE would need SUNDIALS on the host, so it is refused unless
--allow-cvode is given.

Usage: python scripts/fmu/compile_fmu_from_source.py IN.fmu OUT.fmu [--generator "MinGW Makefiles"] [--drop-existing-binaries]
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def host_platform() -> str:
    system = {"Windows": "win", "Linux": "linux", "Darwin": "darwin"}[platform.system()]
    return f"{system}64"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fmu", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--generator", default=None, help="CMake generator (default: CMake's own choice)")
    ap.add_argument("--allow-cvode", action="store_true", help="build a CVODE Co-Simulation FMU (needs SUNDIALS on this machine)")
    ap.add_argument("--drop-existing-binaries", action="store_true", help="remove the binaries already in the FMU (tests)")
    ap.add_argument("--runtime-deps", choices=("none", "modelica", "all"), default="none",
                    help="OpenModelica's RUNTIME_DEPENDENCIES_LEVEL: 'all' also copies system runtime libraries into the FMU "
                         "(Windows/MinGW: libgcc, libwinpthread), so it runs on a machine without the compiler")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="spade_fmu_src_") as tmp:
        root = Path(tmp) / "fmu"
        with zipfile.ZipFile(args.fmu) as z:
            z.extractall(root)
        cml = root / "sources" / "CMakeLists.txt"
        if not cml.exists():
            print(f"{args.fmu.name} has no sources/CMakeLists.txt: not an OpenModelica source-code FMU", file=sys.stderr)
            return 2
        need_cvode = re.search(r'set\(NEED_CVODE\s+ON', cml.read_text()) is not None
        md = (root / "modelDescription.xml").read_text()
        cosim_cvode = need_cvode and "<CoSimulation" in md
        if cosim_cvode and not args.allow_cvode:
            print(f"{args.fmu.name} is a Co-Simulation FMU built with CVODE: compiling it needs SUNDIALS here (use its Model Exchange variant)", file=sys.stderr)
            return 3
        if args.drop_existing_binaries and (root / "binaries").exists():
            shutil.rmtree(root / "binaries")
        build = Path(tmp) / "build"
        cfg = ["cmake", "-S", str(root / "sources"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               f"-DNEED_CVODE={'ON' if cosim_cvode else 'OFF'}", f"-DRUNTIME_DEPENDENCIES_LEVEL={args.runtime_deps}"]
        if args.generator:
            cfg += ["-G", args.generator]
        for cmd in (cfg, ["cmake", "--build", str(build), "--config", "Release", "--parallel"], ["cmake", "--install", str(build), "--config", "Release"]):
            print("+", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-4000:], r.stderr[-4000:], file=sys.stderr)
                return 4
        plat = host_platform()
        built = sorted(p.name for p in (root / "binaries" / plat).glob("*")) if (root / "binaries" / plat).exists() else []
        if not built:
            print(f"no binary installed in binaries/{plat}", file=sys.stderr)
            return 5
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
        print(f"wrote {args.out} with binaries/{plat}: {built}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
