#!/usr/bin/env bash
# Builds data/fmu/<Model>.fmu with the pinned OpenModelica (Docker) and writes data/fmu/manifest.json
# with each FMU's SHA-256, the SHA-256 of modelica/SPADE/package.mo, and the toolchain versions.
#   scripts/fmu/build_fmus.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
data="$(readlink -f "${SPADE_DATA_DIR:-$repo/data}")"   # SPADE_DATA_DIR: another checkout's data/ (git worktrees)
export SPADE_DATA_DIR="$data"
rm -rf "$data/fmu/build" && mkdir -p "$data/fmu/build"
"$here/om.sh" scripts/fmu/build_fmus.mos | tee "$data/fmu/build/omc.log"
for m in BuckAveraged BuckSwitching DcMotorAveraged DcMotorSwitching; do
  test -f "$data/fmu/build/$m.fmu" || { echo "missing $m.fmu (see $data/fmu/build/omc.log)" >&2; exit 1; }
  cp "$data/fmu/build/$m.fmu" "$data/fmu/$m.fmu"
done
python3 - "$here/OPENMODELICA_VERSION" "$data" <<'PY'
import hashlib, json, platform, subprocess, sys
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
data = Path(sys.argv[2])
cfg = dict(l.split("=", 1) for l in Path(sys.argv[1]).read_text().splitlines() if "=" in l and not l.startswith("#"))
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    commit = None
models = ["BuckAveraged", "BuckSwitching", "DcMotorAveraged", "DcMotorSwitching"]
doc = {"fmi_version": "2.0", "fmu_type": "CoSimulation", "solver": "CVODE (--fmiFlags=s:cvode)", "platform": f"{platform.system().lower()}-{platform.machine()}",
       "openmodelica": cfg["OM_VERSION"], "openmodelica_image": f"{cfg['OM_IMAGE']}@{cfg['OM_IMAGE_DIGEST']}", "msl": cfg["MSL_VERSION"],
       "source_mo_sha256": sha("modelica/SPADE/package.mo"), "commit": commit,
       "fmus": {m: {"file": f"{m}.fmu", "sha256": sha(data / "fmu" / f"{m}.fmu"), "bytes": (data / "fmu" / f"{m}.fmu").stat().st_size} for m in models}}
(data / "fmu" / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
PY
