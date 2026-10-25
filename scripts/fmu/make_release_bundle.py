"""Builds the asset folder of an FMU GitHub Release (task B8; .github/workflows/fmu-release.yml).

Only FMUs whose B3 GO/NO-GO verdict, measured on THIS build and platform, is GO are attached, minus
the models the user excluded (G2: DcMotorAveraged, reference use only). Each FMU is renamed
<model>-<platform>.fmu. fmu_manifest.json records the model, platform, OpenModelica and MSL
versions, the SHA-256 of every FMU and of modelica/SPADE/package.mo, the commit and the B3 verdict.
The licence texts of licenses/fmu/ go along as THIRD_PARTY_NOTICES.txt (all texts concatenated)
and licenses.zip.

Usage: python scripts/fmu/make_release_bundle.py --fmu-dir data/fmu --b3 ci_out/fmu/b3/b3_om_fmu_spike.json --out release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: decision G2 (2026-10-25): not distributed this round even if GO
EXCLUDED = {"DcMotorAveraged": "G2: reference use only (B3 post-hoc analysis); not distributed in this round"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmu_platforms(p: Path) -> list[str]:
    with zipfile.ZipFile(p) as z:
        return sorted({n.split("/")[1] for n in z.namelist() if n.startswith("binaries/") and n.count("/") >= 2 and n.split("/")[1]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fmu-dir", type=Path, default=REPO / "data" / "fmu")
    ap.add_argument("--b3", type=Path, required=True, help="b3_om_fmu_spike.json produced on this build")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    build = json.loads((args.fmu_dir / "manifest.json").read_text())
    b3 = json.loads(args.b3.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fmus, skipped = [], {}
    for model, rec in b3["models"].items():
        src = args.fmu_dir / f"{model}.fmu"
        if model in EXCLUDED:
            skipped[model] = EXCLUDED[model]
            continue
        if rec["verdict"] != "GO":
            skipped[model] = f"B3 verdict {rec['verdict']} on this build (failed criteria {rec.get('failed_criteria')})"
            continue
        if sha256(src) != build["fmus"][model]["sha256"]:
            raise SystemExit(f"{model}.fmu changed after the build manifest was written")
        for plat in fmu_platforms(src):
            name = f"{model}-{plat}.fmu"
            shutil.copy(src, args.out / name)
            fmus.append({"model": model, "platform": plat, "file": name, "sha256": sha256(args.out / name), "bytes": (args.out / name).stat().st_size,
                         "b3_verdict": rec["verdict"]})
    if not fmus:
        raise SystemExit(f"no FMU qualifies for the release: {skipped}")
    manifest = {"fmi_version": build["fmi_version"], "fmu_type": build["fmu_type"], "solver": build["solver"], "openmodelica": build["openmodelica"],
                "openmodelica_image": build["openmodelica_image"], "msl": build["msl"], "source_mo_sha256": build["source_mo_sha256"], "commit": build["commit"],
                "b3_report": "docs/audit/plecs_om/b3_om_fmu_spike.md (re-run on this build by the release workflow)",
                "licenses": "THIRD_PARTY_NOTICES.txt / licenses.zip (licenses/fmu/ in the repository)", "fmus": fmus, "not_attached": skipped}
    (args.out / "fmu_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lic = REPO / "licenses" / "fmu"
    parts = []
    with zipfile.ZipFile(args.out / "licenses.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(lic.iterdir()):
            z.write(f, f"licenses/{f.name}")
            parts.append(f"{'=' * 78}\n{f.name}\n{'=' * 78}\n{f.read_text()}\n")
    (args.out / "THIRD_PARTY_NOTICES.txt").write_text("".join(parts))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
