"""Prebuilt OpenModelica FMUs of SPADE (round PLECS/OpenModelica, task B8).

``python -m driveflow.fmu fetch [--release TAG] [--model NAME] [--allow-mismatch]`` downloads the
FMUs of a GitHub Release built by .github/workflows/fmu-release.yml, with the standard library only:

- ``fmu_manifest.json`` of the Release lists, per FMU: model, platform, file, SHA-256 and B3
  verdict, plus the SHA-256 of the ``.mo`` the FMUs were built from.
- Each FMU is downloaded to a temporary folder and its SHA-256 checked. A mismatch deletes the file
  and fails; nothing half-downloaded reaches the FMU folder.
- If the release was built from another ``modelica/SPADE/package.mo`` than this checkout's, the FMU
  is not installed ("FMU built from a different model version") unless ``--allow-mismatch`` is
  given; that choice is recorded in ``fetch_log.json`` next to the FMUs.
- An entry may name a CPU ``machine`` (x86_64, arm64): it is installed only on that CPU, so the
  macOS FMU compiled on arm64 is not installed on an Intel Mac (X1).
- No FMU for this platform: the message says so and points to the local build
  (docs/guides/BUILD_FMUS_LOCALLY.md).

Licence texts that go with the FMUs: licenses/fmu/ (review: docs/audit/plecs_om/b8_license_review.md).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform as _platform
import shutil
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO = "lucianoradrigan/SPADE"
MODEL_SOURCE = REPO_ROOT / "modelica" / "SPADE" / "package.mo"
GUIDE = "docs/guides/BUILD_FMUS_LOCALLY.md"
MISMATCH = "FMU built from a different model version"


class FetchError(RuntimeError):
    pass


def fmi_platform() -> str:
    """The FMI binaries folder name for this machine (linux64, win64, darwin64, …)."""
    system = {"Linux": "linux", "Windows": "win", "Darwin": "darwin"}.get(_platform.system(), _platform.system().lower())
    machine = _platform.machine().lower()
    bits = "64" if machine in ("x86_64", "amd64", "arm64", "aarch64") or machine.endswith("64") else "32"
    return f"{system}{bits}"


def machine() -> str:
    """This CPU as release manifests name it: x86_64 or arm64."""
    m = _platform.machine().lower()
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(m, m)


def fmu_dir() -> Path:
    return Path(os.environ.get("SPADE_FMU_DIR", REPO_ROOT / "data" / "fmu"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_source_sha256(path: Path | None = None) -> str | None:
    """SHA-256 of the .mo with LF line endings, as the release build (Linux) hashes it: a Windows git
    checkout converts the file to CRLF, which is the same model version."""
    path = Path(path) if path is not None else MODEL_SOURCE
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def release_base_url(tag: str, repo: str = DEFAULT_REPO) -> str:
    """$SPADE_FMU_RELEASE_URL/<tag> when set (tests, mirrors), else the GitHub Release download URL."""
    override = os.environ.get("SPADE_FMU_RELEASE_URL")
    return f"{override.rstrip('/')}/{tag}" if override else f"https://github.com/{repo}/releases/download/{tag}"


def latest_tag(repo: str = DEFAULT_REPO) -> str:
    override = os.environ.get("SPADE_FMU_RELEASE_URL")
    url = f"{override.rstrip('/')}/latest.json" if override else f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())["tag_name"]
    except Exception as exc:
        raise FetchError(f"cannot find the latest published FMU release ({exc}); pass --release <tag>") from None


def _download(url: str, dest: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as exc:
        raise FetchError(f"download failed: {url}: {exc}") from None


def fetch(release: str | None = None, model: str | None = None, allow_mismatch: bool = False, dest: Path | None = None,
          repo: str = DEFAULT_REPO, log=print) -> list[dict]:
    """Installs the matching FMUs into ``dest`` (default: data/fmu or $SPADE_FMU_DIR). Returns one
    record per FMU considered: {model, file, status (installed / skipped_mismatch), sha256}."""
    tag = release or latest_tag(repo)
    base = release_base_url(tag, repo)
    dest = Path(dest) if dest else fmu_dir()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spade_fetch_") as tmp:
        man_path = Path(tmp) / "fmu_manifest.json"
        _download(f"{base}/fmu_manifest.json", man_path)
        try:
            manifest = json.loads(man_path.read_text())
        except json.JSONDecodeError as exc:
            raise FetchError(f"fmu_manifest.json of {tag} is not valid JSON: {exc}") from None
        plat = fmi_platform()
        cpu = machine()
        entries = [e for e in manifest.get("fmus", []) if e["platform"] == plat and e.get("machine", cpu) == cpu and (model is None or e["model"] == model)]
        if not entries:
            have = sorted({f"{e['model']} ({e['platform']}{', ' + e['machine'] if e.get('machine') else ''})" for e in manifest.get("fmus", [])})
            raise FetchError(f"release {tag} has no prebuilt FMU{' ' + model if model else ''} for this platform ({plat}, {cpu}); it has {have or 'none'}. "
                             f"Build it locally: {GUIDE}")
        local_mo = model_source_sha256()
        mismatch = manifest.get("source_mo_sha256") != local_mo
        records = []
        for e in entries:
            if mismatch and not allow_mismatch:
                log(f"WARNING: {MISMATCH} ({e['model']}: release .mo {str(manifest.get('source_mo_sha256'))[:12]}…, this checkout {str(local_mo)[:12]}…). "
                    "Not installed; pass --allow-mismatch to install it anyway.")
                records.append({"model": e["model"], "file": e["file"], "status": "skipped_mismatch", "sha256": e["sha256"]})
                continue
            tmp_fmu = Path(tmp) / e["file"]
            _download(f"{base}/{e['file']}", tmp_fmu)
            got = sha256_file(tmp_fmu)
            if got != e["sha256"]:
                tmp_fmu.unlink()
                raise FetchError(f"SHA-256 mismatch for {e['file']}: expected {e['sha256']}, got {got}. The file was deleted.")
            target = dest / f"{e['model']}.fmu"
            shutil.move(str(tmp_fmu), target)
            if mismatch:
                log(f"WARNING: {MISMATCH}: {e['model']} installed because --allow-mismatch was given (recorded in fetch_log.json).")
            log(f"installed {target} (SHA-256 {got[:12]}…, B3 verdict {e.get('b3_verdict')}{' — ' + e['b3_note'] if e.get('b3_note') else ''}, "
                f"{e.get('interface', 'CoSimulation')}, release {tag})")
            records.append({"model": e["model"], "file": e["file"], "status": "installed", "sha256": got})
        _append_log(dest, {"at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "release": tag, "platform": plat,
                           "allow_mismatch": allow_mismatch, "mo_mismatch": mismatch, "records": records})
    return records


def _append_log(dest: Path, entry: dict) -> None:
    p = dest / "fetch_log.json"
    try:
        log = json.loads(p.read_text()) if p.exists() else []
    except json.JSONDecodeError:
        log = []
    log.append(entry)
    p.write_text(json.dumps(log, indent=2) + "\n")
