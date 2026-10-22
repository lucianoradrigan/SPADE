"""PLECS researcher kit (task A2, round PLECS/OM, route a).

One command, run on the researcher's computer with PLECS Standalone and its RPC server enabled:

    python experiments/plecs_researcher_run.py --circuit my_buck.plecs --manifest buck_example.yaml --out results/

It tests the RPC connection, hashes the circuit, runs the template's small declared sweep as a job,
fits the A1 surrogate when the template asks for it, compares the surrogate with the PLECS runs that
were not used to fit it, and packages everything into ``<out>/plecs_evidence_<UTC>.zip``. If a step
fails, the package is still written, with the cause; nothing is retried behind the user's back.

The package is ingested on our side with ``python experiments/ingest_plecs_evidence.py <zip>``,
which checks every file's SHA-256 and the schema before copying anything.

A template is YAML with two blocks: ``manifest`` (the interface manifest of
integrations/plecs/manifest.py) and ``kit`` (the sweep: ``sweep`` {ModelVar: [low, high, levels]},
``fixed``, ``derived`` {ModelVar: arithmetic expression of other ModelVars}, ``time_span_s``,
``sample_s``, ``time_limit_s``, ``fit_surrogate``).
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import operator
import os
import platform
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path

import yaml

from driveflow.integrations.plecs import manifest as mf

EVIDENCE_SCHEMA = 1
REQUIRED_MEMBERS = ("evidence.json", "hashes.json", "log.txt", "template.yaml")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


class KitError(ValueError):
    pass


# ----------------------------------------------------------------------------- template
def safe_eval(expr: str, names: dict) -> float:
    """Arithmetic of numbers and ModelVar names only (+ - * / ** and unary minus); anything else is refused."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise KitError(f"derived expression {expr!r}: unknown name {node.id!r}")
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise KitError(f"derived expression {expr!r}: only numbers, ModelVar names and + - * / ** are allowed")
    try:
        return ev(ast.parse(str(expr), mode="eval"))
    except SyntaxError:
        raise KitError(f"derived expression {expr!r} is not valid arithmetic") from None


def load_template(path) -> tuple[mf.InterfaceManifest, dict, str]:
    """(manifest, kit block, raw text). Raises KitError / ManifestError with what to fix."""
    text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    if "manifest" not in data or "kit" not in data:
        raise KitError("the template needs a 'manifest' block and a 'kit' block (see configs/plecs/manifest_templates/)")
    man = mf.from_dict(data["manifest"])
    kit = dict(data["kit"])
    for key in ("sweep", "time_span_s", "sample_s"):
        if key not in kit:
            raise KitError(f"kit.{key} is required")
    names = {v.name for v in man.model_vars}
    for block in ("sweep", "fixed", "derived"):
        unknown = sorted(set(kit.get(block) or {}) - names)
        if unknown:
            raise KitError(f"kit.{block} refers to undeclared ModelVars {unknown}")
    if kit.get("fit_surrogate") and man.surrogate is None:
        raise KitError("kit.fit_surrogate needs the manifest's 'excitation' and 'surrogate' blocks")
    return man, kit, text


def design(man: mf.InterfaceManifest, kit: dict) -> list[dict]:
    from driveflow.integrations.plecs import batch

    ranges = {k: tuple(v) for k, v in kit["sweep"].items()}
    conds = batch.design_conditions(man, ranges, "grid", fixed=kit.get("fixed") or {})
    for c in conds:
        for name, expr in (kit.get("derived") or {}).items():
            c[name] = safe_eval(expr, c)
        man.check_values(c)
    return conds


# ----------------------------------------------------------------------------- run
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _spade_version() -> str:
    root = Path(__file__).resolve().parents[4]
    try:
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from importlib.metadata import version

        return version("driveflow")
    except Exception:
        return "unknown"


def run_kit(circuit, template, out_dir, rpc_url: str = "http://localhost:1080", command: str = "") -> Path:
    """Runs the whole kit and ALWAYS returns the path of the evidence zip (with the cause when a step failed)."""
    from driveflow.integrations.plecs import batch
    from driveflow.integrations.plecs.rpc_client import PlecsRpcClient, PlecsRpcError

    out = Path(out_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = out / f"plecs_evidence_{stamp}"
    stage.mkdir(parents=True, exist_ok=False)
    work = stage / "work"
    log_lines: list[str] = []

    def log(msg):
        line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  {msg}"
        log_lines.append(line)
        print(line, flush=True)

    ev = {"schema": EVIDENCE_SCHEMA, "created_utc": stamp, "command": command, "rpc_url": rpc_url,
          "versions": {"spade": _spade_version(), "python": platform.python_version(), "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
                       "plecs": None},
          "steps": {}, "status": "failed", "failure": None}
    try:
        shutil.copyfile(template, stage / "template.yaml")
        man, kit, _ = load_template(template)
        ev["template_sha256"] = _sha256(template)
        circuit = Path(circuit).expanduser().resolve()
        if not circuit.is_file():
            raise KitError(f"circuit not found: {circuit}")
        ev["circuit"] = {"name": circuit.name, "sha256": _sha256(circuit), "bytes": circuit.stat().st_size}
        log(f"circuit {circuit.name} sha256 {ev['circuit']['sha256'][:16]}…")

        rep = PlecsRpcClient(rpc_url).test_connection().to_dict()
        ev["steps"]["connection"] = rep
        ev["versions"]["plecs"] = rep.get("plecs_version")
        if not rep["reachable"]:
            raise KitError(f"RPC endpoint not reachable: {rep.get('message')} {rep.get('hint')}".strip())
        log(f"RPC endpoint answered ({rep.get('latency_ms')} ms); PLECS version: {rep.get('plecs_version')}")

        os.environ["SPADE_WORKSPACE"] = str(work / "workspace")
        os.environ["SPADE_PLECS_DATA"] = str(work / "plecs_data")
        os.environ["SPADE_SURROGATE_DATA"] = str(work / "surrogates")
        from driveflow.jobs.service import JobService
        from driveflow.jobs.worker import Worker
        from driveflow.runs.schemas import ExperimentSpec
        from driveflow.runs.store import RunStore, workspace_root

        profile_id = "kit_" + "".join(ch if ch.isalnum() else "_" for ch in circuit.stem.lower())[:40]
        RunStore().save_profile(batch.create_profile(profile_id, circuit, man, rpc_url=rpc_url))
        conds = design(man, kit)
        n_pts = int(round(float(kit["time_span_s"]) / float(kit["sample_s"]))) + 1
        s = JobService()
        job = s.submit(ExperimentSpec.create("plecs_sweep", plant_profile_id=profile_id, conditions=conds, seed=0, algorithm="grid",
                                             params={"time_span_s": float(kit["time_span_s"]), "time_limit_s": float(kit.get("time_limit_s", 120.0)), "n_output_points": n_pts}))
        log(f"sweep: {len(conds)} conditions, {kit['time_span_s']} s each, {n_pts} samples")
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
        sweep = s.run_for(s.get(job.job_id))
        n_ok = sum(c["status"] == "ok" for c in sweep.conditions)
        ev["steps"]["sweep"] = {"run_id": sweep.run_id, "status": sweep.status, "n_conditions": len(conds), "n_ok": n_ok,
                                "failures": [{"index": c["index"], "condition": c["condition"], "error": c.get("error")} for c in sweep.conditions if c["status"] != "ok"]}
        ev["profile_id"] = profile_id
        log(f"sweep {sweep.run_id}: {sweep.status}, {n_ok}/{len(conds)} ok")
        data_dir = Path(os.environ["SPADE_PLECS_DATA"]) / profile_id / sweep.run_id
        if data_dir.exists():
            shutil.copytree(data_dir, stage / "dataset")
        if n_ok == 0:
            raise KitError("no sweep condition completed; see the failures above and log.txt")

        if kit.get("fit_surrogate"):
            fj = s.submit(ExperimentSpec.create("surrogate_fit", seed=0, params={"profile_id": profile_id, "sweep_run_id": sweep.run_id, "epochs": 30}))
            Worker(workspace_root()).serve(once=True, poll_s=0.05)
            fit = s.run_for(s.get(fj.job_id))
            if fit.status != "completed":
                raise KitError(f"surrogate fit {fit.status}: {[c.get('error') for c in fit.conditions if c['status'] != 'ok'][:3]}")
            sur_dir = Path(fit.metrics["surrogate_dir"])
            shutil.copyfile(sur_dir / "surrogate.json", stage / "surrogate.json")
            rows = json.loads((sur_dir / "surrogate.json").read_text())["rows"]
            held = [r for r in rows if r["split"] in ("validation", "test")]
            ev["steps"]["surrogate"] = {"run_id": fit.run_id, "env_id": fit.metrics["env_id"], "check_env_passed": fit.metrics["check_env_passed"],
                                        "default_family": fit.metrics["default_family"], "validation": fit.metrics["validation"], "test": fit.metrics["test"]}
            ev["steps"]["compare_with_spade"] = {"what": "surrogate open-loop rollout under the duty PLECS applied vs. the PLECS trajectory, on runs NOT used to fit it",
                                                 "rows": held}
            log(f"surrogate {fit.metrics['env_id']}: default {fit.metrics['default_family']}, check_env {fit.metrics['check_env_passed']}")
        else:
            ev["steps"]["surrogate"] = {"skipped": "the template does not ask for a surrogate (dataset only)"}
        ev["status"] = "completed" if n_ok == len(conds) else "partially_completed"
    except Exception as exc:
        ev["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=4)}
        log(f"FAILED: {type(exc).__name__}: {exc}")
    finally:
        (stage / "log.txt").write_text("\n".join(log_lines) + "\n")
        (stage / "evidence.json").write_text(json.dumps(ev, indent=2, default=str))
        shutil.rmtree(work, ignore_errors=True)
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): _sha256(p) for p in files}
        (stage / "hashes.json").write_text(json.dumps(hashes, indent=2))
        zpath = out / f"{stage.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage).as_posix())
        shutil.rmtree(stage, ignore_errors=True)
    return zpath


# ----------------------------------------------------------------------------- ingest
def ingest(zip_path, docs_root, data_root) -> dict:
    """Validates a returned evidence package (every file's SHA-256, the schema) and copies the
    evidence to <docs_root>/<package>/ and the dataset to <data_root>/<profile>/<run_id>/.
    Raises KitError on any mismatch; nothing is copied then."""
    zp = Path(zip_path)
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
        missing = [m for m in REQUIRED_MEMBERS if m not in names]
        if missing:
            raise KitError(f"package lacks {missing}")
        hashes = json.loads(z.read("hashes.json"))
        for name in names - {"hashes.json"}:
            if name not in hashes:
                raise KitError(f"{name} is not listed in hashes.json")
            if hashlib.sha256(z.read(name)).hexdigest() != hashes[name]:
                raise KitError(f"SHA-256 mismatch for {name}: the package was modified after it was written")
        absent = sorted(set(hashes) - names)
        if absent:
            raise KitError(f"hashes.json lists files that are not in the package: {absent}")
        ev = json.loads(z.read("evidence.json"))
        if ev.get("schema") != EVIDENCE_SCHEMA or "steps" not in ev or "versions" not in ev:
            raise KitError(f"evidence.json does not follow schema {EVIDENCE_SCHEMA}")
        if any(n.startswith("dataset/") for n in names):
            meta = json.loads(z.read("dataset/manifest.json"))
            if ev.get("circuit", {}).get("sha256") != meta.get("circuit_sha256"):
                raise KitError("the dataset's circuit SHA-256 differs from the evidence's")
        dest_docs = Path(docs_root) / zp.stem
        if dest_docs.exists():
            raise KitError(f"{dest_docs} already exists: this package was ingested before")
        dest_docs.mkdir(parents=True)
        dest_data = None
        for name in sorted(names):
            if name.startswith("dataset/"):
                dest_data = Path(data_root) / ev["profile_id"] / ev["steps"]["sweep"]["run_id"]
                target = dest_data / name.removeprefix("dataset/")
            else:
                target = dest_docs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    return {"package": zp.name, "status": ev["status"], "docs": str(dest_docs), "data": str(dest_data) if dest_data else None,
            "circuit_sha256": ev.get("circuit", {}).get("sha256"), "plecs_version": ev["versions"].get("plecs"), "failure": ev.get("failure")}
