"""Model Registry view-model: every run directory that actually exists under configs/, with
statuses derived from files on disk (guide Sec. 3.4 Model Registry, AUDIT AI-10).

Status vocabulary (each one needs its own evidence; none implies another):
- Promoted        -- configs/registry.yaml currently resolves (domain, tier, block) to this run.
- Superseded      -- was promoted at some point per the registry history, no longer is.
- Candidate       -- a transfer-learning run saved with lineage.json, not promoted; its gate
                     status (APPROVED / BLOCKED) is shown separately.
- Validated       -- ONLY for runs whose persisted gate report is APPROVED. From-scratch runs of
                     experiments/train_model.py have no gate and are shown as "Not gated".
- Exported        -- model.tflite exists (quantization read from the file itself).
- Hardware verified -- only with a deployment_evidence.json saying so (none exists today).

Audit findings (audit_findings.json, ai/audit_findings.py) are listed with the run whenever they
exist: a recorded test metric is never shown without the measured evidence that qualifies it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from driveflow.ai.audit_findings import load_audit_findings
from driveflow.ai.deployment_status import deployment_status
from driveflow.ai.promotion import is_transfer_learning_run, load_gate_report
from driveflow.ai.provenance import file_sha256
from driveflow.ai.registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, _detect_block, _load_index, registry_history

CONFIG_ROOTS = (REPO_ROOT / "configs" / "classifiers", REPO_ROOT / "configs" / "regressors")
ARTIFACT_NAMES = ("config.yaml", "metrics.json", "model.weights.h5", "model.tflite", "lineage.json", "gate_report.json", "validation_report.json", "run_metadata.json", "split_manifest.json", "audit_findings.json")


@lru_cache(maxsize=256)
def _sha(path: str, mtime: float, size: int) -> str:
    return file_sha256(path)


def artifacts(run_dir: Path) -> list:
    out = []
    for name in ARTIFACT_NAMES:
        p = run_dir / name
        if p.exists():
            st_ = p.stat()
            out.append({"file": name, "size_bytes": st_.st_size, "sha256": _sha(str(p), st_.st_mtime, st_.st_size), "path": p})
    return out


def _headline(metrics: dict) -> str:
    if metrics.get("test_macro_f1") is not None:
        return f"macro-F1 {metrics['test_macro_f1']:.3f} · acc {metrics.get('test_accuracy', float('nan')):.3f} (n_test={metrics.get('n_test', '?')})"
    if metrics.get("test_rmse") is not None:
        return f"RMSE {metrics['test_rmse']:.4f} normalized (n_test={metrics.get('n_test', '?')})"
    return "no test metric recorded"


def list_runs(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list:
    entries = _load_index(registry_path)
    current = {str((REPO_ROOT / v).resolve()): k for k, v in entries.items()}
    ever_promoted = {str((REPO_ROOT / e["new"]).resolve()) for e in registry_history(registry_path) if e.get("new") and str(e.get("action", "")).startswith("promote")}
    rows = []
    for root in CONFIG_ROOTS:
        if not root.exists():
            continue
        # <config>/<date>_runNN/ plus the GUI's custom/<name>/<date>_runNN/ and external/<name>/ runs.
        paths = set(root.glob("*/*/metrics.json")) | set(root.glob("custom/*/*/metrics.json")) | set(root.glob("external/*/*/metrics.json"))
        for metrics_path in sorted(paths):
            run_dir = metrics_path.parent
            try:
                metrics = json.loads(metrics_path.read_text())
                raw_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
                block = _detect_block(raw_cfg)
            except Exception as exc:  # malformed run directory: listed, flagged, never hidden
                rows.append({"run_dir": run_dir, "error": str(exc), "domain": "?", "tier": "?", "block": "?", "status": "Unreadable"})
                continue
            key = current.get(str(run_dir.resolve()))
            tl = is_transfer_learning_run(run_dir)
            gate = load_gate_report(run_dir) if tl else None
            lineage = json.loads((run_dir / "lineage.json").read_text()) if (run_dir / "lineage.json").exists() else None
            dep = deployment_status(run_dir)
            if key:
                status = "Promoted (override)" if (lineage or {}).get("status") == "promoted_override" else "Promoted"
            elif str(run_dir.resolve()) in ever_promoted:
                status = "Superseded"
            elif tl:
                status = "Candidate"
            else:
                status = "Trained (not promoted)"
            rows.append({
                "run_dir": run_dir,
                "run": f"{run_dir.parent.name}/{run_dir.name}",
                "domain": raw_cfg.get("domain"),
                "tier": raw_cfg.get("tier"),
                "block": block,
                "status": status,
                "validation": (gate or {}).get("status", "BLOCKED" if tl else "Not gated"),
                "origin": "transfer learning" if tl else ("imported third-party model" if metrics.get("origin") == "EXTERNAL"
                                                          else "trained in AI Diagnostics" if metrics.get("trained_from") == "ai_diagnostics.custom_training"
                                                          else "distilled from " + metrics["distilled_from"] if metrics.get("distilled_from") else "trained from scratch"),
                "exported": dep["stages"]["exported"] == "done",
                "quantization": ((dep.get("artifact") or {}).get("quantization")) if dep["stages"]["exported"] == "done" else None,
                "hardware_verified": dep["stages"]["hardware_verified"] == "done",
                "headline": _headline(metrics),
                "audit_findings": load_audit_findings(run_dir),
                "metrics": metrics,
                "config": raw_cfg,
                "lineage": lineage,
                "gate": gate,
                "deployment": dep,
            })
    return rows


def c2000_summary(kit: dict) -> str:
    """One line on a C2000 kit, from its manifest and, when measured, the board's own report."""
    memory = kit["manifest"]["memory"]
    text = f"Arena {memory['arena_bytes'] / 1024:.1f} KB, weights {memory['weights_bytes'] / 1024:.1f} KB"
    run = kit["run"]
    if run is None:
        return f"{text}. Not measured on the board."
    budget = "meets" if run["verdicts"]["C5_real_time"] else "MISSES"
    return f"{text}. Measured on the {run['measured_on']}: {run['worst_latency_ms']:.1f} ms worst latency, {budget} the {run['latency_budget_ms']:.1f} ms window; max error vs. Keras {run['worst_max_abs_err']:.1e}. Simulated windows replayed from flash, no real signal."


def class_coverage_warnings(rows: list) -> list[str]:
    """Promoted classifiers with fewer classes than another promoted classifier of the same domain
    (X4, round PLECS/OM: the ESP32 keeps a 3-class model while PC and RPi5 have 5). A fault of a
    missing class is answered as one of the known classes, often with high confidence (AI-N13)."""
    promoted = [r for r in rows if "error" not in r and r["status"].startswith("Promoted") and r["block"] == "classifier"]
    out = []
    for r in promoted:
        classes = set((r["metrics"] or {}).get("classes") or [])
        wider = [o for o in promoted if o is not r and o["domain"] == r["domain"] and classes < set((o["metrics"] or {}).get("classes") or [])]
        if wider:
            missing = sorted(set().union(*[set(o["metrics"]["classes"]) for o in wider]) - classes)
            out.append(f"{r['domain']}/{r['tier']}/classifier ({r['run']}) has {len(classes)} classes while "
                       + ", ".join(f"{o['tier']} has {len(o['metrics']['classes'])}" for o in wider)
                       + f". A {' or '.join(missing)} fault on this tier is assigned to one of its {len(classes)} known classes, "
                       "probably with high confidence; no OOD method catches it (AI-N13).")
    return out
