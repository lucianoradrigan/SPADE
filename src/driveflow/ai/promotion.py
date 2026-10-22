"""Promotion gate for fine-tuned (transfer-learning) models (AUDIT AI-06, AUDIT_ANSWERS.md Q19).

Before: the TL tab showed validation results but its "Promote to registry" button was always
enabled, the threshold it used (`max_accuracy_drop=1.0`) accepted any accuracy in [0, 1], and
driveflow.ai.registry.promote() accepted any run directory. A model that failed every check could
silently replace the production entry.

Now the decision lives in the backend, not in a button's `disabled=` flag:

1. PromotionGate.evaluate() turns the evidence of one TL run (dataset integrity, split
   independence, the once-only test evaluation, the task-specific ValidationReport, proof that the
   base weights were untouched, lineage, and -- by default -- an explicit human approval) into a
   GateDecision. FAIL-CLOSED: any required item that is missing, INCOMPLETE or FAIL blocks.
2. save_candidate() writes the run as a CANDIDATE (lineage.json + gate_report.json next to the
   weights). A candidate is inspectable and downloadable but never resolved as production.
3. driveflow.ai.registry.promote() refuses any TL-produced run whose persisted gate_report.json is
   not APPROVED (RegistryError) -- so the gate also holds for scripts, other UIs and API callers,
   not only for this dashboard. The only bypass is an explicit `override=` argument with a
   non-empty reason and author; it is recorded in the registry history as an override and the
   run's status becomes "promoted_override", never "validated".
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

GATE_REPORT_FILE = "gate_report.json"
LINEAGE_FILE = "lineage.json"

APPROVED = "APPROVED"
BLOCKED = "BLOCKED"


@dataclass
class GateItem:
    name: str
    status: str  # PASS | FAIL | PENDING | INCOMPLETE | NOT_APPLICABLE
    detail: str
    required: bool = True


@dataclass
class GateDecision:
    status: str
    items: list = field(default_factory=list)
    evaluated_utc: str = ""
    approved_by: str | None = None

    @property
    def approved(self) -> bool:
        return self.status == APPROVED

    @property
    def blocking(self) -> list:
        return [i for i in self.items if i.required and i.status not in ("PASS", "NOT_APPLICABLE")]

    def to_dict(self) -> dict:
        return {"status": self.status, "evaluated_utc": self.evaluated_utc, "approved_by": self.approved_by, "items": [asdict(i) for i in self.items]}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class PromotionGate:
    """Evaluates the evidence for one fine-tuned run. Every argument is the *evidence object*, not
    a pre-computed boolean, so the gate re-derives each verdict itself."""

    def __init__(self, require_human_approval: bool = True):
        self.require_human_approval = require_human_approval

    def evaluate(
        self,
        *,
        dataset_summary: dict | None,
        split_manifest: dict | None,
        test_evaluation: dict | None,
        validation_report: dict | None,
        base_unchanged: bool | None,
        lineage: dict | None,
        human_approval: dict | None = None,
    ) -> GateDecision:
        items = []

        if not dataset_summary:
            items.append(GateItem("dataset_integrity", "PENDING", "no dataset report"))
        else:
            ok = bool(dataset_summary.get("ok")) and bool(dataset_summary.get("sha256"))
            items.append(GateItem("dataset_integrity", "PASS" if ok else "FAIL", f"{dataset_summary.get('n_acquisitions', '?')} acquisition(s), sha256 {str(dataset_summary.get('sha256'))[:12]}" if ok else "dataset failed validation or has no content hash"))

        if not split_manifest:
            items.append(GateItem("split_independence", "PENDING", "no split manifest"))
        else:
            n = split_manifest.get("n_groups", {})
            ok = bool(split_manifest.get("disjoint")) and all(n.get(k, 0) >= 1 for k in ("train", "val", "test"))
            items.append(GateItem("split_independence", "PASS" if ok else "FAIL", f"groups train/val/test = {n.get('train')}/{n.get('val')}/{n.get('test')}, disjoint={split_manifest.get('disjoint')}"))

        if not test_evaluation:
            items.append(GateItem("independent_test_evaluated", "PENDING", "independent test set not evaluated yet"))
        else:
            # A new class vocabulary (feature transfer with a new head) has no base score to compare;
            # that must be recorded explicitly (base_comparable False), never left implicit.
            not_comparable = test_evaluation.get("base_comparable") is False
            ok = test_evaluation.get("n_evaluations") == 1 and not test_evaluation.get("used_for_selection", True) and (test_evaluation.get("same_test_for_base") or not_comparable)
            detail = ("evaluated once after training, not used for model selection; base not comparable (new class vocabulary)" if not_comparable
                      else "evaluated once after training, same test set for base and fine-tuned, not used for model selection")
            items.append(GateItem("independent_test_evaluated", "PASS" if ok else "FAIL", detail if ok else f"test-set protocol violated: {test_evaluation}"))

        if not validation_report:
            items.append(GateItem("validation_checks", "PENDING", "no validation report"))
        else:
            for c in validation_report.get("checks", []):
                status = {"PASS": "PASS", "FAIL": "FAIL", "NOT_APPLICABLE": "NOT_APPLICABLE"}.get(c["status"], "INCOMPLETE")
                items.append(GateItem(f"validation:{c['name']}", status, c["detail"], required=c.get("required", True)))

        if base_unchanged is None:
            items.append(GateItem("base_model_unchanged", "PENDING", "base weights fingerprint not verified"))
        else:
            items.append(GateItem("base_model_unchanged", "PASS" if base_unchanged else "FAIL", "base weights bit-identical after fine-tuning" if base_unchanged else "base model weights were modified"))

        required_lineage = ("base_run_dir", "base_weights_sha256", "dataset_sha256", "strategy", "seed", "split")
        # `is None`, not falsiness: seed=0 is a perfectly recorded seed.
        missing = [k for k in required_lineage if (lineage or {}).get(k) in (None, "", {}, [])]
        items.append(GateItem("lineage_recorded", "PASS" if lineage and not missing else ("PENDING" if not lineage else "FAIL"), "lineage complete" if lineage and not missing else f"missing lineage field(s): {missing}"))

        if self.require_human_approval:
            ha = human_approval or {}
            ok = bool(ha.get("approved")) and bool(str(ha.get("approved_by", "")).strip())
            items.append(GateItem("human_approval", "PASS" if ok else "PENDING", f"approved by {ha.get('approved_by')}" if ok else "explicit reviewer approval required"))
        else:
            items.append(GateItem("human_approval", "NOT_APPLICABLE", "not required by this gate configuration", required=False))

        blocking = [i for i in items if i.required and i.status not in ("PASS", "NOT_APPLICABLE")]
        return GateDecision(
            status=APPROVED if not blocking else BLOCKED,
            items=items,
            evaluated_utc=_now(),
            approved_by=(human_approval or {}).get("approved_by") if not blocking else None,
        )


def save_candidate(run_dir, lineage: dict, decision: GateDecision, extra_files: dict | None = None) -> Path:
    """Persists lineage + the gate decision next to an already-saved run (config/weights/metrics).
    The run is a CANDIDATE regardless of the decision -- promotion is a separate, explicit step."""
    run_dir = Path(run_dir)
    lineage = dict(lineage)
    lineage["status"] = "candidate"
    (run_dir / LINEAGE_FILE).write_text(json.dumps(lineage, indent=2, default=str))
    (run_dir / GATE_REPORT_FILE).write_text(json.dumps(decision.to_dict(), indent=2, default=str))
    for name, payload in (extra_files or {}).items():
        (run_dir / name).write_text(json.dumps(payload, indent=2, default=str))
    return run_dir


def load_gate_report(run_dir) -> dict | None:
    path = Path(run_dir) / GATE_REPORT_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def is_transfer_learning_run(run_dir) -> bool:
    """A run produced by fine-tuning (TL tab or any script using its layout): it has a lineage
    file, its metrics name a base run, or it lives under a `<config>_tl/` directory."""
    run_dir = Path(run_dir)
    if (run_dir / LINEAGE_FILE).exists() or run_dir.parent.name.endswith("_tl"):
        return True
    metrics = run_dir / "metrics.json"
    if metrics.exists():
        try:
            return bool(json.loads(metrics.read_text()).get("fine_tuned_from"))
        except json.JSONDecodeError:
            return True  # unreadable metrics -> treat as untrusted, gate applies
    return False


def reevaluate_persisted(run_dir, require_human_approval: bool = True) -> GateDecision | None:
    """Re-derives the gate decision from the evidence persisted in run_dir (lineage.json,
    validation_report.json, split manifest inside lineage, approver from gate_report.json) instead
    of trusting the stored status string. None if the evidence files are absent."""
    run_dir = Path(run_dir)
    lineage_path, report_path = run_dir / LINEAGE_FILE, run_dir / "validation_report.json"
    gate = load_gate_report(run_dir)
    if not (lineage_path.exists() and report_path.exists() and gate):
        return None
    lineage = json.loads(lineage_path.read_text())
    approver = gate.get("approved_by")
    return PromotionGate(require_human_approval=require_human_approval).evaluate(
        dataset_summary=lineage.get("dataset_summary"),
        split_manifest=lineage.get("split"),
        test_evaluation=lineage.get("test_protocol"),
        validation_report=json.loads(report_path.read_text()),
        base_unchanged=lineage.get("base_unchanged"),
        lineage=lineage,
        human_approval={"approved": bool(approver), "approved_by": approver},
    )
