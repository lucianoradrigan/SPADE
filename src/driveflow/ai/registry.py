"""Model registry (docs/design_ai_layer_transversal.md Sec. 7, Sec. 8 step 6): the single place
that resolves (domain, tier, block) -> the promoted training run's directory (weights + config +
metrics together, produced by experiments/train_model.py, Sec. 6.4). No caller outside this
module should hardcode a run path -- the dashboard and any deployment script only ask the
registry which run to use for a given combination, so domain/tier conditional logic isn't
duplicated at every call site (Sec. 7).

Promotion is a deliberate, separate action from training (Sec. 6.4: "hasta que uno se promueve a
'el que usa producción'") -- train_model.py never calls promote() itself; a human (or a later
promotion-policy script, not built here) decides which run is production-worthy. The registry
itself is one committed YAML file (configs/registry.yaml), not a marker file scattered per run
directory, so the full promoted state is visible/diffable in one place and one commit -- promoting
a run overwrites any previous entry for the same (domain, tier, block): this names ONE current
production run, not a history (the run directories under configs/<block>/<config_stem>/ are
already that history).

Built now (Sec. 8 step 6) only once a real trained artifact exists to register -- see
docs/design_ai_layer_transversal.md's status note for which run is registered and how it was
produced.

Safety additions (AUDIT AI-06): writes are atomic (temp file + os.replace, so a crash can never
leave a half-written registry.yaml); every promotion / rollback is appended to an audit history
(<registry>_history.yaml next to the registry -- registry.yaml's own format is unchanged); and a
run produced by transfer learning is only accepted if its persisted gate_report.json is APPROVED
(driveflow.ai.promotion). The history is what makes a promotion reversible (rollback()).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from driveflow.ai.promotion import APPROVED, LINEAGE_FILE, is_transfer_learning_run, load_gate_report, reevaluate_persisted

from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import load_classifier_config
from driveflow.models.regressors.builder import build_forecaster
from driveflow.models.regressors.schemas import load_forecaster_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "registry.yaml"

ALLOWED_BLOCKS = frozenset({"classifier", "regressor"})


class RegistryError(ValueError):
    """No promoted run exists for the requested (domain, tier, block), or the registry file / a
    run directory is malformed -- raised so a caller gets a clear "not available" signal instead
    of a raw KeyError/FileNotFoundError."""


class PromotionRejected(RegistryError):
    """A transfer-learning run without an APPROVED gate report was offered for promotion."""


@dataclass(frozen=True)
class RegistryEntry:
    domain: str
    tier: str
    block: str
    run_dir: Path


def _detect_block(raw_config: dict) -> str:
    """Same 'blocks' vs 'recurrent_type' heuristic as experiments/train_model.py's
    _load_any_config -- kept independent (not imported from there) since library code
    (src/driveflow) should not depend on a script (experiments/)."""
    if "blocks" in raw_config:
        return "classifier"
    if "recurrent_type" in raw_config:
        return "regressor"
    raise RegistryError("cannot tell if this run's config.yaml is a classifier ('blocks') or a regressor ('recurrent_type')")


def _key(domain: str, tier: str, block: str) -> str:
    if block not in ALLOWED_BLOCKS:
        raise RegistryError(f"block {block!r} not in {sorted(ALLOWED_BLOCKS)}")
    return f"{domain}/{tier}/{block}"


def _load_index(registry_path: Path) -> dict:
    if not registry_path.exists():
        return {}
    with registry_path.open() as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get("entries", {}))


def _atomic_write_yaml(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _save_index(registry_path: Path, entries: dict) -> None:
    _atomic_write_yaml(registry_path, {"entries": entries})


def history_path(registry_path: Path = DEFAULT_REGISTRY_PATH) -> Path:
    registry_path = Path(registry_path)
    return registry_path.with_name(f"{registry_path.stem}_history.yaml")


def registry_history(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list:
    path = history_path(registry_path)
    if not path.exists():
        return []
    with path.open() as f:
        return list((yaml.safe_load(f) or {}).get("events", []))


def _append_history(registry_path: Path, event: dict) -> None:
    events = registry_history(registry_path)
    events.append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **event})
    _atomic_write_yaml(history_path(registry_path), {"events": events})


def _rel(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        return str(run_dir)


def promote(run_dir, registry_path: Path = DEFAULT_REGISTRY_PATH, promoted_by: str | None = None, override: dict | None = None) -> RegistryEntry:
    """Marks run_dir (a directory produced by experiments/train_model.py -- config.yaml +
    model.weights.h5 + metrics.json) as the production run for its (domain, tier, block), the
    latter two read from the run's own config.yaml/metrics.json, not passed by the caller.

    Transfer-learning runs (driveflow.ai.promotion.is_transfer_learning_run) additionally need an
    APPROVED gate_report.json -- otherwise PromotionRejected, and registry.yaml is not touched.
    `override={"reason": ..., "by": ...}` is the only, audited bypass (never shown as validated)."""
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise RegistryError(f"{run_dir}: missing config.yaml -- not a valid train_model.py run directory")
    if not (run_dir / "model.weights.h5").exists():
        raise RegistryError(f"{run_dir}: missing model.weights.h5 -- not a valid train_model.py run directory")
    if not (run_dir / "metrics.json").exists():
        raise RegistryError(f"{run_dir}: missing metrics.json -- not a valid train_model.py run directory")

    with config_path.open() as f:
        raw = yaml.safe_load(f)
    domain, tier = raw.get("domain"), raw.get("tier")
    if not domain or not tier:
        raise RegistryError(f"{config_path}: missing domain/tier")
    block = _detect_block(raw)

    gate_status = None
    action = "promote"
    if is_transfer_learning_run(run_dir):
        gate = load_gate_report(run_dir)
        # Re-derive the decision from the persisted evidence; a stored "APPROVED" string whose
        # evidence no longer supports it (edited file, missing report) is not accepted.
        recomputed = reevaluate_persisted(run_dir)
        gate_status = gate.get("status") if gate else None
        if gate_status == APPROVED and (recomputed is None or not recomputed.approved):
            gate_status = "STALE_OR_UNSUPPORTED"
        if gate_status != APPROVED:
            reason = str((override or {}).get("reason", "")).strip()
            by = str((override or {}).get("by", "")).strip()
            if not (reason and by):
                raise PromotionRejected(
                    f"{run_dir}: transfer-learning run has no APPROVED gate report (status={gate_status!r}) -- "
                    "promotion refused; registry unchanged. Fix the failing checks, or pass an explicit, audited override."
                )
            action = "promote_override"

    key = _key(domain, tier, block)
    entries = _load_index(registry_path)
    previous = entries.get(key)
    rel_path = _rel(run_dir)
    entries[key] = rel_path
    _save_index(registry_path, entries)
    _append_history(registry_path, {
        "action": action, "key": key, "previous": previous, "new": rel_path, "by": promoted_by or (override or {}).get("by"),
        "gate_status": gate_status, "override_reason": (override or {}).get("reason") if action == "promote_override" else None,
    })
    lineage_path = run_dir / LINEAGE_FILE
    if lineage_path.exists():
        lineage = json.loads(lineage_path.read_text())
        lineage["status"] = "promoted" if action == "promote" else "promoted_override"
        lineage_path.write_text(json.dumps(lineage, indent=2, default=str))
    return RegistryEntry(domain=domain, tier=tier, block=block, run_dir=run_dir)


def rollback(domain: str, tier: str, block: str, registry_path: Path = DEFAULT_REGISTRY_PATH, by: str | None = None, reason: str | None = None) -> str | None:
    """Restores the entry that was active before the most recent promotion of (domain, tier,
    block), per the audit history. Returns the restored path (None = the key is removed because
    nothing was promoted before). Raises RegistryError if there is no promotion to undo."""
    key = _key(domain, tier, block)
    entries = _load_index(registry_path)
    events = [e for e in registry_history(registry_path) if e.get("key") == key]
    current = entries.get(key)
    # The latest promotion that installed the CURRENT entry -- after an earlier rollback this
    # walks further back in history instead of re-undoing the rollback itself.
    target_event = next((e for e in reversed(events) if e.get("action", "").startswith("promote") and e.get("new") == current), None)
    if target_event is None:
        raise RegistryError(f"no recorded promotion of {key!r} to roll back")
    restored = target_event.get("previous")
    if restored is None:
        entries.pop(key, None)
    else:
        entries[key] = restored
    _save_index(registry_path, entries)
    _append_history(registry_path, {"action": "rollback", "key": key, "previous": current, "new": restored, "by": by, "reason": reason})
    return restored


def resolve(domain: str, tier: str, block: str, registry_path: Path = DEFAULT_REGISTRY_PATH) -> Path:
    """Returns the promoted run directory for (domain, tier, block). Raises RegistryError if
    nothing has been promoted for that combination, or if the registered path no longer exists."""
    key = _key(domain, tier, block)
    entries = _load_index(registry_path)
    if key not in entries:
        raise RegistryError(f"no promoted run for {key!r} -- call driveflow.ai.registry.promote() on a train_model.py run directory first")
    run_dir = (REPO_ROOT / entries[key]).resolve()
    if not run_dir.exists():
        raise RegistryError(f"{key!r} points at {run_dir}, which no longer exists")
    return run_dir


def load_promoted_model(domain: str, tier: str, block: str, registry_path: Path = DEFAULT_REGISTRY_PATH) -> tuple:
    """Resolves + loads a ready-to-use keras.Model for (domain, tier, block), plus its config and
    metrics -- what viz/ai_dashboard.py needs without knowing anything about run-directory paths
    or which builder/schema module to call. n_channels comes from metrics.json's "channels" list
    (saved by train_model.py) rather than being re-derived from the dataset.

    Returns (model, config, metrics: dict).
    """
    return load_run_model(resolve(domain, tier, block, registry_path), block)


def load_run_model(run_dir, block: str) -> tuple:
    """Same as load_promoted_model, for any run directory (promoted or a candidate).
    Returns (model, config, metrics: dict)."""
    run_dir = Path(run_dir)
    with (run_dir / "metrics.json").open() as f:
        metrics = json.load(f)
    n_channels = len(metrics["channels"])

    if block == "classifier":
        config = load_classifier_config(run_dir / "config.yaml")
        model = build_classifier(config, n_channels=n_channels)
    else:
        config = load_forecaster_config(run_dir / "config.yaml")
        model = build_forecaster(config, n_channels=n_channels)

    model.load_weights(run_dir / "model.weights.h5")
    return model, config, metrics
