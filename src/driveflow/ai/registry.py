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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

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


def _save_index(registry_path: Path, entries: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w") as f:
        yaml.safe_dump({"entries": entries}, f, sort_keys=True)


def promote(run_dir, registry_path: Path = DEFAULT_REGISTRY_PATH) -> RegistryEntry:
    """Marks run_dir (a directory produced by experiments/train_model.py -- config.yaml +
    model.weights.h5 + metrics.json) as the production run for its (domain, tier, block), the
    latter two read from the run's own config.yaml/metrics.json, not passed by the caller."""
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

    key = _key(domain, tier, block)
    entries = _load_index(registry_path)
    try:
        rel_path = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        rel_path = str(run_dir)
    entries[key] = rel_path
    _save_index(registry_path, entries)
    return RegistryEntry(domain=domain, tier=tier, block=block, run_dir=run_dir)


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
    run_dir = resolve(domain, tier, block, registry_path)
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
