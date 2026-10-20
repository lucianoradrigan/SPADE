"""Workspace status from real checks (guide Sec. 3.4 Overview: no "Engine ready" without health
checks). Every item is a filesystem/registry fact verified on each call -- nothing is assumed."""

from __future__ import annotations

from importlib import metadata

from driveflow.ai.registry import DEFAULT_REGISTRY_PATH, RegistryError, _load_index, resolve
from driveflow.datagen.runner import _CALIBRATION_PATH, _DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH
from driveflow.monitoring.rules.schema import load_all_rulesets
from driveflow.viz.adapters.monitoring_adapter import RULES_DIR


def workspace_status() -> dict:
    checks = []
    entries = _load_index(DEFAULT_REGISTRY_PATH)
    resolvable = 0
    for key in entries:
        domain, tier, block = key.split("/")
        try:
            resolve(domain, tier, block)
            resolvable += 1
        except RegistryError:
            pass
    checks.append({"name": "Model registry", "ok": bool(entries) and resolvable == len(entries), "detail": f"{resolvable}/{len(entries)} promoted entries resolve to an existing run directory"})
    checks.append({"name": "DPC network weights", "ok": _DPC_WEIGHTS_PATH.exists() and _DPC_NOMINAL_WEIGHTS_PATH.exists(), "detail": f"{_DPC_NOMINAL_WEIGHTS_PATH.name} + {_DPC_WEIGHTS_PATH.name}"})
    checks.append({"name": "Vibration calibration (Module B)", "ok": _CALIBRATION_PATH.exists(), "detail": f"{_CALIBRATION_PATH.name} -- 1 measured axis, Y/Z assumed (see Documentation)"})
    try:
        rulesets = load_all_rulesets(RULES_DIR)
        checks.append({"name": "Monitoring rules", "ok": bool(rulesets), "detail": f"rule sets for: {', '.join(sorted(rulesets)) or 'none'}"})
    except Exception as exc:
        checks.append({"name": "Monitoring rules", "ok": False, "detail": f"failed to load: {exc}"})
    try:
        version = metadata.version("driveflow")
    except metadata.PackageNotFoundError:
        version = "dev"
    return {
        "version": version,
        "checks": checks,
        "n_promoted": len(entries),
        "data_source": "simulation",  # no hardware ingestion exists in this repository
        "domains": ["dc_motor", "vsc_dpc"],
    }
