"""Transfer-learning workflow, end to end and Streamlit-free (AUDIT AI-01..AI-06).

The TL dashboard used to do this orchestration inline, and that is where several of the audit's
P0 findings lived (row-level mixing, metadata loss, val == test for regression, training the
cached base model, unconditional promotion). Keeping the whole path here makes every step
unit-testable and lets the UI only collect inputs and render results:

    prepare_dataset()  -> TLDataset   merge whole acquisitions, window, split train/val/test by group
    finetune()         -> TLResult    train a CLONE, then evaluate base and candidate ONCE on test
    gate_decision()    -> GateDecision fail-closed promotion gate over all of the above
    save_candidate()   -> run_dir     artifacts + lineage + gate report, status "candidate"
    promote_candidate()               registry.promote(), which re-checks the persisted gate

Protocol for the independent test set (Q09/Q10): validation data is the only data seen during
training (loss curves); there is no early stopping or checkpoint selection, so no decision is
taken on validation either. After training ends, the base model and the fine-tuned model are both
evaluated exactly once on the same test windows, with the same preprocessing.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from driveflow.ai.promotion import GateDecision, PromotionGate
from driveflow.ai.promotion import save_candidate as _persist_candidate
from driveflow.ai.provenance import file_sha256, run_metadata, set_global_seed
from driveflow.ai.registry import promote
from driveflow.ai.transfer.data_merger import DataMerger
from driveflow.ai.transfer.dataset_validation import dataframe_sha256
from driveflow.ai.transfer.pipeline import FineTuneStrategy, TransferLearningPipeline
from driveflow.ai.transfer.validator import DEFAULT_THRESHOLDS, ModelValidator
from driveflow.models.common.splits import grouped_train_val_test_split
from driveflow.models.common.windowing import build_classification_windows, build_direct_forecast_windows

TASK_BY_BLOCK = {"classifier": "classification", "regressor": "regression"}
#: The simulator's sampling rate (driveflow.datagen.runner.TAU = 1e-4 s). External data must match.
SIM_FS_HZ = 10_000.0


@dataclass
class TLDataset:
    domain: str
    block: str
    windows: dict  # {"train"|"val"|"test": {"X", "y"}}
    split: dict  # SplitManifest.to_dict()
    merge_stats: dict
    sha256: str
    summary: dict
    integrity: list = field(default_factory=list)  # [(name, status, detail)]


@dataclass
class TLResult:
    pipeline: TransferLearningPipeline
    history: dict
    report: dict  # ValidationReport.to_dict()
    base_unchanged: bool
    test_evaluation: dict
    lineage: dict
    thresholds: dict


def _window(block: str, df: pd.DataFrame, channels: list, classes: list | None, input_window: int, horizon: int | None):
    missing = [c for c in channels if c not in df.columns or df[c].isna().all()]
    if missing:
        raise ValueError(f"Missing channel(s) the base model needs: {', '.join(missing)}.")
    if block == "classifier":
        X, y_str, groups = build_classification_windows(df, classes, channels, input_window)
        if X is None:
            raise ValueError("No classification windows could be built -- generate more/longer scenarios.")
        index = {c: i for i, c in enumerate(sorted(classes))}  # LabelEncoder order used at training time
        return X, np.array([index[c] for c in y_str]), groups
    X, Y, groups = build_direct_forecast_windows(df, channels, input_window, horizon)
    if X is None:
        raise ValueError("No forecast windows could be built -- generate more/longer scenarios.")
    return X, Y, groups


def prepare_dataset(
    domain: str,
    block: str,
    base_metrics: dict,
    input_window: int,
    horizon: int | None,
    simulated_df: pd.DataFrame | None,
    external_df: pd.DataFrame | None = None,
    external_summary: dict | None = None,
    mix_ratio: float = 1.0,
    seed: int = 0,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> TLDataset:
    """Merges whole acquisitions, windows them against the BASE model's channels/classes and
    splits by acquisition. Raises ValueError / InsufficientGroupsError with an actionable message
    instead of falling back to anything leaky."""
    channels = list(base_metrics["channels"])
    classes = list(base_metrics.get("classes") or []) or None
    seg_len = input_window + (horizon or 0) if block == "regressor" else input_window
    merger = DataMerger(simulated_df, external_df, mix_ratio=mix_ratio, window_samples=seg_len, expected_fs_hz=SIM_FS_HZ)
    merged = merger.merge(seed=seed)
    X, y, groups = _window(block, merged, channels, classes, input_window, horizon)
    manifest = grouped_train_val_test_split(groups, val_frac=val_frac, test_frac=test_frac, seed=seed, y=y if block == "classifier" else None)
    windows = {name: {"X": X[idx], "y": y[idx]} for name, idx in (("train", manifest.train_idx), ("val", manifest.val_idx), ("test", manifest.test_idx))}

    stats = merger.get_statistics()
    integrity = []
    origins_ok = all(str(g).startswith(("sim::", "ext::")) for g in np.unique(groups))
    integrity.append(("unique_acquisition_ids", "PASS" if origins_ok else "FAIL", "every acquisition ID carries its origin prefix (sim::/ext::)"))
    integrity.append(("temporal_order_preserved", "PASS", "whole acquisitions merged; rows never re-sampled or re-sorted; windows never cross a gap"))
    integrity.append(("no_group_leakage", "PASS" if manifest.is_disjoint() else "FAIL", "train/val/test share no acquisition and no window"))
    excluded = stats.get("excluded_groups") or []
    integrity.append(("excluded_acquisitions", "PASS" if not excluded else "INFO", f"{len(excluded)} acquisition(s) excluded by integrity checks" + (f": {excluded[:3]}" if excluded else "")))

    summary = {
        "ok": True,
        "n_acquisitions": int(len(np.unique(groups))),
        "n_windows": int(len(X)),
        "sha256": dataframe_sha256(merged),
        "external": external_summary,
        "mix": {k: stats.get(k) for k in ("mix_unit", "target_mix_ratio", "realized_mix_ratio", "units")},
    }
    return TLDataset(domain, block, windows, manifest.to_dict(), stats, summary["sha256"], summary, integrity)


def _configure(pipeline: TransferLearningPipeline, strategy_cfg: dict) -> None:
    strategy = strategy_cfg["strategy"]
    if strategy is FineTuneStrategy.FEATURE_EXTRACTOR:
        pipeline.freeze_backbone(strategy_cfg.get("frozen_layers", 0))
        pipeline.train_config = {"learning_rate": strategy_cfg.get("learning_rate", 1e-4)}
    elif strategy is FineTuneStrategy.DISCRIMINATIVE_LR:
        weighted = [layer for layer in pipeline.model.layers if layer.weights]
        pipeline.apply_discriminative_lr(strategy_cfg.get("base_lr", 1e-4), {weighted[-1].name: strategy_cfg.get("head_multiplier", 10.0)})
        pipeline.train_config = {}
    else:
        pipeline.train_config = {"adapter_layers": strategy_cfg.get("adapter_layers", 1), "learning_rate": strategy_cfg.get("learning_rate", 1e-3)}


def finetune(base: dict, dataset: TLDataset, strategy_cfg: dict, seed: int = 0, thresholds: dict | None = None, on_epoch_end=None) -> TLResult:
    """`base` is a ModelLoader.load_latest() result ({"model", "config", "metrics", "run_dir"}).
    The model in it is never trained (the pipeline clones it)."""
    set_global_seed(seed)
    task = TASK_BY_BLOCK[dataset.block]
    thresholds = dict(DEFAULT_THRESHOLDS[task] if thresholds is None else thresholds)
    pipeline = TransferLearningPipeline.from_loaded_model(base["model"], strategy=strategy_cfg["strategy"])
    _configure(pipeline, strategy_cfg)
    pipeline.train_config = {**pipeline.train_config, "seed": seed, **({"on_epoch_end": on_epoch_end} if on_epoch_end else {})}
    history = pipeline.train(dataset.windows["train"], dataset.windows["val"], epochs=strategy_cfg.get("epochs", 10), batch_size=strategy_cfg.get("batch_size", 16))

    # Single evaluation of the independent test set, base and candidate on identical windows.
    validator = ModelValidator(baseline_metrics=base["metrics"], thresholds=thresholds)
    report = validator.evaluate(pipeline.model, dataset.windows["test"], task=task, baseline_model=pipeline.base_model, history=history)
    test_evaluation = {"n_evaluations": 1, "used_for_selection": False, "same_test_for_base": True, "n_test_windows": int(len(dataset.windows["test"]["X"]))}
    base_unchanged = pipeline.verify_base_unchanged()

    weights_path = Path(base["run_dir"]) / "model.weights.h5"
    lineage = {
        "base_run_dir": str(base["run_dir"]),
        "base_weights_sha256": file_sha256(weights_path) if weights_path.exists() else None,
        "base_weights_fingerprint": pipeline.base_fingerprint,
        "dataset_sha256": dataset.sha256,
        "dataset_summary": dataset.summary,
        "strategy": strategy_cfg["strategy"].value,
        "strategy_params": {k: v for k, v in strategy_cfg.items() if k != "strategy"},
        "seed": seed,
        "split": dataset.split,
        "thresholds": thresholds,
        "test_protocol": test_evaluation,
        "base_unchanged": base_unchanged,
        "domain": dataset.domain,
        "block": dataset.block,
    }
    return TLResult(pipeline, history, report.to_dict(), base_unchanged, test_evaluation, lineage, thresholds)


def gate_decision(dataset: TLDataset | None, result: TLResult | None, human_approval: dict | None, require_human_approval: bool = True) -> GateDecision:
    return PromotionGate(require_human_approval=require_human_approval).evaluate(
        dataset_summary=dataset.summary if dataset else None,
        split_manifest=dataset.split if dataset else None,
        test_evaluation=result.test_evaluation if result else None,
        validation_report=result.report if result else None,
        base_unchanged=result.base_unchanged if result else None,
        lineage=result.lineage if result else None,
        human_approval=human_approval,
    )


def next_run_dir(config_path: Path) -> Path:
    """Same <config_dir>/<config_stem>/<date>_runNN/ layout experiments/train_model.py produces."""
    run_root = config_path.parent / config_path.stem
    date_str = dt.date.today().isoformat()
    n = 1
    while True:
        candidate = run_root / f"{date_str}_run{n:02d}"
        if not candidate.exists():
            return candidate
        n += 1


def candidate_run_dir(base_run_dir: Path) -> Path:
    """configs/<block>/<base_config>_tl/<date>_runNN/ -- a sibling of the base config's runs."""
    base_config_dir = Path(base_run_dir).parent
    return next_run_dir(base_config_dir.parent / f"{base_config_dir.name}_tl.yaml")


def save_candidate(result: TLResult, base: dict, decision: GateDecision, run_dir: Path | None = None, model_card: dict | None = None) -> Path:
    """Writes config/weights/metrics plus lineage, gate report, validation report and run
    metadata. Status is always "candidate": nothing here touches the registry."""
    run_dir = Path(run_dir) if run_dir is not None else candidate_run_dir(base["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(base["run_dir"]) / "config.yaml", run_dir / "config.yaml")
    result.pipeline.save_checkpoint(str(run_dir / "model.weights.h5"))
    m = result.report["metrics"]
    metrics = {k: base["metrics"][k] for k in ("kind", "domain", "tier", "channels", "classes") if k in base["metrics"]}
    if result.report["task"] == "classification":
        metrics.update({"test_accuracy": m.get("accuracy"), "test_macro_f1": m.get("macro_f1")})
    else:
        metrics.update({"test_rmse": m.get("rmse"), "test_mae": m.get("mae")})
    metrics.update({"n_test": result.test_evaluation["n_test_windows"], "fine_tuned_from": str(base["run_dir"]), "validation_passed": bool(result.report["passed"])})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    lineage = {**result.lineage, "model_card": model_card or {}}
    _persist_candidate(
        run_dir, lineage, decision,
        extra_files={"validation_report.json": result.report, "split_manifest.json": result.lineage["split"], "run_metadata.json": run_metadata()},
    )
    return run_dir


def update_gate_report(run_dir: Path, decision: GateDecision) -> None:
    """Re-persists the gate decision (e.g. after the reviewer approves a saved candidate)."""
    from driveflow.ai.promotion import GATE_REPORT_FILE

    (Path(run_dir) / GATE_REPORT_FILE).write_text(json.dumps(decision.to_dict(), indent=2, default=str))


def promote_candidate(run_dir: Path, approved_by: str, registry_path=None):
    """Delegates to registry.promote(), which re-reads gate_report.json and refuses anything not
    APPROVED -- the UI's disabled button is a convenience, this is the enforcement."""
    kwargs = {"promoted_by": approved_by}
    if registry_path is not None:
        kwargs["registry_path"] = Path(registry_path)
    return promote(run_dir, **kwargs)
