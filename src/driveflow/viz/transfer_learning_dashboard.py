"""Transfer Learning tab (Phase 3, docs/design_ai_layer_transversal.md's AI layer is the prior
art this extends): fine-tune an already-promoted PC-tier model against new data (simulated,
uploaded, or a mix of both), right from the dashboard. Kept in its own module -- same reasoning
as viz/ai_dashboard.py's own module docstring -- so dashboard.py only gains a landing-page card +
one-line dispatch. Must not import from driveflow.viz.dashboard (nor from ai_dashboard.py, whose
own sidebar helpers are private to that module) -- keeps the dependency one-directional.

Scope: fine-tunes the PC-tier model only. Edge-tier (Raspberry Pi 5/ESP32) distillation from a
newly-promoted fine-tuned run is the EXISTING `experiments/train_model.py --distill` path -- this
tab doesn't re-implement that, it produces a normal promotable PC-tier run that path already
knows how to consume. "Architecture selection" from the original GUI spec (choosing a NEW
architecture like Stacked CNN+LSTM) is deliberately not offered here: fine-tuning modifies an
existing model's WEIGHTS, not its architecture -- picking a different architecture there would
mean training from scratch, which is what Fase A/B + experiments/train_model.py's own default
(non---distill) path is already for.
"""

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai.registry import RegistryError, promote
from driveflow.ai.transfer.data_merger import DataMerger
from driveflow.ai.transfer.loader import ModelLoader
from driveflow.ai.transfer.pipeline import FineTuneStrategy, TransferLearningPipeline
from driveflow.ai.transfer.validator import ModelValidator
from driveflow.datagen import Scenario, run_scenario
from driveflow.models.common.splits import grouped_split, prepare_classification_splits
from driveflow.models.common.windowing import build_classification_windows, build_direct_forecast_windows
from driveflow.viz.dpc_upload_validation import validate_dc_motor_upload, validate_vsc_dpc_forecast_upload

_DOMAIN_LABELS = {"dc_motor": "Fase A -- DC motor diagnosis", "vsc_dpc": "Fase B -- VSC / DPC"}
#: Which block is actually trained+registered for each domain today (Sec. 8's own status note) --
#: the other block per domain has nothing promoted to fine-tune FROM yet.
_BLOCK_BY_DOMAIN = {"dc_motor": "classifier", "vsc_dpc": "regressor"}
_STRATEGY_LABELS = {
    FineTuneStrategy.FEATURE_EXTRACTOR: "Feature Extractor (freeze early layers)",
    FineTuneStrategy.DISCRIMINATIVE_LR: "Discriminative LR (per-layer learning rate)",
    FineTuneStrategy.ADAPTER: "Adapter (freeze all but the last layers)",
}


def _next_run_dir(config_path: Path) -> Path:
    """Same <config_dir>/<config_stem>/<date>_runNN/ layout experiments/train_model.py's own
    _next_run_dir produces, so a fine-tuned run is indistinguishable from a freshly-trained one
    to promote()/resolve() -- and to a future --distill run using it as teacher."""
    run_root = config_path.parent / config_path.stem
    date_str = dt.date.today().isoformat()
    n = 1
    while True:
        candidate = run_root / f"{date_str}_run{n:02d}"
        if not candidate.exists():
            return candidate
        n += 1


def _save_finetuned_artifacts(run_dir: Path, base_config_path: Path, pipeline: TransferLearningPipeline, metrics: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(base_config_path, run_dir / "config.yaml")
    pipeline.save_checkpoint(str(run_dir / "model.weights.h5"))
    import json

    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def _generate_simulated_dataframe(domain: str, n_scenarios: int, duration_s: float, fault_types: list) -> pd.DataFrame:
    """A batch of short runs (not one single run, unlike ai_dashboard.py's own sample generator --
    transfer learning wants VOLUME, not a single quick preview) covering the requested fault
    types, one seed per scenario so each is a distinguishable "group" for prepare_classification_splits."""
    records = []
    for seed, fault in enumerate((fault_types * n_scenarios)[:n_scenarios]):
        if domain == "dc_motor":
            scenario = Scenario(scenario_id=f"tl_sample_{seed}", fault_type=None if fault == "healthy" else fault, duration_s=duration_s, seed=seed)
        else:
            scenario = Scenario(scenario_id=f"tl_sample_{seed}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=duration_s, seed=seed)
        records.extend(run_scenario(scenario))
    df = pd.DataFrame.from_records(records)
    if domain == "dc_motor":
        df["plant_config_id"] = "dc_perm_ex_v1"
        df = load_diagnosis_dataset_from_df(df)
    return df


def load_diagnosis_dataset_from_df(df: pd.DataFrame) -> pd.DataFrame:
    from driveflow.models.common.dataset import filter_diagnosis_domain

    return filter_diagnosis_domain(df)


@st.cache_resource
def _cached_loader_result(registry_path: str, domain: str, block: str, tier: str):
    loader = ModelLoader(registry_path=registry_path, domain=domain, block=block, tier=tier)
    return loader.load_latest()


def _render_model_selection(domain: str, block: str):
    st.sidebar.markdown('<div class="df-sidebar-title">1. Base model</div>', unsafe_allow_html=True)
    from driveflow.ai.registry import DEFAULT_REGISTRY_PATH

    try:
        result = _cached_loader_result(str(DEFAULT_REGISTRY_PATH), domain, block, "pc")
    except RegistryError:
        st.sidebar.info(f"No PC-tier {block} registered yet for {_DOMAIN_LABELS[domain]} -- nothing to fine-tune.")
        return None
    metrics = result["metrics"]
    headline = f"{metrics['test_accuracy']:.1%} accuracy" if "test_accuracy" in metrics else f"{metrics.get('test_rmse', float('nan')):.3f} RMSE"
    st.sidebar.caption(f"**{result['run_dir'].name}** -- {headline}, {len(metrics['channels'])} channels, trained {result['run_dir'].parent.name}.")
    return result


def _render_data_source(domain: str):
    st.sidebar.markdown('<div class="df-sidebar-title">2. Data</div>', unsafe_allow_html=True)
    n_scenarios = st.sidebar.slider("Simulated scenarios", 4, 40, 12, key="tl_n_scenarios")
    duration_s = st.sidebar.slider("Duration per scenario (s)", 0.05, 0.5, 0.15, key="tl_duration")
    if domain == "dc_motor":
        fault_types = st.sidebar.multiselect("Fault types", ["healthy", "outer_race", "inner_race"], default=["healthy", "outer_race"], key="tl_faults")
    else:
        fault_types = ["healthy"]

    uploaded = st.sidebar.file_uploader("External data (optional, CSV/JSON)", type=["csv", "json"], key="tl_upload")
    external_df = None
    if uploaded is not None:
        try:
            df_raw = pd.read_json(uploaded) if uploaded.name.lower().endswith(".json") else pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not parse '{uploaded.name}': {e}")
            df_raw = None
        if df_raw is not None:
            validator = validate_dc_motor_upload if domain == "dc_motor" else validate_vsc_dpc_forecast_upload
            external_df, _, messages = validator(df_raw)
            for level, text in messages:
                getattr(st.sidebar, level)(text)

    mix_ratio = st.sidebar.slider("Mix ratio (simulated fraction)", 0.0, 1.0, 0.7, key="tl_mix_ratio", disabled=external_df is None)
    return {"n_scenarios": n_scenarios, "duration_s": duration_s, "fault_types": fault_types or ["healthy"], "external_df": external_df, "mix_ratio": mix_ratio}


def _render_strategy_controls():
    st.sidebar.markdown('<div class="df-sidebar-title">3. Fine-tune strategy</div>', unsafe_allow_html=True)
    strategy = st.sidebar.radio("Strategy", list(_STRATEGY_LABELS), format_func=lambda s: _STRATEGY_LABELS[s], key="tl_strategy")
    params = {"strategy": strategy}
    if strategy is FineTuneStrategy.FEATURE_EXTRACTOR:
        params["frozen_layers"] = st.sidebar.slider("Frozen layers", 0, 6, 2, key="tl_frozen_layers")
        params["learning_rate"] = st.sidebar.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="tl_lr_fe")
    elif strategy is FineTuneStrategy.DISCRIMINATIVE_LR:
        params["base_lr"] = st.sidebar.select_slider("Base learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="tl_base_lr")
        params["head_multiplier"] = st.sidebar.slider("Output-layer LR multiplier", 1.0, 20.0, 10.0, key="tl_head_mult", help="The last layer trains at base_lr * this; every other layer trains at base_lr * 1.0.")
    else:
        params["adapter_layers"] = st.sidebar.slider("Trainable output layers", 1, 3, 1, key="tl_adapter_layers")
        params["learning_rate"] = st.sidebar.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-3, key="tl_lr_adapter")
    params["epochs"] = st.sidebar.slider("Epochs", 1, 50, 15, key="tl_epochs")
    params["batch_size"] = st.sidebar.select_slider("Batch size", [4, 8, 16, 32], value=8, key="tl_batch_size")
    return params


def _prepare_windows(block: str, df: pd.DataFrame, input_window: int, horizon: int | None, channels: list, classes: list | None):
    """channels/classes come from the BASE model's own metrics.json, not re-discovered from the
    new data -- the loaded model's input/output layers have a FIXED shape from when it was first
    trained (n_channels, num_classes), so fine-tuning data must be windowed against the exact
    same channel list and class vocabulary, or model.fit() fails with a shape mismatch (found by
    testing the actual Train button, not by inspection: a fresh small simulated sample can
    legitimately "discover" fewer live channels than the original training set did, e.g. a
    near-constant torque_nm in a 4-scenario sample where the original run saw more variation)."""
    missing = [c for c in channels if c not in df.columns or df[c].isna().all()]
    if missing:
        return None, f"Missing channel(s) the base model needs: {', '.join(missing)}."

    if block == "classifier":
        X, y_str, groups = build_classification_windows(df, classes, channels, input_window)
        if X is None:
            return None, "No classification windows could be built -- generate more/longer scenarios."
        X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_classification_splits(X, y_str, classes, groups, seed=0)
        return {
            "train": {"X": X_train, "y": y_train}, "val": {"X": X_val, "y": y_val}, "test": {"X": X_test, "y": y_test},
        }, None

    X, Y, groups = build_direct_forecast_windows(df, channels, input_window, horizon)
    if X is None:
        return None, "No forecast windows could be built -- generate more/longer scenarios."
    dummy = np.zeros(len(X), dtype=int)
    idx_train, idx_test = grouped_split(np.arange(len(X)), dummy, groups, test_frac=0.2, seed=0)
    return {"train": {"X": X[idx_train], "y": Y[idx_train]}, "val": {"X": X[idx_test], "y": Y[idx_test]}, "test": {"X": X[idx_test], "y": Y[idx_test]}}, None


def _render_fase_tl():
    st.sidebar.markdown(
        '<div class="df-sidebar-title">Transfer Learning</div>'
        '<div class="df-sidebar-hint">Fine-tune an already-promoted PC-tier model against new data</div>',
        unsafe_allow_html=True,
    )
    domain = st.sidebar.selectbox("Domain", list(_DOMAIN_LABELS), format_func=lambda d: _DOMAIN_LABELS[d], key="tl_domain")
    block = _BLOCK_BY_DOMAIN[domain]

    base = _render_model_selection(domain, block)
    if base is None:
        st.info("Choose a domain with a promoted PC-tier model in the sidebar to get started.")
        return
    data_cfg = _render_data_source(domain)
    strategy_cfg = _render_strategy_controls()

    tab_train, tab_compare, tab_deploy = st.tabs(["1. Training", "2. Comparison", "3. Promote & Deploy"])

    with tab_train:
        st.markdown("##### Training")
        if st.button("Train", type="primary", key="tl_train_button"):
            with st.spinner("Generating data..."):
                sim_df = _generate_simulated_dataframe(domain, data_cfg["n_scenarios"], data_cfg["duration_s"], data_cfg["fault_types"])
            merger = DataMerger(simulated_data=sim_df, external_data=data_cfg["external_df"], mix_ratio=data_cfg["mix_ratio"])
            merged = merger.merge()

            config = base["config"]
            input_window = config.input_window
            horizon = getattr(config, "horizon", None)
            base_channels = base["metrics"]["channels"]
            base_classes = base["metrics"].get("classes")
            windows, error = _prepare_windows(block, merged, input_window, horizon, base_channels, base_classes)
            if windows is None:
                st.error(error or f"Not enough rows to build even one {input_window}-sample window -- generate more scenarios or a longer duration.")
                return

            pipeline = TransferLearningPipeline.from_loaded_model(base["model"], strategy=strategy_cfg["strategy"])
            if strategy_cfg["strategy"] is FineTuneStrategy.FEATURE_EXTRACTOR:
                pipeline.freeze_backbone(strategy_cfg["frozen_layers"])
                pipeline.train_config = {"learning_rate": strategy_cfg["learning_rate"]}
            elif strategy_cfg["strategy"] is FineTuneStrategy.DISCRIMINATIVE_LR:
                weighted = [layer for layer in pipeline.model.layers if layer.weights]
                pipeline.apply_discriminative_lr(strategy_cfg["base_lr"], {weighted[-1].name: strategy_cfg["head_multiplier"]})
            else:
                pipeline.train_config = {"adapter_layers": strategy_cfg["adapter_layers"], "learning_rate": strategy_cfg["learning_rate"]}

            progress = st.progress(0.0, text="Training...")
            chart_placeholder = st.empty()
            loss_so_far = []

            def _on_epoch_end(epoch, logs):
                loss_so_far.append(logs["loss"])
                progress.progress((epoch + 1) / strategy_cfg["epochs"], text=f"Epoch {epoch + 1}/{strategy_cfg['epochs']} -- loss {logs['loss']:.4f}")
                fig = go.Figure(go.Scatter(y=loss_so_far, mode="lines+markers", name="loss"))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                chart_placeholder.plotly_chart(fig, width="stretch", key=f"tl_loss_chart_{epoch}")

            pipeline.train_config = {**pipeline.train_config, "on_epoch_end": _on_epoch_end}
            history = pipeline.train(windows["train"], windows["val"], epochs=strategy_cfg["epochs"], batch_size=strategy_cfg["batch_size"])
            progress.progress(1.0, text="Done.")

            validator = ModelValidator(baseline_metrics=base["metrics"], thresholds={"max_accuracy_drop": 1.0})  # informational only -- see Promote tab for the real gate
            passed, val_results = validator.validate(pipeline.model, windows["test"])

            st.session_state["tl_result"] = {
                "pipeline": pipeline, "history": history, "val_results": val_results,
                "base": base, "windows": windows, "domain": domain, "block": block,
            }
            st.success("Training complete -- see the Comparison and Promote & Deploy tabs.")
        elif "tl_result" not in st.session_state:
            st.caption("Configure the sidebar and click Train.")

    result = st.session_state.get("tl_result")

    with tab_compare:
        if result is None or result["domain"] != domain:
            st.caption("Train a model first.")
        else:
            base_metrics = result["base"]["metrics"]
            new_results = result["val_results"]
            st.markdown("##### Base model vs. fine-tuned")
            rows = []
            if "test_accuracy" in base_metrics and "accuracy" in new_results:
                rows.append(("Accuracy", f"{base_metrics['test_accuracy']:.1%}", f"{new_results['accuracy']:.1%}"))
            if "test_rmse" in base_metrics:
                rows.append(("Baseline RMSE", f"{base_metrics['test_rmse']:.4f}", "--"))
            st.table(pd.DataFrame(rows, columns=["Metric", "Base", "Fine-tuned"]))
            fig = go.Figure(go.Scatter(y=result["history"]["loss"], mode="lines", name="loss"))
            if result["history"].get("val_loss"):
                fig.add_trace(go.Scatter(y=result["history"]["val_loss"], mode="lines", name="val_loss"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

    with tab_deploy:
        if result is None or result["domain"] != domain:
            st.caption("Train a model first.")
        else:
            st.markdown("##### Validation")
            all_ok = True
            for check, ok in result["val_results"].items():
                if check.endswith("_ok"):
                    all_ok = all_ok and ok
                    (st.success if ok else st.error)(f"{check.replace('_ok', '')}: {'OK' if ok else 'FAILED'}")
            if not all_ok:
                st.warning("Validation didn't fully pass -- promoting anyway will still replace whatever is currently in production for this (domain, tier, block).")
            if st.button("Promote to registry", key="tl_promote_button"):
                metrics = dict(result["base"]["metrics"])
                metrics.update({k: v for k, v in result["val_results"].items() if not k.endswith("_ok")})
                metrics["fine_tuned_from"] = str(result["base"]["run_dir"])
                # Fine-tuned runs live as a SIBLING of the base config's own run directory --
                # configs/classifiers/pc_server_tl/<date>_runNN/, next to
                # configs/classifiers/pc_server/<date>_runNN/ -- so they're visually distinguished
                # from from-scratch runs of the same config without needing a real
                # "pc_server_tl.yaml" file to exist (_next_run_dir only reads the path's own
                # parent/stem, not the file itself).
                base_config_dir = result["base"]["run_dir"].parent
                tl_stem_path = base_config_dir.parent / f"{base_config_dir.name}_tl.yaml"
                run_dir = _next_run_dir(tl_stem_path)
                _save_finetuned_artifacts(run_dir, result["base"]["run_dir"] / "config.yaml", result["pipeline"], metrics)
                entry = promote(run_dir)
                st.success(f"Promoted {run_dir.name} -> {entry.domain}/{entry.tier}/{entry.block}")
                st.download_button("Download weights", (run_dir / "model.weights.h5").read_bytes(), file_name=f"{domain}_{block}_finetuned.weights.h5", key="tl_download")
