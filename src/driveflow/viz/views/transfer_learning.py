"""Transfer Learning workbench (guide Sec. 3.4, figures/gui/concept_v2/03_transfer_learning_workbench.png).

A 4-step wizard over driveflow.ai.transfer.workflow -- the UI only collects inputs and renders
results; every decision is made in the backend:

1 Data     validate external data (training contract), merge whole acquisitions, window, split
           train/val/test by acquisition -> dataset hash, split manifest, integrity checks.
2 Setup    transfer mode: FULL transfer learning (every layer: one learning rate, or a larger rate
           on the output layer) or FEATURE transfer learning (reuse the base model's features:
           freeze the first N layers, train only the last N -- not LoRA --, or a new output head,
           which also allows a new set of classes); epochs, seed, thresholds (proposed defaults,
           flagged as requiring confirmation).
3 Validate train a CLONE of the base model; base and candidate are evaluated ONCE on the same
           independent test set; task-specific checks with PASS/FAIL/INCOMPLETE/N/A.
4 Promote  PromotionGate decision (fail-closed, human approval required). "Save as candidate"
           writes artifacts without touching the registry; "Promote" is disabled unless the gate
           is APPROVED and registry.promote() re-checks the persisted evidence anyway.

The base model is the promoted one, any other run on this machine (custom-trained in AI Diagnostics,
an earlier transfer-learning candidate, a historical run) or a third-party Keras model imported here
(driveflow.ai.transfer.external; the supported formats are listed on the page).

Side effects happen only inside explicit button handlers; navigating between steps or any rerun
never trains, saves or promotes anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai import custom_training as ct
from driveflow.ai.registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, RegistryError, resolve
from driveflow.ai.transfer import external as ext
from driveflow.ai.transfer import workflow as wf
from driveflow.ai.transfer.data_merger import InsufficientDataError
from driveflow.ai.transfer.dataset_validation import validate_training_dataset
from driveflow.ai.transfer.loader import ModelLoader
from driveflow.ai.transfer.pipeline import FULL_TRANSFER, FineTuneStrategy
from driveflow.ai.transfer.validator import DEFAULT_THRESHOLDS
from driveflow.datagen import Scenario, run_scenario
from driveflow.models.common.dataset import filter_diagnosis_domain
from driveflow.models.common.splits import InsufficientGroupsError
from driveflow.viz.adapters.session import record_run
from driveflow.viz.adapters.simulation_adapter import resolve_fault
from driveflow.viz.components.presets import BUILTIN_FAULT_TYPES, load_custom_faults
from driveflow.viz.components.metric_cards import check_list, kpi_row
from driveflow.viz.components.model_comparison import model_comparison
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, status_badge_html
from driveflow.viz.components.promotion_gate import promotion_gate
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure
from driveflow.viz.theme.tokens import SERIES

DOMAIN_LABELS = {"dc_motor": "DC motor diagnosis (classifier)", "vsc_dpc": "VSC / DPC (forecaster or classifier)"}
BLOCK_BY_DOMAIN = {"dc_motor": "classifier", "vsc_dpc": "regressor"}
BLOCK_LABELS = {"regressor": "Forecaster (regressor)", "classifier": "Classifier (capacitor degradation)"}
STRATEGY_LABELS = {
    FineTuneStrategy.FEATURE_EXTRACTOR: "Feature transfer · freeze the first N weighted layers, retrain the rest",
    FineTuneStrategy.ADAPTER: "Feature transfer · simplified adapter: train only the last N layers (NOT LoRA)",
    FineTuneStrategy.NEW_HEAD: "Feature transfer · new output layer on the frozen features (new classes allowed)",
    FineTuneStrategy.FULL_FINETUNE: "Full transfer · fine-tune every layer with one learning rate",
    FineTuneStrategy.DISCRIMINATIVE_LR: "Full transfer · every layer, larger learning rate on the output layer",
}
BASE_SOURCES = ["Promoted model (registry)", "Another model on this machine", "Import a model file"]
STEPS = [("Data", "Source & metadata"), ("Setup", "Strategy & thresholds"), ("Validate", "Independent metrics"), ("Promote", "Gate & approval")]
MAX_UPLOAD_MB = 50
_RESET_KEYS = ("tl_dataset", "tl_result", "tl_candidate_dir", "tl_promoted", "tl_external")


@st.cache_data(show_spinner=False)
def _base_metadata(domain: str, block: str, registry_key: str, run_dir: str | None = None) -> dict | None:
    """Metadata only (config + metrics + run dir) -- immutable, safe to cache. The keras model is
    loaded fresh, per training click, and then cloned (AUDIT AI-04). run_dir None = promoted."""
    try:
        meta = ModelLoader(DEFAULT_REGISTRY_PATH, domain, block).get_metadata(run_dir)
    except (RegistryError, OSError):
        return None
    if meta["metrics"].get("domain") not in (None, domain):
        return None
    return {"run_dir": str(meta["run_dir"]), "config": meta["config"], "metrics": meta["metrics"]}


def _load_base(domain: str, block: str, run_dir: str | None) -> dict:
    loader = ModelLoader(DEFAULT_REGISTRY_PATH, domain, block)
    return loader.load_latest() if run_dir is None else loader.load_by_generation(run_dir)


def _rel(path) -> str:
    p = str(path)
    return p.split("configs/", 1)[-1] if "configs/" in p else p


def _import_form(domain: str, block: str) -> None:
    st.dataframe(pd.DataFrame(ext.SUPPORTED_FORMATS), hide_index=True, width="stretch")
    if block != "classifier":
        st.info("Importing is available for classifiers. Forecasters must come from a SPADE run.", icon=":material/info:")
        return
    st.caption("The model must take one input of shape (window, channels) and end in a softmax Dense layer. Declare what its inputs and outputs mean: "
               "SPADE feeds it windows sampled at 10 kHz, z-scored per window and channel (its original preprocessing is unknown to SPADE).")
    up = st.file_uploader("Model file (.keras / .h5)", type=["keras", "h5", "hdf5"], key="tl_import_file", help=f"At most {ext.MAX_MODEL_MB} MB. Loaded in Keras safe mode.")
    allowed = list(ct.DC_CHANNELS) + ["current_s", "current_t"] if domain == "dc_motor" else list(ct.VSC_CHANNELS)
    chans = st.multiselect("Input channels, in the model's input order", allowed, key="tl_import_channels")
    classes = st.text_input("Output classes, in the model's output order (comma-separated)", key="tl_import_classes",
                            placeholder="normal, outer_race, inner_race", help="Use 'normal' for the healthy class so SPADE's simulated labels match.")
    if st.button("Import model", key="tl_import_button", icon=":material/upload:", disabled=up is None):
        try:
            with st.spinner("Checking and loading the model…"):
                run_dir = ext.import_model(up.getvalue(), up.name, domain, chans, [c for c in classes.split(",") if c.strip()])
            st.session_state["_tl_base_next"] = str(run_dir)
            st.rerun()
        except ext.ExternalModelError as exc:
            st.error(f"Not imported: {exc}", icon=":material/error:")


def _base_choice(domain: str, block: str) -> str | None | bool:
    """None = promoted, a run directory string, or False when nothing is selected yet."""
    if "_tl_base_next" in st.session_state:  # set by Import, applied before the widgets exist
        st.session_state["tl_base_source"] = BASE_SOURCES[1]
        st.session_state["tl_base_run"] = st.session_state.pop("_tl_base_next")
    source = st.radio("Base model", BASE_SOURCES, horizontal=True, key="tl_base_source",
                      help="Start from the promoted model, from any other model on this machine (e.g. one trained in AI Diagnostics → Train a model), or import a third-party Keras model.")
    if source == BASE_SOURCES[0]:
        return None
    if source == BASE_SOURCES[2]:
        with st.container(border=True):
            st.markdown("**Import a model file** — supported formats")
            _import_form(domain, block)
        return False
    runs = ct.list_model_runs(domain, "classifier" if block == "classifier" else "forecaster")
    try:
        promoted = resolve(domain, "pc", block).resolve()
    except RegistryError:
        promoted = None
    options = [str(r["run_dir"]) for r in runs if r["run_dir"].resolve() != promoted]
    if not options:
        st.info("No other model of this kind on this machine yet. Train one in AI Diagnostics → Train a model, or import one.", icon=":material/info:")
        return False
    labels = {str(r["run_dir"]): f"{r['label']} · {r['origin']}" + (f" · {len(r['classes'])} classes" if r.get("classes") else "") for r in runs}
    if st.session_state.get("tl_base_run") not in options:
        st.session_state.pop("tl_base_run", None)
    return st.selectbox("Model", options, format_func=lambda o: labels.get(o, o), key="tl_base_run")


def _steps_html(active: int, done: set) -> str:
    items = []
    for i, (title, sub) in enumerate(STEPS, start=1):
        cls = "spade-step" + (" is-active" if i == active else "") + (" is-done" if i in done else "")
        items.append(f'<div class="{cls}"><div class="spade-step-num">{"✓" if i in done and i != active else i}</div><div><div class="spade-step-title">{title}</div><div class="spade-step-sub">{sub}</div></div></div>')
    return '<div class="spade-root spade-steps">' + "".join(items) + "</div>"


def _simulate(domain: str, n: int, duration: float, faults: list, opts: dict | None = None) -> pd.DataFrame:
    """n acquisitions cycling over `faults` (GUI labels). opts: speed range and max noise (dc_motor),
    capacitor scale (vsc_dpc classifier). A user-defined fault keeps its own name as label."""
    opts = opts or {}
    rng = np.random.default_rng(int(opts.get("seed", 0)))
    custom = load_custom_faults()
    frames = []
    for seed, fault in enumerate((faults * n)[:n]):
        if domain == "dc_motor":
            fault_type, override, extras = resolve_fault(fault, custom)
            lo, hi = opts.get("speed_range", (150.0, 150.0))
            noise = float(opts.get("noise_max_pct", 0.0))
            kw = {"omega_ref_rad_s": float(rng.uniform(lo, hi)), "electrical_noise_pct": float(rng.uniform(0, noise)), "mechanical_noise_pct": float(rng.uniform(0, noise))}
            if fault_type is not None and fault_type not in ct.DC_BUILTIN_FAULTS:
                kw.update(fault_order_override=override, extra_faults=extras or None, mechanical_severity=0.02)
            sc = Scenario(scenario_id=f"tl_sample_{seed}", fault_type=fault_type, duration_s=duration, seed=seed, **kw)
        elif fault == ct.VSC_DEGRADED:
            sc = Scenario(scenario_id=f"tl_sample_{seed}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=duration, seed=seed,
                          vsc_capacitor_scale=float(opts.get("capacitor_scale", 0.6)))
        else:
            sc = Scenario(scenario_id=f"tl_sample_{seed}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=duration, seed=seed)
        df = pd.DataFrame.from_records(run_scenario(sc))
        if domain == "dc_motor" and fault not in ("healthy", ct.HEALTHY):
            df["label"] = fault
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return filter_diagnosis_domain(df) if domain == "dc_motor" else df


# ----------------------------------------------------------------------------------- step 1
def _step_data(domain: str, block: str, base: dict) -> None:
    left, right = st.columns([1, 1.6], gap="medium")
    metrics = base["metrics"]
    task = wf.TASK_BY_BLOCK[block]
    with left:
        with st.container(border=True):
            st.markdown("**Simulated data**")
            n = st.slider("Simulated scenarios (acquisitions)", 3, 40, 12, key="tl_n_scenarios", help="One acquisition per scenario. Splits are by acquisition: classification needs ≥ 3 per class.")
            duration = st.slider("Duration per scenario (s)", 0.1, 5.0, 1.0, 0.05, key="tl_duration")
            if domain == "dc_motor" and duration < 1.0:
                st.warning("Runs shorter than ~1 s are dominated by the start-up transient (speed settles in ~0.45 s). The promoted dc_motor classifier was trained on 5 s runs (experiments/generate_diagnosis_dataset.py) — short runs are out of its distribution.", icon=":material/warning:")
            opts = {"seed": 0}
            base_classes = list(metrics.get("classes") or [])
            if block == "classifier" and domain == "dc_motor":
                options = BUILTIN_FAULT_TYPES + sorted(load_custom_faults())
                default = [("healthy" if c == ct.HEALTHY else c) for c in base_classes if ("healthy" if c == ct.HEALTHY else c) in options] or ["healthy", "outer_race", "inner_race"]
                faults = st.multiselect("Fault conditions (classes)", options, default=default[:3], key="tl_faults",
                                        help="Classes outside the base model's vocabulary are allowed: only feature transfer with a new output head can learn them.")
                opts["speed_range"] = st.slider("Speed setpoint range ω_ref (rad/s)", 10.0, 350.0, (150.0, 150.0), key="tl_speed_range", help="Each acquisition draws its setpoint uniformly in this range.")
                opts["noise_max_pct"] = st.slider("Max sensor noise (%)", 0.0, 100.0, 0.0, key="tl_noise")
            elif block == "classifier":
                faults = st.multiselect("Classes", [ct.HEALTHY, ct.VSC_DEGRADED], default=[ct.HEALTHY, ct.VSC_DEGRADED], key="tl_faults")
                opts["capacitor_scale"] = st.slider("Degraded capacitance (fraction of nominal)", 0.05, 0.95, 0.6, key="tl_cap_scale")
            else:
                faults = ["healthy"]
            targets = [ct.HEALTHY if f == "healthy" else f for f in faults] if block == "classifier" else None
            new = sorted(set(targets or []) - set(base_classes)) if block == "classifier" else []
            if new:
                st.info(f"New class(es) for this base model: {', '.join(new)}. Only **feature transfer with a new output layer** can learn them (step 2); the base model cannot be compared on them.", icon=":material/info:")
        with st.container(border=True):
            st.markdown("**External training data (optional)**")
            uploaded = st.file_uploader("CSV / JSON with channels + label + source_file + timestamp_s", type=["csv", "json"], key="tl_upload", help=f"At most {MAX_UPLOAD_MB} MB. Must be sampled at {wf.SIM_FS_HZ:,.0f} Hz (no resampling is implemented).")
            report = None
            if uploaded is not None:
                raw = uploaded.getvalue()
                if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"File is {len(raw) / 1e6:.1f} MB — the limit is {MAX_UPLOAD_MB} MB.")
                else:
                    try:
                        df_raw = pd.read_json(uploaded) if uploaded.name.lower().endswith(".json") else pd.read_csv(uploaded)
                        report = validate_training_dataset(df_raw, domain, task, metrics["channels"], metrics.get("classes"), expected_fs_hz=wf.SIM_FS_HZ)
                    except Exception as e:
                        st.error(f"Could not parse '{uploaded.name}': {e}")
                    if report is not None:
                        for level, text in report.messages:
                            getattr(st, level)(text)
            ext_ok = report is not None and report.ok
            mix = st.slider("Mix ratio — fraction of training windows from simulation", 0.0, 1.0, 0.7 if ext_ok else 1.0, key="tl_mix_ratio", disabled=not ext_ok, help="Measured in non-overlapping windows; whole acquisitions are selected, so the realized ratio is reported after preparation.")
        with st.container(border=True):
            st.markdown("**Split (by acquisition)**")
            c1, c2, c3 = st.columns(3)
            val = c1.number_input("Validation", 0.1, 0.4, 0.2, 0.05, key="tl_val_frac")
            test = c2.number_input("Test", 0.1, 0.4, 0.2, 0.05, key="tl_test_frac")
            seed = c3.number_input("Seed", value=0, step=1, key="tl_seed")
        if st.button("Prepare dataset", type="primary", key="tl_prepare_button", width="stretch", icon=":material/dataset:"):
            for k in _RESET_KEYS:
                st.session_state.pop(k, None)
            try:
                with st.spinner("Simulating and windowing…"):
                    sim_df = _simulate(domain, n, duration, faults or ["healthy"], {**opts, "seed": int(seed)}) if mix > 0 or not ext_ok else None
                    cfg = base["config"]
                    ds = wf.prepare_dataset(domain, block, metrics, cfg["input_window"], cfg.get("horizon"), sim_df, report.df if ext_ok else None, report.summary() if ext_ok else None, mix_ratio=mix if ext_ok else 1.0, seed=int(seed), val_frac=float(val), test_frac=float(test),
                                            target_classes=targets)
                st.session_state["tl_dataset"] = ds
                st.session_state["tl_external"] = report.summary() if ext_ok else None
                record_run("tl_data", {"domain": domain, "n": n, "duration": duration, "faults": faults, "mix": mix, "seed": int(seed)}, {"windows": ds.summary["n_windows"], "dataset": ds.sha256[:12]})
                st.session_state["_tl_step_next"] = "2 · Setup"
                st.rerun()
            except (InsufficientGroupsError, InsufficientDataError, ValueError) as exc:
                st.error(f"Dataset not prepared: {exc}")
    with right:
        _dataset_panel(base, st.session_state.get("tl_dataset"))


def _dataset_panel(base: dict, ds) -> None:
    with st.container(border=True):
        st.markdown("##### Dataset & model lineage")
        cfg = base["config"]
        win = f"{cfg['input_window']} samples" + (f" + {cfg['horizon']} horizon" if cfg.get("horizon") else "")
        src = "—" if ds is None else ("simulated + external" if (ds.summary.get("external")) else "simulated only")
        st.html(
            '<div class="spade-root spade-flow">'
            f'<div class="spade-flow-node"><b>Source data</b><span>{src}</span></div><div class="spade-flow-arrow">→</div>'
            f'<div class="spade-flow-node"><b>Preprocessing</b><span>per-window z-score · {win} · gap-aware windows</span></div><div class="spade-flow-arrow">→</div>'
            f'<div class="spade-flow-node"><b>Base model (read-only)</b><span>{_rel(base["run_dir"])}</span></div></div>'
        )
    if ds is None:
        st.info("Prepare a dataset to see its split, hash and integrity checks.", icon=":material/info:")
        return
    sp = ds.split
    n = sp["n_windows"]
    total = max(1, sum(n.values()))
    with st.container(border=True):
        st.markdown("##### Train / validation / test split")
        st.html(
            '<div class="spade-root"><div class="spade-split">'
            + "".join(f'<div style="width:{100 * n[k] / total:.1f}%;background:{SERIES[c]}" title="{k}"></div>' for k, c in (("train", "training"), ("val", "validation"), ("test", "test")))
            + "</div></div>"
        )
        kpi_row([
            {"label": "Training", "value": f"{n['train']}", "unit": "windows", "sub": f"{sp['n_groups']['train']} acquisition(s)", "tone": SERIES["training"]},
            {"label": "Validation", "value": f"{n['val']}", "unit": "windows", "sub": f"{sp['n_groups']['val']} acquisition(s)", "tone": SERIES["validation"]},
            {"label": "Test (independent)", "value": f"{n['test']}", "unit": "windows", "sub": f"{sp['n_groups']['test']} acquisition(s)", "tone": SERIES["test"]},
        ])
        if sp.get("class_counts"):
            st.dataframe(pd.DataFrame(sp["class_counts"]).fillna(0).astype(int), width="stretch")
        mix = ds.summary["mix"]
        provenance_table({
            "Split method": f"{sp['method']} · seed {sp['seed']}", "Dataset SHA-256": ds.sha256, "Acquisitions": ds.summary["n_acquisitions"],
            "Mix unit": mix.get("mix_unit"), "Target / realized mix": f"{mix.get('target_mix_ratio')} / {mix.get('realized_mix_ratio') if mix.get('realized_mix_ratio') is None else round(mix['realized_mix_ratio'], 3)}",
        })
    with st.container(border=True):
        st.markdown("##### Data integrity checks")
        check_list([(name.replace("_", " ").capitalize(), status.lower(), detail) for name, status, detail in ds.integrity])


# ----------------------------------------------------------------------------------- step 2
def _step_setup(domain: str, block: str, base: dict) -> None:
    ds = st.session_state.get("tl_dataset")
    if ds is None:
        st.info("Prepare a dataset in step 1 first.", icon=":material/info:")
        return
    task = wf.TASK_BY_BLOCK[block]
    left, right = st.columns([1, 1.6], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("**Transfer mode and strategy**")
            st.caption("**Full transfer learning** fine-tunes every layer of the base model on your data. **Feature transfer learning** keeps the base model's learned "
                       "features (most layers frozen) and trains only part of it; with a new output layer it can also learn classes the base model never had.")
            choices = list(STRATEGY_LABELS)
            if task != "classification":
                choices.remove(FineTuneStrategy.NEW_HEAD)
            if ds.new_vocabulary:
                choices = [FineTuneStrategy.NEW_HEAD]
                st.info(f"The dataset's classes ({', '.join(ds.classes)}) differ from the base model's, so only a new output layer can learn them.", icon=":material/info:")
            if st.session_state.get("tl_strategy") not in choices:
                st.session_state.pop("tl_strategy", None)
            strategy = st.radio("Strategy", choices, format_func=STRATEGY_LABELS.get, key="tl_strategy")
            st.caption(f"Mode: **{'full' if strategy in FULL_TRANSFER else 'feature'} transfer learning**.")
            cfg = {"strategy": strategy}
            if strategy is FineTuneStrategy.FEATURE_EXTRACTOR:
                cfg["frozen_layers"] = st.slider("Frozen layers", 0, 6, 2, key="tl_frozen_layers")
                cfg["learning_rate"] = st.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="tl_lr_fe")
            elif strategy is FineTuneStrategy.FULL_FINETUNE:
                cfg["learning_rate"] = st.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="tl_lr_full",
                                                        help="Every layer is trained; a small rate keeps the base model's features from being overwritten early.")
            elif strategy is FineTuneStrategy.NEW_HEAD:
                cfg["unfreeze_last"] = st.slider("Also train the last N feature layers", 0, 3, 0, key="tl_unfreeze_last")
                cfg["learning_rate"] = st.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-3, key="tl_lr_head")
                st.caption("The base model's output layer is replaced by a new one with one unit per class of the dataset; the layers before it are reused as a frozen feature extractor.")
            elif strategy is FineTuneStrategy.DISCRIMINATIVE_LR:
                cfg["base_lr"] = st.select_slider("Base learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="tl_base_lr")
                cfg["head_multiplier"] = st.slider("Output-layer LR multiplier", 1.0, 20.0, 10.0, key="tl_head_mult")
            else:
                cfg["adapter_layers"] = st.slider("Trainable output layers", 1, 3, 1, key="tl_adapter_layers")
                cfg["learning_rate"] = st.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-3, key="tl_lr_adapter")
                st.caption("Freezes every weighted layer except the last N. This is not LoRA: no low-rank adapter modules are inserted.")
            cfg["epochs"] = st.slider("Epochs", 1, 50, 15, key="tl_epochs")
            cfg["batch_size"] = st.select_slider("Batch size", [4, 8, 16, 32], value=8, key="tl_batch_size")
        with st.container(border=True):
            st.markdown("**Acceptance thresholds**")
            st.caption("Proposed defaults — REQUIRE RESEARCHER CONFIRMATION. They are compared against the base model on the same independent test set.")
            if task == "classification":
                thr = {"max_macro_f1_drop": st.number_input("Max macro-F1 drop vs. base", 0.0, 0.5, DEFAULT_THRESHOLDS["classification"]["max_macro_f1_drop"], 0.01, key="tl_thr_f1_drop")}
                min_recall = st.number_input("Min per-class recall (0 = not required)", 0.0, 1.0, 0.0, 0.05, key="tl_thr_min_recall")
                thr["min_class_recall"] = min_recall or None
                if ds.new_vocabulary:
                    thr["min_macro_f1"] = st.number_input("Min macro-F1 on the test set (required: the base model is not comparable)", 0.0, 1.0, value=None, step=0.05,
                                                          key="tl_thr_min_f1", placeholder="enter a value", help="No value is proposed. Without one the check stays INCOMPLETE and promotion is blocked.")
            else:
                thr = {"max_rmse_increase": st.number_input("Max relative RMSE increase vs. base", 0.0, 1.0, DEFAULT_THRESHOLDS["regression"]["max_rmse_increase"], 0.01, key="tl_thr_rmse_inc")}
        seed = int(st.session_state.get("tl_seed", 0))
        st.caption(f"Cost estimate: {len(ds.windows['train']['X'])} training windows × {cfg['epochs']} epochs, CPU.")
        clicked = st.button("Train candidate", type="primary", key="tl_train_button", width="stretch", icon=":material/model_training:")
    with right:
        chart = st.empty()
        progress = st.empty()
        if clicked:
            losses = {"loss": [], "val_loss": []}

            def on_epoch_end(epoch, logs):
                losses["loss"].append(float(logs["loss"]))
                if logs.get("val_loss") is not None:
                    losses["val_loss"].append(float(logs["val_loss"]))
                progress.progress((epoch + 1) / cfg["epochs"], text=f"Epoch {epoch + 1}/{cfg['epochs']} — loss {logs['loss']:.4f}")
                chart.plotly_chart(_loss_figure(losses), width="stretch", key=f"tl_live_loss_{epoch}")

            with st.spinner("Loading base model (fresh instance) and training a clone…"):
                base_loaded = _load_base(domain, block, st.session_state.get("_tl_base_run_dir"))
                result = wf.finetune(base_loaded, ds, cfg, seed=seed, thresholds=thr, on_epoch_end=on_epoch_end)
            st.session_state["tl_result"] = result
            st.session_state["tl_base_loaded"] = base_loaded
            st.session_state.pop("tl_candidate_dir", None)
            st.session_state.pop("tl_promoted", None)
            record_run("tl_train", {"domain": domain, "strategy": strategy.value, **{k: v for k, v in cfg.items() if k != "strategy"}, "seed": seed}, {"validation_passed": result.report["passed"], "base_unchanged": result.base_unchanged})
            st.session_state["_tl_step_next"] = "3 · Validate"
            st.rerun()
        elif st.session_state.get("tl_result") is not None:
            chart.plotly_chart(_loss_figure(st.session_state["tl_result"].history), width="stretch", config=PLOTLY_CONFIG, key="tl_loss_setup")
        else:
            st.info("Configure the strategy and click **Train candidate**. Training uses train windows; validation windows only produce the loss curve; the test set is untouched until training ends.", icon=":material/info:")


def _loss_figure(history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.get("loss", []), x=list(range(1, len(history.get("loss", [])) + 1)), mode="lines+markers", name="training loss", line=dict(color=series_color("training"))))
    if history.get("val_loss"):
        fig.add_trace(go.Scatter(y=history["val_loss"], x=list(range(1, len(history["val_loss"]) + 1)), mode="lines+markers", name="validation loss", line=dict(color=series_color("validation"))))
    style_figure(fig, height=300, x_title="Epoch", y_title="Loss")
    return fig


# ----------------------------------------------------------------------------------- step 3
def _step_validate(base: dict) -> None:
    result = st.session_state.get("tl_result")
    if result is None:
        st.info("Train a candidate in step 2 first.", icon=":material/info:")
        return
    rep = result.report
    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("##### Training & validation loss")
            st.plotly_chart(_loss_figure(result.history), width="stretch", config=PLOTLY_CONFIG, key="tl_loss_validate")
            st.caption("No early stopping or checkpoint selection: the final epoch is the candidate, so no decision is taken on validation or test data.")
        with st.container(border=True):
            model_comparison(rep, result.test_evaluation, sorted(base["metrics"].get("classes") or []), key="tl_rmse_horizon")
    with right:
        with st.container(border=True):
            st.markdown("##### Validation checks")
            st.html('<div class="spade-root spade-badges">' + status_badge_html("pass" if rep["passed"] else "fail") + "</div>")
            check_list([(c["name"].replace("_", " "), c["status"].lower(), c["detail"]) for c in rep["checks"]])
            check_list([("base model unchanged", "pass" if result.base_unchanged else "fail", "weights fingerprint identical before/after fine-tuning")])
            st.caption("N/A = the check does not exist for this task (never counted as PASS). INCOMPLETE = applies but could not be decided (blocks promotion).")


# ----------------------------------------------------------------------------------- step 4
def _step_promote(domain: str, block: str) -> None:
    ds, result = st.session_state.get("tl_dataset"), st.session_state.get("tl_result")
    if result is None:
        st.info("Train and validate a candidate first.", icon=":material/info:")
        return
    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("**Model card & approval**")
            notes = st.text_area("Model card — intended use, data, known limitations", key="tl_model_card", placeholder="e.g. adapted to rig B bearings; simulated + 3 external recordings; not validated on real machines")
            approver = st.text_input("Reviewer", key="tl_approver", placeholder="name of the person approving")
            approved = st.checkbox("I reviewed the dataset, the split and the independent test metrics", key="tl_approve")
    human = {"approved": bool(approved), "approved_by": approver.strip() or None, "notes": notes}
    decision = wf.gate_decision(ds, result, human)
    with right:
        with st.container(border=True):
            promotion_gate(decision)
    c1, c2, c3 = st.columns(3)
    saved = st.session_state.get("tl_candidate_dir")
    if c1.button("Save as candidate", key="tl_save_candidate", disabled=saved is not None, width="stretch", icon=":material/save:", help="Writes weights + lineage + gate report under configs/<block>s/<base>_tl/. Does not touch the registry."):
        run_dir = wf.save_candidate(result, st.session_state["tl_base_loaded"], decision, model_card={"notes": notes, "reviewer": approver})
        st.session_state["tl_candidate_dir"] = str(run_dir)
        st.rerun()
    if c2.button("Promote to registry", key="tl_promote_button", type="primary", disabled=not decision.approved or st.session_state.get("tl_promoted") is not None, width="stretch", icon=":material/rocket_launch:"):
        run_dir = saved or str(wf.save_candidate(result, st.session_state["tl_base_loaded"], decision, model_card={"notes": notes, "reviewer": approver}))
        st.session_state["tl_candidate_dir"] = run_dir
        wf.update_gate_report(run_dir, decision)
        try:
            entry = wf.promote_candidate(run_dir, approved_by=approver.strip())
            st.session_state["tl_promoted"] = f"{entry.domain}/{entry.tier}/{entry.block}"
        except RegistryError as exc:
            st.error(f"Promotion refused by the registry: {exc}")
        st.rerun()
    if saved:
        from pathlib import Path

        weights = Path(saved) / "model.weights.h5"
        if weights.exists():
            c3.download_button("Download candidate weights", weights.read_bytes(), file_name=f"{domain}_{block}_candidate.weights.h5", key="tl_download", width="stretch")
        st.success(f"Candidate saved: {saved.split('configs/')[-1]}" + (" — status: promoted" if st.session_state.get("tl_promoted") else " — status: candidate (not active)"))
    if st.session_state.get("tl_promoted"):
        st.success(f"Promoted to {st.session_state['tl_promoted']}. Roll back from the Model Registry page if needed.")


#: Step widgets whose values outlive the step that renders them (see render()).
_PERSISTED_WIDGET_KEYS = (
    "tl_n_scenarios", "tl_duration", "tl_faults", "tl_mix_ratio", "tl_val_frac", "tl_test_frac", "tl_seed", "tl_speed_range", "tl_noise", "tl_cap_scale",
    "tl_strategy", "tl_frozen_layers", "tl_lr_fe", "tl_base_lr", "tl_head_mult", "tl_adapter_layers", "tl_lr_adapter", "tl_lr_full", "tl_unfreeze_last", "tl_lr_head",
    "tl_epochs", "tl_batch_size", "tl_thr_f1_drop", "tl_thr_min_recall", "tl_thr_rmse_inc", "tl_thr_min_f1",
    "tl_model_card", "tl_approver", "tl_approve",
)


def render() -> None:
    page_header(
        "Intelligence / Transfer learning",
        "Adapt a model safely",
        "Guided fine-tuning of a promoted PC-tier model with explicit dataset checks, an independent test set and a fail-closed promotion gate.",
        [badge_html("FAIL-CLOSED GATE", "critical", "■"), badge_html("HUMAN APPROVAL REQUIRED", "warning", "▲")],
    )
    c1, c2 = st.columns([1, 1])
    domain = c1.selectbox("Domain", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="tl_domain")
    block = BLOCK_BY_DOMAIN[domain] if domain == "dc_motor" else c2.radio("Model kind", ["regressor", "classifier"], format_func=BLOCK_LABELS.get, horizontal=True, key="tl_block")
    choice = _base_choice(domain, block)
    if choice is False:
        return
    if st.session_state.get("_tl_domain_prev") != (domain, block, choice):
        for k in _RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state["_tl_domain_prev"] = (domain, block, choice)
    st.session_state["_tl_base_run_dir"] = choice
    base = _base_metadata(domain, block, str(DEFAULT_REGISTRY_PATH.stat().st_mtime) if DEFAULT_REGISTRY_PATH.exists() else "", choice)
    if base is None:
        st.info(f"No PC-tier {block} is promoted for {domain} — choose another model on this machine, or import one." if choice is None else "The selected model could not be read.",
                icon=":material/info:")
        return
    m = base["metrics"]
    head = (f"macro-F1 {m['test_macro_f1']:.3f}" if m.get("test_macro_f1") is not None else f"RMSE {m['test_rmse']:.4f} (normalized)" if m.get("test_rmse") is not None
            else "not evaluated in SPADE")
    st.caption(f"Base model: **{_rel(base['run_dir'])}** — {head}, channels {', '.join(m['channels'])}" + (f", classes {', '.join(m['classes'])}" if m.get("classes") else "")
               + ". Loaded read-only; fine-tuning trains a copy." + (" **Imported third-party model.**" if m.get("origin") == "EXTERNAL" else ""))

    # Each step renders only its own widgets, and Streamlit drops the state of a widget that is not
    # rendered in a run. Re-assigning the keys that exist keeps a step's settings when the user moves
    # to another step and back (the pattern Streamlit documents for this). It also keeps those keys
    # valid for Streamlit <= 1.61's AppTest, which otherwise raised a KeyError for a step-2 widget
    # after jumping to step 4 (docs/audit/install/macos_intel_emulated_on_linux.txt; TODO P2).
    for key in _PERSISTED_WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    done = {i for i, k in ((1, "tl_dataset"), (2, "tl_result"), (3, "tl_result"), (4, "tl_promoted")) if st.session_state.get(k) is not None}
    labels = [f"{i} · {t}" for i, (t, _) in enumerate(STEPS, start=1)]
    # A step change requested by a button handler is applied here, BEFORE the radio exists in
    # this run (Streamlit forbids writing a widget's key after it has been instantiated).
    if "_tl_step_next" in st.session_state:
        st.session_state["tl_step"] = st.session_state.pop("_tl_step_next")
    step = st.radio("Step", labels, horizontal=True, key="tl_step", label_visibility="collapsed")
    idx = labels.index(step) + 1
    st.html(_steps_html(idx, done))
    if idx == 1:
        _step_data(domain, block, base)
    elif idx == 2:
        _step_setup(domain, block, base)
    elif idx == 3:
        _step_validate(base)
    else:
        _step_promote(domain, block)
    st.caption("Gate and registry are enforced in the backend (driveflow.ai.promotion / driveflow.ai.registry); this page cannot bypass them.")
