"""Transfer Learning workbench (guide Sec. 3.4, figures/gui/concept_v2/03_transfer_learning_workbench.png).

A 4-step wizard over driveflow.ai.transfer.workflow -- the UI only collects inputs and renders
results; every decision is made in the backend:

1 Data     validate external data (training contract), merge whole acquisitions, window, split
           train/val/test by acquisition -> dataset hash, split manifest, integrity checks.
2 Setup    strategy (Feature extractor / Discriminative LR / simplified Adapter -- not LoRA),
           epochs, seed, thresholds (proposed defaults, flagged as requiring confirmation).
3 Validate train a CLONE of the base model; base and candidate are evaluated ONCE on the same
           independent test set; task-specific checks with PASS/FAIL/INCOMPLETE/N/A.
4 Promote  PromotionGate decision (fail-closed, human approval required). "Save as candidate"
           writes artifacts without touching the registry; "Promote" is disabled unless the gate
           is APPROVED and registry.promote() re-checks the persisted evidence anyway.

Side effects happen only inside explicit button handlers; navigating between steps or any rerun
never trains, saves or promotes anything.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai.registry import DEFAULT_REGISTRY_PATH, RegistryError
from driveflow.ai.transfer import workflow as wf
from driveflow.ai.transfer.data_merger import InsufficientDataError
from driveflow.ai.transfer.dataset_validation import validate_training_dataset
from driveflow.ai.transfer.loader import ModelLoader
from driveflow.ai.transfer.pipeline import FineTuneStrategy
from driveflow.ai.transfer.validator import DEFAULT_THRESHOLDS
from driveflow.datagen import Scenario, run_scenario
from driveflow.models.common.dataset import filter_diagnosis_domain
from driveflow.models.common.splits import InsufficientGroupsError
from driveflow.viz.adapters.session import record_run
from driveflow.viz.components.metric_cards import check_list, kpi_row
from driveflow.viz.components.model_comparison import model_comparison
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, status_badge_html
from driveflow.viz.components.promotion_gate import promotion_gate
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure
from driveflow.viz.theme.tokens import SERIES

DOMAIN_LABELS = {"dc_motor": "DC motor diagnosis (classifier)", "vsc_dpc": "VSC / DPC forecasting (regressor)"}
BLOCK_BY_DOMAIN = {"dc_motor": "classifier", "vsc_dpc": "regressor"}
STRATEGY_LABELS = {
    FineTuneStrategy.FEATURE_EXTRACTOR: "Feature extractor — freeze the first N weighted layers",
    FineTuneStrategy.DISCRIMINATIVE_LR: "Discriminative LR — larger learning rate on the output layer",
    FineTuneStrategy.ADAPTER: "Simplified adapter — train only the last N layers (NOT LoRA)",
}
STEPS = [("Data", "Source & metadata"), ("Setup", "Strategy & thresholds"), ("Validate", "Independent metrics"), ("Promote", "Gate & approval")]
MAX_UPLOAD_MB = 50
_RESET_KEYS = ("tl_dataset", "tl_result", "tl_candidate_dir", "tl_promoted", "tl_external")


@st.cache_data(show_spinner=False)
def _base_metadata(domain: str, block: str, registry_key: str) -> dict | None:
    """Metadata only (config + metrics + run dir) -- immutable, safe to cache. The keras model is
    loaded fresh, per training click, and then cloned (AUDIT AI-04)."""
    try:
        meta = ModelLoader(DEFAULT_REGISTRY_PATH, domain, block).get_metadata()
    except RegistryError:
        return None
    return {"run_dir": str(meta["run_dir"]), "config": meta["config"], "metrics": meta["metrics"]}


def _steps_html(active: int, done: set) -> str:
    items = []
    for i, (title, sub) in enumerate(STEPS, start=1):
        cls = "spade-step" + (" is-active" if i == active else "") + (" is-done" if i in done else "")
        items.append(f'<div class="{cls}"><div class="spade-step-num">{"✓" if i in done and i != active else i}</div><div><div class="spade-step-title">{title}</div><div class="spade-step-sub">{sub}</div></div></div>')
    return '<div class="spade-root spade-steps">' + "".join(items) + "</div>"


def _simulate(domain: str, n: int, duration: float, faults: list) -> pd.DataFrame:
    records = []
    for seed, fault in enumerate((faults * n)[:n]):
        if domain == "dc_motor":
            sc = Scenario(scenario_id=f"tl_sample_{seed}", fault_type=None if fault == "healthy" else fault, duration_s=duration, seed=seed)
        else:
            sc = Scenario(scenario_id=f"tl_sample_{seed}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=duration, seed=seed)
        records.extend(run_scenario(sc))
    df = pd.DataFrame.from_records(records)
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
            faults = st.multiselect("Fault conditions", ["healthy", "outer_race", "inner_race"], default=["healthy", "outer_race", "inner_race"], key="tl_faults") if domain == "dc_motor" else ["healthy"]
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
                    sim_df = _simulate(domain, n, duration, faults or ["healthy"]) if mix > 0 or not ext_ok else None
                    cfg = base["config"]
                    ds = wf.prepare_dataset(domain, block, metrics, cfg["input_window"], cfg.get("horizon"), sim_df, report.df if ext_ok else None, report.summary() if ext_ok else None, mix_ratio=mix if ext_ok else 1.0, seed=int(seed), val_frac=float(val), test_frac=float(test))
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
            f'<div class="spade-flow-node"><b>Base model (read-only)</b><span>{base["run_dir"].split("configs/")[-1]}</span></div></div>'
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
            st.markdown("**Fine-tuning strategy**")
            strategy = st.radio("Strategy", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get, key="tl_strategy")
            cfg = {"strategy": strategy}
            if strategy is FineTuneStrategy.FEATURE_EXTRACTOR:
                cfg["frozen_layers"] = st.slider("Frozen layers", 0, 6, 2, key="tl_frozen_layers")
                cfg["learning_rate"] = st.select_slider("Learning rate", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="tl_lr_fe")
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
                base_loaded = ModelLoader(DEFAULT_REGISTRY_PATH, domain, block).load_latest()
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
    "tl_n_scenarios", "tl_duration", "tl_faults", "tl_mix_ratio", "tl_val_frac", "tl_test_frac", "tl_seed",
    "tl_strategy", "tl_frozen_layers", "tl_lr_fe", "tl_base_lr", "tl_head_mult", "tl_adapter_layers", "tl_lr_adapter",
    "tl_epochs", "tl_batch_size", "tl_thr_f1_drop", "tl_thr_min_recall", "tl_thr_rmse_inc",
    "tl_model_card", "tl_approver", "tl_approve",
)


def render() -> None:
    page_header(
        "Intelligence / Transfer learning",
        "Adapt a model safely",
        "Guided fine-tuning of a promoted PC-tier model with explicit dataset checks, an independent test set and a fail-closed promotion gate.",
        [badge_html("FAIL-CLOSED GATE", "critical", "■"), badge_html("HUMAN APPROVAL REQUIRED", "warning", "▲")],
    )
    domain = st.selectbox("Domain / base model", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="tl_domain")
    if st.session_state.get("_tl_domain_prev") != domain:
        for k in _RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state["_tl_domain_prev"] = domain
    block = BLOCK_BY_DOMAIN[domain]
    base = _base_metadata(domain, block, str(DEFAULT_REGISTRY_PATH.stat().st_mtime) if DEFAULT_REGISTRY_PATH.exists() else "")
    if base is None:
        st.info(f"No PC-tier {block} is promoted for {domain} — nothing to fine-tune.", icon=":material/info:")
        return
    m = base["metrics"]
    head = f"macro-F1 {m['test_macro_f1']:.3f}" if m.get("test_macro_f1") is not None else f"RMSE {m.get('test_rmse', float('nan')):.4f} (normalized)"
    st.caption(f"Base model: **{base['run_dir'].split('configs/')[-1]}** — {head}, channels {', '.join(m['channels'])}" + (f", classes {', '.join(m['classes'])}" if m.get("classes") else "") + ". Loaded read-only; fine-tuning trains a copy.")

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
