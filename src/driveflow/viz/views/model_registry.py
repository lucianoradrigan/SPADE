"""Model Registry page (guide Sec. 3.4, figures/gui/concept_v2/05_model_registry.png).

Lists only run directories that exist on disk; statuses come from evidence (see
adapters/registry_adapter.py). Artifacts are offered for download with size + SHA-256, and only
the files that actually exist -- no invented .onnx/.pt formats. Rollback uses the registry's audit
history and requires an explicit confirmation."""


import pandas as pd
import streamlit as st

from driveflow.ai.c2000_export import kit_zip
from driveflow.ai.registry import RegistryError, registry_history, rollback
from driveflow.viz.adapters.registry_adapter import c2000_summary, class_coverage_warnings, list_runs
from driveflow.viz.components.artifact_table import artifact_table
from driveflow.viz.components.metric_cards import check_list, kpi_row
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table
from driveflow.ai.audit_findings import SYNTHETIC_LABEL, is_synthetic_class

_STATUS_TONE = {"Promoted": "healthy", "Promoted (override)": "warning", "Candidate": "warning", "Superseded": "unknown", "Trained (not promoted)": "unknown", "Unreadable": "critical"}


def render() -> None:
    page_header(
        "Intelligence / Model registry",
        "Model registry & artifacts",
        "Every trained run on disk, with its promotion status, lineage, validation gate and edge-export evidence.",
        [badge_html("HARDWARE VERIFICATION: NONE RECORDED", "warning", "▲", "No run has real-signal evidence. C2000 kits record their bench measurements on the board.")],
    )
    rows = list_runs()
    ok_rows = [r for r in rows if "error" not in r]
    kpi_row([
        {"label": "Runs on disk", "value": str(len(rows)), "sub": "configs/classifiers + configs/regressors", "tone": "primary"},
        {"label": "Promoted", "value": str(sum(r["status"].startswith("Promoted") for r in ok_rows)), "sub": "active registry entries", "tone": "healthy"},
        {"label": "Exported (.tflite)", "value": str(sum(r["exported"] for r in ok_rows)), "sub": "export only — not hardware-verified", "tone": "training"},
        {"label": "Hardware verified", "value": str(sum(r["hardware_verified"] for r in ok_rows)), "sub": "requires deployment evidence", "tone": "unknown"},
    ])
    c1, c2, c3 = st.columns(3)
    f_domain = c1.multiselect("Domain", sorted({r["domain"] for r in ok_rows}), key="reg_f_domain")
    f_tier = c2.multiselect("Platform / tier", sorted({r["tier"] for r in ok_rows}), key="reg_f_tier")
    f_status = c3.multiselect("Status", sorted({r["status"] for r in rows}), key="reg_f_status")
    coverage = class_coverage_warnings(rows)
    shown = [r for r in rows if (not f_domain or r["domain"] in f_domain) and (not f_tier or r["tier"] in f_tier) and (not f_status or r["status"] in f_status)]
    table = pd.DataFrame([{"run": r.get("run", str(r["run_dir"])), "domain": r["domain"], "tier": r["tier"], "block": r["block"], "status": r["status"], "validation gate": r.get("validation"), "origin": r.get("origin"), "export": r.get("quantization") or "—", "test metric": r.get("headline", r.get("error")), "audit": f"⚠ {len(r['audit_findings'])} finding(s)" if r.get("audit_findings") else "—",
                           "class limitation": "fewer classes than other tiers (AI-N13)" if any(f"({r.get('run')})" in w for w in coverage) else SYNTHETIC_LABEL if any(is_synthetic_class(c) for c in ((r.get("metrics") or {}).get("classes") or [])) else "—"} for r in shown])
    for warning in coverage:
        st.warning(f"**Class coverage:** {warning}", icon=":material/warning:")
    left, right = st.columns([1.5, 1], gap="medium")
    with left:
        st.markdown(f"##### Models ({len(shown)})")
        event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="reg_table")
        sel = event.selection.rows[0] if event and event.selection.rows else None
        if sel is None:
            st.caption("Select a row to inspect it.")
    with right:
        if sel is None or sel >= len(shown):
            st.info("Select a model in the table.", icon=":material/info:")
        else:
            _detail(shown[sel])
    with st.expander("Registry history & rollback"):
        history = registry_history()
        if not history:
            st.caption("No promotion recorded through the audited path yet (configs/registry_history.yaml does not exist). Earlier promotions were made with experiments/promote_run.py before history existed.")
        else:
            st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        keys = sorted({e["key"] for e in history})
        if keys:
            key = st.selectbox("Entry to roll back", keys, key="reg_rb_key")
            reason = st.text_input("Reason", key="reg_rb_reason")
            confirm = st.checkbox("I understand this changes the production model for this entry", key="reg_rb_confirm")
            if st.button("Roll back to previous", key="reg_rb_button", disabled=not (confirm and reason.strip())):
                domain, tier, block = key.split("/")
                try:
                    restored = rollback(domain, tier, block, by="dashboard user", reason=reason.strip())
                    st.success(f"{key} now points to {restored or '(nothing — entry removed)'}.")
                except RegistryError as exc:
                    st.error(str(exc))


def _detail(r: dict) -> None:
    with st.container(border=True):
        st.markdown(f"##### {r.get('run', r['run_dir'].name)}")
        if "error" in r:
            st.error(f"Unreadable run directory: {r['error']}")
            return
        st.html('<div class="spade-root spade-badges">' + badge_html(r["status"], _STATUS_TONE.get(r["status"], "unknown")) + badge_html(f"gate: {r['validation']}", "healthy" if r["validation"] == "APPROVED" else "unknown") + badge_html(r["tier"].upper(), "training") + "</div>")
        for f in r.get("audit_findings", []):
            st.warning(f"**Audit finding ({f.get('date') or 'undated'}):** {f.get('summary')}  \nEvidence: `{f.get('evidence')}`", icon=":material/warning:")
        tabs = st.tabs(["Overview", "Artifacts", "Metrics", "Lineage", "Deployment"])
        m = r["metrics"]
        with tabs[0]:
            if m.get("residual_detector"):
                d = m["residual_detector"]
                st.caption(f"Forecast-residual detector (E14): threshold {d['threshold']:.3f} = {d['quantile']:.0%} quantile on {d['fitted_on']['n']} healthy validation windows; "
                           f"scores {', '.join(d['target_channels'])}. Trained on '{m.get('label_filter')}' runs only.")
            provenance_table({"Domain / block": f"{r['domain']} / {r['block']}", "Origin": r["origin"], "Input channels": ", ".join(m.get("channels", [])), "Classes": ", ".join(m.get("classes", []) or []) or "— (regressor)", "Train / val / test windows": f"{m.get('n_train', '?')} / {m.get('n_val', '—')} / {m.get('n_test', '?')}", "Headline": r["headline"], "Evaluation set": "simulated windows, grouped-by-run test split (see metrics.json)"})
        with tabs[1]:
            artifact_table(r["run_dir"], r["run"])
        with tabs[2]:
            st.json(m, expanded=False)
        with tabs[3]:
            if r["lineage"]:
                st.json(r["lineage"], expanded=False)
            else:
                st.caption("No lineage.json (from-scratch run of experiments/train_model.py). Its config.yaml + metrics.json are the record; run_metadata.json exists only for runs trained after the reproducibility update.")
        with tabs[4]:
            stages = r["deployment"]["stages"]
            check_list([(s.replace("_", " "), "pass" if v == "done" else ("not_applicable" if v == "not_exported" else "pending"), "evidence recorded" if v == "done" else ("no .tflite" if v == "not_exported" else "no evidence recorded")) for s, v in stages.items()])
            art = r["deployment"].get("artifact")
            if art:
                st.caption(f"Quantization (read from the file): {art.get('quantization')} · input {art.get('input_dtypes')} · {art['size_bytes'] / 1024:.1f} KB. Latency, RAM/Flash and post-quantization accuracy on hardware: not measured; host-CPU parity vs. Keras in docs/audit/tflite_parity_report.txt.")
                if art.get("flex_ops"):
                    st.warning(f"Contains TF Select (Flex) ops {art['flex_ops']}: needs the Flex delegate — not runnable on TFLite Micro or the plain tflite-runtime/LiteRT interpreter.", icon=":material/warning:")
            st.caption("To record hardware evidence, add deployment_evidence.json to the run directory (see Documentation).")
            kit = r["deployment"].get("c2000")
            if kit:
                st.markdown("**TI C2000 (LAUNCHXL-F28379D)**")
                st.caption(c2000_summary(kit))
                st.download_button("Download C2000 kit (.zip)", kit_zip(kit["kit_dir"]), file_name=f"{r['run'].replace('/', '_')}_c2000.zip", key=f"registry_c2000_{r['run']}", icon=":material/download:")
