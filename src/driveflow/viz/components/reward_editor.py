"""RewardEditor (spec Sec. 5 step 2, Sec. 9): the reward as named terms with weight, formula and
units. It edits and validates; it never draws a curve. The breakdown is shown only after a real
episode has been run (``reward_preview``)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driveflow.envs import reward as rw

_UNITS = {"tracking": "(reference − output) / output scale", "overcurrent": "|i| / current limit", "action_rate": "Δaction / action width"}


def reward_editor(key: str = "reward") -> tuple[list[dict] | None, str | None]:
    """(terms, error). terms is a list of RewardTerm dicts, or None when the edit is invalid."""
    defaults = pd.DataFrame([{"Term": t.kind, "Enabled": t.enabled, "Weight": t.weight, "Threshold": t.threshold if t.kind == "overcurrent" else None,
                              "Formula (penalty)": rw.TERM_KINDS[t.kind][1], "Signal": _UNITS[t.kind]} for t in rw.DEFAULT_TERMS])
    edited = st.data_editor(defaults, hide_index=True, width="stretch", key=key, disabled=["Term", "Formula (penalty)", "Signal"],
                            column_config={"Weight": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.3f"),
                                           "Threshold": st.column_config.NumberColumn(min_value=0.01, step=0.05, help="Overcurrent only: fraction of the limit where the penalty starts")})
    terms = [{"kind": r["Term"], "enabled": bool(r["Enabled"]), "weight": float(r["Weight"]),
              **({"threshold": float(r["Threshold"])} if r["Term"] == "overcurrent" and pd.notna(r["Threshold"]) else {})} for _, r in edited.iterrows()]
    try:
        rw.parse_terms(terms)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    st.caption("Reward r = −Σ wₖ · penaltyₖ over the enabled terms (normalized, dimensionless signals). The per-term breakdown is in info['reward_terms'] at every step.")
    return terms, None


def reward_preview(breakdown: dict | None, total: float | None) -> None:
    if breakdown is None:
        st.info("No evaluated episode yet. Run **Preview on baseline episode** to see each term's contribution.", icon=":material/info:")
        return
    df = pd.DataFrame([{"Term": k, "Sum of weighted penalties": v, "Share of −reward": (v / max(-total, 1e-12)) if total else None} for k, v in breakdown.items()])
    st.dataframe(df, hide_index=True, width="stretch", column_config={"Share of −reward": st.column_config.NumberColumn(format="percent")})
    st.caption(f"Episode reward {total:.3f} (baseline controller, one executed episode).")
