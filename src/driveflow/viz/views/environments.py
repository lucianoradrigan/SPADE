"""Workspace / Environments (spec Sec. 8.5; expansion task E1).

One row per Gymnasium environment SPADE registers, read from a real instance through
adapters/environments_adapter.py: plant, version, source, observation (shape and units), action
(type and bounds), default reward, physical limits, termination conditions and compatible
algorithms. The Python snippet is offered only for IDs that are registered in Gymnasium;
environments other tasks will add are listed as "Not registered", with no snippet.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driveflow.viz.adapters import environments_adapter as ea
from driveflow.viz.components.page_header import badge_html, badges_row, page_header, source_badge_html
from driveflow.viz.components.state_panel import ComponentView, render_state


def _fmt(x: float) -> str:
    return f"{x:g}"


def observation_text(c: dict) -> str:
    o = c["observation"]
    return f"{o['shape']} {o['dtype']}: " + ", ".join(f"{n} [{u}]" for n, u in zip(o["names"], o["units"]))


def action_text(c: dict) -> str:
    a = c["action"]
    bounds = ", ".join(f"{n} ∈ [{_fmt(lo)}, {_fmt(hi)}] {u}" for n, lo, hi, u in zip(a["names"], a["low"], a["high"], a["units"]))
    return f"{a['type']}{a['shape']}: {bounds}"


def reward_text(c: dict) -> str:
    return " + ".join(f"{t['weight']:g}·{t['kind']}" for t in c["reward_terms"] if t["enabled"]) + "  (penalties, reward = −sum)"


def surrogate_text(c: dict) -> str:
    s = c.get("surrogate")
    if not s:
        return "—"
    e = s["fit_error"]
    cl = e.get("closed_loop_rmse_v_out_v")
    closed = f", closed-loop v_out {cl:.3f} V RMSE" if cl is not None else ", closed loop not compared (no closed-loop reference for this source)"
    return (f"{s['family']}: open-loop v_out {e['rollout_rel_rmse_v_out']:.1%} rel. RMSE{closed} (median over validation conditions); "
            f"valid for V_in {s['validity_domain']['v_in_v']} V, R {s['validity_domain']['load_resistance_ohm']} Ω")


def table_rows(contracts: list[dict], planned: list[dict]) -> list[dict]:
    rows = []
    for c in contracts:
        if "error" in c:
            rows.append({"Environment ID": c["env_id"], "Plant": "—", "Status": f"Error: {c['error']}"})
            continue
        if "unavailable" in c:
            rows.append({"Environment ID": c["env_id"], "Plant": c["display_name"], "Version": f"v{c['version']}", "Source": c["source"],
                         "Status": f"Not available: {c['unavailable']}"})
            continue
        rows.append({"Environment ID": c["env_id"], "Plant": c["display_name"], "Version": f"v{c['version']}", "Source": c["source"],
                     "Status": "Registered" if c["registered"] else "Not registered",
                     "Observation": observation_text(c), "Action": action_text(c), "Default reward": reward_text(c),
                     "Physical limits": ", ".join(f"{k} {_fmt(v)}" for k, v in c["limits"].items()),
                     "Step / horizon": f"{c['tau_s'] * 1e6:g} µs × {c['horizon_steps']}",
                     "Terminations": "; ".join(c["terminations"]), "Compatible algorithms": "; ".join(c["compatible_algorithms"]),
                     "Fit error (validation)": surrogate_text(c)})
    for p in planned:
        rows.append({"Environment ID": p["name"], "Plant": p["plant"], "Status": f"Not registered (task: {p['task']})"})
    return rows


def _capability_lines(cap) -> list[str]:
    return [f"{'✓' if getattr(cap, f) else '✗'} {f.removeprefix('can_').replace('_', ' ')}"
            for f in ("can_simulate_batch", "can_step_interactively", "can_reset_episode", "can_train_rl")]


def _details(c: dict) -> None:
    env_id = c["env_id"]
    with st.expander(f"{c['display_name']} — {env_id}", icon=":material/sports_esports:"):
        badges_row([source_badge_html(c["source"]), badge_html("REGISTERED", "healthy", "●") if c["registered"] else badge_html("NOT REGISTERED", "unknown", "○")])
        tabs = st.tabs(["Contract", "Reward", "Capabilities", "Python snippet"])
        with tabs[0]:
            o, a = c["observation"], c["action"]
            st.dataframe(pd.DataFrame({"Observation": o["names"], "Unit": o["units"], "Scale (obs = value / scale)": o["scales"]}), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame({"Action": a["names"], "Unit": a["units"], "Low": a["low"], "High": a["high"]}), hide_index=True, width="stretch")
            st.caption(f"Step: {c['tau_s'] * 1e6:g} µs (one plant simulate() call) · default horizon {c['horizon_steps']} steps · "
                       f"reference policy: {c['baseline_controller']} · plant profile {c['plant_profile_id']}")
            st.markdown("**Episode end** — terminated: " + "; ".join(t for t in c["terminations"] if "truncated" not in t) + ". Truncated: horizon reached.")
            st.caption("Actions outside the bounds are clipped; info['action_applied'] and info['action_clipped'] show what reached the plant. A non-finite action is rejected.")
            if c.get("note"):
                st.info(c["note"], icon=":material/info:")
            if c.get("surrogate"):
                st.warning(f"**Surrogate, not the plant.** Fitted to SPADE's own averaged buck (not to PLECS); run {c['surrogate']['run']}. {surrogate_text(c)}. "
                           "Outside that range every step reports info['out_of_domain'] (Surrogate outside validated operating range).", icon=":material/model_training:")
        with tabs[1]:
            st.dataframe(pd.DataFrame([{"Term": t["kind"], "Weight": t["weight"], "Penalty": t["formula"],
                                        "Threshold": t["threshold"] if t["kind"] == "overcurrent" else None} for t in c["reward_terms"]]),
                         hide_index=True, width="stretch")
            st.caption("reward = −Σ weight · penalty. Pass reward_terms=[...] to gym.make to change it; the per-step breakdown is in info['reward_terms'].")
        with tabs[2]:
            declared = ea.declared_capability(env_id)
            if declared is not None:
                st.markdown("**Declared on the plant profile** (backed by tests/envs/test_native_envs.py)")
                st.text("\n".join(_capability_lines(declared)))
            if st.button("Run contract check now", key=f"env_verify_{env_id}", icon=":material/fact_check:",
                         help="Runs gymnasium's check_env and a seeded determinism check on a fresh instance."):
                st.session_state[f"env_verify_result_{env_id}"] = ea.verify(env_id)
            result = st.session_state.get(f"env_verify_result_{env_id}")
            if result is None:
                st.caption("Not checked in this session.")
            elif "error" in result:
                render_state(ComponentView("error", "Contract check could not run", result["error"]))
            else:
                cap = result["capability"]
                ok = cap.can_step_interactively and cap.can_reset_episode
                render_state(ComponentView("success" if ok else "negative", "Contract check passed" if ok else "Contract check failed",
                                           f"Checked {cap.validated_at} UTC"))
                st.text("\n".join(_capability_lines(cap)))
                st.caption("\n\n".join(cap.diagnostic_messages))
        with tabs[3]:
            code = ea.snippet(env_id)
            if code is None:
                st.caption("Not registered — no snippet.")
            else:
                st.caption("Copy Python snippet (copy icon at the top right of the block). It instantiates the registered ID.")
                st.code(code, language="python")


def render() -> None:
    ids = ea.env_ids()
    contracts = [ea.contract(e) for e in ids]
    n_ok = sum(1 for c in contracts if "error" not in c and "unavailable" not in c and c["registered"])
    page_header("Workspace / Environments", "Environments",
                "Gymnasium environments over SPADE's simulated plants: the contract each one exposes to Python algorithms "
                "(observation, action, reward, limits, episode end). Values are read from the registered environments, not typed by hand.",
                [badge_html(f"{n_ok} REGISTERED", "healthy" if n_ok else "unknown", "●"), badge_html(f"{len(ea.planned())} PLANNED", "unknown", "○")])
    if not contracts:
        render_state(ComponentView("empty", "No environments registered"))
    st.dataframe(pd.DataFrame(table_rows(contracts, ea.planned())).fillna("—"), hide_index=True, width="stretch")
    st.markdown("##### Details")
    for c in contracts:
        if "error" in c:
            render_state(ComponentView("error", f"{c['env_id']} could not be instantiated", c["error"]))
        elif "unavailable" in c:
            render_state(ComponentView("empty", f"{c['env_id']} is not available here", c["unavailable"]))
        else:
            _details(c)
    st.caption("Planned environments are listed for orientation only: they are not registered and offer no snippet until their task lands.")
