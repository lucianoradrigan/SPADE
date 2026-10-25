"""PLECS Plant Studio → Create Plant → "Use as Gym plant" with the engine "OpenModelica FMU"
(task B5; spec Sec. 3.4 six stages, 3.8 state machine, 3.17 blocking rules).

Stages: Goal → Engine → Circuit → Signals → Gym behavior → Validate & publish, with the fixed
"Environment readiness" column. The FMUs listed are the ones built in data/fmu/ (or
$SPADE_FMU_DIR); a model whose B3 verdict is NO-GO is shown with its verdict and a link to the
report and can never be published. Verification runs as a persistent job; "Publish to
Environments" is enabled only by a passing verification for the current FMU hash and
configuration. The PLECS Co-Simulation FMU engine is listed as not supported in this version.
"""

from __future__ import annotations

import streamlit as st

from driveflow.viz.adapters import fmu_studio_adapter as fa
from driveflow.viz.adapters import jobs_adapter as ja
from driveflow.viz.components.page_header import badge_html, source_badge_html

_TONE = {"Validated": ("healthy", "✓"), "Needs attention": ("warning", "!"), "Failed": ("critical", "✗"), "Not supported": ("unknown", "⊘"), "Not configured": ("unknown", "○")}
_PRESETS = ("DC-DC voltage regulation", "Motor speed regulation", "Current control", "Custom research task")


def _profile():
    pid = st.session_state.get("fmu_profile")
    if not pid:
        return None
    try:
        return fa.get_profile(pid)
    except Exception:
        return None


def _readiness(p) -> None:
    with st.container(border=True):
        st.markdown("**Environment readiness**")
        st.caption(f"Profile state: **{fa.state(p)}**" + (f" · `{p.profile_id}` v{p.version}" if p else ""))
        for item, (state, text) in fa.readiness(p).items():
            tone, glyph = _TONE[state]
            st.html('<div class="spade-root spade-badges">' + badge_html(f"{item}: {state}", tone, glyph, text or None) + "</div>")
            if text:
                st.caption(text)


def _goal() -> None:
    st.markdown("#### What will SPADE control?")
    st.selectbox("Task preset (fills configuration fields only, no circuit is pre-imported)", _PRESETS, key="fmu_preset")
    with st.container(border=True):
        st.markdown("**Why does Gym need step-by-step control?**")
        st.caption("At every control interval the agent sends an action, the plant advances by Ts, and SPADE reads the observation and computes the reward. "
                   "That needs a plant that can be stepped and reset — here, an FMI 2.0 Co-Simulation FMU built with OpenModelica.")
    st.caption("Next: 2. Engine.")


def _engine() -> None:
    engine = st.radio("Engine", ["OpenModelica FMU", "PLECS Co-Simulation FMU"], key="fmu_engine", horizontal=True)
    if engine != "OpenModelica FMU":
        st.info("Not supported in this version: a PLECS Co-Simulation FMU needs PLECS 5.0 with its runtime and licence on this computer. "
                "Use the OpenModelica FMU engine, or the dataset route with PLECS RPC.", icon=":material/block:")
        return
    if st.button("Test capabilities", key="fmu_test_engine", icon=":material/fact_check:"):
        st.session_state["fmu_engine_report"] = fa.engine_report()
    rep = st.session_state.get("fmu_engine_report") or fa.engine_report()
    if rep["fmpy"]:
        st.success(f"fmpy {rep['fmpy_version']} available · platform {rep['platform']}. Interactive control is confirmed only by the verification in stage 6.", icon=":material/check_circle:")
    else:
        st.error(rep["message"], icon=":material/error:")
        st.code('pip install "driveflow[fmu]"', language="bash")
    st.caption(f"FMU folder: `{rep['fmu_dir']}`. Checked: {', '.join(rep['checked'])}. Not checked: {', '.join(rep['not_checked'])}.")


def _circuit() -> None:
    rows = fa.candidates()
    if not rows:
        st.info(f"No FMU found in `{fa.engine_report()['fmu_dir']}`. Build them with scripts/fmu/build_fmus.sh (docs/guides/BUILD_FMUS_LOCALLY.md).", icon=":material/info:")
        return
    st.dataframe(fa.candidates_table(rows), hide_index=True, width="stretch")
    for r in rows:
        if (r.get("b3") or {}).get("verdict") == "NO-GO":
            st.warning(f"**{r['name']}: B3 verdict NO-GO** — not publishable. Report: [{fa.B3_LINK}]({fa.B3_LINK})" + (f" · **{r['note']}**" if r.get("note") else ""),
                       icon=":material/block:")
    names = [r["name"] for r in rows]
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("FMU", names, key="fmu_pick")
    pid = c2.text_input("Profile ID", key="fmu_new_id", placeholder="e.g. om_buck")
    disp = c3.text_input("Display name", key="fmu_new_name")
    cand = next(r for r in rows if r["name"] == name)
    unsupported = cand["status"] in ("Not supported", "Failed")
    if unsupported:
        st.error(f"{name}: {cand['status']} — {cand['reason']}", icon=":material/block:")
    if st.button("Import circuit", key="fmu_import", icon=":material/add:", disabled=unsupported or not pid):
        try:
            p = fa.import_fmu(pid, disp, name)
        except ValueError as exc:
            st.error(f"Not imported: {exc}", icon=":material/error:")
        else:
            st.session_state["fmu_profile"] = p.profile_id
            st.success(f"Imported **{p.display_name}** as `{p.profile_id}` v{p.version} (SHA-256 {p.file_hash[:12]}…).", icon=":material/check_circle:")
    existing = fa.profiles()
    if existing:
        ids = [p.profile_id for p in existing]
        cur = st.session_state.get("fmu_profile")
        sel = st.selectbox("Open an FMU plant profile", ids, index=ids.index(cur) if cur in ids else 0, key="fmu_open")
        if sel != cur and st.button("Open profile", key="fmu_open_btn"):
            st.session_state["fmu_profile"] = sel
            st.rerun()


def _signals(p) -> None:
    if p is None:
        st.info("Import an FMU first (3. Circuit).", icon=":material/info:")
        return
    st.caption("SPADE writes → the FMU simulates → SPADE reads. " + fa.space_text(p))
    m = p.parameters.get("mapping") or st.session_state.get(f"fmu_suggested_{p.profile_id}")
    if st.button("Auto-suggest mapping", key="fmu_suggest", help="Proposes names from the FMU; nothing is saved until you press Validate mapping."):
        st.session_state[f"fmu_suggested_{p.profile_id}"] = m = fa.suggest_mapping(p)
    if m is None:
        m = fa.suggest_mapping(p)
        st.html('<div class="spade-root spade-badges">' + badge_html("SUGGESTED MAPPING — CONFIRM BEFORE SAVING", "unknown", "◇") + "</div>")
    ins = p.parameters["fmu_inputs"]
    outs = [o["name"] for o in p.parameters["fmu_outputs"]]
    units = {o["name"]: o["unit"] or "1" for o in p.parameters["fmu_outputs"]}
    k = p.profile_id
    st.markdown("**What can SPADE control? — Action**")
    c1, c2, c3, c4 = st.columns(4)
    a_in = c1.selectbox("FMU input", ins, index=ins.index(m["action"]["input"]) if m["action"]["input"] in ins else 0, key=f"fmu_a_in_{k}")
    a_name = c2.text_input("SPADE name", m["action"]["name"] or a_in, key=f"fmu_a_name_{k}")
    a_lo = c3.number_input("Low", value=float(m["action"]["low"]), key=f"fmu_a_lo_{k}")
    a_hi = c4.number_input("High", value=float(m["action"]["high"]), key=f"fmu_a_hi_{k}")
    st.markdown("**What can SPADE observe? — Observations** (in this order; observation = value / scale)")
    prev = [o["output"] for o in m["observations"] if o["output"] in outs]
    chosen = st.multiselect("FMU outputs", outs, default=prev, key=f"fmu_obs_{k}")
    scales = {o["output"]: o["scale"] for o in m["observations"]}
    obs = []
    for name in chosen:
        sc = st.number_input(f"Scale of {name} ({units[name]})", min_value=1e-9, value=float(scales.get(name, 1.0)), key=f"fmu_sc_{k}_{name}")
        obs.append({"output": name, "unit": units[name], "scale": float(sc)})
    st.markdown("**What changes during an episode? — Reference**")
    c1, c2 = st.columns(2)
    ref_opts = ["(none)"] + chosen
    ref0 = (m.get("reference") or {}).get("output")
    ref_out = c1.selectbox("Tracked output", ref_opts, index=ref_opts.index(ref0) if ref0 in ref_opts else 0, key=f"fmu_ref_{k}")
    ref_val = c2.number_input("Reference value", value=float((m.get("reference") or {}).get("value") or 0.0), key=f"fmu_refv_{k}")
    st.markdown("**What must SPADE protect? — Safety**")
    c1, c2 = st.columns(2)
    s_opts = ["(none)"] + outs
    s0 = (m.get("safety") or {}).get("output")
    s_out = c1.selectbox("Safety output", s_opts, index=s_opts.index(s0) if s0 in s_opts else 0, key=f"fmu_saf_{k}")
    s_rated = c2.number_input("Rated value (episode ends above 2 × rated)", min_value=1e-9, value=float((m.get("safety") or {}).get("rated") or 1.0), key=f"fmu_safr_{k}")
    if st.button("Validate mapping", key="fmu_save_mapping", type="primary", icon=":material/rule:"):
        mapping = {"action": {"input": a_in, "name": a_name, "unit": "1", "low": a_lo, "high": a_hi}, "observations": obs,
                   "reference": {"output": ref_out, "value": ref_val} if ref_out != "(none)" else None,
                   "safety": {"output": s_out, "rated": s_rated} if s_out != "(none)" else None}
        try:
            fa.save_mapping(p, mapping)
        except ValueError as exc:
            for e in str(exc).split("; "):
                st.error(e, icon=":material/error:")
        else:
            st.success("Mapping saved. Any earlier verification of this profile is invalidated.", icon=":material/check_circle:")
            st.rerun()


def _gym(p) -> None:
    if p is None or not p.parameters.get("mapping"):
        st.info("Map the signals first (4. Signals).", icon=":material/info:")
        return
    g = fa.default_gym(p)
    k = p.profile_id
    t = st.tabs(["A. Control timing", "B. Episode reset", "C. Reward", "D. Safety & episode end", "E. Environment export"])
    with t[0]:
        c1, c2 = st.columns(2)
        tau_us = c1.number_input("Controller sample time Ts (µs)", 1.0, 1e6, float(g["tau_s"] * 1e6), key=f"fmu_tau_{k}")
        dur_ms = c2.number_input("Episode duration (ms)", 0.001, 1e6, float(g["duration_s"] * 1e3), key=f"fmu_dur_{k}")
        st.caption("Zero-order hold: the action is held for Ts; one doStep per action. The agent sees obs(t), decides action(t), the FMU advances to t + Ts. "
                   "The FMU's internal solver step (CVODE, adaptive) is not the control step.")
    with t[1]:
        st.caption("reset(seed) re-instantiates the FMU and sets these parameters at initialization (fixed initial conditions).")
        params = {}
        for q in p.parameters["fmu_parameters"]:
            params[q["name"]] = st.number_input(f"{q['name']} ({q['unit'] or '1'})", value=float(g["parameters"].get(q["name"], q["start"] or 0.0)), format="%.6g",
                                                key=f"fmu_par_{k}_{q['name']}")
    with t[2]:
        terms = {tm["kind"]: tm for tm in g["reward_terms"]}
        new_terms = []
        for kind, label in (("tracking", "Tracking error"), ("overcurrent", "Unsafe current"), ("action_rate", "Action variation")):
            c1, c2 = st.columns([1, 2])
            on = c1.checkbox(label, value=kind in terms and terms[kind].get("enabled", True), key=f"fmu_rw_on_{k}_{kind}")
            w = c2.number_input(f"Weight · {label}", 0.0, 1e6, float(terms.get(kind, {}).get("weight", 1.0)), key=f"fmu_rw_w_{k}_{kind}")
            new_terms.append({"kind": kind, "weight": w, "enabled": on, **({"threshold": float(terms.get(kind, {}).get("threshold", 0.8))} if kind == "overcurrent" else {})})
        st.caption("reward = −Σ weight · penalty (envs/reward.py). The per-term breakdown is shown only after the verification evaluates it.")
    with t[3]:
        policy = st.radio("Action outside the bounds", ["clip", "reject"], index=["clip", "reject"].index(g.get("action_out_of_range", "clip")),
                          format_func={"clip": "Clip and log", "reject": "Reject action"}.get, key=f"fmu_oor_{k}", horizontal=True)
        st.caption("terminated: safety output above 2 × rated, non-finite state, or FMU solver failure (never a success; reward −100). truncated: horizon reached.")
    with t[4]:
        st.caption(fa.space_text(p))
        st.code("# Example API contract; the environment exists only after Publish\nobs, info = env.reset(seed=42)\nobs, reward, terminated, truncated, info = env.step(action)\nenv.close()",
                language="python")
    if st.button("Validate definition", key="fmu_save_gym", type="primary", icon=":material/rule:"):
        cfg = {"tau_s": tau_us * 1e-6, "duration_s": dur_ms * 1e-3, "parameters": params, "reward_terms": new_terms, "action_out_of_range": policy}
        try:
            fa.save_gym(p, cfg)
        except ValueError as exc:
            for e in str(exc).split("; "):
                st.error(e, icon=":material/error:")
        else:
            st.success("Gym behavior saved. Any earlier verification of this profile is invalidated.", icon=":material/check_circle:")
            st.rerun()


def _validate(p) -> None:
    if p is None or not p.parameters.get("gym"):
        st.info("Complete 4. Signals and 5. Gym behavior first.", icon=":material/info:")
        return
    job = fa.verification_job(p)
    running = job is not None and job["state"] in ("queued", "preparing", "running", "cancelling")
    c1, c2 = st.columns([1, 1])
    if c1.button("Run verification", key="fmu_verify", type="primary", icon=":material/play_arrow:", disabled=running or p.status == "unsupported"):
        j = fa.submit_verification(p)
        st.success(f"Queued as job `{j.job_id}`. Follow it in Experiment Runs.", icon=":material/schedule:")
        if not ja.live_workers():
            st.caption(":material/warning: No worker is running: the verification waits in the queue until one starts (Experiment Runs → Start local worker).")
        st.rerun()
    c2.button("Refresh", key="fmu_refresh", icon=":material/refresh:")
    if job is not None:
        st.caption(f"Last verification job `{job['job_id']}`: {job['state']}.")
    rep = fa.verification_report(p)
    if rep is None:
        st.info("No verification has been run for this configuration.", icon=":material/info:")
    else:
        stale = fa.state(p) == "STALE"
        head = "Stale — Revalidate" if stale else ("Validated" if rep["status"] == "passed" else "Failed")
        (st.warning if stale else st.success if rep["status"] == "passed" else st.error)(
            f"**{head}**: {rep['n_passed']}/{rep['n_checks']} checks passed in {rep['wall_s']:g} s (FMU {rep['fmu_sha256'][:12]}…, configuration {rep['config_hash'][:12]}…).")
        st.dataframe(fa.checks_table(rep), hide_index=True, width="stretch")
    blockers = fa.blockers(p)
    for b in blockers:
        st.caption(f":material/block: {b}")
    if p.status == "published" and not blockers:
        env_id = p.parameters["published_env_id"]
        st.success(f"Published as `{env_id}` (Workspace → Environments). Simulated, not hardware.", icon=":material/check_circle:")
        st.code(fa.snippet(env_id), language="python")
    elif st.button("Publish to Environments", key="fmu_publish", disabled=bool(blockers), icon=":material/publish:"):
        try:
            env_id = fa.publish(p)
        except ValueError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            st.success(f"Published as `{env_id}`.", icon=":material/check_circle:")
            st.rerun()
    st.caption("Save draft is automatic: every validated stage is stored on the profile. Publishing does not deploy a controller to hardware.")


def render_gym_route() -> None:
    st.html('<div class="spade-root spade-badges">' + source_badge_html("SIMULATED · OPENMODELICA", "Simulated with an OpenModelica FMU; not hardware") + "</div>")
    stage = st.radio("Stage", fa.STAGES, key="fmu_stage", horizontal=True)
    p = _profile()
    main, side = st.columns([3, 1], gap="medium")
    with main:
        {fa.STAGES[0]: _goal, fa.STAGES[1]: _engine, fa.STAGES[2]: _circuit}.get(stage, lambda: None)()
        if stage == fa.STAGES[3]:
            _signals(p)
        elif stage == fa.STAGES[4]:
            _gym(p)
        elif stage == fa.STAGES[5]:
            _validate(p)
    with side:
        _readiness(p)
