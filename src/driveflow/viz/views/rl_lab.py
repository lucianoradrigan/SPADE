"""Design & Control / RL Lab (spec Sec. 5; expansion task E8).

Tabs Environment → Reward → Training → Benchmark:
- training runs as a job in the worker (``rl_training``);
- every chart comes from an executed run: the reward preview from a real baseline episode, the
  playback from the evaluation episodes the job recorded;
- the benchmark compares the policy with the plant's classical controllers on the same grid, and
  is shown even when RL loses;
- "Save policy candidate" deploys nothing, and "Use as default" is blocked (decision D4).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from driveflow.viz.adapters import environments_adapter as ea
from driveflow.viz.adapters import jobs_adapter as ja
from driveflow.viz.adapters import rl_adapter as ra
from driveflow.viz.adapters.session import record_run
from driveflow.viz.components.approval_gate import approval_gate
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.components.reward_editor import reward_editor, reward_preview
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure


def _environment() -> tuple[str, str, int | None]:
    envs = ra.env_ids()
    env_id = st.selectbox("Environment (registered Gymnasium ID)", envs, key="rl_env")
    c = ea.contract(env_id)
    st.html('<div class="spade-root spade-badges">' + source_badge_html(c["source"]) + "</div>")
    st.caption(f"Observation {c['observation']['shape']} ({', '.join(c['observation']['names'])}); action {c['action']['type']}{c['action']['shape']} "
               f"{', '.join(c['action']['names'])} in [{', '.join(f'{x:g}' for x in c['action']['low'])}] … [{', '.join(f'{x:g}' for x in c['action']['high'])}]; "
               f"step {c['tau_s'] * 1e6:g} µs × {c['horizon_steps']}.")
    with st.expander("Preview environment contract / Copy Python snippet", icon=":material/code:"):
        st.code(ea.snippet(env_id), language="python")
    c1, c2 = st.columns(2)
    algo = c1.radio("Algorithm", ["ddpg", "dqn"], format_func={"ddpg": "DDPG (continuous actions)", "dqn": "DQN (discrete actions)"}.get, key="rl_algo", horizontal=True)
    levels = None
    if algo == "dqn":
        levels = c2.slider("Discretize the action: levels per dimension", 3, 41 if len(c["action"]["low"]) == 1 else 11, 11 if len(c["action"]["low"]) == 1 else 5, key="rl_levels",
                           help="DQN needs discrete actions: the continuous action is replaced by a grid of levels, saved with the policy.")
        c2.caption(f"{levels ** len(c['action']['low'])} discrete actions, evenly spaced over the action bounds.")
    return env_id, algo, levels


def _reward(env_id: str):
    terms, err = reward_editor(key=f"rl_reward_{env_id}")
    if err:
        st.error(err, icon=":material/error:")
        return None
    if st.button("Preview on baseline episode", key="rl_preview", icon=":material/play_circle:", help=f"Runs one episode of {ra.baseline_name(env_id)} with these terms."):
        with st.spinner("Running one baseline episode…"):
            pv = ra.preview_baseline(env_id, terms)
        st.session_state["rl_preview_result"] = {"env": env_id, "terms": terms, **pv, "run": record_run("rl_reward_preview", {"env_id": env_id, "terms": terms}, {"reward": pv["total"]})}
    pv = st.session_state.get("rl_preview_result")
    fresh = pv is not None and pv["env"] == env_id and pv["terms"] == terms
    if pv is not None and not fresh:
        st.caption("The terms or the environment changed since the last preview: run it again.")
    reward_preview(pv["breakdown"] if fresh else None, pv["total"] if fresh else None)
    if fresh:
        st.caption(f"Baseline: {pv['controller']}, {pv['steps']} steps{', terminated: ' + pv['termination'] if pv['termination'] else ''}. Evaluation run {pv['run']['run_id']}.")
    return terms


def _training(env_id: str, algo: str, levels, terms) -> None:
    c1, c2, c3 = st.columns(3)
    episodes = c1.number_input("Episodes", 1, 2000, 100, 10, key="rl_episodes")
    seed = c2.number_input("Seed", 0, 2**31 - 1, 0, key="rl_seed")
    eval_every = c3.number_input("Evaluate and checkpoint every N episodes", 1, 500, 10, key="rl_eval_every")
    if not ja.live_workers():
        st.caption(":material/warning: No worker is running: the job will wait in the queue (Experiment Runs → Start local worker).")
    if st.button("Start training", type="primary", key="rl_start", icon=":material/play_arrow:", disabled=terms is None):
        job = ra.submit(env_id, algo, int(episodes), int(seed), terms, levels, int(eval_every))
        st.success(f"Queued as job `{job.job_id}`.", icon=":material/schedule:")
    for j in ra.active_jobs():
        st.caption(f":material/schedule: Job `{j['job_id']}`: {j['state']}" + (f" ({j['done']}/{j['total']} episodes)" if j["total"] else ""))
    st.button("Refresh", key="rl_refresh", icon=":material/refresh:")
    rs = ra.runs()
    if not rs:
        st.info("No simulation has been executed for this configuration.", icon=":material/info:")
        return
    run = next(r for r in rs if r.run_id == st.selectbox("Training run", [r.run_id for r in rs], key="rl_run",
                                                          format_func=lambda i: f"{i} · {next(f'{r.metrics.get('algorithm')} on {r.metrics.get('env_id')}' for r in rs if r.run_id == i)}"))
    st.session_state["rl_selected_run"] = run.run_id
    h = ra.history(run)
    if h is None:
        st.warning(f"Run {run.run_id} ended as {run.status} without a training history. {run.failure_reason or ''}", icon=":material/error:")
        return
    g1, g2, g3 = st.columns(3)
    fig = go.Figure(go.Scatter(x=h["episode"], y=h["reward"], mode="lines", name="training episode", line=dict(width=1.2, color=series_color("base"))))
    ev = h.dropna(subset=["eval_reward"])
    fig.add_trace(go.Scatter(x=ev["episode"], y=ev["eval_reward"], mode="lines+markers", name="evaluation (greedy)", line=dict(width=2, color=series_color("candidate"))))
    style_figure(fig, height=260, x_title="Episode", y_title="Episode reward")
    g1.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"rl_reward_{run.run_id}")
    fig = go.Figure(go.Scatter(x=h["episode"], y=h["loss"], mode="lines", name="critic / Q loss", line=dict(width=1.6, color=series_color("electrical"))))
    if "actor_loss" in h:
        fig.add_trace(go.Scatter(x=h["episode"], y=h["actor_loss"], mode="lines", name="actor loss", line=dict(width=1.6, color=series_color("mechanical"))))
    style_figure(fig, height=260, x_title="Episode", y_title="Loss")
    g2.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"rl_loss_{run.run_id}")
    fig = go.Figure(go.Scatter(x=h["episode"], y=h["violation_rate_10"], mode="lines", name="terminated episodes (10-episode mean)", line=dict(width=1.6, color=series_color("secondary"))))
    style_figure(fig, height=260, x_title="Episode", y_title="Constraint-violation rate")
    fig.update_yaxes(range=[0, 1])
    g3.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"rl_viol_{run.run_id}")
    eps = ra.playback_episodes(run)
    if eps:
        ep = st.select_slider("Playback actual evaluation episode (recorded by the job)", eps, value=eps[-1], key=f"rl_pb_{run.run_id}")
        pb = ra.playback(run, ep)
        tr = pb["trace"]
        t = [x["t_s"] * 1e3 for x in tr]
        out_key = "v_out_v" if "v_out_v" in tr[0] else "vc_real"
        ref_key = "v_ref_v" if "v_ref_v" in tr[0] else "v_ref_real"
        fig = go.Figure(go.Scatter(x=t, y=[x[out_key] for x in tr], mode="lines", name=out_key, line=dict(width=2, color=series_color("candidate"))))
        fig.add_trace(go.Scatter(x=t, y=[x[ref_key] for x in tr], mode="lines", name=ref_key, line=dict(width=1.4, dash="dash", color=series_color("reference"))))
        style_figure(fig, height=280, x_title="Time (ms)", y_title="Volts")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"rl_pb_fig_{run.run_id}")
        st.caption(f"Evaluation episode after {ep} training episodes (seed {pb['seed']}): reward {pb['reward']:.3f}"
                   f"{', terminated: ' + pb['termination'] if pb['termination'] else ''}; every {pb['stride']}th step shown.")


def _benchmark() -> None:
    rs = ra.runs()
    run_id = st.session_state.get("rl_selected_run")
    run = next((r for r in rs if r.run_id == run_id), rs[0] if rs else None)
    if run is None:
        st.info("No simulation has been executed for this configuration.", icon=":material/info:")
        return
    b = ra.benchmark(run)
    if b is None:
        st.info(f"Run {run.run_id} has no benchmark (it did not finish training).", icon=":material/info:")
        return
    st.markdown(f"##### {run.run_id} — same environment, same grid ({len(b['grid'])} conditions) for every controller")
    summary = b["summary"]
    rl = next(k for k in summary if k.startswith("RL"))
    rows = [{"Controller": k, "Mean episode reward": v["mean_reward"], "Median tracking RMSE (V)": v["median_tracking_rmse"],
             "Terminated / diverged": f"{v['n_terminated']}/{v['n_conditions']}"} for k, v in summary.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    loses = [k for k, v in summary.items() if k != rl and v["mean_reward"] > summary[rl]["mean_reward"]]
    if loses:
        st.warning(f"The RL policy performs worse than {', '.join(loses)} on this grid. Shown as measured.", icon=":material/trending_down:")
    with st.expander("Per condition (failed and divergent conditions included)", icon=":material/table:"):
        st.dataframe(ra.benchmark_table(b), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    if c1.button("Save policy candidate", key=f"rl_save_{run.run_id}", icon=":material/bookmark:"):
        art = ra.save_candidate(run)
        st.success(f"Saved next to the run ({art['path']}). Nothing was deployed.", icon=":material/check:")
    with c2:
        st.markdown("**Use as default**")
        gate = ra.gate_report(run)
        approval_gate(f"RL policy for {run.metrics.get('env_id')}", run.run_id, gate, lambda who, d: ra.record_decision(run, gate, who, d), key=f"rl_gate_{run.run_id}")
    provenance_table({"Run ID": run.run_id, "Algorithm": run.metrics.get("algorithm"), "Environment": run.metrics.get("env_id"), "Seed": run.seed,
                      "Configuration hash": run.config_hash, "Reproduce": run.command, "Status": b["status"]})


def render() -> None:
    page_header("Design & Control / RL Lab", "RL Lab",
                "Train DQN or DDPG agents on SPADE's Gymnasium environments and compare them with the classical controllers on the same condition grid. "
                "Policies are candidates: nothing is deployed or used as a default.",
                [source_badge_html("simulated"), badge_html("CANDIDATES ONLY", "warning", "▲", ra.NOT_AUTHORIZED)])
    t_env, t_rew, t_train, t_bench = st.tabs(["Environment", "Reward", "Training", "Benchmark"])
    with t_env:
        env_id, algo, levels = _environment()
    with t_rew:
        terms = _reward(env_id)
    with t_train:
        _training(env_id, algo, levels, terms)
    with t_bench:
        _benchmark()
