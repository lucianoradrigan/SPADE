"""Simulation stability & controller-performance battery -> docs/audit/stability/.

Produces every number and figure quoted in docs/SIMULATION_STABILITY_REPORT.md: results.json
(raw), results.md (tables), and PNG figures (plotly + kaleido). Each experiment records its wall
time and peak Python-heap memory (tracemalloc). Deterministic: re-running gives the same numbers
except wall time / memory.

Usage:  python experiments/run_stability_battery.py [--out-dir docs/audit/stability] [--no-figures] [--quick]
  --quick skips the MPC runs and shortens the long simulations (for a fast sanity pass).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

warnings.simplefilter("ignore")

from driveflow.control.classical import PICascadeController  # noqa: E402
from driveflow.control.classical.pmsm_foc import EULER_NSTEPS, MOTOR_PARAMS, mtpa_id_iq  # noqa: E402
from driveflow.datagen import Scenario, run_scenario  # noqa: E402
from driveflow.sim import EulerSolver  # noqa: E402
from driveflow.sim.solvers import ScipySolveIvpSolver  # noqa: E402
from driveflow.sim.vsc_system import load_feedback_spectral_radius  # noqa: E402
from driveflow.validation import (  # noqa: E402
    build_dc_system,
    dc_motor_linear_matrix,
    euler_max_stable_speed_pmsm,
    euler_spectral_radius,
    pmsm_dq_matrix,
    run_dc_closed_loop,
    run_dc_open_loop,
    run_pmsm_current_step,
    run_pmsm_open_loop,
    run_vsc_closed_loop,
    step_response_metrics,
)
from driveflow.validation.benches import DC_LOAD_PARAMETER, pmsm_short_circuit_current  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
#: Reference categorical slots 1-4 (dataviz skill palette, light surface), fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DASHES = ["solid", "dash", "dot", "dashdot"]
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def rk45():
    return ScipySolveIvpSolver(method="RK45", rtol=1e-9, atol=1e-9)


def measured(fn, *a, **kw):
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        out = fn(*a, **kw)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return out, {"wall_time_s": round(time.perf_counter() - t0, 3), "peak_heap_mb": round(peak / 1e6, 2)}


def m2d(m):
    return {k: (None if v is None else (round(v, 6) if isinstance(v, float) else v)) for k, v in m.as_dict().items()}


# ----------------------------------------------------------------------------------- DC motor
def dc_convergence(duration=0.2):
    ref, cost_ref = measured(run_dc_closed_loop, "PI", 150.0, duration_s=duration, ode_solver=rk45())
    rows, prev = [], None
    for n in (1, 2, 4, 10, 100):
        x, cost = measured(run_dc_closed_loop, "PI", 150.0, duration_s=duration, ode_solver=EulerSolver(nsteps=n))
        ew, ei = float(np.max(np.abs(x["omega"] - ref["omega"]))), float(np.max(np.abs(x["i"] - ref["i"])))
        rows.append({"euler_substeps": n, "h_s": 1e-4 / n, "max_err_omega_rad_s": ew, "max_err_i_A": ei,
                     "ratio_vs_previous": None if prev is None else round(prev / ew, 3), **cost})
        prev = ew
    return {"reference": {"solver": "RK45 rtol=atol=1e-9", **cost_ref}, "rows": rows, "_traces": (ref, None)}


def dc_euler_boundary():
    s = build_dc_system()
    mp = s.electrical_motor.motor_parameter
    a = dc_motor_linear_matrix(mp["r_a"], mp["l_a"], mp["psi_e"], s.mechanical_load.j_total, DC_LOAD_PARAMETER["b"])
    eig = np.linalg.eigvals(a)
    tau_max = 2.0 / np.max(np.abs(eig))
    rows = []
    for f in (0.04, 0.5, 0.93, 1.05, 1.2):
        tau = f * tau_max
        r = run_dc_open_loop(0.5, duration_s=0.4, tau=tau)
        rows.append({"tau_s": tau, "tau_over_limit": f, "rho_I_plus_tauA": euler_spectral_radius(a, tau),
                     "final_omega": float(r["omega"][-1]), "finite": bool(np.all(np.isfinite(r["omega"])))})
    u = 32.5
    w_ss = (mp["psi_e"] * u / mp["r_a"] - DC_LOAD_PARAMETER["a"]) / (DC_LOAD_PARAMETER["b"] + mp["psi_e"] ** 2 / mp["r_a"])
    return {"eigenvalues": [float(e.real) for e in eig], "euler_tau_limit_s": tau_max, "analytic_open_loop_omega": w_ss, "rows": rows}


def dc_steps(controllers):
    out = []
    for c in controllers:
        for ref in (50.0, 150.0, 300.0):
            r, cost = measured(run_dc_closed_loop, c, ref, duration_s=0.6)
            out.append({"controller": c, "omega_ref": ref, **m2d(step_response_metrics(r["t"], r["omega"], ref, 0.0)), "peak_current_A": float(r["i"].max()), **cost})
    return out


def dc_small_step_vs_linear():
    s = build_dc_system()
    c = PICascadeController(s)
    psi, j, b = c.psi_e, s.mechanical_load.j_total, DC_LOAD_PARAMETER["b"]
    den = [j, b + psi * c.kp_omega, psi * c.ki_omega]
    t = np.linspace(0, 0.4, 40001)
    _, y = signal.step(signal.lti([psi * c.kp_omega, psi * c.ki_omega], den), T=t)
    lin = step_response_metrics(t, 150 + 2 * y, 152, 150)
    r = run_dc_closed_loop("PI", lambda tt: 150.0 if tt < 0.4 else 152.0, duration_s=0.8)
    sim = step_response_metrics(r["t"], r["omega"], 152, 150, t_step=0.4)
    _, yd = signal.step(signal.lti([-1.0, 0.0], den), T=t)
    rd = run_dc_closed_loop("PI", 150.0, duration_s=1.0, load_step_nm=10.0, load_step_t_s=0.5)
    return {"linear": m2d(lin), "simulated": m2d(sim), "disturbance_dip_linear": float(-10 * yd.min()),
            "disturbance_dip_simulated": float(150 - rd["omega"][rd["t"] >= 0.5].min()), "_trace": (t, y, r)}


def dc_load_step(controllers):
    rows, traces = [], {}
    for c in controllers:
        r, cost = measured(run_dc_closed_loop, c, 150.0, duration_s=1.0, load_step_nm=10.0, load_step_t_s=0.5)
        post = r["t"] >= 0.5
        dev = np.abs(r["omega"][post] - 150.0)
        band = 0.005 * 150.0
        outside = np.flatnonzero(dev > band)
        recovery = None if outside.size == 0 else (None if outside[-1] == dev.size - 1 else float(r["t"][post][outside[-1] + 1] - 0.5))
        rows.append({"controller": c, "load_step_Nm": 10.0, "max_speed_dip_rad_s": float(dev.max()), "recovery_to_0.5pct_s": recovery if outside.size else 0.0,
                     "residual_offset_rad_s": float(np.mean(dev[-1000:])), "current_after_A": float(r["i"][-1]), **cost})
        traces[c] = r
    return rows, traces


def dc_param_variation(controllers):
    rows = []
    for c in controllers:
        for name, ov in (("nominal", None), ("r_a +50%", {"r_a": 0.024}), ("l_a x2", {"l_a": 38e-6}), ("psi_e -20%", {"psi_e": 0.132}), ("j_rotor x3", {"j_rotor": 0.075})):
            r = run_dc_closed_loop(c, 150.0, duration_s=1.0, motor_parameter=ov)
            m = step_response_metrics(r["t"], r["omega"], 150.0, 0.0)
            rows.append({"controller": c, "plant": name, **m2d(m)})
    return rows


def dc_long(duration):
    r, cost = measured(run_dc_closed_loop, "PI", 150.0, duration_s=duration)
    k1 = 9999
    return {"duration_s": duration, "steps": len(r["t"]), "finite": bool(np.all(np.isfinite(r["omega"])) and np.all(np.isfinite(r["i"]))),
            "omega_at_1s": float(r["omega"][k1]), "omega_end": float(r["omega"][-1]), "i_at_1s": float(r["i"][k1]), "i_end": float(r["i"][-1]),
            "max_abs_omega_deviation_after_1s": float(np.max(np.abs(r["omega"][k1:] - 150.0))), **cost}


def reproducibility():
    rows = []
    for name, kw in (("DC PI outer_race seed 7", dict(fault_type="outer_race", seed=7, duration_s=0.2)),
                     ("DC MPC seed 0", dict(controller_type="MPC", duration_s=0.02)),
                     ("VSC DPC seed 3", dict(controller_type="DPC", plant_config_id="vsc_dpc_v1", seed=3, duration_s=0.05))):
        digests = []
        for _ in range(3):
            recs = run_scenario(Scenario(scenario_id="rep", **kw))
            arr = np.array([[r[k] for k in ("current_r", "rpm", "acc_x", "vc_real", "i_f_imag")] for r in recs], dtype=float)
            digests.append(hashlib.sha256(np.nan_to_num(arr, nan=-1.2345e300).tobytes()).hexdigest()[:16])
        rows.append({"scenario": name, "runs": 3, "identical": len(set(digests)) == 1, "sha256_prefix": digests[0]})
    return rows


# --------------------------------------------------------------------------------------- PMSM
def pmsm_open_loop():
    rows, traces = [], {}
    w_lim = euler_max_stable_speed_pmsm(MOTOR_PARAMS, 1e-4)
    w_lim_default = euler_max_stable_speed_pmsm(MOTOR_PARAMS, 1e-4 / EULER_NSTEPS, 5000.0)
    for w in (100.0, 200.0, 250.0, 260.0, 270.0, 300.0, 3e3 * np.pi / 30):
        a = pmsm_dq_matrix(w * MOTOR_PARAMS["p"], MOTOR_PARAMS["r_s"], MOTOR_PARAMS["l_d"], MOTOR_PARAMS["l_q"])
        e = run_pmsm_open_loop(w, duration_s=0.4, ode_solver=EulerSolver())
        d = run_pmsm_open_loop(w, duration_s=0.4)
        k = run_pmsm_open_loop(w, duration_s=0.4, ode_solver=ScipySolveIvpSolver(method="RK45", rtol=1e-8, atol=1e-8))
        me, mk, md = np.hypot(e["i_d"], e["i_q"])[-500:], np.hypot(k["i_d"], k["i_q"])[-500:], np.hypot(d["i_d"], d["i_q"])[-500:]
        rows.append({"omega_mech_rad_s": w, "rho_euler": euler_spectral_radius(a, 1e-4), "analytic_A": float(np.hypot(*pmsm_short_circuit_current(w))),
                     "rk45_tail_min_max_A": [float(mk.min()), float(mk.max())], "euler_tail_min_max_A": [float(me.min()), float(me.max())],
                     "default_solver_tail_min_max_A": [float(md.min()), float(md.max())]})
        traces[w] = (e, k, d)
    return {"euler_stable_up_to_rad_s": w_lim, "default_solver_nsteps": EULER_NSTEPS, "default_solver_stable_up_to_rad_s": w_lim_default, "nominal_speed_rad_s": 3e3 * np.pi / 30, "rows": rows}, traces


def pmsm_closed_loop():
    rows = []
    for i_s in (100.0, 200.0, 380.0):
        i_d, i_q = mtpa_id_iq(i_s)
        e = run_pmsm_current_step(i_d, i_q)
        k = run_pmsm_current_step(i_d, i_q, ode_solver=ScipySolveIvpSolver(method="RK45", rtol=1e-8, atol=1e-8))
        rows.append({"i_s_A": i_s, "i_d_ref": float(i_d), "i_q_ref": float(i_q), "max_euler_vs_rk45_A": float(max(np.max(np.abs(e["i_d"] - k["i_d"])), np.max(np.abs(e["i_q"] - k["i_q"])))),
                     "final_error_A": float(np.hypot(e["i_d"][-1] - i_d, e["i_q"][-1] - i_q)), "finite": bool(np.all(np.isfinite(e["i_d"])))})
    return rows


# ---------------------------------------------------------------------------------------- VSC
def vsc_sweep():
    rows = []
    for r_ohm in (2.0, 3.0, 3.5, 4.0, 6.0, 8.0064, 12.0, 20.0, 50.0, 100.0, 1000.0):
        r, cost = measured(run_vsc_closed_loop, r_ohm, duration_s=0.2)
        e = np.hypot(r["vc_real"] - r["v_ref_real"], r["vc_imag"] - r["v_ref_imag"])
        fin = bool(np.all(np.isfinite(e)))
        rows.append({"R_ohm": r_ohm, "open_loop_rho": load_feedback_spectral_radius(r_ohm), "finite": fin,
                     "settled_rmse_V": float(np.sqrt(np.mean(e[50:] ** 2))) if fin else None,
                     "max_vc_V": float(np.nanmax(np.hypot(r["vc_real"], r["vc_imag"]))) if np.any(np.isfinite(r["vc_real"])) else None, **cost})
    return rows


def vsc_long(duration):
    r, cost = measured(run_vsc_closed_loop, 8.0064, duration_s=duration)
    e = np.hypot(r["vc_real"] - r["v_ref_real"], r["vc_imag"] - r["v_ref_imag"])
    n = len(e) // 4
    return {"duration_s": duration, "finite": bool(np.all(np.isfinite(e))), "rmse_first_quarter_after_warmup": float(np.sqrt(np.mean(e[50:n] ** 2))),
            "rmse_last_quarter": float(np.sqrt(np.mean(e[-n:] ** 2))), **cost}


# ------------------------------------------------------------------------------------ figures
def _style(fig, title, x, y, log=False):
    fig.update_layout(title=dict(text=title, font=dict(color=INK, size=16)), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=MUTED, size=12),
                      width=900, height=480, margin=dict(l=70, r=30, t=60, b=60), legend=dict(bgcolor=SURFACE, font=dict(color=INK)))
    fig.update_xaxes(title=x, gridcolor=GRID, zeroline=False, linecolor=MUTED, type="log" if log else None)
    fig.update_yaxes(title=y, gridcolor=GRID, zeroline=False, linecolor=MUTED, type="log" if log else None)
    return fig


def figures(out, res, load_traces, pmsm_traces):
    import plotly.graph_objects as go

    rows = res["dc"]["convergence"]["rows"]
    f = go.Figure()
    for i, (key, name) in enumerate((("max_err_omega_rad_s", "speed error (rad/s)"), ("max_err_i_A", "current error (A)"))):
        f.add_trace(go.Scatter(x=[r["h_s"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name,
                               line=dict(color=COLORS[i], width=2, dash=DASHES[i]), marker=dict(size=9)))
    h = np.array([r["h_s"] for r in rows])
    f.add_trace(go.Scatter(x=h, y=rows[0]["max_err_i_A"] * h / h[0], mode="lines", name="slope 1 (first order)", line=dict(color=MUTED, width=1, dash="dot")))
    _style(f, "DC motor: Euler error vs RK45 reference shrinks linearly with the step", "Integration step h (s)", "Max abs error over 0.2 s", log=True).write_image(out / "dc_convergence.png")

    f = go.Figure()
    for i, (c, r) in enumerate(load_traces.items()):
        f.add_trace(go.Scatter(x=r["t"], y=r["omega"], mode="lines", name=c, line=dict(color=COLORS[i], width=2, dash=DASHES[i])))
    f.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"), annotation_text="+10 N·m load step", annotation_font_color=INK)
    f.update_xaxes(range=[0.45, 0.75])
    f.update_yaxes(range=[147.5, 150.5])
    _style(f, "DC motor: speed under a 10 N·m load-torque step", "Time (s)", "Speed (rad/s)").write_image(out / "dc_load_step.png")

    t, y, r = res["dc"]["small_step_vs_linear"]["_trace"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=t * 1e3, y=150 + 2 * y, mode="lines", name="linear design model", line=dict(color=COLORS[0], width=2)))
    m = r["t"] >= 0.4
    f.add_trace(go.Scatter(x=(r["t"][m] - 0.4) * 1e3, y=r["omega"][m], mode="lines", name="simulated PI cascade", line=dict(color=COLORS[1], width=2, dash="dash")))
    f.update_xaxes(range=[0, 150])
    _style(f, "PI speed loop, 2 rad/s step: simulation vs design model (~15 % overshoot)", "Time after step (ms)", "Speed (rad/s)").write_image(out / "dc_small_step_vs_linear.png")

    e, k, d = pmsm_traces[300.0]
    f = go.Figure()
    f.add_trace(go.Scatter(x=k["t"], y=np.hypot(k["i_d"], k["i_q"]), mode="lines", name="RK45 (reference)", line=dict(color=COLORS[0], width=2)))
    f.add_trace(go.Scatter(x=e["t"], y=np.hypot(e["i_d"], e["i_q"]), mode="lines", name="Euler, 1 step per τ = 1e-4 s", line=dict(color=COLORS[1], width=2, dash="dash")))
    f.add_trace(go.Scatter(x=d["t"], y=np.hypot(d["i_d"], d["i_q"]), mode="lines", name=f"Euler, {EULER_NSTEPS} substeps (default)", line=dict(color=COLORS[2], width=2, dash="dot")))
    f.add_hline(y=float(np.hypot(*pmsm_short_circuit_current(300.0))), line=dict(color=MUTED, width=1, dash="dot"), annotation_text="analytic", annotation_font_color=INK)
    _style(f, "PMSM at 300 rad/s, zero voltage: single-step Euler diverges, the physics does not", "Time (s)", "|i_dq| (A)").write_image(out / "pmsm_euler_vs_rk45_300rads.png")

    _vsc_figure(out, res)


def _vsc_figure(out, res):
    import plotly.graph_objects as go

    rows = [r for r in res["vsc"]["r_sweep"] if r["settled_rmse_V"] is not None]
    f = go.Figure(go.Scatter(x=[r["R_ohm"] for r in rows], y=[r["settled_rmse_V"] for r in rows], mode="lines+markers", name="settled RMSE", line=dict(color=COLORS[0], width=2), marker=dict(size=9)))
    # On log axes plotly positions shapes in data units but annotations in log10 units.
    f.add_vline(x=8.0064, line=dict(color=MUTED, width=1, dash="dot"))
    f.add_vrect(x0=4.0, x1=20.0, fillcolor=GRID, opacity=0.35, line_width=0)
    f.add_annotation(x=np.log10(8.0064), y=1, yref="paper", text="training R", showarrow=False, yanchor="bottom", font=dict(color=INK))
    f.add_annotation(x=np.log10(4.0), y=0.93, yref="paper", text="GUI slider range", showarrow=False, xanchor="left", font=dict(color=INK))
    _style(f, "VSC + DPC: tracking error vs load resistance (R < 3.37 Ω diverges, not plotted)", "Load resistance R (Ω)", "Settled RMSE of v_c (V)", log=True)
    f.update_xaxes(range=[np.log10(3.0), np.log10(1500.0)])
    f.write_image(out / "vsc_rmse_vs_r.png")


def _md(res) -> str:
    L = [f"Generated by `{res['command'].strip()}` on {res['environment']['platform']} (Python {res['environment']['python']}); total {res['total_wall_time_s']} s. Wall times and peak heap were measured under tracemalloc, which slows pure-Python loops (the MPC most); use them for relative comparison only.\n"]
    c = res["dc"]["convergence"]
    L += ["## DC motor -- step convergence (closed-loop PI, 150 rad/s, 0.2 s; reference RK45 rtol=atol=1e-9)\n",
          "| Euler substeps | h (s) | max err ω (rad/s) | max err i (A) | error ratio vs previous row (step ratio 2, 2, 2.5, 10) | wall (s) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['euler_substeps']} | {r['h_s']:.0e} | {r['max_err_omega_rad_s']:.3e} | {r['max_err_i_A']:.3e} | {r['ratio_vs_previous'] or '-'} | {r['wall_time_s']} |" for r in c["rows"]]
    b = res["dc"]["euler_boundary"]
    L += [f"\n## DC motor -- Euler stability boundary (open loop, duty 0.5)\n\nEigenvalues {['%.1f' % e for e in b['eigenvalues']]} 1/s; predicted limit τ < {b['euler_tau_limit_s'] * 1e3:.3f} ms; analytic ω_ss = {b['analytic_open_loop_omega']:.3f} rad/s.\n",
          "| τ / limit | τ (ms) | ρ(I+τA) | final ω (rad/s) | finite |", "|---|---|---|---|---|"]
    L += [f"| {r['tau_over_limit']} | {r['tau_s'] * 1e3:.3f} | {r['rho_I_plus_tauA']:.4f} | {r['final_omega']:.4g} | {r['finite']} |" for r in b["rows"]]
    L += ["\n## DC motor -- reference steps from standstill (0.6 s)\n", "| Controller | ω_ref | rise (s) | settling 2 % (s) | overshoot % | SSE (rad/s) | RMSE | max |e| | peak i (A) | wall (s) | peak heap (MB) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['controller']} | {r['omega_ref']:.0f} | {r['rise_time_s']} | {r['settling_time_s']} | {r['overshoot_pct']:.2f} | {r['steady_state_error']:.2e} | {r['rmse']:.2f} | {r['max_abs_error']:.1f} | {r['peak_current_A']:.1f} | {r['wall_time_s']} | {r['peak_heap_mb']} |" for r in res["dc"]["steps"]]
    s = res["dc"]["small_step_vs_linear"]
    L += ["\n## PI vs its linear design model (2 rad/s step at 150 rad/s; 10 N·m load step)\n", "| Quantity | linear model | simulation |", "|---|---|---|"]
    for k in ("rise_time_s", "settling_time_s", "overshoot_pct"):
        L.append(f"| {k} | {s['linear'][k]:.4g} | {s['simulated'][k]:.4g} |")
    L.append(f"| load-step speed dip (rad/s) | {s['disturbance_dip_linear']:.4g} | {s['disturbance_dip_simulated']:.4g} |")
    L += ["\n## DC motor -- 10 N·m load-torque step at t = 0.5 s (ω_ref = 150 rad/s)\n", "| Controller | max dip (rad/s) | recovery into ±0.5 % (s) | residual offset (rad/s) | i after (A) | wall (s) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['controller']} | {r['max_speed_dip_rad_s']:.3f} | {'never left the band' if r['recovery_to_0.5pct_s'] == 0.0 else ('did not recover' if r['recovery_to_0.5pct_s'] is None else format(r['recovery_to_0.5pct_s'], '.4f'))} | {r['residual_offset_rad_s']:.3e} | {r['current_after_A']:.2f} | {r['wall_time_s']} |" for r in res["dc"]["load_step"]]
    L += ["\n## DC motor -- plant parameter drift (controller tuned on the nominal plant, 150 rad/s, 1 s)\n", "| Controller | Plant | rise (s) | settling (s) | overshoot % | SSE (rad/s) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['controller']} | {r['plant']} | {r['rise_time_s']} | {r['settling_time_s']} | {r['overshoot_pct']:.2f} | {r['steady_state_error']:.2e} |" for r in res["dc"]["param_variation"]]
    lg = res["dc"]["long"]
    L.append(f"\n## DC motor -- long run\n\n{lg['duration_s']} s ({lg['steps']} steps), finite={lg['finite']}; ω(1 s)={lg['omega_at_1s']!r}, ω(end)={lg['omega_end']!r}; i(1 s)={lg['i_at_1s']!r}, i(end)={lg['i_end']!r}; max |ω-150| after 1 s = {lg['max_abs_omega_deviation_after_1s']:.2e}; wall {lg['wall_time_s']} s, peak heap {lg['peak_heap_mb']} MB.\n")
    L += ["## Reproducibility (3 repeated runs, same parameters and seed)\n", "| Scenario | identical | SHA-256 prefix |", "|---|---|---|"]
    L += [f"| {r['scenario']} | {r['identical']} | `{r['sha256_prefix']}` |" for r in res["reproducibility"]]
    p = res["pmsm"]["open_loop"]
    L += [f"\n## PMSM -- open-loop dq dynamics at held speed, zero voltage (0.4 s)\n\nOne Euler step per τ=1e-4 s is stable up to {p['euler_stable_up_to_rad_s']:.1f} rad/s; the default solver ({p['default_solver_nsteps']} substeps) up to {p['default_solver_stable_up_to_rad_s']:.1f} rad/s; motor nominal speed {p['nominal_speed_rad_s']:.1f} rad/s.\n",
          "| ω (rad/s) | ρ(I+τA), 1 step | analytic |i| (A) | RK45 tail min/max (A) | Euler 1 step tail min/max (A) | default solver tail min/max (A) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['omega_mech_rad_s']:.0f} | {r['rho_euler']:.5f} | {r['analytic_A']:.2f} | {r['rk45_tail_min_max_A'][0]:.2f} / {r['rk45_tail_min_max_A'][1]:.2f} | {r['euler_tail_min_max_A'][0]:.4g} / {r['euler_tail_min_max_A'][1]:.4g} | {r['default_solver_tail_min_max_A'][0]:.2f} / {r['default_solver_tail_min_max_A'][1]:.2f} |" for r in p["rows"]]
    L += ["\n## PMSM -- closed-loop MTPA current steps (30 ms, rotor near standstill)\n", "| i_s (A) | max |Euler − RK45| (A) | final |error| (A) | finite |", "|---|---|---|---|"]
    L += [f"| {r['i_s_A']:.0f} | {r['max_euler_vs_rk45_A']:.4f} | {r['final_error_A']:.3f} | {r['finite']} |" for r in res["pmsm"]["closed_loop"]]
    L += ["\n## VSC + DPC -- load-resistance sweep (0.2 s, |v_ref| = 50 V)\n", "| R (Ω) | open-loop ρ | finite | settled RMSE (V) | max |v_c| (V) | wall (s) |", "|---|---|---|---|---|---|"]
    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    L += [f"| {r['R_ohm']} | {r['open_loop_rho']:.4f} | {r['finite']} | {fmt(r['settled_rmse_V'], '.3f')} | {fmt(r['max_vc_V'], '.4g')} | {r['wall_time_s']} |" for r in res["vsc"]["r_sweep"]]
    v = res["vsc"]["long"]
    L.append(f"\nLong run at R = 8.0064 Ω: {v['duration_s']} s, finite={v['finite']}, RMSE first quarter {v['rmse_first_quarter_after_warmup']:.3f} V vs last quarter {v['rmse_last_quarter']:.3f} V; wall {v['wall_time_s']} s.\n")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "docs" / "audit" / "stability")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="re-render results.md (and the VSC figure) from an existing results.json")
    args = ap.parse_args(argv)
    if args.render_only:
        res = json.loads((args.out_dir / "results.json").read_text())
        (args.out_dir / "results.md").write_text(_md(res))
        if not args.no_figures:
            _vsc_figure(args.out_dir, res)
        return
    t0 = time.time()
    ctrls = ["PI"] if args.quick else ["PI", "MPC"]
    import tensorflow as tf

    res = {"command": "python experiments/run_stability_battery.py " + " ".join(argv or []),
           "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__}}
    print("DC convergence...", flush=True)
    res["dc"] = {"convergence": dc_convergence(), "euler_boundary": dc_euler_boundary()}
    print("DC steps...", flush=True)
    res["dc"]["steps"] = dc_steps(ctrls)
    res["dc"]["small_step_vs_linear"] = dc_small_step_vs_linear()
    load_rows, load_traces = dc_load_step(ctrls)
    res["dc"]["load_step"] = load_rows
    print("DC parameter drift...", flush=True)
    res["dc"]["param_variation"] = dc_param_variation(ctrls)
    print("DC long run...", flush=True)
    res["dc"]["long"] = dc_long(5.0 if args.quick else 60.0)
    res["reproducibility"] = reproducibility()
    print("PMSM...", flush=True)
    pol, pmsm_traces = pmsm_open_loop()
    res["pmsm"] = {"open_loop": pol, "closed_loop": pmsm_closed_loop()}
    print("VSC...", flush=True)
    res["vsc"] = {"r_sweep": vsc_sweep(), "long": vsc_long(0.5 if args.quick else 2.0)}
    res["total_wall_time_s"] = round(time.time() - t0, 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures(args.out_dir, res, load_traces, pmsm_traces)
    res["dc"]["convergence"].pop("_traces")
    res["dc"]["small_step_vs_linear"].pop("_trace")
    (args.out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    md = _md(res)
    (args.out_dir / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
