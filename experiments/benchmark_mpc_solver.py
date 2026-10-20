"""C3: CPU cost of the DC-motor MPC, original L-BFGS-B solver vs. the exact QP option --
TODO_CLAUDE_CODE.md C3, docs/SIMULATION_STABILITY_REPORT.md §5.1.

Runs four closed loops on driveflow.validation's DC bench with MpcController(qp_solver="lbfgsb")
(the default, unchanged) and MpcController(qp_solver="exact"), and reports wall time, the speed-up
and the trajectory differences (max |d omega|, max and RMS |d i|), plus the PI wall time for the
"MPC costs ~65-75x the PI" baseline. Single-threaded BLAS for both (threadpoolctl).

Usage:

    python experiments/benchmark_mpc_solver.py --out docs/audit/stability/mpc_exact_solver.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import threadpoolctl

from driveflow.validation import run_dc_closed_loop

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "startup_0_to_150_rad_s_0.6s": dict(omega_ref=150.0, duration_s=0.6),
    "speed_steps_100_200_150_rad_s_0.9s": dict(omega_ref=lambda t: 100.0 if t < 0.3 else (200.0 if t < 0.6 else 150.0), duration_s=0.9),
    "load_step_10Nm_at_0.5s_0.8s": dict(omega_ref=150.0, duration_s=0.8, load_step_nm=10.0, load_step_t_s=0.5),
    "torque_10Nm_0.3s": dict(omega_ref=0.0, duration_s=0.3, torque_ref_nm=10.0),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "audit" / "stability" / "mpc_exact_solver.json")
    args = ap.parse_args(argv)
    res = {"machine": f"{platform.processor() or platform.machine()}, Python {platform.python_version()}", "cases": {}}
    with threadpoolctl.threadpool_limits(limits=1):
        res["pi_wall_s_0.6s"] = run_dc_closed_loop("PI", 150.0, duration_s=0.6)["wall_time_s"]
        for name, kw in CASES.items():
            a = run_dc_closed_loop("MPC", **kw, controller_kwargs=dict(qp_solver="lbfgsb"))
            b = run_dc_closed_loop("MPC", **kw, controller_kwargs=dict(qp_solver="exact"))
            res["cases"][name] = {
                "wall_s_lbfgsb": a["wall_time_s"], "wall_s_exact": b["wall_time_s"], "speedup": a["wall_time_s"] / b["wall_time_s"],
                "max_abs_d_omega_rad_s": float(np.max(np.abs(a["omega"] - b["omega"]))),
                "max_abs_d_i_a": float(np.max(np.abs(a["i"] - b["i"]))), "rms_d_i_a": float(np.sqrt(np.mean((a["i"] - b["i"]) ** 2))),
                "max_i_a_lbfgsb": float(np.max(a["i"])), "max_i_a_exact": float(np.max(b["i"])),
            }
            print(name, {k: round(v, 4) for k, v in res["cases"][name].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
