"""C1, option 1: select DPC v3 or v4 by operating point, without retraining -- TODO_CLAUDE_CODE.md C1.

Design and evaluation are separate:
- DESIGN grid (used to choose REGION): R = 5, 6, 7, 7.5, 8.0064, 8.5, 9, 10, 12 ohm x
  |v_ref| = 10, 25, 35, 50, 75, 100 V. REGION marks where v3's settled RMSE was below v4's there.
- EVALUATION points (never used to choose REGION): the C1 target points (8.0064 ohm at 50 and
  10 V; 4, 20, 1000 ohm at 50 V) and held-out points between and outside the design levels.
- LOAD / SETPOINT STEPS: a run whose load jumps 8.0064 -> 20 -> 8.0064 ohm, and one whose
  reference jumps 50 -> 75 -> 50 V, so the controller crosses REGION's boundary at run time. The
  transient after each switch is compared with v4 alone under the same steps.

C1 targets: 8 ohm <= 1.3 V; |v_ref| = 10 V <= 2.0 V; 4, 20, 1000 ohm <= 3.5 V; no divergence at any
R > R* = 3.3707 ohm.

Usage:

    python experiments/dpc_operating_point_selection.py --out-dir docs/audit/stability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driveflow.control.dpc.controller import DEFAULT_SCHEDULE_REGION, DpcController, ScheduledDpcController  # noqa: E402
from driveflow.control.dpc.reference import RotatingReference  # noqa: E402
from driveflow.sim.vsc_system import VscSystem  # noqa: E402

V3 = ROOT / "configs" / "dpc_trained_v3_closed_loop.weights.h5"
V4 = ROOT / "configs" / "dpc_trained_v4_r_range.weights.h5"
DESIGN_R = (5.0, 6.0, 7.0, 7.5, 8.0064, 8.5, 9.0, 10.0, 12.0)
DESIGN_V = (10.0, 25.0, 35.0, 50.0, 75.0, 100.0)
#: Chosen from the DESIGN grid only. Each boundary lies halfway between the last design level where
#: v3 beat v4 and the first where it lost: R between 7.0/7.5 and 8.5/9.0 ohm at 50 V, |v_ref|
#: between 50 and 75 V; at 10 V v3 won at every design load (5-12 ohm), so that band is the
#: tested range. A first rule put the boundaries ON the winning levels (7.5-8.5 ohm, 50 V); the
#: load-step run showed that with hysteresis it then never re-entered the band at the nominal
#: 50 V, so it was revised -- from the design grid, not from the evaluation points (see report).
REGION = DEFAULT_SCHEDULE_REGION  # defined next to the controller, the default since 2026-10-20
RULE_1 = {"r_min_ohm": 7.5, "r_max_ohm": 8.5, "band_v_max": 50.0, "small_v_max": 10.0, "small_r_min_ohm": 5.0, "small_r_max_ohm": 12.0}
TARGETS = [(8.0064, 50.0, 1.3), (8.0064, 10.0, 2.0), (4.0, 50.0, 3.5), (20.0, 50.0, 3.5), (1000.0, 50.0, 3.5)]
HELD_OUT_R = (3.5, 4.0, 6.5, 7.75, 8.25, 9.5, 11.0, 15.0, 20.0, 40.0, 1000.0)
HELD_OUT_V = (15.0, 30.0, 45.0, 60.0)
N_STEPS = 2000
WARMUP = 50


def settled(controller, r_ohm: float, magnitude_v: float) -> float | None:
    system, reference = VscSystem(load_resistance_ohm=r_ohm), RotatingReference(magnitude_v=magnitude_v)
    state = system.reset()
    controller.reset()
    err = np.empty(N_STEPS)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(N_STEPS):
            state = system.simulate(*controller.control(state, reference, k))
            vr = reference.at_step(k)
            err[k] = np.hypot(state.vc_real - vr[0], state.vc_imag - vr[1])
    return float(np.sqrt(np.mean(err[WARMUP:] ** 2))) if np.all(np.isfinite(err)) else None


def stepped_run(controller, schedule: list, scheduled: bool) -> dict:
    """schedule: [(start_step, r_ohm, magnitude_v), ...]. Returns the error trace summary."""
    n = schedule[-1][0] + 1000
    system = VscSystem(load_resistance_ohm=schedule[0][1])
    state = system.reset()
    controller.reset()
    err = np.empty(n)
    seg = 0
    reference = RotatingReference(magnitude_v=schedule[0][2])
    for k in range(n):
        if seg + 1 < len(schedule) and k == schedule[seg + 1][0]:
            seg += 1
            _start, r, mag = schedule[seg]
            system.r = r
            reference.magnitude_v = mag
            if scheduled:
                controller.set_load(r)
            else:
                controller.r = r
        state = system.simulate(*controller.control(state, reference, k))
        vr = reference.at_step(k)
        err[k] = np.hypot(state.vc_real - vr[0], state.vc_imag - vr[1])
    out = {"finite": bool(np.all(np.isfinite(err)))}
    for i, (start, r, mag) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else n
        w = err[start:end]
        out[f"segment{i}_R{r:g}_V{mag:g}"] = {
            "peak_error_first_100_steps_v": float(np.max(w[:100])),
            "settled_rmse_after_100_steps_v": float(np.sqrt(np.mean(w[100:] ** 2))),
        }
    if scheduled:
        out["switches"] = controller.switches
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "audit" / "stability")
    ap.add_argument("--design-json", type=Path, default=None, help="reuse a previous design-grid result")
    args = ap.parse_args(argv)

    if args.design_json and args.design_json.exists():
        design = json.loads(args.design_json.read_text())
    else:
        design = {}
        for r in DESIGN_R:
            for v in DESIGN_V:
                design[f"{r:g},{v:g}"] = [settled(DpcController(V3, r), r, v), settled(DpcController(V4, r), r, v)]
                print("design", r, v, design[f"{r:g},{v:g}"], flush=True)

    def make_sched(r):
        return ScheduledDpcController(V3, V4, r, REGION)

    evaluation = {}
    points = [(r, v) for r, v, _t in TARGETS] + [(r, v) for r in HELD_OUT_R for v in HELD_OUT_V]
    for r, v in points:
        row = {"v3": settled(DpcController(V3, r), r, v), "v4": settled(DpcController(V4, r), r, v), "scheduled": settled(make_sched(r), r, v)}
        evaluation[f"{r:g},{v:g}"] = row
        print("eval", r, v, row, flush=True)

    targets = {f"R={r:g},V={v:g} <= {t:g} V": (evaluation[f"{r:g},{v:g}"]["scheduled"] is not None and evaluation[f"{r:g},{v:g}"]["scheduled"] <= t) for r, v, t in TARGETS}
    diverged = [k for k, row in evaluation.items() if row["scheduled"] is None]

    load_steps = [(0, 8.0064, 50.0), (1000, 20.0, 50.0), (2000, 8.0064, 50.0)]
    setpoint_steps = [(0, 8.0064, 50.0), (1000, 8.0064, 75.0), (2000, 8.0064, 50.0)]
    steps = {
        "load_step": {"schedule": load_steps, "scheduled": stepped_run(make_sched(8.0064), load_steps, True), "v4_only": stepped_run(DpcController(V4, 8.0064), load_steps, False)},
        "setpoint_step": {"schedule": setpoint_steps, "scheduled": stepped_run(make_sched(8.0064), setpoint_steps, True), "v4_only": stepped_run(DpcController(V4, 8.0064), setpoint_steps, False)},
    }

    res = {"region": REGION, "rule_1_superseded": RULE_1, "design_grid_v3_v4": design, "evaluation": evaluation, "targets_met": targets, "diverged_points": diverged, "steps": steps}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dpc_operating_point_selection.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"targets_met": targets, "diverged": diverged, "steps": steps}, indent=1))


if __name__ == "__main__":
    main()
