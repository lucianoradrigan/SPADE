"""Refit the dc_motor residual threshold on steady-state healthy validation windows -- design change
of 2026-10-21 (AI-N18), declared in docs/audit/ai/dc_motor_residual_gate.md.

The forecaster weights are NOT retrained. A new run directory is written next to the source run
(``<config>/<date>_run02``) with the same weights and config, and metrics.json["residual_detector"]
refitted:
1. the gate is derived before looking at any window, from nameplate parameters only
   (ai/residual.steady_state_gate_s): current-limited acceleration from rest to DELTA_OMEGA_RAD_S
   (250 rad/s, the highest speed evaluated in E14) + 5 tau_m;
2. the 99th percentile is recomputed only on the HEALTHY validation runs recorded in the source
   run's metrics.json["val_groups"], using only windows that start >= gate_s after the run start.
The source run is left untouched.

    python experiments/refit_residual_gate.py --run configs/regressors/dc_motor_residual_pc/2026-10-21_run01 [--dataset data/diagnosis_grid_dataset.parquet]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from driveflow.ai import residual  # noqa: E402
from driveflow.ai.registry import load_run_model  # noqa: E402

DELTA_OMEGA_RAD_S = 250.0


def plant_gate() -> dict:
    """The gate from the dc_motor plant that generated the datasets (datagen.runner.build_plant)."""
    from driveflow.control.classical import PICascadeController
    from driveflow.datagen.runner import build_plant
    from driveflow.datagen.scenario import Scenario

    system = build_plant(Scenario(scenario_id="gate", omega_ref_rad_s=150.0))
    motor = system.electrical_motor.motor_parameter
    load = getattr(system.mechanical_load, "base_load", system.mechanical_load).load_parameter
    ctrl = PICascadeController(system)
    i_limit = float(system.limits[system.state_names.index("i")])
    gate = residual.steady_state_gate_s(system.mechanical_load.j_total, motor["r_a"], motor["psi_e"], ctrl.current_safety_margin * i_limit,
                                        load["a"], load["b"], DELTA_OMEGA_RAD_S)
    gate["source"] = "nameplate parameters of datagen.runner.build_plant: " + ", ".join(f"{k}={v:g}" for k, v in
                                                                                       {"j_total": system.mechanical_load.j_total, "r_a": motor["r_a"], "psi_e": motor["psi_e"], "i_limit": i_limit, "load_a": load["a"], "load_b": load["b"]}.items())
    return gate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "diagnosis_grid_dataset.parquet")
    ap.add_argument("--new-name", default=None, help="name of the new run directory (default: <date>_run02 of the source)")
    args = ap.parse_args(argv)

    src = args.run.resolve()
    model, _config, metrics = load_run_model(src, "regressor")
    old = metrics["residual_detector"]
    gate = plant_gate()
    df = pd.read_parquet(args.dataset)
    val_df = df[df["source_file"].isin(metrics["val_groups"])]
    channels, window, horizon = list(metrics["channels"]), int(metrics["input_window"]), int(metrics["horizon"])
    seg, _, labels, t0 = residual.raw_windows(val_df, channels, window, horizon, start_s=gate["gate_s"])
    pred = residual.forecast(model, seg, channels, window, metrics.get("level_scaling"))
    rmse = residual.channel_rmse(seg, pred, channels, list(old["target_channels"]), window)
    det = residual.fit_threshold(rmse, labels, list(old["target_channels"]), quantile=old["quantile"])
    det["gate"] = gate
    det["fitted_on"]["windows"] = f"validation, starting >= {gate['gate_s']:.4f} s after the run start (steady-state gate)"
    det["previous_threshold"] = {"run": src.name, "threshold": old["threshold"], "channel_scale": old["channel_scale"], "n": old["fitted_on"]["n"]}
    det["design_change"] = ("2026-10-21: threshold refitted on steady-state healthy validation windows after the E14 test results showed that start-up "
                            "windows inflated it (AI-N18). Decided after seeing test results; gate fixed from physics before refitting; no faulted window used.")

    dst = src.parent / (args.new_name or src.name.rsplit("_run", 1)[0] + "_run02")
    if dst.exists():
        sys.exit(f"{dst} exists; refusing to overwrite")
    dst.mkdir()
    for f in src.iterdir():
        if f.name != "metrics.json" and f.is_file():
            shutil.copy2(f, dst / f.name)
    new_metrics = dict(metrics, residual_detector=det, derived_from={"run": str(src.relative_to(ROOT)), "what": "same weights; residual threshold refitted with a steady-state gate",
                                                                    "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()})
    (dst / "metrics.json").write_text(json.dumps(new_metrics, indent=2, default=float))
    print(json.dumps({"new_run": str(dst.relative_to(ROOT)), "gate": gate, "threshold_old": old["threshold"], "threshold_new": det["threshold"],
                      "n_old": old["fitted_on"]["n"], "n_new": det["fitted_on"]["n"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
