"""R2 evidence: healthy and faulted runs at AI Diagnostics' defaults (0.82 s, PI, seed 0, no noise) for the old and new
PC and RPi5 classifiers, at 80, 220, 250, 300 and 350 rad/s; the training-grid window (as the page)."""
import json, sys
import pandas as pd
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.viz.adapters import ai_adapter as da
from driveflow.viz.adapters.simulation_adapter import DC_PLANT
RUNS = {"PC old": ("pc", "configs/classifiers/pc_server_5class_w2048/2026-11-04_run01"), "PC new": ("pc", "configs/classifiers/pc_server_5class_w2048/2026-11-10_run01"),
        "RPi5 old": ("rpi5", "configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01"), "RPi5 new": ("rpi5", "configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-10_run01")}
out = []
for fault in (None, "outer_race", "inner_race", "ball", "cage"):
    for w in (80.0, 220.0, 250.0, 300.0, 350.0):
        if fault and w not in (300.0,):
            continue
        kw = dict(scenario_id="ai_tab_sample_dc", plant_config_id=DC_PLANT, controller_type="PI", fault_type=fault, omega_ref_rad_s=w, duration_s=0.82, seed=0, electrical_noise_pct=0.0, mechanical_noise_pct=0.0)
        if fault:
            kw.update(electrical_severity=8.0, mechanical_severity=float(CALIBRATED_MECHANICAL_SEVERITY.get(fault, 0.02)))
        df = pd.DataFrame.from_records(run_scenario(Scenario(**kw)))
        for name, (tier, run) in RUNS.items():
            a, _, _ = da.assess_training_window("dc_motor", df, tier, run_dir=run)
            out.append({"case": fault or "healthy", "speed_rad_s": w, "model": name, "state": a.state, "top_class": a.top_class, "confidence": round(float(a.confidence), 3)})
print(json.dumps(out))
