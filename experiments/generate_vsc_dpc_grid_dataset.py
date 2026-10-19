"""vsc_dpc forecaster dataset over a grid of operating conditions (docs/AI_TECHNICAL_AUDIT.md
AI-N11).

experiments/generate_vsc_dpc_dataset.py varies only the seed, i.e. the reference start phase: its
15 runs are phase-shifted copies of one periodic waveform (time-shifted correlation 0.9999), so the
run-grouped test split measured interpolation of that waveform. Here every run is a DIFFERENT
operating condition -- load resistance x reference magnitude -- so the run-grouped split of
experiments/train_model.py is a split by condition: test conditions are never seen in training.

R stays inside the plant's open-loop stable region (R > R* = 3.3707 ohm) and the reference
frequency at the grid's 50 Hz (the DPC network's loss assumes it, S-05).

Usage:
    python experiments/generate_vsc_dpc_grid_dataset.py --out data/vsc_dpc_grid_dataset.parquet
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "vsc_dpc_grid_dataset.parquet"
R_OHM = (4.0, 5.0, 6.0, 7.0, 8.0064, 10.0, 13.0, 16.0, 20.0)
MAGNITUDES_V = (30.0, 50.0, 75.0, 100.0)
DURATION_S = 0.3


def _simulate(job: tuple) -> pd.DataFrame:
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    i, r, m = job
    sc = Scenario(scenario_id=f"vsc_grid_R{r:g}_V{m:g}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=DURATION_S, seed=i,
                  load_resistance_ohm=r, reference_magnitude_v=m)
    df = records_to_dataframe(run_scenario(sc))
    df["grid_load_resistance_ohm"], df["grid_reference_magnitude_v"] = r, m
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    jobs = [(i, r, m) for i, (r, m) in enumerate((r, m) for r in R_OHM for m in MAGNITUDES_V)]
    print(f"{len(jobs)} runs ({len(R_OHM)} loads x {len(MAGNITUDES_V)} reference magnitudes), {DURATION_S} s each")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        df = pd.concat(pool.map(_simulate, jobs), ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows to {args.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
