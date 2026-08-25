"""Generates the first test dataset per INSTRUCTIONS.md A.5's closeout criterion:

    "Generar un primer dataset de prueba: escenario normal + al menos 2 tipos de falla,
    bajo controlador PI, con vibracion sintetica poblada. Verificar (notebook) que las
    frecuencias de falla inyectadas aparecen tanto en el espectro de corriente (MCSA)
    como en el de vibracion sintetica."

The frequency-domain verification itself is asserted rigorously in
tests/test_datagen.py::TestA5CloseoutCriterion (compares against a healthy-baseline run, not just
a raw peak -- see that test's docstring for why). This script is the "notebook" artifact: it
prints the same check in a readable form and, optionally, saves the dataset.

Usage:
    python experiments/generate_first_dataset.py [--out /path/to/dataset.parquet]

The output .parquet is NOT committed to the repo (Principle of design #5 -- datasets are
regenerated on demand, not versioned); pass --out only to inspect it locally.
"""

import argparse

import numpy as np
import pandas as pd

from driveflow.datagen import Scenario, export_parquet, healthy_and_faulted_grid, run_scenarios
from driveflow.datagen.runner import TAU
from driveflow.sim.vibration import bearing_frequencies as bf


def verify_fault_frequencies(df: pd.DataFrame):
    print("\n=== A.5 verification: injected fault frequencies vs. MCSA/vibration spectra ===")
    healthy = df[df["label"] == "normal"]
    for label in sorted(df["label"].unique()):
        if label == "normal":
            continue
        faulty = df[df["label"] == label]
        settled = faulty.iloc[len(faulty) // 2 :]
        f_r_hz = settled["rpm"].mean() / 60.0
        expected_hz = bf.fault_order(label, bf.KAT_DATACENTER_6203_GEOMETRY) * f_r_hz

        def spectrum(frame, column):
            settled_frame = frame.iloc[len(frame) // 2 :]
            sig = settled_frame[column].to_numpy() - settled_frame[column].mean()
            return np.fft.rfftfreq(len(sig), d=TAU), np.abs(np.fft.rfft(sig))

        cur_freqs, cur_spec = spectrum(faulty, "current_r")
        cur_peak_hz = cur_freqs[np.argmax(cur_spec)]

        def band_energy(frame, column, center_hz, half_width_hz=15.0):
            freqs, spec = spectrum(frame, column)
            mask = np.abs(freqs - center_hz) <= half_width_hz
            return float(np.sum(spec[mask] ** 2))

        vib_faulty_energy = band_energy(faulty, "acc_x", expected_hz)
        vib_healthy_energy = band_energy(healthy, "acc_x", expected_hz)

        print(f"\n{label}: expected fault frequency ~= {expected_hz:.1f} Hz (order-tracked, f_r={f_r_hz:.2f} Hz)")
        print(f"  current_r raw-spectrum peak:            {cur_peak_hz:.1f} Hz")
        print(
            f"  acc_x band energy @ fault freq: faulty={vib_faulty_energy:.3e}  "
            f"healthy={vib_healthy_energy:.3e}  ratio={vib_faulty_energy / max(vib_healthy_energy, 1e-30):.1f}x"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Optional path to save the dataset as .parquet")
    parser.add_argument("--duration-s", type=float, default=0.4)
    args = parser.parse_args()

    scenarios = healthy_and_faulted_grid(
        "driveflow_v1_test",
        fault_types=("outer_race", "inner_race"),
        # electrical_severity defaults to 8.0 Nm (matches A.4/A.5's original MCSA validation);
        # mechanical_severity defaults to the per-fault-type value calibrated against real
        # Paderborn separability in experiments/verify_vibration_separability_auc.py -- see
        # docs/patch2_retiro_modulo_C.md Sec. 4 and scenario.py's CALIBRATED_MECHANICAL_SEVERITY.
        seeds=(0,),
        duration_s=args.duration_s,
        omega_ref_rad_s=150.0,
    )
    print(f"Running {len(scenarios)} scenarios ({[s.label for s in scenarios]})...")
    runs = run_scenarios(scenarios)
    total_records = sum(len(r) for r in runs)
    print(f"Generated {total_records} records ({args.duration_s}s each @ {1/TAU:.0f} Hz).")

    if args.out:
        df = export_parquet(runs, args.out)
        print(f"Saved {args.out}")
    else:
        from driveflow.datagen.export_parquet import records_to_dataframe

        df = records_to_dataframe([r for run in runs for r in run])

    print("\nvibration_source values:", sorted(df["vibration_source"].unique()))
    print("acc_x/y/z populated (no NaN):", not df[["acc_x", "acc_y", "acc_z"]].isna().any().any())

    verify_fault_frequencies(df)


if __name__ == "__main__":
    main()
