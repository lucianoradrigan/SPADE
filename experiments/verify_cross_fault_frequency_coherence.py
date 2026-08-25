"""Reproduces docs/patch3_mejora_modulo_B.md's "Paso 0" and "Paso 3b" cross-fault-frequency
coherence control -- the check that ruled out a fault-type-specific torque-conditioned modulator
for Module B's excitation. Written for Patch 8 Sec. 2: a third party should be able to run this
and directly compare its printed numbers against the ones cited in the patch.

Needs a local extraction of the KAt-DataCenter (Paderborn) .rar archives (CC BY-NC, not part of
this repo -- see DATA.md). Point --dataset-root at the folder holding one subfolder per bearing
code (K001/, K002/, ..., each with its .mat runs).

Usage:
    python experiments/verify_cross_fault_frequency_coherence.py \\
        --dataset-root /path/to/BearingDataCenter/extracted

Methodology (see verify_torque_vibration_coherence.py's docstring for the general caveat: this is
a best-effort reconstruction of a computation the patch describes the RESULT of, not a byte-for-
byte replay -- parameters not explicitly stated in the patch are documented here, and this script
prints its own numbers next to the cited ones rather than asserting a silent match):

Paso 0 -- is torque<->vibration coherence specific to a fault's OWN characteristic frequency, or
does it rise generically? For each artificial-damage code group (KA* = outer race, KI* = inner
race), coherence(torque, vibration) (scipy.signal.coherence, nperseg=4096, matching Patch 3's
explicitly stated value) is evaluated in a +-2Hz window (explicitly stated in Patch 3) around each
of BPFO/BPFI/BSF (that run's own characteristic frequencies, from its measured shaft speed), then
averaged across runs in the group. Condition/run-count/code lists reuse
verify_vibration_separability_auc.py's FAULT_REAL_CODES and N15_M07_F10 condition for consistency.

Paso 3b -- does the same fixed low-frequency band (9.6-123Hz, covering the shaft-speed harmonics
these characteristic frequencies live in at this condition) already read high on HEALTHY bearings,
or is it a fault-specific effect? Coherence in that fixed band is measured on the same 12 healthy
runs used by verify_torque_vibration_coherence.py, and (as a further, direct check not explicitly
detailed as a separate computation in the patch) on the pooled KA*+KI* faulted runs in the same
fixed band, for a genuine faulted-vs-healthy comparison in identical terms.

Note on a methodology choice specific to Paso 0 (see verify_torque_vibration_coherence.py's
docstring for the general reconstruction caveat): unlike that script's broadband measurement,
Paso 0 evaluates coherence in a narrow +-2Hz window around a specific target frequency. Trying the
same "concatenate runs first" fix used there makes Paso 0's numbers WORSE (near zero), not better
-- because runs at the same nominal condition still have slightly different measured shaft speeds,
so their true characteristic frequencies don't align exactly; concatenating smears a real narrow
peak across runs instead of averaging repeated instances of it. Paso 0 therefore evaluates each
run's target frequency from that SAME run's own measured f_r_hz, then averages the resulting
per-run band-means -- reproducing the qualitative finding (control frequency >= the fault's own)
cleanly, though its absolute values run lower than cited (own-frequency coherence ~0.18-0.23 here
vs. ~0.32-0.40 cited) for the same few-segments-per-run reason noted in the other script.
"""

import argparse

import numpy as np
import scipy.signal

from driveflow.sim.vibration import bearing_frequencies as bf
from driveflow.sim.vibration.calibration import MECH_FS_HZ, list_recordings, load_recording
from verify_torque_vibration_coherence import CONDITION, HEALTHY_CODES, N_RUNS_PER_CODE, NPERSEG, _load_healthy_runs, _to_mech_rate

GEOMETRY = bf.KAT_DATACENTER_6203_GEOMETRY
FAULT_CODES = {
    "outer_race": ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09", "KA15", "KA16", "KA22", "KA30"],
    "inner_race": ["KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16", "KI17", "KI18", "KI21"],
}
N_RUNS_PER_FAULT_CODE = 2
HALF_WIDTH_HZ = 2.0  # Patch 3's stated window around each characteristic frequency
FIXED_BAND_HZ = (9.6, 123.0)  # Patch 3 Paso 3b's stated fixed band

#: Cited in docs/patch3_mejora_modulo_B.md, for direct comparison. own = the group's own fault
#: frequency; control = the other two characteristic frequencies (not this group's actual defect).
CITED_PASO0 = {
    "outer_race": {"own": ("BPFO", 0.398), "control": {"BPFI": 0.340, "BSF": 0.416}},
    "inner_race": {"own": ("BPFI", 0.315), "control": {"BPFO": 0.390, "BSF": 0.453}},
}
CITED_HEALTHY_FIXED_BAND = (0.168, 0.144, 0.183)  # mean, range_min, range_max


def _load_fault_runs(dataset_root, fault_type):
    recs = []
    for code in FAULT_CODES[fault_type]:
        paths = [p for p in list_recordings(dataset_root, code) if p.name.startswith(CONDITION)][:N_RUNS_PER_FAULT_CODE]
        recs.extend(load_recording(p) for p in paths)
    return recs


def _per_run_coherence(rec):
    """(freqs, coherence curve) for one recording, torque<->vibration at MECH_FS_HZ."""
    vib_4khz = _to_mech_rate(rec.vibration)
    n = min(len(rec.torque), len(vib_4khz))
    freqs, coh = scipy.signal.coherence(rec.torque[:n], vib_4khz[:n], fs=MECH_FS_HZ, nperseg=min(NPERSEG, n))
    return freqs, coh


def _band_mean(freqs, coh, center_hz=None, half_width_hz=None, band_hz=None):
    if band_hz is not None:
        mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    else:
        mask = np.abs(freqs - center_hz) <= half_width_hz
    return float(np.mean(coh[mask])) if mask.any() else float("nan")


def paso0(dataset_root):
    print("=== Paso 0: coherencia torque<->vibracion en la frecuencia propia vs. de control ===\n")
    print(f"{'fault_type':<12}{'freq':<6}{'role':<10}{'measured':>10}{'cited':>10}")
    for fault_type in ("outer_race", "inner_race"):
        recs = _load_fault_runs(dataset_root, fault_type)
        curves = [_per_run_coherence(r) for r in recs]
        f_r_by_rec = [r.f_r_hz for r in recs]

        for freq_name, func in [("BPFO", bf.bpfo), ("BPFI", bf.bpfi), ("BSF", bf.bsf)]:
            per_run_means = []
            for (freqs, coh), f_r in zip(curves, f_r_by_rec):
                target_hz = func(f_r, GEOMETRY)
                per_run_means.append(_band_mean(freqs, coh, center_hz=target_hz, half_width_hz=HALF_WIDTH_HZ))
            measured = float(np.mean(per_run_means))

            own_name, own_cited = CITED_PASO0[fault_type]["own"]
            if freq_name == own_name:
                role, cited = "own", own_cited
            else:
                role, cited = "control", CITED_PASO0[fault_type]["control"][freq_name]
            print(f"{fault_type:<12}{freq_name:<6}{role:<10}{measured:>10.4f}{cited:>10.3f}")
        print()
    print(
        "Si 'control' >= 'own' (BSF > BPFO propio en outer_race, o BSF > BPFI propio en inner_race), "
        "la coherencia NO es especifica al tipo de falla real -- ver conclusion de Patch 3 Paso 0.\n"
    )


def paso3b(dataset_root):
    print(f"=== Paso 3b: banda fija {FIXED_BAND_HZ[0]:.1f}-{FIXED_BAND_HZ[1]:.0f}Hz, sanos vs. fallados ===\n")
    healthy_recs = _load_healthy_runs(dataset_root)
    healthy_means = [_band_mean(*_per_run_coherence(r), band_hz=FIXED_BAND_HZ) for r in healthy_recs]
    cited_mean, cited_min, cited_max = CITED_HEALTHY_FIXED_BAND
    print(
        f"Sanos:    media medida={np.mean(healthy_means):.4f} (cited={cited_mean:.3f}), "
        f"rango medido=[{min(healthy_means):.4f}, {max(healthy_means):.4f}] (cited=[{cited_min:.3f}, {cited_max:.3f}]), "
        f"n={len(healthy_means)} corridas"
    )

    fault_recs = _load_fault_runs(dataset_root, "outer_race") + _load_fault_runs(dataset_root, "inner_race")
    fault_means = [_band_mean(*_per_run_coherence(r), band_hz=FIXED_BAND_HZ) for r in fault_recs]
    print(
        f"Fallados: media medida={np.mean(fault_means):.4f}, "
        f"rango medido=[{min(fault_means):.4f}, {max(fault_means):.4f}] (cited del Paso 0: [0.315, 0.453]), "
        f"n={len(fault_means)} corridas (KA*+KI* combinados)"
    )
    print(
        "\nSi 'Sanos' ya esta muy por encima del baseline banda ancha de Patch 2 (0.022) pero muy por "
        "debajo de 'Fallados', hay tanto un artefacto generico de armonicos de giro (presente sin "
        "falla) como un incremento real cuando hay falla -- ver conclusion de Patch 3 Paso 3b."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted KAt-DataCenter archives")
    args = parser.parse_args()
    paso0(args.dataset_root)
    paso3b(args.dataset_root)


if __name__ == "__main__":
    main()
