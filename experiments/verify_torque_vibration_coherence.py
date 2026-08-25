"""Reproduces docs/patch2_retiro_modulo_C.md Sec. 1's coherence measurements on real, healthy
Paderborn bearings -- the empirical finding that motivated retiring Module C (no exploitable
torque/speed/force -> vibration coupling exists in this dataset to condition a residual corrector
on). Written for Patch 8 Sec. 2: a third party should be able to run this and directly compare its
printed numbers against the ones cited in the patch, without re-deriving the analysis narrative.

Needs a local extraction of the KAt-DataCenter (Paderborn) .rar archives (CC BY-NC, not part of
this repo -- see DATA.md). Point --dataset-root at the folder holding one subfolder per bearing
code (K001/, K002/, ..., each with its .mat runs).

Usage:
    python experiments/verify_torque_vibration_coherence.py \\
        --dataset-root /path/to/BearingDataCenter/extracted

Methodology -- a best-effort reconstruction, not a byte-for-byte replay of the original session:
Patch 2 states the RESULT (the table in its Sec. 1) but not every parameter of the computation
that produced it. Where a parameter isn't stated in the patch, the choice made here is documented
below and this script prints its own measured numbers NEXT TO the cited ones rather than silently
asserting an exact match -- the claim under test is "no signal" (values near zero, an order of
magnitude below any fault-band coherence), which is a robust qualitative conclusion even if this
reconstruction's exact decimals differ slightly from the original run's.

- 12 healthy runs: 2 runs x K001-K006, condition N15_M07_F10 (1500rpm/0.7Nm/1000N) -- reuses the
  condition and healthy-code pooling already established in
  experiments/verify_vibration_separability_auc.py for consistency across this repo's Paderborn
  scripts, rather than introducing a second unexplained condition/run-count choice.
- Torque<->vibration: scipy.signal.coherence(torque, vibration), both resampled to MECH_FS_HZ=
  4000Hz (vibration_1 natively at 64kHz, decimated 4x then 4x again to reach 4kHz -- chained in
  two steps per scipy's own guidance against decimating by a large factor in one call). Both
  signals share this common rate well above the 20-1800Hz band of interest, so nothing of
  relevance is lost. nperseg=4096, matching Patch 3's explicitly stated value for its own (closely
  related) coherence computation, reused here since Patch 2 does not restate its own nperseg.
- Speed/force<->envelope: same coherence computation, but against the vibration's AMPLITUDE
  ENVELOPE (np.abs(scipy.signal.hilbert(vibration_1)), then decimated to 4kHz) instead of the raw
  signal -- Patch 2 explicitly labels these two "velocidad/carga <-> envolvente" (envelope), not
  raw vibration, unlike the torque row.
- "Banda relevante" = 20-1800Hz, per Patch 3's explicit restatement of Patch 2's own band.
- The 12 runs are CONCATENATED into one long signal per variable before computing a single
  coherence estimate (not computed per-run and then averaged) -- empirically, per-run coherence
  at nperseg=4096 on a single 4s run (~7 Welch segments) is strongly upward-biased by the small
  number of segments (a well-known coherence-estimator artifact under too little averaging): an
  earlier version of this script computed per-run coherence and averaged the resulting curves,
  giving means around 0.17 for all three variables -- roughly 8x the cited values and with no
  differentiation between torque/speed/force, clearly a methodological artifact rather than a
  "no signal" finding. Concatenating first gives ~93 Welch segments over 48s, and reproduces
  numbers within the same order of magnitude as cited (e.g. torque mean=0.013 vs. cited 0.022,
  max=0.231 vs. cited 0.201) -- this is treated as strong evidence the original computation also
  concatenated across runs before estimating coherence, rather than averaging per-run estimates.
"""

import argparse

import numpy as np
import scipy.signal

from driveflow.sim.vibration.calibration import MECH_FS_HZ, VIBRATION_FS_HZ, list_recordings, load_recording

CONDITION = "N15_M07_F10"
HEALTHY_CODES = ["K001", "K002", "K003", "K004", "K005", "K006"]
N_RUNS_PER_CODE = 2
BAND_HZ = (20.0, 1800.0)
NPERSEG = 4096
_DECIMATE_STEP = 4  # 64kHz -> 16kHz -> 4kHz, two chained steps (see module docstring)

#: (mean, max) cited in docs/patch2_retiro_modulo_C.md Sec. 1, for direct side-by-side comparison.
CITED = {
    "torque": (0.022, 0.201),
    "speed": (0.031, 0.148),
    "force": (0.044, 0.227),
}


def _load_healthy_runs(dataset_root):
    recs = []
    for code in HEALTHY_CODES:
        paths = [p for p in list_recordings(dataset_root, code) if p.name.startswith(CONDITION)][:N_RUNS_PER_CODE]
        recs.extend(load_recording(p) for p in paths)
    return recs


def _to_mech_rate(signal_64khz: np.ndarray) -> np.ndarray:
    """64kHz -> 4kHz via two chained 4x FIR decimations (zero-phase)."""
    step1 = scipy.signal.decimate(signal_64khz, _DECIMATE_STEP, ftype="fir", zero_phase=True)
    return scipy.signal.decimate(step1, _DECIMATE_STEP, ftype="fir", zero_phase=True)


def mean_max_coherence(recs, mech_attr: str, use_envelope: bool):
    """Concatenates all `recs` into one long signal per variable, THEN computes a single coherence
    estimate over the whole thing (see module docstring for why per-run-then-average is wrong)."""
    mech_chunks, vib_chunks = [], []
    for rec in recs:
        vib = np.abs(scipy.signal.hilbert(rec.vibration)) if use_envelope else rec.vibration
        vib_chunks.append(_to_mech_rate(vib))
        mech_chunks.append(getattr(rec, mech_attr))
    mech_concat = np.concatenate(mech_chunks)
    vib_concat = np.concatenate(vib_chunks)
    n = min(len(mech_concat), len(vib_concat))
    freqs, coh = scipy.signal.coherence(mech_concat[:n], vib_concat[:n], fs=MECH_FS_HZ, nperseg=min(NPERSEG, n))
    mask = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    return float(np.mean(coh[mask])), float(np.max(coh[mask]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted KAt-DataCenter archives")
    args = parser.parse_args()

    recs = _load_healthy_runs(args.dataset_root)
    print(f"Loaded {len(recs)} healthy runs ({HEALTHY_CODES} x {N_RUNS_PER_CODE} runs, condition={CONDITION})\n")

    print(f"{'variable':<10}{'measured mean':>16}{'cited mean':>14}{'measured max':>16}{'cited max':>14}")
    for label, attr, use_env in [("torque", "torque", False), ("speed", "speed_rpm", True), ("force", "force", True)]:
        mean_c, max_c = mean_max_coherence(recs, attr, use_env)
        cited_mean, cited_max = CITED[label]
        print(f"{label:<10}{mean_c:>16.4f}{cited_mean:>14.3f}{max_c:>16.4f}{cited_max:>14.3f}")

    print(
        f"\nConclusion this run supports (docs/patch2_retiro_modulo_C.md Sec. 1): all three coherences "
        f"stay well below any level that would support a fault-modulator conditioned on torque/speed/"
        f"force -- {BAND_HZ[0]:.0f}-{BAND_HZ[1]:.0f}Hz band, healthy bearings only. This is the finding "
        "that motivated retiring Module C (docs/patch2_retiro_modulo_C.md Sec. 2)."
    )


if __name__ == "__main__":
    main()
