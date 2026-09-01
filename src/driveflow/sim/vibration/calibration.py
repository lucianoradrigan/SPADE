"""Calibration of Module B's {omega_n,k, zeta_k, gain_k} against real PSDs from the
KAt-DataCenter (Paderborn) dataset, per docs/addendum_vibracion_v1.md Sec. 3/9.

Licensing note (Addendum Sec. 9 / Principle of design #5): the dataset is CC BY-NC. This module
only ever reads it from a local path the caller supplies (``dataset_root``) -- nothing here
downloads, redistributes or commits any of it. Only the *fitted* {omega_n,k, zeta_k, gain_k}
scalars (small numbers, not the recordings) are meant to be persisted/versioned.

Known limitation: KAt-DataCenter records a single accelerometer channel (``vibration_1``, one
radial axis) per run, not 3-axis (x, y, z). Calibration here therefore only fits one axis; the
same fitted Mode set is reused across x/y/z by the caller (see ``calibrate_module_b``'s
``axis_gains``) until/unless a 3-axis reference becomes available.

File layout expected under ``dataset_root`` (one subfolder per bearing code, as produced by
extracting the KAt-DataCenter .rar archives)::

    <dataset_root>/K001/N15_M07_F10_K001_1.mat
    <dataset_root>/K001/N15_M07_F10_K001_2.mat
    ...

Filename convention (Paderborn): ``N<speed_code>_M<load_code>_F<force_code>_<bearing_code>_<run>.mat``.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.signal import csd, find_peaks, welch

from .modal_model import Mode

#: Sample rates hard-coded in every KAt-DataCenter .mat file (verified against the raw data:
#: 4 seconds of recording -> 16001 mechanical samples / 256001 vibration+current samples).
MECH_FS_HZ = 4000.0
VIBRATION_FS_HZ = 64000.0

_FILENAME_RE = re.compile(r"^N(?P<speed_code>\d+)_M(?P<load_code>\d+)_F(?P<force_code>\d+)_(?P<bearing_code>[A-Za-z]+\d+)_(?P<run>\d+)$")

#: KAt-DataCenter's healthy (undamaged, run-in) bearing codes. Module B's modal filter must only
#: ever be fit against these -- it represents the structural response of the housing/bearing
#: itself (docs/patch3_mejora_modulo_B.md, "Paso 2"), and mixing in fault-energy-contaminated
#: segments would conflate "structural resonance" with "fault content", which the two dedicated
#: excitation sources (background_noise.py, fault_impulses.py) are meant to keep separate.
HEALTHY_BEARING_CODES = ["K001", "K002", "K003", "K004", "K005", "K006"]


def _assert_healthy_only(recordings, caller: str):
    bad = sorted({rec.bearing_code for rec in recordings if rec.bearing_code not in HEALTHY_BEARING_CODES})
    if bad:
        raise ValueError(
            f"{caller} received recordings from non-healthy bearing(s) {bad}. Module B's modal "
            f"filter (and the background-noise gain) must be calibrated only against "
            f"{HEALTHY_BEARING_CODES} -- see docs/patch3_mejora_modulo_B.md, 'Paso 2'."
        )


@dataclass
class PaderbornRecording:
    """One loaded .mat measurement (4 s run)."""

    bearing_code: str
    speed_code: str
    load_code: str
    force_code: str
    run: int
    vibration: np.ndarray  # g (or sensor-native unit; KAt-DataCenter does not label it), 64 kHz
    speed_rpm: np.ndarray  # min^-1, 4 kHz
    torque: np.ndarray  # Nm, 4 kHz
    force: np.ndarray  # N, 4 kHz -- radial load cell, see the "force" channel in the raw .mat.
    path: Path

    @property
    def f_r_hz(self) -> float:
        """Mean shaft rotational frequency over the run."""
        return float(np.mean(self.speed_rpm)) / 60.0


def parse_filename(mat_path) -> dict:
    stem = Path(mat_path).stem
    match = _FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"{mat_path} does not follow the KAt-DataCenter Nxx_Mxx_Fxx_<code>_<run>.mat convention")
    return match.groupdict()


def load_recording(mat_path) -> PaderbornRecording:
    """Loads one KAt-DataCenter .mat file into a PaderbornRecording."""
    mat_path = Path(mat_path)
    meta = parse_filename(mat_path)
    raw = loadmat(mat_path, simplify_cells=True)
    inner = raw[mat_path.stem]
    channels = {ch["Name"]: np.asarray(ch["Data"], dtype=float) for ch in inner["Y"]}
    return PaderbornRecording(
        bearing_code=meta["bearing_code"],
        speed_code=meta["speed_code"],
        load_code=meta["load_code"],
        force_code=meta["force_code"],
        run=int(meta["run"]),
        vibration=channels["vibration_1"],
        speed_rpm=channels["speed"],
        torque=channels["torque"],
        force=channels["force"],
        path=mat_path,
    )


def list_recordings(dataset_root, bearing_code: str):
    """Paths of every .mat run available for one bearing code under dataset_root."""
    folder = Path(dataset_root) / bearing_code
    return sorted(folder.glob(f"*_{bearing_code}_*.mat"))


def compute_psd(signal: np.ndarray, fs: float, nperseg: int = 8192):
    """Welch PSD of a vibration signal. Returns (freqs_hz, psd)."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    return freqs, psd


def averaged_psd(recordings, fs: float = VIBRATION_FS_HZ, nperseg: int = 8192):
    """Welch PSD of ``vibration``, averaged across multiple recordings (reduces noise, per-run
    variance)."""
    psds = []
    freqs = None
    for rec in recordings:
        freqs, psd = compute_psd(rec.vibration, fs=fs, nperseg=nperseg)
        psds.append(psd)
    return freqs, np.mean(psds, axis=0)


def _modal_bank_psd(freqs_hz: np.ndarray, params: np.ndarray, n_modes: int) -> np.ndarray:
    """Analytic force->acceleration PSD of a bank of n_modes independent modes under a shared
    white-noise excitation (PSD level folded into each mode's gain), evaluated at freqs_hz.

    For a single mode x'' + 2*zeta*wn*x' + wn**2*x = gain*F(t), the force->acceleration transfer
    function is H(w) = -gain*w**2 / (wn**2 - w**2 + j*2*zeta*wn*w), so under a unit-PSD white-noise
    F(t), the acceleration PSD contribution is |H(w)|**2. params packs [wn_1, zeta_1, gain_1, ...].
    """
    w = 2.0 * np.pi * freqs_hz
    total = np.zeros_like(freqs_hz)
    for k in range(n_modes):
        wn, zeta, gain = params[3 * k : 3 * k + 3]
        denom = (wn**2 - w**2) ** 2 + (2.0 * zeta * wn * w) ** 2
        total += (gain**2 * w**4) / np.maximum(denom, 1e-30)
    return total


def _pack_params(modes):
    return np.concatenate([[2 * np.pi * m.natural_freq_hz, m.damping_ratio, m.gain] for m in modes])


def _unpack_params(params, n_modes):
    modes = []
    for k in range(n_modes):
        wn, zeta, gain = params[3 * k : 3 * k + 3]
        modes.append(Mode(natural_freq_hz=wn / (2 * np.pi), damping_ratio=float(zeta), gain=float(gain)))
    return modes


def _initial_guess_from_peaks(freqs_hz, psd, n_modes, freq_bounds):
    mask = (freqs_hz >= freq_bounds[0]) & (freqs_hz <= freq_bounds[1])
    f_band, psd_band = freqs_hz[mask], psd[mask]
    peak_idx, _ = find_peaks(psd_band, distance=max(1, len(psd_band) // (4 * n_modes)))
    if len(peak_idx) == 0:
        peak_freqs = np.linspace(freq_bounds[0], freq_bounds[1], n_modes)
    else:
        order = np.argsort(psd_band[peak_idx])[::-1][:n_modes]
        peak_freqs = np.sort(f_band[peak_idx][order])
        if len(peak_freqs) < n_modes:
            filler = np.linspace(freq_bounds[0], freq_bounds[1], n_modes - len(peak_freqs))
            peak_freqs = np.sort(np.concatenate([peak_freqs, filler]))
    gain0 = np.sqrt(np.median(psd_band[psd_band > 0]))
    modes0 = [Mode(natural_freq_hz=float(f), damping_ratio=0.02, gain=float(gain0)) for f in peak_freqs]
    return modes0


def fit_modes_to_psd(
    freqs_hz: np.ndarray,
    psd_measured: np.ndarray,
    n_modes: int,
    freq_bounds=(200.0, 10000.0),
    damping_bounds=(1e-3, 0.2),
):
    """Fits an n_modes-mode bank to a measured acceleration PSD by nonlinear least squares on the
    log-PSD (peaks span orders of magnitude, so fitting in log domain weights them comparably).

    Args:
        freqs_hz, psd_measured: Output of compute_psd/averaged_psd on a real recording.
        n_modes: Number of modes to fit.
        freq_bounds: (min_hz, max_hz) frequency range considered for both peak-picking the
            initial guess and computing the fit residual.
        damping_bounds: Bounds on each mode's damping ratio.

    Returns:
        list[Mode]: The fitted modes, directly usable in a ModalAxis.
    """
    mask = (freqs_hz >= freq_bounds[0]) & (freqs_hz <= freq_bounds[1])
    f_fit, psd_fit = freqs_hz[mask], np.maximum(psd_measured[mask], 1e-30)

    modes0 = _initial_guess_from_peaks(freqs_hz, psd_measured, n_modes, freq_bounds)
    x0 = _pack_params(modes0)

    wn_lo, wn_hi = 2 * np.pi * freq_bounds[0], 2 * np.pi * freq_bounds[1]
    lower = np.tile([wn_lo, damping_bounds[0], 1e-6], n_modes)
    upper = np.tile([wn_hi, damping_bounds[1], np.inf], n_modes)

    def residuals(params):
        psd_model = np.maximum(_modal_bank_psd(f_fit, params, n_modes), 1e-30)
        return np.log10(psd_model) - np.log10(psd_fit)

    result = least_squares(residuals, x0, bounds=(lower, upper))
    return _unpack_params(result.x, n_modes)


def psd_log_rmse(freqs_hz, psd_measured, modes, freq_bounds=(200.0, 10000.0)) -> float:
    """Validation metric: RMSE in log10(PSD) between a fitted mode bank and a (typically holdout)
    measured PSD -- use with a recording NOT used for fitting, per the addendum's Sec. 9
    validation requirement."""
    mask = (freqs_hz >= freq_bounds[0]) & (freqs_hz <= freq_bounds[1])
    f_eval, psd_eval = freqs_hz[mask], np.maximum(psd_measured[mask], 1e-30)
    params = _pack_params(modes)
    psd_model = np.maximum(_modal_bank_psd(f_eval, params, len(modes)), 1e-30)
    return float(np.sqrt(np.mean((np.log10(psd_model) - np.log10(psd_eval)) ** 2)))


def calibrate_module_b(
    fit_recordings,
    n_modes: int = 4,
    freq_bounds=(200.0, 10000.0),
    damping_bounds=(1e-3, 0.2),
    nperseg: int = 8192,
):
    """End-to-end calibration: averages the PSD of ``fit_recordings`` and fits an n_modes bank.

    Args:
        fit_recordings: list[PaderbornRecording] to calibrate against (e.g. several healthy runs
            of the same/different bearing codes -- NOT the holdout used for validation).

    Returns:
        (list[Mode], freqs_hz, psd_measured): the fitted modes plus the PSD they were fit to
        (handy for plotting/inspection).
    """
    _assert_healthy_only(fit_recordings, "calibrate_module_b")
    freqs_hz, psd_measured = averaged_psd(fit_recordings, nperseg=nperseg)
    modes = fit_modes_to_psd(freqs_hz, psd_measured, n_modes, freq_bounds, damping_bounds)
    return modes, freqs_hz, psd_measured


def fit_background_noise_gain(fit_recordings, modes, n_substeps: int = 1) -> float:
    """Fits BackgroundNoiseGenerator's overall RMS gain so that, once shaped by the calibrated
    modal filter (the SAME modes fit by calibrate_module_b -- see this module's "Paso 2"), the
    resulting synthetic acceleration's RMS matches real healthy vibration's RMS.

    Args:
        fit_recordings: list[PaderbornRecording], healthy bearings only (see HEALTHY_BEARING_CODES).
        modes: The Module B modes already fit by calibrate_module_b (same recordings, ideally).
        n_substeps: Substeps per output sample for ModalAxis -- 1 is enough here since white
            noise has no structure to lose between adjacent samples (unlike a real force signal
            with content that could alias if under-sampled).

    Returns:
        float: the gain to pass to BackgroundNoiseGenerator(gain=...).
    """
    _assert_healthy_only(fit_recordings, "fit_background_noise_gain")
    from .modal_model import ModalAxis  # local import: avoids a module-level cycle with modal_model

    real_rms = float(np.sqrt(np.mean(np.concatenate([rec.vibration for rec in fit_recordings]) ** 2)))

    axis = ModalAxis([Mode(m.natural_freq_hz, m.damping_ratio, m.gain) for m in modes])
    dt_sub = 1.0 / VIBRATION_FS_HZ
    rng = np.random.default_rng(0)
    n_probe = 200_000  # a few seconds' worth at 64 kHz, enough for a stable RMS estimate
    white_noise = rng.normal(0.0, 1.0, size=n_probe)
    synthetic = np.array([axis.step(white_noise[k : k + 2], dt_sub) for k in range(n_probe - 1)])
    synthetic_rms_at_unit_gain = float(np.sqrt(np.mean(synthetic**2)))

    if synthetic_rms_at_unit_gain < 1e-30:
        raise ValueError("Modal filter produced ~zero output for unit-gain white noise input -- check the fitted modes.")
    return real_rms / synthetic_rms_at_unit_gain


# ---------------------------------------------------------------------------------------------
# Empirical transfer function (FRF) calibration -- phase-preserving, unlike the PSD fit above.
#
# fit_modes_to_psd/calibrate_module_b fit the *magnitude* of the vibration spectrum assuming an
# (unverified) white-noise force input. That throws away phase: it says "the frequency content
# has a similar shape" but nothing about *when* a given torque fluctuation shows up in the
# vibration signal. Driving those modes with the REAL torque(t) and comparing sample-by-sample
# to the real vibration(t) confirmed this empirically -- see experiments/_archive/train_module_c.py
# and experiments/_archive/train_module_c_envelope.py's reports (archived, not runnable -- Patch 2
# retired Module C; moved out of the active experiments/ tree in Patch 11, see
# docs/patch11_archivado_modulo_c.md): even the best-fit scalar gain correction gave
# 0.0% RMSE improvement over no correction, i.e. Module B's raw output is essentially
# uncorrelated with reality in time, only similar in aggregate spectral shape.
#
# The fix is standard system identification: estimate the EMPIRICAL frequency response function
# H(f) = Pxy(f)/Pxx(f) between the recording's own simultaneous torque(t) and vibration(t)
# (Pxy = cross-spectral density, Pxx = torque's auto-spectral density), which is complex --
# magnitude AND phase -- then fit Module B's analytic torque->acceleration transfer function to
# that complex target instead of to a magnitude-only PSD. This directly calibrates "how does
# THIS real torque signal turn into THIS real vibration signal", which a white-noise-driven PSD
# fit never established.
#
# Bandwidth caveat: torque is only sampled at MECH_FS_HZ (4 kHz) in KAt-DataCenter, i.e. content
# above 2 kHz (its Nyquist) is absent from that channel by construction -- the FRF can only be
# identified up to that limit, well below some of the PSD-fitted modes (up to ~7.8 kHz). This is
# a real ceiling on what phase-accurate calibration can achieve with this dataset's mechanical
# channel, not a fitting shortcoming.
#
# OUTCOME (Patch 3, docs/patch3_mejora_modulo_B.md): this also failed -- fit_modes_to_frf's own
# normalized error was ~95% even on the TRAINING recordings, before any holdout check. The root
# cause turned out to be even more basic than "PSD fit throws away phase": torque and vibration
# have essentially zero coherence in this dataset for healthy bearings (mean 0.022, broadband;
# ~0.03-0.04 for speed/force too), so there was never a real transfer function to identify,
# phase-aware or not. Patch 3 additionally checked coherence specifically at the bearing fault
# frequencies (BPFO/BPFI/BSF/FTF) on FAULTED bearings and found it elevated (0.3-0.45) but NOT
# fault-type-specific (BSF reads as high as a bearing's own BPFO even when only the outer race is
# damaged) -- ruling out a fault-conditioned torque coupling too. Module B's excitation is
# therefore torque-free (sim/vibration/background_noise.py + fault_impulses.py). This FRF code is
# kept for reference/possible reuse against a richer dataset -- calibrate_module_b() and
# fit_background_noise_gain() above are what the live pipeline actually calls.
# ---------------------------------------------------------------------------------------------


def _torque_at_vibration_rate(recording: PaderbornRecording):
    """Upsamples the recording's torque (MECH_FS_HZ) onto the vibration signal's own sample
    times (VIBRATION_FS_HZ) via linear interpolation. Introduces no spurious content above
    MECH_FS_HZ/2 (that band is simply absent from the source channel)."""
    t_mech = np.arange(len(recording.torque)) / MECH_FS_HZ
    t_vib = np.arange(len(recording.vibration)) / VIBRATION_FS_HZ
    return np.interp(t_vib, t_mech, recording.torque)


def empirical_frf(torque: np.ndarray, vibration: np.ndarray, fs: float = VIBRATION_FS_HZ, nperseg: int = 8192):
    """Empirical torque->acceleration frequency response function H(f) = Pxy(f)/Pxx(f).

    Args:
        torque: Torque signal, same length and sample rate as vibration (see
            ``_torque_at_vibration_rate``).
        vibration: Measured acceleration signal.

    Returns:
        (freqs_hz, H): H is complex, shape matching freqs_hz.
    """
    freqs, p_xy = csd(torque, vibration, fs=fs, nperseg=min(nperseg, len(torque)))
    _, p_xx = csd(torque, torque, fs=fs, nperseg=min(nperseg, len(torque)))
    return freqs, p_xy / np.where(np.abs(p_xx) > 1e-30, p_xx, 1e-30)


def averaged_frf(recordings, fs: float = VIBRATION_FS_HZ, nperseg: int = 8192):
    """Empirical FRF averaged (real/imag parts separately) across several recordings."""
    freqs = None
    h_sum = None
    for rec in recordings:
        torque_up = _torque_at_vibration_rate(rec)
        freqs, h = empirical_frf(torque_up, rec.vibration, fs=fs, nperseg=nperseg)
        h_sum = h.copy() if h_sum is None else h_sum + h
    return freqs, h_sum / len(recordings)


def _modal_bank_frf(freqs_hz: np.ndarray, params: np.ndarray, n_modes: int) -> np.ndarray:
    """Analytic torque->acceleration transfer function of an n_modes mode bank (complex).

    For mode k, x_k'' + 2*zeta_k*wn_k*x_k' + wn_k**2*x_k = gain_k*Torque(t) (a single free scale
    per mode, avoiding a redundant/degenerate extra coupling-gain parameter), so
    Accel(jw)/Torque(jw) = -gain_k*w**2 / (wn_k**2 - w**2 + j*2*zeta_k*wn_k*w), summed over modes.
    Historical/exploratory (see this module's FRF section header) -- not used by the live
    calibration pipeline.
    """
    w = 2.0 * np.pi * freqs_hz
    total = np.zeros_like(freqs_hz, dtype=complex)
    for k in range(n_modes):
        wn, zeta, gain = params[3 * k : 3 * k + 3]
        denom = (wn**2 - w**2) + 1j * (2.0 * zeta * wn * w)
        total += -gain * w**2 / denom
    return total


def fit_modes_to_frf(
    freqs_hz: np.ndarray,
    h_measured: np.ndarray,
    n_modes: int,
    freq_bounds=(20.0, 1800.0),
    damping_bounds=(1e-3, 0.3),
):
    """Fits an n_modes mode bank to a measured (complex) torque->acceleration FRF by nonlinear
    least squares on the stacked real/imaginary residuals (phase-aware, unlike fit_modes_to_psd).

    Args:
        freqs_hz, h_measured: Output of empirical_frf/averaged_frf.
        freq_bounds: Defaults to (20, 1800) Hz -- stays under the torque channel's ~2 kHz Nyquist
            (see module docstring); pass a narrower/explicit range if the mechanical channel's
            own Nyquist differs.

    Returns:
        list[Mode].
    """
    mask = (freqs_hz >= freq_bounds[0]) & (freqs_hz <= freq_bounds[1])
    f_fit, h_fit = freqs_hz[mask], h_measured[mask]

    modes0 = _initial_guess_from_peaks(freqs_hz, np.abs(h_measured) ** 2, n_modes, freq_bounds)
    x0 = _pack_params(modes0)

    wn_lo, wn_hi = 2 * np.pi * freq_bounds[0], 2 * np.pi * freq_bounds[1]
    lower = np.tile([wn_lo, damping_bounds[0], -np.inf], n_modes)
    upper = np.tile([wn_hi, damping_bounds[1], np.inf], n_modes)

    def residuals(params):
        h_model = _modal_bank_frf(f_fit, params, n_modes)
        diff = h_model - h_fit
        return np.concatenate([diff.real, diff.imag])

    result = least_squares(residuals, x0, bounds=(lower, upper))
    return _unpack_params(result.x, n_modes)


def frf_fit_error(freqs_hz, h_measured, modes, freq_bounds=(20.0, 1800.0)) -> float:
    """Validation metric: normalized RMSE between a fitted mode bank's FRF and a (typically
    holdout) measured FRF -- the phase-aware analogue of psd_log_rmse."""
    mask = (freqs_hz >= freq_bounds[0]) & (freqs_hz <= freq_bounds[1])
    f_eval, h_eval = freqs_hz[mask], h_measured[mask]
    params = _pack_params(modes)
    h_model = _modal_bank_frf(f_eval, params, len(modes))
    return float(np.sqrt(np.mean(np.abs(h_model - h_eval) ** 2)) / np.sqrt(np.mean(np.abs(h_eval) ** 2)))


def calibrate_module_b_frf(
    fit_recordings,
    n_modes: int = 4,
    freq_bounds=(20.0, 1800.0),
    damping_bounds=(1e-3, 0.3),
    nperseg: int = 8192,
):
    """End-to-end phase-preserving calibration: averages the empirical torque->acceleration FRF
    of ``fit_recordings`` and fits an n_modes bank to it.

    Returns:
        (list[Mode], freqs_hz, h_measured).
    """
    freqs_hz, h_measured = averaged_frf(fit_recordings, nperseg=nperseg)
    modes = fit_modes_to_frf(freqs_hz, h_measured, n_modes, freq_bounds, damping_bounds)
    return modes, freqs_hz, h_measured
