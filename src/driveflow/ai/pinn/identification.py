"""Physics-informed identification of the VSC filter parameters L, R_f and C -- expansion task E10b.

Origin: Fund. ``5_PIML/PINN/pinn_ode.ipynb`` (XinzeLee/Fundamentals_of_AI_for_PE, Apache 2.0 code):
an inverse-problem PINN, i.e. a network u(t) fitted to measurements while an ODE residual with
trainable coefficients is driven to zero, so the coefficients become the estimate. That notebook is
PyTorch; this is a reimplementation in Keras/TensorFlow, no code was copied.

PLANT (sim/vsc_params.py, the model behind ADF/BDF): state [i_f, v_c] per axis (real and imaginary
parts of the stationary-frame phasor obey the same, uncoupled equations)

    L  di_f/dt = v_o - v_c - R_f i_f
    C  dv_c/dt = i_f - i_load,        i_load = v_c / R

SPADE's plant is the exact zero-order-hold discretization of this model at Ts = 100 us with v_o and
i_load HELD over each period (sim/vsc_system.py: i_load uses the state at the start of the step). A
row k of a run holds the input v_o[k] applied during step k and the state at the END of step k. The
residual below uses the same held inputs, so the physics the PINN enforces is the plant's own and
not a continuous-load approximation of it. R (the load) is an operating condition and is GIVEN, not
estimated: with i_load unmeasured, C and R enter only as 1/C and 1/(RC).

ESTIMATORS
- ``pinn``: an MLP N(t) -> [i_f, v_c] (both axes) trained on the measured samples plus the residual
  at collocation points inside every sample period; the parameters are log-scaled offsets from the
  nominal values, initialised AT nominal (an informative start, which the validation report states).
- ``least_squares``: a reference without a network. Nonlinear least squares of the one-step
  prediction error x[k+1] - (A_d x[k] + B_d [v_o[k], i_load[k]]) with A_d, B_d = discretize_zoh(L,
  R_f, C) (both axes share them). Exact on noise-free data. With noisy states it is an
  errors-in-variables problem and is biased. (A LINEAR regression on the four discrete
  regressors does not work: i_load = v_c/R is collinear with v_c.)

UNCERTAINTY: both use the same declared method, a moving-block bootstrap. The window is split
into blocks of ``block_len`` samples, blocks are resampled with replacement, and the estimator is
refitted on each resample. For ``pinn`` each refit also uses a new network seed, so the spread
includes the optimiser's spread (a bootstrap ensemble). The interval is the 5-95 % percentile of the
refits; with few refits (the PINN default is 5) it is coarse, and its empirical coverage is
measured in docs/audit/ai/pinn_identification.md, not assumed.

What this is NOT: a validated fault diagnosis. A low C estimate is consistent with capacitor aging
(sim/vsc_faults.py) but is a parameter estimate under this model; an ESR, which this model does not
contain, shows up as a biased R_f/C.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd

from driveflow.sim.vsc_params import CAPACITANCE_F, FILTER_RESISTANCE_OHM, INDUCTANCE_H, SAMPLE_PERIOD_S

PARAMS = ("L", "R_f", "C")
NOMINAL = {"L": INDUCTANCE_H, "R_f": FILTER_RESISTANCE_OHM, "C": CAPACITANCE_F}
UNITS = {"L": "H", "R_f": "Ω", "C": "F"}
ESTIMATORS = ("pinn", "least_squares")
UNCERTAINTY_METHOD = "moving-block bootstrap, 5-95 % percentile interval"
DISCLAIMER = "Parameter estimate — not a validated fault diagnosis."
COLUMNS = ("vc_real", "vc_imag", "i_f_real", "i_f_imag", "v_o_real", "v_o_imag")


@dataclasses.dataclass(frozen=True)
class IdentificationData:
    """One window, rows k0..k0+n-1 of a run, plus the state of the row before it (x0).

    i_f, v_c: (n, 2) state at the END of each step; v_o: (n, 2) input during each step;
    x0_v: (2,) v_c at the start of the first step (needed for its held i_load)."""

    t: np.ndarray
    i_f: np.ndarray
    v_c: np.ndarray
    v_o: np.ndarray
    x0_i: np.ndarray
    x0_v: np.ndarray
    load_resistance_ohm: float
    ts: float = SAMPLE_PERIOD_S

    @property
    def n(self) -> int:
        return len(self.t)

    def i_load(self) -> np.ndarray:
        """(n, 2) held load current of each step: v_c at the step's start over R."""
        v_start = np.vstack([self.x0_v[np.newaxis], self.v_c[:-1]])
        return v_start / self.load_resistance_ohm


def from_dataframe(df: pd.DataFrame, load_resistance_ohm: float, start_s: float, length_s: float, ts: float = SAMPLE_PERIOD_S) -> IdentificationData:
    """Window [start_s, start_s + length_s) of a VSC run (datagen/runner.py columns). The row
    before the window supplies the initial state; start_s must leave at least one row before it."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing VSC column(s): {', '.join(missing)}")
    if not (np.isfinite(load_resistance_ohm) and load_resistance_ohm > 0):
        raise ValueError(f"load resistance must be > 0, got {load_resistance_ohm!r}")
    t = df["timestamp_s"].to_numpy(dtype=float) if "timestamp_s" in df.columns else (np.arange(len(df)) + 1) * ts
    k0 = int(np.searchsorted(t, start_s - 1e-12))
    n = int(round(length_s / ts))
    if k0 < 1:
        raise ValueError("the window must start after the first sample (its initial state comes from the row before it)")
    if n < 20 or k0 + n > len(df):
        raise ValueError(f"window needs 20..{len(df) - k0} samples from {start_s:g} s, got {n}")
    rows = df.iloc[k0 - 1:k0 + n]
    vals = {c: rows[c].to_numpy(dtype=float) for c in COLUMNS}
    if not all(np.isfinite(v).all() for v in vals.values()):
        raise ValueError("the window contains non-finite values")
    i_f = np.stack([vals["i_f_real"], vals["i_f_imag"]], axis=1)
    v_c = np.stack([vals["vc_real"], vals["vc_imag"]], axis=1)
    v_o = np.stack([vals["v_o_real"], vals["v_o_imag"]], axis=1)
    return IdentificationData(t=t[k0:k0 + n], i_f=i_f[1:], v_c=v_c[1:], v_o=v_o[1:], x0_i=i_f[0], x0_v=v_c[0], load_resistance_ohm=float(load_resistance_ohm), ts=ts)


def add_noise(data: IdentificationData, noise_pct: float, rng: np.random.Generator) -> IdentificationData:
    """Gaussian noise on the measured states (i_f, v_c, and the initial state), with standard
    deviation noise_pct % of each signal's RMS over the window. v_o is the controller's own command
    and is taken as known exactly."""
    if noise_pct <= 0:
        return data

    def noisy(x, ref):
        rms = np.sqrt(np.mean(ref ** 2))
        return x + rng.normal(0.0, noise_pct / 100 * rms, size=x.shape)

    return dataclasses.replace(data, i_f=noisy(data.i_f, data.i_f), v_c=noisy(data.v_c, data.v_c), x0_i=noisy(data.x0_i, data.i_f), x0_v=noisy(data.x0_v, data.v_c))


# ----------------------------------------------------------------------------- least squares
def _pairs(data: IdentificationData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x[k], u[k], x[k+1]) of every step, both axes stacked: (2n, 2) each."""
    x_prev = np.concatenate([np.stack([np.r_[data.x0_i[a], data.i_f[:-1, a]], np.r_[data.x0_v[a], data.v_c[:-1, a]]], axis=1) for a in range(2)])
    u = np.concatenate([np.stack([data.v_o[:, a], data.i_load()[:, a]], axis=1) for a in range(2)])
    x_next = np.concatenate([np.stack([data.i_f[:, a], data.v_c[:, a]], axis=1) for a in range(2)])
    return x_prev, u, x_next


def _ls_fit(x_prev: np.ndarray, u: np.ndarray, x_next: np.ndarray, ts: float) -> dict:
    """Nonlinear least squares of the one-step prediction error of the exact ZOH model, over the
    log-scale of (L, R_f, C) from nominal. Each state's error is scaled by that state's RMS."""
    from scipy.optimize import least_squares

    from driveflow.sim.vsc_params import discretize_zoh

    scale = np.sqrt(np.mean(x_next ** 2, axis=0))
    scale[scale == 0] = 1.0

    def resid(theta):
        L, r_f, c = (NOMINAL[k] * np.exp(v) for k, v in zip(PARAMS, theta))
        ad, bd = discretize_zoh(L, r_f, c, ts)
        return ((x_prev @ ad.T + u @ bd.T - x_next) / scale).ravel()

    sol = least_squares(resid, np.zeros(3), method="lm", x_scale=1.0)
    return {k: float(NOMINAL[k] * np.exp(v)) for k, v in zip(PARAMS, sol.x)}


# ----------------------------------------------------------------------------- PINN
def _pinn_fit(data: IdentificationData, seed: int, weights: np.ndarray | None, steps: int, collocation_per_step: int, hidden: int, lr: float) -> dict:
    """One PINN fit; ``weights`` (n,) are per-sample bootstrap multiplicities for the data term and
    for the residual inside that sample's period (None = all ones)."""
    import tensorflow as tf
    import keras

    keras.utils.set_random_seed(seed)
    n, ts = data.n, data.ts
    t_end = n * ts
    i_s = float(np.sqrt(np.mean(data.i_f ** 2))) or 1.0
    v_s = float(np.sqrt(np.mean(data.v_c ** 2))) or 1.0
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.mean()

    # data points: the end of each step, normalised time in (0, 1]; plus the initial state at 0
    t_data = np.concatenate([[0.0], (np.arange(n) + 1) / n])[:, None].astype(np.float32)
    y_data = np.concatenate([np.concatenate([data.x0_i, data.x0_v])[np.newaxis], np.concatenate([data.i_f, data.v_c], axis=1)])
    y_data = (y_data / np.array([i_s, i_s, v_s, v_s])).astype(np.float32)
    w_data = np.concatenate([[1.0], w]).astype(np.float32)[:, None]

    # collocation: fixed interior points of every step, with that step's held inputs
    frac = (np.arange(collocation_per_step) + 0.5) / collocation_per_step
    k_idx = np.repeat(np.arange(n), collocation_per_step)
    t_col = ((k_idx + np.tile(frac, n)) / n)[:, None].astype(np.float32)
    v_o = data.v_o[k_idx].astype(np.float32)
    w_col = w[k_idx].astype(np.float32)[:, None]

    model = keras.Sequential([keras.Input((1,))] + [keras.layers.Dense(hidden, activation="tanh") for _ in range(3)] + [keras.layers.Dense(4)])
    log_p = tf.Variable(np.zeros(3, dtype=np.float32), name="log_scale")  # [L, R_f, C] = nominal * exp(.)
    nom = tf.constant([NOMINAL["L"], NOMINAL["R_f"], NOMINAL["C"]], dtype=tf.float32)
    opt = keras.optimizers.Adam(learning_rate=lr)
    variables = model.trainable_variables + [log_p]
    t_col_tf, t_data_tf, y_data_tf = tf.constant(t_col), tf.constant(t_data), tf.constant(y_data)
    v_o_tf, w_col_tf, w_data_tf = tf.constant(v_o), tf.constant(w_col), tf.constant(w_data)
    k_idx_tf = tf.constant(k_idx)
    r_load = float(data.load_resistance_ohm)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            y_fit = model(t_data_tf)
            data_loss = tf.reduce_mean(w_data_tf * tf.square(y_fit - y_data_tf))
            p = nom * tf.exp(log_p)
            # held load current of each step from the NETWORK's v_c at the step start (t_data[k]),
            # not from the noisy measurement: a noisy input makes the C equation errors-in-variables
            i_load_col = tf.gather(y_fit[:, 2:], k_idx_tf) * (v_s / r_load)
            with tf.GradientTape(persistent=True) as g2:
                g2.watch(t_col_tf)
                yc = model(t_col_tf)
                yi_r, yi_i, yv_r, yv_i = yc[:, 0], yc[:, 1], yc[:, 2], yc[:, 3]
            d = [g2.gradient(q, t_col_tf)[:, 0] for q in (yi_r, yi_i, yv_r, yv_i)]
            del g2
            res = []
            for a in range(2):
                i_f, v_c = yc[:, a] * i_s, yc[:, 2 + a] * v_s
                di, dv = d[a] * (i_s / t_end), d[2 + a] * (v_s / t_end)
                res.append((p[0] * di - (v_o_tf[:, a] - v_c - p[1] * i_f)) / v_s)
                res.append((p[2] * dv - (i_f - i_load_col[:, a])) / i_s)
            phys = tf.reduce_mean(w_col_tf[:, 0] * tf.add_n([tf.square(r) for r in res]) / 4.0)
            loss = data_loss + phys
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return data_loss, phys

    for _ in range(steps):
        data_loss, phys = train_step()
    p = NOMINAL["L"] * np.exp(float(log_p[0])), NOMINAL["R_f"] * np.exp(float(log_p[1])), NOMINAL["C"] * np.exp(float(log_p[2]))
    return {"L": float(p[0]), "R_f": float(p[1]), "C": float(p[2]), "data_loss": float(data_loss), "physics_loss": float(phys)}


# ----------------------------------------------------------------------------- public API
def _block_bootstrap_weights(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicity of every sample under one moving-block bootstrap resample."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    w = np.zeros(n)
    for s in starts:
        w[s:s + block_len] += 1
    return w


def identify(data: IdentificationData, estimator: str = "pinn", n_bootstrap: int | None = None, block_len: int = 20, seed: int = 0,
             steps: int = 3000, collocation_per_step: int = 2, hidden: int = 32, lr: float = 3e-3) -> dict:
    """Point estimate (fit on the whole window) and the bootstrap interval of L, R_f and C.

    n_bootstrap: refits on block-bootstrap resamples (default 5 for the PINN, 50 for least squares)."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if estimator == "least_squares":
        n_boot = 50 if n_bootstrap is None else n_bootstrap
        x_prev, u, x_next = _pairs(data)
        point = _ls_fit(x_prev, u, x_next, data.ts)
        boots = []
        for _ in range(n_boot):
            w = _block_bootstrap_weights(data.n, block_len, rng)
            idx = np.repeat(np.arange(data.n), w.astype(int))
            idx = np.concatenate([idx, idx + data.n])  # both axes of each resampled step
            boots.append(_ls_fit(x_prev[idx], u[idx], x_next[idx], data.ts))
        extra = {}
    else:
        n_boot = 5 if n_bootstrap is None else n_bootstrap
        kw = dict(steps=steps, collocation_per_step=collocation_per_step, hidden=hidden, lr=lr)
        point = _pinn_fit(data, seed, None, **kw)
        extra = {"data_loss": point.pop("data_loss"), "physics_loss": point.pop("physics_loss")}
        boots = []
        for b in range(n_boot):
            fit = _pinn_fit(data, seed + 1 + b, _block_bootstrap_weights(data.n, block_len, rng), **kw)
            boots.append({p: fit[p] for p in PARAMS})
    out = {"estimator": estimator, "uncertainty_method": UNCERTAINTY_METHOD, "n_bootstrap": n_boot, "block_len": block_len,
           "window_samples": data.n, "load_resistance_ohm": data.load_resistance_ohm, "elapsed_s": None, "params": {}, **extra}
    for p in PARAMS:
        vals = np.array([b[p] for b in boots]) if boots else np.array([point[p]])
        lo, hi = (np.percentile(vals, [5, 95]) if len(vals) > 1 else (np.nan, np.nan))
        out["params"][p] = {"estimate": point[p], "low": float(lo), "high": float(hi), "nominal": NOMINAL[p], "unit": UNITS[p],
                            "error_vs_nominal_pct": 100 * (point[p] - NOMINAL[p]) / NOMINAL[p]}
    out["elapsed_s"] = time.perf_counter() - t0
    return out
