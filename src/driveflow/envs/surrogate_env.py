"""Surrogate buck plant and ``SPADE/BuckSurrogate-v0`` -- expansion task E5 (round 2026-10-21, D3).

A surrogate is a model FITTED to simulation data of a plant and used in its place as a fast
environment. The intended source is PLECS sweeps (task E2). No PLECS data exists yet, so the source
here is SPADE's own averaged buck (native:buck_v1, sim/buck_system.py), labelled SIMULATED. This
validates the surrogate MECHANICS (data → fit → validity domain → environment → comparison), not
fidelity to any PLECS circuit. The environment is labelled ``SIMULATED · SURROGATE``.

Action: the duty cycle, as in SPADE/Buck-v0. For a PLECS circuit the duty is expected to be exposed
as a ModelVar ``D`` or as an input of the atomic subsystem, declared in the interface manifest
(integrations/plecs/manifest.py).

DATA: open-loop excitation of the native buck. The duty is piecewise constant with random levels in
[0.05, 0.85] held for 25–150 steps, from rest, over a (V_in × R) condition grid. Conditions, never
windows, are split into train / validation / test (``SPLIT``). Conditions outside the training
ranges are only used to show the out-of-domain warning.

FAMILIES
- ``zoh``: grey-box. The averaged buck structure with unknown L, C, r_L, identified by nonlinear
  least squares of the one-step prediction error of the exact ZOH model (the method of E10b,
  ai/pinn/identification.least_squares); V_in and R are known operating conditions.
- ``mlp``: black-box. A small Keras network (2 × 32 tanh) mapping (i_L, v_out, duty, V_in, log R),
  scaled, to the one-step state increment. Trained with a fixed seed, then run step by step with a
  numpy forward pass of the exported weights.

VALIDITY DOMAIN: the ranges of V_in, R and duty seen in training. The environment reports
``info["out_of_domain"]``, and the GUI says "Surrogate outside validated operating range".
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from driveflow.envs.native import BuckEnv
from driveflow.sim.buck_system import TAU, BuckParams, BuckState, BuckSystem

ROOT = Path(__file__).resolve().parents[3]
SURROGATE_DIR = ROOT / "configs" / "surrogates" / "buck_surrogate"
FAMILIES = ("zoh", "mlp")
VIN_GRID = (24.0, 36.0, 48.0, 60.0)
R_GRID = (3.0, 6.0, 12.0, 24.0)
#: (V_in, R) -> split; everything else on the grid is training. Fixed before fitting.
SPLIT = {(36.0, 6.0): "validation", (60.0, 24.0): "validation", (48.0, 12.0): "test", (24.0, 3.0): "test"}
EXTRAPOLATION = ((72.0, 12.0), (48.0, 40.0))
DUTY_RANGE = (0.05, 0.85)
N_STEPS = 2500  # 50 ms per condition
OUT_OF_DOMAIN = "Surrogate outside validated operating range"


# ----------------------------------------------------------------------------- data
def excitation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d, out = [], np.empty(n)
    k = 0
    while k < n:
        hold = int(rng.integers(25, 151))
        out[k:k + hold] = rng.uniform(*DUTY_RANGE)
        k += hold
    return out


def simulate_native(vin: float, r: float, duty: np.ndarray, params: BuckParams | None = None) -> np.ndarray:
    """(n + 1, 2) states [i_L, v_out] of the native averaged buck from rest under the duty sequence."""
    p = dataclasses.replace(params or BuckParams(), v_in_v=vin, load_resistance_ohm=r)
    sys = BuckSystem(p)
    s = sys.reset()
    x = [(s.i_l, s.v_out)]
    for d in duty:
        s = sys.simulate(float(d))
        x.append((s.i_l, s.v_out))
    return np.array(x)


def condition_split(vin: float, r: float) -> str:
    if (vin, r) in SPLIT:
        return SPLIT[(vin, r)]
    if vin in VIN_GRID and r in R_GRID:
        return "train"
    return "extrapolation"


def build_dataset(seed: int = 0) -> list[dict]:
    """One record per condition: its split, the excitation and the native trajectory."""
    out = []
    conds = [(v, r) for v in VIN_GRID for r in R_GRID] + list(EXTRAPOLATION)
    for i, (vin, r) in enumerate(conds):
        duty = excitation(N_STEPS, seed + 1000 + i)
        out.append({"v_in_v": vin, "load_resistance_ohm": r, "split": condition_split(vin, r), "duty": duty, "x": simulate_native(vin, r, duty)})
    return out


def transitions(records: list[dict]) -> dict:
    X, D, V, R, Xn = [], [], [], [], []
    for rec in records:
        X.append(rec["x"][:-1])
        Xn.append(rec["x"][1:])
        D.append(rec["duty"])
        V.append(np.full(len(rec["duty"]), rec["v_in_v"]))
        R.append(np.full(len(rec["duty"]), rec["load_resistance_ohm"]))
    return {"x": np.vstack(X), "d": np.concatenate(D), "vin": np.concatenate(V), "r": np.concatenate(R), "x_next": np.vstack(Xn)}


# ----------------------------------------------------------------------------- families
def fit_zoh(records: list[dict]) -> dict:
    """L, C, r_L by nonlinear least squares of the one-step prediction (exact ZOH per condition)."""
    from scipy.optimize import least_squares

    from driveflow.sim.buck_system import continuous_matrices, zoh

    nom = BuckParams()
    tr = transitions(records)
    scale = np.sqrt(np.mean(tr["x_next"] ** 2, axis=0))
    keys = sorted({(v, r) for v, r in zip(tr["vin"], tr["r"])})
    masks = {k: (tr["vin"] == k[0]) & (tr["r"] == k[1]) for k in keys}

    def resid(theta):
        L, C, rl = nom.l_h * math.exp(theta[0]), nom.c_f * math.exp(theta[1]), nom.r_l_ohm * math.exp(theta[2])
        res = np.empty_like(tr["x_next"])
        for (vin, r), m in masks.items():
            p = BuckParams(v_in_v=vin, l_h=L, c_f=C, r_l_ohm=rl, load_resistance_ohm=r)
            ad, bd = zoh(*continuous_matrices(p, r), TAU)
            res[m] = tr["x"][m] @ ad.T + np.outer(tr["d"][m], bd) - tr["x_next"][m]
        return (res / scale).ravel()

    # Start away from the truth: the nominal values are the plant's own, so starting there would
    # say nothing about the identification.
    sol = least_squares(resid, np.log([1.5, 0.7, 2.0]), method="lm")
    return {"l_h": nom.l_h * math.exp(sol.x[0]), "c_f": nom.c_f * math.exp(sol.x[1]), "r_l_ohm": nom.r_l_ohm * math.exp(sol.x[2]),
            "start": "x1.5 L, x0.7 C, x2 r_L of nominal", "cost": float(sol.cost)}


I_S, V_S = 15.0, 60.0


def _features(x: np.ndarray, d: np.ndarray, vin: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.column_stack([x[:, 0] / I_S, x[:, 1] / V_S, d, vin / V_S, np.log(r / 6.0)]).astype(np.float32)


def fit_mlp(records: list[dict], seed: int = 0, epochs: int = 30) -> dict:
    import keras

    keras.utils.set_random_seed(seed)
    tr = transitions(records)
    F = _features(tr["x"], tr["d"], tr["vin"], tr["r"])
    dX = tr["x_next"] - tr["x"]
    y_scale = dX.std(axis=0)
    Y = (dX / y_scale).astype(np.float32)
    model = keras.Sequential([keras.Input((F.shape[1],)), keras.layers.Dense(32, activation="tanh"), keras.layers.Dense(32, activation="tanh"), keras.layers.Dense(2)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(F, Y, epochs=epochs, batch_size=256, verbose=0, shuffle=True)
    return {"weights": [w.tolist() for w in model.get_weights()], "y_scale": y_scale.tolist(), "epochs": epochs, "seed": seed,
            "architecture": "Dense(32, tanh) -> Dense(32, tanh) -> Dense(2); inputs i_L/15, v_out/60, duty, V_in/60, log(R/6)"}


def mlp_forward(weights: list, f: np.ndarray) -> np.ndarray:
    W = [np.asarray(w, dtype=np.float64) for w in weights]
    h = np.tanh(f @ W[0] + W[1])
    h = np.tanh(h @ W[2] + W[3])
    return h @ W[4] + W[5]


# ----------------------------------------------------------------------------- plants
class ZohSurrogatePlant(BuckSystem):
    """The averaged buck with the IDENTIFIED L, C, r_L (same interface as BuckSystem)."""

    def __init__(self, fit: dict, vin: float, load_ohm: float, i_max_a: float = 15.0):
        super().__init__(BuckParams(v_in_v=vin, l_h=fit["l_h"], c_f=fit["c_f"], r_l_ohm=fit["r_l_ohm"], load_resistance_ohm=load_ohm, i_max_a=i_max_a))


class MlpSurrogatePlant:
    """One-step network surrogate with BuckSystem's reset/simulate/set_load interface."""

    def __init__(self, fit: dict, vin: float, load_ohm: float, i_max_a: float = 15.0):
        self.weights, self.y_scale = fit["weights"], np.asarray(fit["y_scale"])
        self.params = BuckParams(v_in_v=vin, load_resistance_ohm=load_ohm, i_max_a=i_max_a)
        self.tau = TAU
        self.load_resistance_ohm = float(load_ohm)
        self.state: BuckState = None

    def set_load(self, load_resistance_ohm: float) -> None:
        if not (np.isfinite(load_resistance_ohm) and load_resistance_ohm > 0):
            raise ValueError(f"load resistance must be > 0, got {load_resistance_ohm!r}")
        self.load_resistance_ohm = float(load_resistance_ohm)

    def reset(self, i_l: float = 0.0, v_out: float = 0.0) -> BuckState:
        self.state = BuckState(float(i_l), float(v_out))
        return self.state

    def simulate(self, duty: float) -> BuckState:
        d = min(1.0, max(0.0, float(duty)))
        x = np.array([[self.state.i_l, self.state.v_out]])
        f = _features(x, np.array([d]), np.array([self.params.v_in_v]), np.array([self.load_resistance_ohm])).astype(np.float64)
        dx = mlp_forward(self.weights, f)[0] * self.y_scale
        self.state = BuckState(float(self.state.i_l + dx[0]), float(self.state.v_out + dx[1]))
        return self.state


def make_plant(surrogate: dict, family: str, vin: float, load_ohm: float, i_max_a: float = 15.0):
    if family == "zoh":
        return ZohSurrogatePlant(surrogate["families"]["zoh"]["fit"], vin, load_ohm, i_max_a)
    if family == "mlp":
        return MlpSurrogatePlant(surrogate["families"]["mlp"]["fit"], vin, load_ohm, i_max_a)
    raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")


def domain_violations(domain: dict, vin: float, load_ohm: float, duty: float | None = None) -> list[str]:
    out = []
    for name, value in (("v_in_v", vin), ("load_resistance_ohm", load_ohm), ("duty", duty)):
        if value is None:
            continue
        lo, hi = domain[name]
        if not lo - 1e-9 <= value <= hi + 1e-9:
            out.append(f"{name} = {value:g} outside [{lo:g}, {hi:g}]")
    return out


def latest_surrogate(path: Path | None = None) -> tuple[dict, Path]:
    """The fitted surrogate (surrogate.json) of the given run directory, or of the newest run."""
    if path is None:
        runs = sorted(p for p in SURROGATE_DIR.glob("*/surrogate.json"))
        if not runs:
            raise FileNotFoundError(f"no fitted buck surrogate under {SURROGATE_DIR} -- run experiments/fit_buck_surrogate.py")
        path = runs[-1].parent
    path = Path(path)
    return json.loads((path / "surrogate.json").read_text()), path


# ----------------------------------------------------------------------------- environment
class BuckSurrogateEnv(BuckEnv):
    """``SPADE/BuckSurrogate-v0``: SPADE/Buck-v0's contract (observation, action, reward,
    terminations), with the plant replaced by the fitted surrogate. ``family`` None = the
    surrogate's default family (chosen on validation conditions)."""

    metadata = {**BuckEnv.metadata, "plant_profile_id": "surrogate:buck_surrogate", "origin": "SIMULATED · SURROGATE"}

    def __init__(self, family: str | None = None, surrogate_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.surrogate, self.surrogate_dir = latest_surrogate(Path(surrogate_path) if surrogate_path else None)
        self.family = family or self.surrogate["default_family"]
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        self.metadata = {**self.metadata, "surrogate_family": self.family, "surrogate_run": self.surrogate_dir.name,
                         "validity_domain": self.surrogate["domain"], "fit_error": self.surrogate["families"][self.family]["validation"]}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _obs, info = super().reset(seed=seed, options=options)
        load = self.plant.load_resistance_ohm
        self.plant = make_plant(self.surrogate, self.family, self.vin_v, load, self.i_max)
        self.plant_state = self.plant.reset()
        viol = domain_violations(self.surrogate["domain"], self.vin_v, load)
        info.update(self._physical(), surrogate_family=self.family, out_of_domain=viol)
        return self._obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["out_of_domain"] = domain_violations(self.surrogate["domain"], self.vin_v, self.plant.load_resistance_ohm, float(info["action_applied"][0]))
        return obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------------- comparison
def rollout(plant, duty: np.ndarray) -> np.ndarray:
    s = plant.reset()
    x = [(s.i_l, s.v_out)]
    for d in duty:
        s = plant.simulate(float(d))
        x.append((s.i_l, s.v_out))
    return np.array(x)


def _rel_rmse(a: np.ndarray, b: np.ndarray) -> list:
    """Per channel RMSE(a - b) / RMS(b); None for a non-finite rollout."""
    if not np.all(np.isfinite(a)):
        return [None, None]
    return [float(np.sqrt(np.mean((a[:, j] - b[:, j]) ** 2)) / max(float(np.sqrt(np.mean(b[:, j] ** 2))), 1e-9)) for j in range(2)]


def closed_loop(plant_factory, vin: float, load_ohm: float, vref: float, step_ohm: float | None, duration_s: float = 0.02) -> dict:
    """PI (BuckVoltageController designed on the NATIVE nominal plant) driving a plant built by
    plant_factory(vin, load); optional load step at half the run."""
    from driveflow.control.classical.buck_pi import BuckVoltageController

    design = BuckSystem(BuckParams(v_in_v=vin, load_resistance_ohm=load_ohm))
    ctrl = BuckVoltageController(design)
    plant = plant_factory(vin, load_ohm)
    s = plant.reset()
    n = int(round(duration_s / TAU))
    t, v, i, d = [], [], [], []
    for k in range(n):
        if step_ohm is not None and k == n // 2:
            plant.set_load(step_ohm)
        duty = ctrl.control(s, vref)
        s = plant.simulate(duty)
        t.append((k + 1) * TAU)
        v.append(s.v_out)
        i.append(s.i_l)
        d.append(duty)
    return {"t": np.array(t), "v_out": np.array(v), "i_l": np.array(i), "duty": np.array(d), "finite": bool(np.all(np.isfinite(v)) and np.all(np.isfinite(i)))}


def compare_condition(surrogate: dict, family: str, vin: float, load_ohm: float, vref: float | None = None, step_ohm: float | None = None,
                      duration_s: float = 0.02) -> dict:
    """Native vs. surrogate under the same PI, reference and load step; aligned time axes."""
    vref = 0.5 * vin if vref is None else vref
    native = closed_loop(lambda v, r: BuckSystem(BuckParams(v_in_v=v, load_resistance_ohm=r)), vin, load_ohm, vref, step_ohm, duration_s)
    sur = closed_loop(lambda v, r: make_plant(surrogate, family, v, r), vin, load_ohm, vref, step_ohm, duration_s)
    out = {"v_in_v": vin, "load_resistance_ohm": load_ohm, "vref_v": vref, "step_ohm": step_ohm, "family": family, "native": native, "surrogate": sur,
           "out_of_domain": list(dict.fromkeys(domain_violations(surrogate["domain"], vin, load_ohm) + (domain_violations(surrogate["domain"], vin, step_ohm) if step_ohm else []))),
           "status": "ok" if sur["finite"] else "failed"}
    if sur["finite"]:
        out["rmse_v_out_v"] = float(np.sqrt(np.mean((sur["v_out"] - native["v_out"]) ** 2)))
        out["rmse_i_l_a"] = float(np.sqrt(np.mean((sur["i_l"] - native["i_l"]) ** 2)))
        out["max_abs_v_out_v"] = float(np.max(np.abs(sur["v_out"] - native["v_out"])))
    return out
