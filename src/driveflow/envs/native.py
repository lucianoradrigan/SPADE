"""Gymnasium environments over SPADE's native plants -- expansion task E1 (spec Sec. 3.7, 8.5).

``DcMotorEnv`` and ``VscEnv`` wrap the same plants ``datagen.runner.run_scenario`` simulates,
built by the same code (``runner.build_plant`` for the DC motor; ``VscSystem`` /
``DegradedVscSystem`` for the VSC) and seeded the same way. The wrappers add no physics: a step is
exactly one ``simulate()`` call of period TAU = 100 us. tests/envs/test_native_envs.py checks
that feeding the runner's own controller actions through ``step()`` reproduces ``run_scenario``
bit for bit.

Contract (Gymnasium 1.x):

- ``reset(seed=None, options=None) -> (obs, info)``. The seed fixes the plant's random components
  and every quantity drawn at reset (reference, load); ``options`` overrides them per episode.
- ``step(action) -> (obs, reward, terminated, truncated, info)``. ``terminated`` = a physical
  limit was exceeded or the state is not finite (``info["termination"]`` says which);
  ``truncated`` = the episode horizon was reached (``info["truncation"] == "horizon"``).
- Actions outside the action space are clipped, and ``info["action_applied"]`` /
  ``info["action_clipped"]`` show what reached the plant ("clip and log", spec Sec. 3.4 D).
  A non-finite action raises ValueError.
- Units: ``metadata["observation_names"]``, ``["observation_units"]``,
  ``["observation_scales"]`` (observation = physical value / scale), ``["action_names"]``,
  ``["action_units"]``; ``info`` carries the physical values.
- Reward: ``envs.reward`` terms (tracking, overcurrent, action rate), breakdown in
  ``info["reward_terms"]``.

Design of the observation, seeded reset and term-wise reward follows Fund.
7_Reinforcement_Learning/RL_buck_control.ipynb (BuckAvgEnv); no code was copied (that class is
PyTorch-side and not a gymnasium.Env).
"""

from __future__ import annotations

import dataclasses
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driveflow.control.dpc.reference import RotatingReference
from driveflow.datagen.runner import TAU, _VSC_R_OHM, build_plant
from driveflow.datagen.scenario import Scenario
from driveflow.envs import reward as rw
from driveflow.sim.vsc_system import VscSystem


def _plant_seed(env: gym.Env, seed: int | None) -> int:
    """The seed used for the plant: the reset seed itself (so ``reset(seed=s)`` matches a
    Scenario with ``seed=s``), else one drawn from the env's generator."""
    return int(seed) if seed is not None else int(env.np_random.integers(0, 2**31 - 1))


class _NativeEnv(gym.Env):
    def _setup_reward(self, reward_terms) -> None:
        self.reward_terms = rw.parse_terms(reward_terms)

    def _finish_step(self, obs, signals: dict, violation: str | None, info: dict):
        reward, breakdown = rw.evaluate(self.reward_terms, signals)
        self._k += 1
        terminated = violation is not None
        truncated = not terminated and self._k >= self.n_steps
        info.update(step=self._k, t_s=self._k * self.tau, reward_terms=breakdown, signals=signals)
        if terminated:
            info["termination"] = violation
        if truncated:
            info["truncation"] = "horizon"
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _as_action(action, shape) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(shape)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"Action must be finite, got {action!r}")
        return a


class DcMotorEnv(_NativeEnv):
    """Permanently excited DC motor on a one-quadrant converter (plant ``dc_perm_ex_v1``), speed
    tracking. Action: duty cycle in [0, 1]. Observation (float32): [omega, i, u, omega_ref,
    omega_ref - omega], each divided by the plant's limit (400 rad/s, 210 A, 60 V)."""

    metadata = {
        "render_modes": [],
        "plant_profile_id": "native:dc_perm_ex_v1",
        "observation_names": ("omega", "i_a", "u_a", "omega_ref", "omega_error"),
        "observation_units": ("rad/s", "A", "V", "rad/s", "rad/s"),
        "action_names": ("duty_cycle",),
        "action_units": ("1",),
        "tau_s": TAU,
    }

    def __init__(self, omega_ref_rad_s: float | None = 150.0, omega_ref_range: tuple = (50.0, 300.0), duration_s: float = 0.5,
                 fault_type: str | None = None, reward_terms=None, render_mode: str | None = None):
        if render_mode is not None:
            raise ValueError("DcMotorEnv has no render modes; plot info signals instead.")
        lo, hi = map(float, omega_ref_range)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
            raise ValueError(f"omega_ref_range must be finite with 0 <= low <= high, got {omega_ref_range!r}")
        if not (math.isfinite(duration_s) and duration_s > 0):
            raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
        self.omega_ref_fixed = None if omega_ref_rad_s is None else float(omega_ref_rad_s)
        self.omega_ref_range = (lo, hi)
        self.fault_type = fault_type
        self.tau = TAU
        self.n_steps = int(round(duration_s / TAU))
        self._setup_reward(reward_terms)
        self.plant = self._build(0, self.omega_ref_fixed if self.omega_ref_fixed is not None else lo)  # validates fault_type now
        self.limits = self.plant.limits.copy()
        self._idx = (self.plant.OMEGA_IDX, self.plant.CURRENTS_IDX[0], self.plant.VOLTAGES_IDX[0])
        omega_lim, i_lim, u_lim = (float(self.limits[i]) for i in self._idx)
        self.metadata = {**self.metadata, "observation_scales": (omega_lim, i_lim, u_lim, omega_lim, omega_lim),
                         "limits": {"omega_rad_s": omega_lim, "current_a": i_lim, "voltage_v": u_lim}}
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.plant_state = None
        self.omega_ref_rad_s = None
        self._k = 0
        self._prev_action = np.zeros(1)

    def _build(self, seed: int, omega_ref: float):
        return build_plant(Scenario(scenario_id="gym-episode", plant_config_id="dc_perm_ex_v1", controller_type="PI",
                                    omega_ref_rad_s=omega_ref, fault_type=self.fault_type, seed=seed))

    def _physical(self) -> dict:
        p = self.plant_state * self.plant.limits
        omega = float(p[self._idx[0]])
        return {"omega_rad_s": omega, "rpm": omega * 60.0 / (2 * np.pi), "current_a": float(p[self._idx[1]]),
                "voltage_v": float(p[self._idx[2]]), "torque_nm": float(p[self.plant.TORQUE_IDX]), "omega_ref_rad_s": self.omega_ref_rad_s}

    def _obs(self, phys: dict) -> np.ndarray:
        s = self.metadata["observation_scales"]
        return np.array([phys["omega_rad_s"] / s[0], phys["current_a"] / s[1], phys["voltage_v"] / s[2], self.omega_ref_rad_s / s[3],
                         (self.omega_ref_rad_s - phys["omega_rad_s"]) / s[4]], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        plant_seed = _plant_seed(self, seed)
        if options.get("omega_ref_rad_s") is not None:
            self.omega_ref_rad_s = float(options["omega_ref_rad_s"])
        elif self.omega_ref_fixed is not None:
            self.omega_ref_rad_s = self.omega_ref_fixed
        else:
            self.omega_ref_rad_s = float(self.np_random.uniform(*self.omega_ref_range))
        # a fresh plant per episode, seeded like datagen.runner._run_dc_motor_scenario
        self.plant = self._build(plant_seed, self.omega_ref_rad_s)
        self.plant.seed(np.random.SeedSequence(plant_seed))
        self.plant_state = self.plant.reset()
        self._k = 0
        self._prev_action = np.zeros(1)
        phys = self._physical()
        return self._obs(phys), {**phys, "plant_seed": plant_seed, "step": 0, "t_s": 0.0}

    def step(self, action):
        if self.plant_state is None:
            raise RuntimeError("Call reset() before step().")
        a = self._as_action(action, (1,))
        applied = np.clip(a, 0.0, 1.0)
        self.plant_state = self.plant.simulate(applied)
        phys = self._physical()
        lim = self.metadata["limits"]
        signals = {"tracking_error": (self.omega_ref_rad_s - phys["omega_rad_s"]) / lim["omega_rad_s"],
                   "current_fraction": phys["current_a"] / lim["current_a"],
                   "action_delta": float(applied[0] - self._prev_action[0])}
        self._prev_action = applied
        violation = None
        if not np.all(np.isfinite(self.plant_state)):
            violation = "non_finite_state"
        elif abs(phys["omega_rad_s"]) > lim["omega_rad_s"]:
            violation = "limit_violation:omega"
        elif abs(phys["current_a"]) > lim["current_a"]:
            violation = "limit_violation:current"
        info = {**phys, "action_applied": applied.copy(), "action_clipped": bool(np.any(applied != a))}
        return self._finish_step(self._obs(phys), signals, violation, info)


class VscEnv(_NativeEnv):
    """Voltage source converter with L-C filter and resistive load (plant ``vsc_dpc_v1``),
    tracking a rotating voltage reference. Action: converter voltage (v_o_real, v_o_imag) in V.
    Observation (float32): [i_f_real, i_f_imag, v_c_real, v_c_imag, v_ref_real, v_ref_imag] for
    the reference of the NEXT step, voltages / 100 V and currents / current_limit_a.

    The native model has no converter voltage rating; ``voltage_limit_v`` (default 2000 V) is the
    action bound, wide enough for the DPC network's start-up transient (~1000 V at 100 V / 8 ohm;
    see sim/vsc_system.py and S-09). Lower it to study saturation.
    """

    V_SCALE = 100.0

    metadata = {
        "render_modes": [],
        "plant_profile_id": "native:vsc_dpc_v1",
        "observation_names": ("i_f_real", "i_f_imag", "v_c_real", "v_c_imag", "v_ref_real", "v_ref_imag"),
        "observation_units": ("A", "A", "V", "V", "V", "V"),
        "action_names": ("v_o_real", "v_o_imag"),
        "action_units": ("V", "V"),
        "tau_s": TAU,
    }

    def __init__(self, reference_magnitude_v: float | None = 50.0, magnitude_range: tuple = (25.0, 100.0),
                 load_resistance_ohm: float | None = _VSC_R_OHM, load_range_ohm: tuple = (4.0, 1000.0), duration_s: float = 0.1,
                 voltage_limit_v: float = 2000.0, current_limit_a: float = 40.0, capacitor_scale: float | None = None,
                 esr_ohm: float | None = None, reward_terms=None, render_mode: str | None = None):
        if render_mode is not None:
            raise ValueError("VscEnv has no render modes; plot info signals instead.")
        for name, value in (("duration_s", duration_s), ("voltage_limit_v", voltage_limit_v), ("current_limit_a", current_limit_a)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be > 0, got {value!r}")
        for name, (lo, hi) in (("magnitude_range", magnitude_range), ("load_range_ohm", load_range_ohm)):
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(f"{name} must be finite with 0 < low <= high, got {(lo, hi)!r}")
        self.magnitude_fixed = None if reference_magnitude_v is None else float(reference_magnitude_v)
        self.magnitude_range = tuple(map(float, magnitude_range))
        self.load_fixed = None if load_resistance_ohm is None else float(load_resistance_ohm)
        self.load_range_ohm = tuple(map(float, load_range_ohm))
        self.voltage_limit_v = float(voltage_limit_v)
        self.current_limit_a = float(current_limit_a)
        self.capacitor_scale, self.esr_ohm = capacitor_scale, esr_ohm
        self.tau = TAU
        self.n_steps = int(round(duration_s / TAU))
        self._setup_reward(reward_terms)
        v, i = self.V_SCALE, self.current_limit_a
        self.metadata = {**self.metadata, "observation_scales": (i, i, v, v, v, v),
                         "limits": {"voltage_v": self.voltage_limit_v, "current_a": self.current_limit_a}}
        self.action_space = spaces.Box(-self.voltage_limit_v, self.voltage_limit_v, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.plant = None
        self.plant_state = None
        self.reference = None
        self.load_resistance_ohm = None
        self._k = 0
        self._prev_action = np.zeros(2)

    def _build(self, r_ohm: float):
        if self.capacitor_scale is not None or self.esr_ohm is not None:  # same switch as runner._run_vsc_scenario
            from driveflow.sim.vsc_faults import DegradedVscSystem

            return DegradedVscSystem(r_ohm, c_scale=self.capacitor_scale or 1.0, esr_ohm=self.esr_ohm or 0.0, tau=TAU)
        return VscSystem(load_resistance_ohm=r_ohm, tau=TAU)

    def _physical(self) -> dict:
        s = self.plant_state
        return {"i_f_real": s.i_f_real, "i_f_imag": s.i_f_imag, "vc_real": s.vc_real, "vc_imag": s.vc_imag,
                "load_resistance_ohm": self.load_resistance_ohm, "reference_magnitude_v": self.reference.magnitude_v}

    def _obs(self, k: int) -> np.ndarray:
        s, sc = self.plant_state, self.metadata["observation_scales"]
        vr, vi = self.reference.at_step(k)
        return np.array([s.i_f_real / sc[0], s.i_f_imag / sc[1], s.vc_real / sc[2], s.vc_imag / sc[3], vr / sc[4], vi / sc[5]], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        plant_seed = _plant_seed(self, seed)
        if options.get("reference_magnitude_v") is not None:
            magnitude = float(options["reference_magnitude_v"])
        else:
            magnitude = self.magnitude_fixed if self.magnitude_fixed is not None else float(self.np_random.uniform(*self.magnitude_range))
        if options.get("load_resistance_ohm") is not None:
            self.load_resistance_ohm = float(options["load_resistance_ohm"])
        elif self.load_fixed is not None:
            self.load_resistance_ohm = self.load_fixed
        else:  # log-uniform: the range spans orders of magnitude
            lo, hi = self.load_range_ohm
            self.load_resistance_ohm = float(math.exp(self.np_random.uniform(math.log(lo), math.log(hi))))
        # the starting phase follows the seed exactly as in runner._run_vsc_scenario
        phase0 = float(options["phase0_rad"]) if options.get("phase0_rad") is not None else (plant_seed % 360) * (2 * np.pi / 360)
        self.reference = RotatingReference(magnitude_v=magnitude, tau=TAU, phase0_rad=phase0)
        self.plant = self._build(self.load_resistance_ohm)
        self.plant_state = self.plant.reset()
        self._k = 0
        self._prev_action = np.zeros(2)
        return self._obs(0), {**self._physical(), "plant_seed": plant_seed, "phase0_rad": phase0, "step": 0, "t_s": 0.0}

    def step(self, action):
        if self.plant_state is None:
            raise RuntimeError("Call reset() before step().")
        a = self._as_action(action, (2,))
        applied = np.clip(a, -self.voltage_limit_v, self.voltage_limit_v)
        self.plant_state = self.plant.simulate(float(applied[0]), float(applied[1]))
        vr, vi = self.reference.at_step(self._k)  # the reference this step was meant to reach
        s = self.plant_state
        signals = {"tracking_error": math.hypot(vr - s.vc_real, vi - s.vc_imag) / self.V_SCALE,
                   "current_fraction": math.hypot(s.i_f_real, s.i_f_imag) / self.current_limit_a,
                   "action_delta": float(np.hypot(*(applied - self._prev_action))) / (2 * self.voltage_limit_v)}
        self._prev_action = applied
        values = (s.i_f_real, s.i_f_imag, s.vc_real, s.vc_imag)
        violation = None
        if not all(math.isfinite(x) for x in values):
            violation = "non_finite_state"
        elif math.hypot(s.vc_real, s.vc_imag) > self.voltage_limit_v:
            violation = "limit_violation:capacitor_voltage"
        info = {**self._physical(), "v_ref_real": vr, "v_ref_imag": vi, "action_applied": applied.copy(), "action_clipped": bool(np.any(applied != a))}
        # the observation carries the reference of the NEXT step (k + 1)
        return self._finish_step(self._obs(self._k + 1), signals, violation, info)


class BuckEnv(_NativeEnv):
    """Averaged synchronous buck converter (plant ``buck_v1``, sim/buck_system.py; expansion task
    E6), output-voltage regulation with an optional load step. Built by
    ``datagen.runner.build_buck`` like run_scenario. Action: duty cycle in [0, 1]. Observation
    (float32): [i_L / i_max, v_out / V_in, v_ref / V_in, (v_ref − v_out) / V_in]. Step: the buck's
    own TAU (20 us, one switching period at 50 kHz)."""

    metadata = {
        "render_modes": [],
        "plant_profile_id": "native:buck_v1",
        "observation_names": ("i_l", "v_out", "v_ref", "v_error"),
        "observation_units": ("A", "V", "V", "V"),
        "action_names": ("duty_cycle",),
        "action_units": ("1",),
    }

    def __init__(self, vout_ref_v: float | None = 24.0, vout_ref_range: tuple = (5.0, 40.0), load_resistance_ohm: float | None = 6.0,
                 load_range_ohm: tuple = (3.0, 30.0), load_step_ohm: float | None = None, load_step_time_s: float | None = None,
                 duration_s: float = 0.02, vin_v: float = 48.0, parameter_overrides: dict | None = None, reward_terms=None,
                 render_mode: str | None = None):
        from driveflow.sim.buck_system import TAU as BUCK_TAU, BuckParams

        if render_mode is not None:
            raise ValueError("BuckEnv has no render modes; plot info signals instead.")
        if not (math.isfinite(duration_s) and duration_s > 0):
            raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
        for name, (lo, hi) in (("vout_ref_range", vout_ref_range), ("load_range_ohm", load_range_ohm)):
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(f"{name} must be finite with 0 < low <= high, got {(lo, hi)!r}")
        if vout_ref_range[1] >= vin_v:
            raise ValueError(f"vout_ref_range must stay below V_in = {vin_v:g} V")
        self.vout_ref_fixed = None if vout_ref_v is None else float(vout_ref_v)
        self.vout_ref_range = tuple(map(float, vout_ref_range))
        self.load_fixed = None if load_resistance_ohm is None else float(load_resistance_ohm)
        self.load_range_ohm = tuple(map(float, load_range_ohm))
        self.load_step_ohm, self.load_step_time_s = load_step_ohm, load_step_time_s
        self.vin_v = float(vin_v)
        self.parameter_overrides = dict(parameter_overrides or {})
        self.duration_s = float(duration_s)
        self.tau = BUCK_TAU
        self.n_steps = int(round(duration_s / BUCK_TAU))
        self._setup_reward(reward_terms)
        params = BuckParams(v_in_v=self.vin_v, **self.parameter_overrides)  # validates the overrides now
        self.i_max = params.i_max_a
        self.metadata = {**self.metadata, "tau_s": BUCK_TAU, "observation_scales": (self.i_max, self.vin_v, self.vin_v, self.vin_v),
                         "limits": {"current_a": self.i_max, "terminate_current_a": 2 * self.i_max, "vin_v": self.vin_v},
                         "parameters": dataclasses.asdict(params)}
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.plant = self.plant_state = self.vout_ref_v = self._step = None
        self._k = 0
        self._prev_action = np.zeros(1)

    def scenario(self, vout_ref_v: float, load_ohm: float, seed: int, load_step_ohm=None, load_step_time_s=None) -> Scenario:
        return Scenario(scenario_id="gym-episode", plant_config_id="buck_v1", controller_type="PI", duration_s=self.duration_s, seed=seed,
                        buck_vout_ref_v=vout_ref_v, buck_vin_v=self.vin_v, load_resistance_ohm=load_ohm,
                        buck_load_step_ohm=load_step_ohm, buck_load_step_time_s=load_step_time_s,
                        buck_parameter_overrides=self.parameter_overrides or None)

    def _physical(self) -> dict:
        s = self.plant_state
        return {"i_l_a": s.i_l, "v_out_v": s.v_out, "v_ref_v": self.vout_ref_v, "load_resistance_ohm": self.plant.load_resistance_ohm}

    def _obs(self) -> np.ndarray:
        s, sc = self.plant_state, self.metadata["observation_scales"]
        return np.array([s.i_l / sc[0], s.v_out / sc[1], self.vout_ref_v / sc[2], (self.vout_ref_v - s.v_out) / sc[3]], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        from driveflow.datagen.runner import build_buck

        super().reset(seed=seed)
        options = options or {}
        plant_seed = _plant_seed(self, seed)
        if options.get("vout_ref_v") is not None:
            vref = float(options["vout_ref_v"])
        else:
            vref = self.vout_ref_fixed if self.vout_ref_fixed is not None else float(self.np_random.uniform(*self.vout_ref_range))
        if options.get("load_resistance_ohm") is not None:
            load = float(options["load_resistance_ohm"])
        elif self.load_fixed is not None:
            load = self.load_fixed
        else:
            lo, hi = self.load_range_ohm
            load = float(math.exp(self.np_random.uniform(math.log(lo), math.log(hi))))
        step_ohm = options.get("load_step_ohm", self.load_step_ohm)
        step_t = options.get("load_step_time_s", self.load_step_time_s)
        self.plant, self.vout_ref_v, self._step = build_buck(self.scenario(vref, load, plant_seed, step_ohm, step_t))
        self.plant_state = self.plant.reset()
        self._k = 0
        self._prev_action = np.zeros(1)
        return self._obs(), {**self._physical(), "plant_seed": plant_seed, "load_step": self._step, "step": 0, "t_s": 0.0}

    def step(self, action):
        if self.plant_state is None:
            raise RuntimeError("Call reset() before step().")
        a = self._as_action(action, (1,))
        applied = np.clip(a, 0.0, 1.0)
        if self._step is not None and self._k == self._step[0]:
            self.plant.set_load(self._step[1])
        self.plant_state = self.plant.simulate(float(applied[0]))
        phys = self._physical()
        signals = {"tracking_error": (self.vout_ref_v - phys["v_out_v"]) / self.vin_v, "current_fraction": phys["i_l_a"] / self.i_max,
                   "action_delta": float(applied[0] - self._prev_action[0])}
        self._prev_action = applied
        violation = None
        if not (math.isfinite(phys["i_l_a"]) and math.isfinite(phys["v_out_v"])):
            violation = "non_finite_state"
        elif abs(phys["i_l_a"]) > 2 * self.i_max:
            violation = "limit_violation:inductor_current"
        info = {**phys, "action_applied": applied.copy(), "action_clipped": bool(np.any(applied != a))}
        return self._finish_step(self._obs(), signals, violation, info)
