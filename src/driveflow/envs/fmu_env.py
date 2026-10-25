"""Gymnasium environments over OpenModelica FMUs -- round PLECS/OpenModelica, task B4.

``FmuGymEnv`` drives one FMI 2.0 Co-Simulation FMU through ``integrations.fmu.FmuAdapter``. What
the environment exposes is read from an ``FmuMapping`` (action input, observation outputs, units,
scales, limits, FMU parameters), never typed again in the class:

- ``reset(seed, options)`` re-instantiates the FMU (FmuAdapter.reset) with the episode's
  parameters; t = 0.
- ``step(action)`` sets the input and advances the SAME instance from t to t + Ts (one doStep).
- Actions outside the action space: ``action_out_of_range="clip"`` (default; clipped, and
  ``info["action_clipped"]`` says so, as the native environments do) or ``"reject"`` (ValueError).
  A non-finite action always raises ValueError.
- ``terminated`` = a physical limit was exceeded, the state is not finite, or the FMU failed
  (``info["termination"]`` says which; a solver failure is ``"solver_failure"`` with
  ``info["solver_error"]``). A solver failure is never a success: its reward is
  ``-solver_failure_penalty`` and the episode ends. ``truncated`` = the horizon was reached.
- Reward: the existing term schema (envs/reward.py) on the same dimensionless signals as the
  native environment of the same plant.

Only ``SPADE/OmBuck-v0`` (BuckAveraged, B3 verdict GO) is registered (decisions G1/G2,
2026-10-25). It replicates the native averaged buck (``SPADE/Buck-v0``): it validates the FMU
integration, it does not add other physics. The FMU's load R is a fixed parameter, so it is set at
reset; a load step inside an episode is not available here (it is in ``SPADE/Buck-v0``).

The environment is registered only when it can run: fmpy installed, the FMU built (data/fmu/ or
$SPADE_FMU_DIR) and binaries for this platform. ``availability()`` returns the reason otherwise.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from driveflow.envs import reward as rw

REPO_ROOT = Path(__file__).resolve().parents[3]


def fmu_dir() -> Path:
    return Path(os.environ.get("SPADE_FMU_DIR", REPO_ROOT / "data" / "fmu"))


@dataclasses.dataclass(frozen=True)
class Signal:
    name: str
    unit: str
    #: FMU output variable this observation reads, or None for a derived one (reference, error)
    fmu_output: str | None
    #: key of the scale (observation = physical value / scale) in the env's scales
    scale: str


@dataclasses.dataclass(frozen=True)
class FmuMapping:
    model: str
    plant_profile_id: str
    tau_s: float
    action_name: str
    action_input: str
    action_low: float
    action_high: float
    action_unit: str
    observations: tuple
    #: env argument -> FMU parameter name
    parameters: dict

    @property
    def fmu_file(self) -> str:
        return f"{self.model}.fmu"


BUCK_MAPPING = FmuMapping(
    model="BuckAveraged", plant_profile_id="openmodelica:BuckAveraged", tau_s=20e-6,  # sim/buck_system.TAU, the B3 communication step
    action_name="duty_cycle", action_input="duty", action_low=0.0, action_high=1.0, action_unit="1",
    observations=(Signal("i_l", "A", "i_L", "i_max"), Signal("v_out", "V", "v_out", "vin"), Signal("v_ref", "V", None, "vin"), Signal("v_error", "V", None, "vin")),
    parameters={"vin_v": "Vin", "load_resistance_ohm": "R", "l_h": "L", "c_f": "C", "r_l_ohm": "rL"},
)


def availability(mapping: FmuMapping = BUCK_MAPPING) -> tuple[bool, str]:
    """(True, path) when the environment can run here, else (False, the reason)."""
    try:
        from driveflow.integrations.fmu.adapter import FmuError, current_platform, inspect_fmu
    except Exception as exc:  # pragma: no cover -- the adapter imports only numpy
        return False, f"FMU adapter unavailable: {exc}"
    path = fmu_dir() / mapping.fmu_file
    try:
        plat = current_platform()
    except FmuError:
        return False, 'fmpy is not installed: pip install "driveflow[fmu]"'
    if not path.is_file():
        return False, f"{mapping.fmu_file} is not built in {fmu_dir()} (scripts/fmu/build_fmus.sh, or download a prebuilt FMU)"
    try:
        info = inspect_fmu(path)
    except FmuError as exc:
        return False, str(exc)
    if plat not in info.platforms:
        return False, f"{mapping.fmu_file} has binaries for {list(info.platforms) or 'no platform'}, not {plat}: Not supported on this machine"
    return True, str(path)


class FmuGymEnv(gym.Env):
    """Base class: FMU life cycle, spaces from the mapping, action-range policy, solver failures.
    Subclasses provide the episode set-up (``_episode``), the derived observations and the reward
    signals."""

    mapping: FmuMapping

    def __init__(self, duration_s: float, reward_terms=None, action_out_of_range: str = "clip", solver_failure_penalty: float = 100.0,
                 slave_factory=None, fmu_path: str | None = None):
        from driveflow.integrations.fmu.adapter import FmuAdapter

        if not (math.isfinite(duration_s) and duration_s > 0):
            raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
        if action_out_of_range not in ("clip", "reject"):
            raise ValueError(f"action_out_of_range must be 'clip' or 'reject', got {action_out_of_range!r}")
        if not (math.isfinite(solver_failure_penalty) and solver_failure_penalty > 0):
            raise ValueError(f"solver_failure_penalty must be finite and > 0, got {solver_failure_penalty!r}")
        m = self.mapping
        self.duration_s = float(duration_s)
        self.tau = m.tau_s
        self.n_steps = int(round(duration_s / m.tau_s))
        self.action_out_of_range = action_out_of_range
        self.solver_failure_penalty = float(solver_failure_penalty)
        self.reward_terms = rw.parse_terms(reward_terms)
        self.action_space = spaces.Box(m.action_low, m.action_high, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(m.observations),), dtype=np.float32)
        self.adapter = FmuAdapter(slave_factory)
        path = Path(fmu_path) if fmu_path else fmu_dir() / m.fmu_file
        self.fmu_info = self.adapter.load(path)
        self._k = 0
        self._outputs: dict | None = None
        self._prev_action = np.zeros(1)

    # -- subclass hooks -------------------------------------------------------------------------
    def _episode(self, options: dict) -> tuple[dict, dict]:
        """(FMU parameter values, episode quantities such as the reference) for this reset."""
        raise NotImplementedError

    def _derived(self, name: str) -> float:
        raise NotImplementedError

    def _signals(self, action: np.ndarray) -> dict:
        raise NotImplementedError

    def _violation(self) -> str | None:
        return None

    def _info(self) -> dict:
        """Physical values put in info (subclasses use the native environment's keys)."""
        return self._physical()

    # -- contract -------------------------------------------------------------------------------
    def _physical(self) -> dict:
        return {s.name: (self._outputs[s.fmu_output] if s.fmu_output else self._derived(s.name)) for s in self.mapping.observations}

    def _obs(self) -> np.ndarray:
        phys, sc = self._physical(), self.scales
        return np.array([phys[s.name] / sc[s.scale] for s in self.mapping.observations], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        params, self.episode = self._episode(options or {})
        self.adapter.reset(params={self.mapping.parameters[k]: v for k, v in params.items()}, seed=seed)
        self._outputs = self.adapter.get_outputs(check_range=False)
        self._k = 0
        self._prev_action = np.zeros(1)
        return self._obs(), {**self._info(), **self.episode, "fmu_parameters": params, "plant_seed": seed, "step": 0, "t_s": 0.0}

    def step(self, action):
        if self._outputs is None:
            raise RuntimeError("Call reset() before step().")
        from driveflow.integrations.fmu.adapter import FmuError

        a = np.asarray(action, dtype=np.float64).reshape(1)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"Action must be finite, got {action!r}")
        applied = np.clip(a, self.action_space.low, self.action_space.high)
        clipped = bool(np.any(applied != a))
        if clipped and self.action_out_of_range == "reject":
            raise ValueError(f"Action {a.tolist()} outside [{self.mapping.action_low}, {self.mapping.action_high}] (action_out_of_range='reject')")
        info = {"action_applied": applied.copy(), "action_clipped": clipped}
        try:
            self.adapter.set_inputs({self.mapping.action_input: float(applied[0])})
            self.adapter.do_step(self._k * self.tau, self.tau)
            self._outputs = self.adapter.get_outputs(check_range=False)
        except FmuError as exc:
            self._k += 1
            info.update(step=self._k, t_s=self._k * self.tau, termination="solver_failure", solver_error=str(exc), solver_error_code=exc.code,
                        reward_terms={"solver_failure": self.solver_failure_penalty})
            return self._obs_or_last(), -self.solver_failure_penalty, True, False, info
        signals = self._signals(applied)
        self._prev_action = applied
        reward, breakdown = rw.evaluate(self.reward_terms, signals)
        self._k += 1
        violation = self._violation()
        truncated = violation is None and self._k >= self.n_steps
        info.update(self._info(), step=self._k, t_s=self._k * self.tau, reward_terms=breakdown, signals=signals)
        if violation is not None:
            info["termination"] = violation
        if truncated:
            info["truncation"] = "horizon"
        return self._obs(), reward, violation is not None, truncated, info

    def _obs_or_last(self) -> np.ndarray:
        try:
            return self._obs()
        except Exception:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def close(self):
        self.adapter.close()
        super().close()


class OmBuckEnv(FmuGymEnv):
    """SPADE/OmBuck-v0: the OpenModelica BuckAveraged FMU with SPADE/Buck-v0's contract (action:
    duty in [0, 1]; observation [i_L / i_max, v_out / V_in, v_ref / V_in, (v_ref − v_out) / V_in];
    step 20 us; terminated when |i_L| > 2 i_max or the state is not finite)."""

    mapping = BUCK_MAPPING
    metadata = {
        "render_modes": [],
        "plant_profile_id": BUCK_MAPPING.plant_profile_id,
        "observation_names": tuple(s.name for s in BUCK_MAPPING.observations),
        "observation_units": tuple(s.unit for s in BUCK_MAPPING.observations),
        "action_names": (BUCK_MAPPING.action_name,),
        "action_units": (BUCK_MAPPING.action_unit,),
        "supports_load_step": False,
    }

    def __init__(self, vout_ref_v: float | None = 24.0, vout_ref_range: tuple = (5.0, 40.0), load_resistance_ohm: float | None = 6.0,
                 load_range_ohm: tuple = (3.0, 30.0), duration_s: float = 0.02, vin_v: float = 48.0, parameter_overrides: dict | None = None,
                 reward_terms=None, action_out_of_range: str = "clip", solver_failure_penalty: float = 100.0, render_mode: str | None = None,
                 slave_factory=None):
        from driveflow.sim.buck_system import BuckParams

        if render_mode is not None:
            raise ValueError("OmBuckEnv has no render modes; plot info signals instead.")
        for name, (lo, hi) in (("vout_ref_range", vout_ref_range), ("load_range_ohm", load_range_ohm)):
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(f"{name} must be finite with 0 < low <= high, got {(lo, hi)!r}")
        if vout_ref_range[1] >= vin_v:
            raise ValueError(f"vout_ref_range must stay below V_in = {vin_v:g} V")
        overrides = dict(parameter_overrides or {})
        unknown = set(overrides) - {"l_h", "c_f", "r_l_ohm", "i_max_a"}
        if unknown:
            raise ValueError(f"parameter_overrides: unknown {sorted(unknown)}; expected l_h, c_f, r_l_ohm, i_max_a")
        params = BuckParams(v_in_v=float(vin_v), **overrides)  # same defaults and validation as the native buck
        self.buck_params = params
        self.vin_v, self.i_max = params.v_in_v, params.i_max_a
        self.vout_ref_fixed = None if vout_ref_v is None else float(vout_ref_v)
        self.vout_ref_range = tuple(map(float, vout_ref_range))
        self.load_fixed = None if load_resistance_ohm is None else float(load_resistance_ohm)
        self.load_range_ohm = tuple(map(float, load_range_ohm))
        self.scales = {"i_max": self.i_max, "vin": self.vin_v}
        self.plant = None
        super().__init__(duration_s, reward_terms, action_out_of_range, solver_failure_penalty, slave_factory)
        self.metadata = {**self.metadata, "tau_s": self.tau, "observation_scales": tuple(self.scales[s.scale] for s in self.mapping.observations),
                         "limits": {"current_a": self.i_max, "terminate_current_a": 2 * self.i_max, "vin_v": self.vin_v},
                         "parameters": dataclasses.asdict(params), "fmu": {"model": self.fmu_info.model_name, "sha256": self.fmu_info.sha256,
                                                                         "path": self.fmu_info.path}}

    def _episode(self, options: dict):
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
        if options.get("load_step_ohm") is not None:
            raise ValueError("SPADE/OmBuck-v0 has no load step: the FMU's R is a fixed parameter (use SPADE/Buck-v0)")
        p = self.buck_params
        self.vout_ref_v = vref
        # what BuckVoltageController (the reference policy) and the RL benchmark read from a buck env
        self.plant = SimpleNamespace(params=dataclasses.replace(p, load_resistance_ohm=load), tau=self.tau, load_resistance_ohm=load)
        return ({"vin_v": p.v_in_v, "load_resistance_ohm": load, "l_h": p.l_h, "c_f": p.c_f, "r_l_ohm": p.r_l_ohm},
                {"v_ref_v": vref, "load_resistance_ohm": load})

    @property
    def plant_state(self):
        return None if self._outputs is None else SimpleNamespace(i_l=self._outputs["i_L"], v_out=self._outputs["v_out"])

    def _info(self) -> dict:
        return {"i_l_a": self._outputs["i_L"], "v_out_v": self._outputs["v_out"], "v_ref_v": self.vout_ref_v, "load_resistance_ohm": self.plant.load_resistance_ohm}

    def _derived(self, name: str) -> float:
        return {"v_ref": self.vout_ref_v, "v_error": self.vout_ref_v - self._outputs["v_out"]}[name]

    def _signals(self, action):
        return {"tracking_error": (self.vout_ref_v - self._outputs["v_out"]) / self.vin_v, "current_fraction": self._outputs["i_L"] / self.i_max,
                "action_delta": float(action[0] - self._prev_action[0])}

    def _violation(self):
        i, v = self._outputs["i_L"], self._outputs["v_out"]
        if not (math.isfinite(i) and math.isfinite(v)):
            return "non_finite_state"
        if abs(i) > 2 * self.i_max:
            return "limit_violation:inductor_current"
        return None


class MappedFmuEnv(FmuGymEnv):
    """An FMU environment published from Plant Studio (task B5): everything comes from the plant
    profile's mapping and Gym configuration (integrations/fmu/gym_profile.py), not from code.

    spec = {"fmu_path", "model", "plant_profile_id", "mapping": {"action": {input, name, unit, low, high},
    "observations": [{output, unit, scale}], "reference": {output, value} | None,
    "safety": {output, rated} | None}, "gym": {tau_s, duration_s, parameters, reward_terms, action_out_of_range}}.

    Observation: each mapped output / its scale, then (with a reference) reference / scale and
    (reference − output) / scale. Reward signals: tracking_error = (reference − output) / scale (0
    without a reference); current_fraction = safety output / rated (0 without one); action_delta.
    Terminated when |safety output| > 2 × rated (the native buck's rule) or the state is not finite."""

    metadata = {"render_modes": []}

    def __init__(self, spec: dict, reward_terms=None, duration_s: float | None = None, render_mode: str | None = None, slave_factory=None):
        if render_mode is not None:
            raise ValueError("MappedFmuEnv has no render modes.")
        m, g = spec["mapping"], spec["gym"]
        a, ref = m["action"], m.get("reference")
        obs = [Signal(o["output"], o["unit"], o["output"], o["output"]) for o in m["observations"]]
        if ref:
            unit = next(o["unit"] for o in m["observations"] if o["output"] == ref["output"])
            obs += [Signal(f"{ref['output']}_ref", unit, None, ref["output"]), Signal(f"{ref['output']}_error", unit, None, ref["output"])]
        self.mapping = FmuMapping(model=spec["model"], plant_profile_id=spec["plant_profile_id"], tau_s=float(g["tau_s"]), action_name=a["name"],
                                  action_input=a["input"], action_low=float(a["low"]), action_high=float(a["high"]), action_unit=a["unit"],
                                  observations=tuple(obs), parameters={k: k for k in g.get("parameters", {})})
        self.spec_dict = spec
        self.scales = {o["output"]: float(o["scale"]) for o in m["observations"]}
        self.reference, self.safety = ref, m.get("safety")
        super().__init__(float(duration_s if duration_s is not None else g["duration_s"]), reward_terms if reward_terms is not None else g.get("reward_terms"),
                         g.get("action_out_of_range", "clip"), float(g.get("solver_failure_penalty", 100.0)), slave_factory, spec["fmu_path"])
        self.metadata = {**MappedFmuEnv.metadata, "plant_profile_id": spec["plant_profile_id"], "tau_s": self.tau,
                         "observation_names": tuple(s.name for s in obs), "observation_units": tuple(s.unit for s in obs),
                         "observation_scales": tuple(self.scales[s.scale] for s in obs), "action_names": (a["name"],), "action_units": (a["unit"],),
                         "limits": ({"safety_output": self.safety["output"], "rated": self.safety["rated"], "terminate_at": 2 * self.safety["rated"]} if self.safety else {}),
                         "fmu": {"model": self.fmu_info.model_name, "sha256": self.fmu_info.sha256, "path": self.fmu_info.path}}

    def _episode(self, options: dict):
        params = dict(self.spec_dict["gym"].get("parameters", {}))
        return params, ({"reference": self.reference["value"]} if self.reference else {})

    def _derived(self, name: str) -> float:
        out, val = self.reference["output"], float(self.reference["value"])
        return val if name.endswith("_ref") else val - self._outputs[out]

    def _signals(self, action):
        track = 0.0
        if self.reference:
            out = self.reference["output"]
            track = (float(self.reference["value"]) - self._outputs[out]) / self.scales[out]
        cur = self._outputs[self.safety["output"]] / float(self.safety["rated"]) if self.safety else 0.0
        return {"tracking_error": track, "current_fraction": cur, "action_delta": float(action[0] - self._prev_action[0])}

    def _violation(self):
        if not all(math.isfinite(v) for v in self._outputs.values()):
            return "non_finite_state"
        if self.safety and abs(self._outputs[self.safety["output"]]) > 2 * float(self.safety["rated"]):
            return f"limit_violation:{self.safety['output']}"
        return None
