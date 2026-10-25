"""Stable Gymnasium IDs for SPADE's environments and what each one can do -- expansion task E1.

``import driveflow.envs`` registers every ID below with ``gymnasium.register`` (idempotent), so
``gymnasium.make("SPADE/DcMotor-v0")`` works after that import. ``PLANNED`` lists environments that
other tasks will add; they are never registered here and the Environments page shows them as
"Not registered".

Capabilities are not declared by hand: ``verify(env_id)`` runs Gymnasium's own
``check_env`` plus a seeded reset/step determinism check on a fresh instance and returns the
resulting EnvironmentCapability with its timestamp. ``can_train_rl`` stays False until a trainer
exists (task E8).
"""

from __future__ import annotations

import dataclasses
import re
import traceback
import warnings

import gymnasium as gym
import numpy as np

from driveflow.envs import reward as rw
from driveflow.runs.schemas import EnvironmentCapability, utc_now

NAMESPACE = "SPADE"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    env_id: str
    entry_point: str
    plant_profile_id: str
    display_name: str
    version: int
    source: str
    compatible_algorithms: tuple
    terminations: tuple
    baseline_controller: str
    defaults: dict = dataclasses.field(default_factory=dict)
    #: "module:function" returning (available, reason); None = always available (native plants)
    availability: str | None = None
    #: shown on the Environments page next to the contract
    note: str | None = None


ENVIRONMENTS = {
    s.env_id: s
    for s in (
        EnvSpec("SPADE/DcMotor-v0", "driveflow.envs.native:DcMotorEnv", "native:dc_perm_ex_v1", "DC motor — speed tracking", 0, "SIMULATED",
                ("Continuous-action RL (DDPG, SAC, PPO)", "DQN after discretizing the duty cycle", "PI cascade (reference policy)"),
                ("|ω| > 400 rad/s", "|i| > 210 A", "non-finite state", "horizon reached → truncated"),
                "control.classical.PICascadeController", {"omega_ref_rad_s": 150.0, "duration_s": 0.5}),
        EnvSpec("SPADE/Vsc-v0", "driveflow.envs.native:VscEnv", "native:vsc_dpc_v1", "VSC with L-C filter — voltage tracking", 0, "SIMULATED",
                ("Continuous-action RL (DDPG, SAC, PPO)", "DPC network (reference policy)"),
                ("|v_c| > voltage limit (2000 V)", "non-finite state", "horizon reached → truncated"),
                "control.dpc.controller.ScheduledDpcController", {"reference_magnitude_v": 50.0, "load_resistance_ohm": 8.0064, "duration_s": 0.1}),
        EnvSpec("SPADE/Buck-v0", "driveflow.envs.native:BuckEnv", "native:buck_v1", "Buck converter (averaged) — voltage regulation", 0, "SIMULATED",
                ("Continuous-action RL (DDPG, SAC, PPO)", "DQN after discretizing the duty cycle", "PI cascade (reference policy)"),
                ("|i_L| > 2 × i_max (30 A)", "non-finite state", "horizon reached → truncated"),
                "control.classical.buck_pi.BuckVoltageController", {"vout_ref_v": 24.0, "load_resistance_ohm": 6.0, "duration_s": 0.02}),
        # E5 (round 2026-10-21): fitted to SPADE's own buck, not to PLECS; see envs/surrogate_env.py.
        EnvSpec("SPADE/BuckSurrogate-v0", "driveflow.envs.surrogate_env:BuckSurrogateEnv", "surrogate:buck_surrogate",
                "Buck surrogate (fitted to the native buck) — voltage regulation", 0, "SIMULATED · SURROGATE",
                ("Continuous-action RL (DDPG, SAC, PPO)", "DQN after discretizing the duty cycle", "PI cascade (reference policy)"),
                ("|i_L| > 2 × i_max (30 A)", "non-finite state", "horizon reached → truncated", "info['out_of_domain'] outside the fitted range (warning, not a termination)"),
                "control.classical.buck_pi.BuckVoltageController", {"vout_ref_v": 24.0, "load_resistance_ohm": 6.0, "duration_s": 0.02}),
        # Round 2026-10-21, decision D10: the existing PMSM simulation (solver unchanged, S-01/S-13).
        EnvSpec("SPADE/Pmsm-v0", "driveflow.envs.native:PmsmEnv", "native:pmsm_v1", "PMSM — dq current tracking (MTPA reference)", 0, "SIMULATED",
                ("Continuous-action RL (DDPG, SAC, PPO)", "FOC DqCurrentController (reference policy)"),
                ("|i_s| > 1.25 × I_LIM (500 A)", "non-finite state", "horizon reached → truncated"),
                "control.classical.pmsm_foc.DqCurrentController", {"current_ref_a": 200.0, "strategy": "mtpa", "duration_s": 0.03}),
        # Round PLECS/OM, B4 (decisions G1/G2): only BuckAveraged, B3 verdict GO.
        EnvSpec("SPADE/OmBuck-v0", "driveflow.envs.fmu_env:OmBuckEnv", "openmodelica:BuckAveraged",
                "Buck converter (averaged, OpenModelica FMU) — voltage regulation", 0, "SIMULATED · OPENMODELICA",
                ("Continuous-action RL (DDPG, SAC, PPO)", "DQN after discretizing the duty cycle", "PI cascade (reference policy)"),
                ("|i_L| > 2 × i_max (30 A)", "non-finite state", "FMU solver failure (never a success; reward −solver_failure_penalty)", "horizon reached → truncated"),
                "control.classical.buck_pi.BuckVoltageController", {"vout_ref_v": 24.0, "load_resistance_ohm": 6.0, "duration_s": 0.02},
                availability="driveflow.envs.fmu_env:availability",
                note=("Replicates the native averaged buck (SPADE/Buck-v0) through the OpenModelica FMU BuckAveraged: it validates the FMU "
                      "integration and adds no other physics (B3 criterion 4: 1.7e-6 / 2.1e-6 NRMS on v_out / i_L). The load R is an FMU "
                      "parameter, set at reset; no load step inside an episode.")),
    )
}

#: Environments other tasks will add; shown as "Not registered", never instantiated.
PLANNED = {
    "PLECS plant environments": ("Circuit from PLECS via FMU co-simulation", "E3/E4"),
}


def register_dynamic(env_id: str, profile_id: str, surrogate_dir: str) -> EnvSpec:
    """A surrogate fitted to a PLECS sweep (task A1): SPADE/<Profile>Surrogate-v0 on that surrogate,
    with SPADE/BuckSurrogate-v0's contract. Added to ENVIRONMENTS so the Environments page, RL Lab
    and describe() treat it like any other environment."""
    spec = EnvSpec(env_id, "driveflow.envs.surrogate_env:BuckSurrogateEnv", f"surrogate:plecs:{profile_id}",
                   f"Buck surrogate fitted to PLECS profile {profile_id} — voltage regulation", 0, "SIMULATED · SURROGATE",
                   ENVIRONMENTS["SPADE/BuckSurrogate-v0"].compatible_algorithms, ENVIRONMENTS["SPADE/BuckSurrogate-v0"].terminations,
                   "control.classical.buck_pi.BuckVoltageController",
                   {**ENVIRONMENTS["SPADE/BuckSurrogate-v0"].defaults, "surrogate_path": surrogate_dir})
    ENVIRONMENTS[env_id] = spec
    if env_id in gym.registry:
        del gym.registry[env_id]
    gym.register(id=env_id, entry_point=spec.entry_point, kwargs=dict(spec.defaults), disable_env_checker=False)
    return spec


def register_fmu_profile(env_id: str, record: dict) -> EnvSpec:
    """An FMU plant published from Plant Studio (task B5): SPADE/Fmu<Profile>-v0 on
    envs.fmu_env.MappedFmuEnv, configured entirely by the profile's mapping and Gym settings."""
    spec = record["spec"]
    env = EnvSpec(env_id, "driveflow.envs.fmu_env:MappedFmuEnv", spec["plant_profile_id"],
                  f"{spec['model']} (OpenModelica FMU, profile {record['profile_id']} v{record['version']})", 0, "SIMULATED · OPENMODELICA",
                  ("Continuous-action RL (DDPG, SAC, PPO)", "DQN after discretizing the action"),
                  ("|safety output| > 2 × rated", "non-finite state", "FMU solver failure (never a success)", "horizon reached → truncated"),
                  "—", {"spec": spec},
                  note=f"Published from PLECS Plant Studio (OpenModelica FMU engine). FMU SHA-256 {record['fmu_sha256'][:12]}…, configuration {record['config_hash'][:12]}…. Simulated, not hardware.")
    ENVIRONMENTS[env_id] = env
    if env_id in gym.registry:
        del gym.registry[env_id]
    gym.register(id=env_id, entry_point=env.entry_point, kwargs=dict(env.defaults), disable_env_checker=False)
    return env


def _published_fmu_profiles() -> dict:
    """Published FMU environments whose FMU file still has the verified SHA-256 (a changed FMU is
    stale and is not registered)."""
    import hashlib
    from pathlib import Path

    from driveflow.integrations.fmu import gym_profile

    out = {}
    for env_id, rec in gym_profile.published().items():
        path = Path(rec["spec"]["fmu_path"])
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == rec["fmu_sha256"]:
            out[env_id] = rec
    return out


def availability(env_id: str) -> tuple[bool, str]:
    """(True, "") for environments that can run here; else (False, why) -- e.g. an FMU-backed
    environment without fmpy or without its built FMU."""
    spec = ENVIRONMENTS[env_id]
    if spec.availability is None:
        return True, ""
    import importlib

    module, func = spec.availability.split(":")
    try:
        ok, reason = getattr(importlib.import_module(module), func)()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return (True, "") if ok else (False, reason)


def register_all() -> list[str]:
    """Registers every ID in ENVIRONMENTS that is not registered yet, plus the published PLECS
    surrogates (envs.surrogate_plecs.published); returns the IDs."""
    from driveflow.envs import surrogate_plecs

    for env_id, rec in surrogate_plecs.published().items():
        if env_id not in ENVIRONMENTS or ENVIRONMENTS[env_id].defaults.get("surrogate_path") != rec["surrogate_dir"]:
            register_dynamic(env_id, rec["profile_id"], rec["surrogate_dir"])
    fmu_published = _published_fmu_profiles()
    for env_id in [e for e, sp in ENVIRONMENTS.items() if sp.entry_point.endswith(":MappedFmuEnv") and e not in fmu_published]:
        ENVIRONMENTS.pop(env_id)  # unpublished, stale, or from another workspace
        if env_id in gym.registry:
            del gym.registry[env_id]
    for env_id, rec in fmu_published.items():
        if env_id not in ENVIRONMENTS or ENVIRONMENTS[env_id].defaults.get("spec") != rec["spec"]:
            register_fmu_profile(env_id, rec)
    for spec in ENVIRONMENTS.values():
        if spec.availability is not None and not availability(spec.env_id)[0]:
            if spec.env_id in gym.registry:  # e.g. the FMU was removed since the last call
                del gym.registry[spec.env_id]
            continue
        if spec.env_id not in gym.registry:
            gym.register(id=spec.env_id, entry_point=spec.entry_point, kwargs=dict(spec.defaults), disable_env_checker=False)
    return list(ENVIRONMENTS)


#: reference controller (EnvSpec.baseline_controller) -> what RL Lab and Tuning Studio can do with an
#: environment that exposes that controller's plant interface (B6: listed by capability, not by name)
_CAPABILITY_BY_BASELINE = {"control.classical.buck_pi.BuckVoltageController": "buck", "control.dpc.controller.ScheduledDpcController": "vsc"}


def rl_kind(env_id: str) -> str | None:
    """"buck" or "vsc" when the RL job can train on and benchmark this environment (its reference
    controller is supported by control.rl.benchmark), else None."""
    spec = ENVIRONMENTS.get(env_id)
    return None if spec is None else _CAPABILITY_BY_BASELINE.get(spec.baseline_controller)


def rl_env_ids() -> list[str]:
    """Registered, available environments with an RL capability."""
    register_all()
    order = {"buck": 0, "vsc": 1}  # buck plants first, as RL Lab has always listed them
    return sorted((e for e in ENVIRONMENTS if rl_kind(e) and is_registered(e)), key=lambda e: order[rl_kind(e)])


def is_registered(env_id: str) -> bool:
    return env_id in ENVIRONMENTS and env_id in gym.registry


def make(env_id: str, **kwargs) -> gym.Env:
    register_all()
    ok, reason = availability(env_id) if env_id in ENVIRONMENTS else (True, "")
    if not ok:
        raise RuntimeError(f"{env_id} is not available here: {reason}")
    return gym.make(env_id, **kwargs)


def describe(env_id: str) -> dict:
    """Contract of a registered environment, read from a real instance (no hand-written copy)."""
    spec = ENVIRONMENTS[env_id]
    env = make(env_id)
    try:
        u = env.unwrapped
        md = u.metadata
        return {
            "env_id": env_id, "display_name": spec.display_name, "plant_profile_id": spec.plant_profile_id, "version": spec.version,
            "source": spec.source, "registered": is_registered(env_id),
            "observation": {"shape": tuple(env.observation_space.shape), "dtype": str(env.observation_space.dtype),
                            "names": md["observation_names"], "units": md["observation_units"], "scales": md["observation_scales"]},
            "action": {"type": type(env.action_space).__name__, "shape": tuple(env.action_space.shape),
                       "low": env.action_space.low.tolist(), "high": env.action_space.high.tolist(),
                       "names": md["action_names"], "units": md["action_units"]},
            "tau_s": md["tau_s"], "horizon_steps": u.n_steps, "limits": md["limits"],
            "reward_terms": [{**t.to_dict(), "formula": rw.TERM_KINDS[t.kind][1]} for t in u.reward_terms],
            "terminations": spec.terminations, "compatible_algorithms": spec.compatible_algorithms,
            "baseline_controller": spec.baseline_controller, "defaults": dict(spec.defaults), "note": spec.note,
            **({"surrogate": {"family": md["surrogate_family"], "run": md["surrogate_run"], "validity_domain": md["validity_domain"], "fit_error": md["fit_error"]}}
               if "surrogate_family" in md else {}),
        }
    finally:
        env.close()


def snippet(env_id: str) -> str:
    if not is_registered(env_id):
        raise KeyError(f"{env_id} is not registered")
    return (
        "import gymnasium as gym\n"
        "import driveflow.envs  # registers the SPADE/* environment IDs\n\n"
        f'env = gym.make("{env_id}")\n'
        "obs, info = env.reset(seed=0)\n"
        "for _ in range(100):\n"
        "    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())\n"
        "    if terminated or truncated:\n"
        "        obs, info = env.reset()\n"
        "env.close()\n"
    )


def _determinism(env_id: str, n: int = 50) -> str | None:
    """None if two fresh instances give identical observations for the same seed and actions."""
    runs = []
    for _ in range(2):
        env = make(env_id)
        env.action_space.seed(123)
        obs, _ = env.reset(seed=7)
        trace = [obs]
        for _ in range(n):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            trace.append(np.append(obs, r))
            if term or trunc:
                break
        env.close()
        runs.append(trace)
    same = len(runs[0]) == len(runs[1]) and all(np.array_equal(a, b) for a, b in zip(*runs))
    return None if same else "Two resets with the same seed and actions diverged."


def verify(env_id: str) -> EnvironmentCapability:
    """Runs the contract checks now and returns the capability they support."""
    from gymnasium.utils.env_checker import check_env

    messages = []
    try:
        env = make(env_id)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env.unwrapped, skip_render_check=True)
        env.close()
        messages.append("gymnasium.utils.env_checker.check_env passed.")
        messages += [f"check_env warning: {_ANSI.sub('', str(w.message)).removeprefix('WARN: ')}" for w in caught]
        problem = _determinism(env_id)
        if problem:
            return EnvironmentCapability(can_simulate_batch=True, validated_at=utc_now(), diagnostic_messages=(*messages, problem))
        messages.append("Seeded reset and step are deterministic.")
    except Exception as exc:
        return EnvironmentCapability(can_simulate_batch=True, validated_at=utc_now(),
                                     diagnostic_messages=(f"Contract check failed: {type(exc).__name__}: {exc}", traceback.format_exc(limit=3)))
    messages.append("RL training: no trainer in SPADE yet (task E8).")
    return EnvironmentCapability(can_simulate_batch=True, can_step_interactively=True, can_reset_episode=True, can_train_rl=False,
                                 validated_at=utc_now(), diagnostic_messages=tuple(messages))
