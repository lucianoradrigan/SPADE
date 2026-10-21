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
    )
}

#: Environments other tasks will add; shown as "Not registered", never instantiated.
PLANNED = {
    "SPADE/Pmsm-v0": ("PMSM — dq current tracking", "none assigned"),
    "PLECS plant environments": ("Circuit from PLECS via FMU co-simulation", "E3/E4"),
}


def register_all() -> list[str]:
    """Registers every ID in ENVIRONMENTS that is not registered yet; returns the IDs."""
    for spec in ENVIRONMENTS.values():
        if spec.env_id not in gym.registry:
            gym.register(id=spec.env_id, entry_point=spec.entry_point, kwargs=dict(spec.defaults), disable_env_checker=False)
    return list(ENVIRONMENTS)


def is_registered(env_id: str) -> bool:
    return env_id in ENVIRONMENTS and env_id in gym.registry


def make(env_id: str, **kwargs) -> gym.Env:
    register_all()
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
            "baseline_controller": spec.baseline_controller, "defaults": dict(spec.defaults),
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
