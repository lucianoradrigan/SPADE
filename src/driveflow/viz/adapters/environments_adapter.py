"""GUI boundary to the Gymnasium environment registry (expansion task E1; spec Sec. 8.5).

Everything shown comes from driveflow.envs.registry: the contract is read from a real instance
(``describe``), the snippet exists only for IDs registered in Gymnasium, and the capability
check runs the actual ``check_env`` (``verify``). Never raises into the page.
"""

from __future__ import annotations

import functools

from driveflow.envs import registry
from driveflow.runs.profiles import NATIVE_PROFILES


def env_ids() -> list[str]:
    registry.register_all()
    return list(registry.ENVIRONMENTS)


@functools.lru_cache(maxsize=None)
def contract(env_id: str) -> dict:
    """{..describe..} or {"error": message}. Cached: the contract depends only on the code."""
    try:
        return registry.describe(env_id)
    except Exception as exc:
        return {"env_id": env_id, "error": f"{type(exc).__name__}: {exc}"}


def declared_capability(env_id: str):
    profile = NATIVE_PROFILES.get(registry.ENVIRONMENTS[env_id].plant_profile_id)
    return profile.capabilities if profile else None


def verify(env_id: str) -> dict:
    """Runs the contract checks now: {"capability": EnvironmentCapability} or {"error": ...}."""
    try:
        return {"capability": registry.verify(env_id)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def snippet(env_id: str) -> str | None:
    """Python code instantiating the registered ID, or None when it is not registered."""
    return registry.snippet(env_id) if registry.is_registered(env_id) else None


def planned() -> list[dict]:
    return [{"name": name, "plant": plant, "task": task} for name, (plant, task) in registry.PLANNED.items()]
