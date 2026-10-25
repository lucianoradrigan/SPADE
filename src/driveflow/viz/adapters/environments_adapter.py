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


def contract(env_id: str) -> dict:
    """{..describe..}, {"unavailable": why} for an environment that cannot run here (e.g. an FMU
    not built), or {"error": message}."""
    ok, reason = registry.availability(env_id)
    if not ok:
        spec = registry.ENVIRONMENTS[env_id]
        return {"env_id": env_id, "display_name": spec.display_name, "source": spec.source, "version": spec.version, "unavailable": reason, "note": spec.note}
    return _contract(env_id)


@functools.lru_cache(maxsize=None)
def _contract(env_id: str) -> dict:
    """Cached: the contract of an available environment depends only on the code."""
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
