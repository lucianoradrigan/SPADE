"""Plant profiles -- expansion task E0 (spec Sec. 2, 3.7, 3.8).

The native plants are described here so that every Run and Job can point at a profile id. Their
only declared capability is batch simulation, which datagen.runner.run_scenario provides and the
existing simulation tests exercise; interactive stepping and RL training stay False until the
Gymnasium wrappers of task E1 pass their own checks.

For file-backed profiles (PLECS, surrogate) the SHA-256 of the file is part of the profile:
``is_stale()`` reports when the file on disk no longer matches, so a validation can never be
reused for a different circuit.
"""

from __future__ import annotations

import dataclasses
import os

from driveflow.runs.schemas import EnvironmentCapability, PlantProfile
from driveflow.runs.store import sha256_file

_BATCH_ONLY = EnvironmentCapability(can_simulate_batch=True, diagnostic_messages=("Batch simulation via datagen.runner.run_scenario.", "Interactive stepping: not registered (task E1)."))

NATIVE_PROFILES = {
    p.profile_id: p
    for p in (
        PlantProfile(profile_id="native:dc_perm_ex_v1", kind="native", engine="driveflow.sim", display_name="DC motor (permanently excited)",
                     signals=("rpm", "torque_nm", "voltage_v", "current_r", "current_s", "current_t"), capabilities=_BATCH_ONLY, status="published"),
        PlantProfile(profile_id="native:vsc_dpc_v1", kind="native", engine="driveflow.sim", display_name="VSC with L-C filter",
                     signals=("vc_real", "vc_imag", "i_f_real", "i_f_imag"), capabilities=_BATCH_ONLY, status="published"),
        PlantProfile(profile_id="native:pmsm_v1", kind="native", engine="driveflow.sim", display_name="PMSM (FOC demo)",
                     signals=("i_d", "i_q", "u_d", "u_q", "omega_rad_s"), capabilities=_BATCH_ONLY, status="published"),
    )
}


def native_profile_for(plant_config_id: str) -> PlantProfile | None:
    return NATIVE_PROFILES.get(f"native:{plant_config_id}")


def is_stale(profile: PlantProfile, path: str | os.PathLike) -> bool:
    """True when the profile's file hash no longer matches the file (or the file is gone)."""
    if profile.file_hash is None:
        return False
    try:
        return sha256_file(path) != profile.file_hash
    except OSError:
        return True


def mark_stale(profile: PlantProfile) -> PlantProfile:
    """New version with status 'stale' and its capabilities reset: 'Revalidation required'."""
    return dataclasses.replace(profile, status="stale", version=profile.version + 1,
                               capabilities=EnvironmentCapability(diagnostic_messages=("Validation is stale. Re-run environment verification.",)))
