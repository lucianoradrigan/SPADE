"""Plant profiles -- expansion task E0 (spec Sec. 2, 3.7, 3.8).

The native plants are described here so that every Run and Job can point at a profile id. Their
declared capabilities are the ones a test shows: batch simulation (datagen.runner.run_scenario and
the simulation tests) for all three, and since task E1 interactive stepping and episode reset for
the DC motor and the VSC (envs/native.py, checked by tests/envs/test_native_envs.py: check_env and
bit-for-bit equivalence with run_scenario). The Environments page re-runs those checks live with
envs.registry.verify. RL training stays False until a trainer exists (task E8); the PMSM has no
Gymnasium wrapper.

For file-backed profiles (PLECS, surrogate) the SHA-256 of the file is part of the profile:
``is_stale()`` reports when the file on disk no longer matches, so a validation can never be
reused for a different circuit.
"""

from __future__ import annotations

import dataclasses
import os

from driveflow.runs.schemas import EnvironmentCapability, PlantProfile
from driveflow.runs.store import sha256_file

_BATCH_ONLY = EnvironmentCapability(can_simulate_batch=True, diagnostic_messages=("Batch simulation via datagen.runner.run_scenario.", "Interactive stepping: no Gymnasium environment registered."))


def _stepping(env_id: str) -> EnvironmentCapability:
    return EnvironmentCapability(can_simulate_batch=True, can_step_interactively=True, can_reset_episode=True,
                                 diagnostic_messages=("Batch simulation via datagen.runner.run_scenario.",
                                                      f"Gymnasium environment {env_id} (tests/envs/test_native_envs.py).",
                                                      "RL training: no trainer in SPADE yet (task E8)."))

NATIVE_PROFILES = {
    p.profile_id: p
    for p in (
        PlantProfile(profile_id="native:dc_perm_ex_v1", kind="native", engine="driveflow.sim", display_name="DC motor (permanently excited)",
                     signals=("rpm", "torque_nm", "voltage_v", "current_r", "current_s", "current_t"), capabilities=_stepping("SPADE/DcMotor-v0"), status="published"),
        PlantProfile(profile_id="native:vsc_dpc_v1", kind="native", engine="driveflow.sim", display_name="VSC with L-C filter",
                     signals=("vc_real", "vc_imag", "i_f_real", "i_f_imag"), capabilities=_stepping("SPADE/Vsc-v0"), status="published"),
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
