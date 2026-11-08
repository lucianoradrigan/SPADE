"""S-21 mitigation in the assistant (decision S21-3, approved by Luciano 2026-11-02; implemented 2026-11-08).

Decision E3 (Luciano, 2026-11-08): S-21 is corrected in the v2 plants (dc_perm_ex_v2 / dc_pmg132_v2, converter
"dcm_exact"), which the GUI and the assistant use; this warning is for the frozen v1 plants only.

S-21 (docs/SIMULATION_STABILITY_REPORT.md, docs/audit/assistant/s21_one_quadrant.md): when the speed PI
asks GEM's one-quadrant converter to brake and friction plus load are small, the simulated speed RISES
instead of coasting. Nothing here changes the plant, a default or a run: it only says so, as a
non-blocking WARNING, while S-21 is open.

Which motors (the motor-level warning, ``motor_reason``):
- ``dc_pmg132_v1`` (its default load is the scaled 2 % friction);
- ``dc_perm_ex_v1`` with a changed motor whose rated torque and speed are known, i.e. the 0b-D1
  friction (``runner.dc_default_load_parameter`` is not None).
The stock motor with its stock friction is not covered by S21-3 (s21_one_quadrant.md: not affected in
the shipped grids). With ``dc_converter_model="dcm"`` (S21-2) the effect is absent: no warning.

Which experiments (``condition_warning``): the motors above, in speed control (no ``torque_ref_nm``),
with a speed reference below the free speed and a load below 1 N*m:
- free speed: the no-load speed at full supply voltage, U_supply / psi_e (the speed the converter
  reaches with duty 1, friction neglected). A reference below it needs a duty below 1, so a speed
  overshoot (the start-up, a lower reference) asks the converter to brake;
- load: ``load_torque_nm`` (the extra passive load; absent = 0). 1 N*m is the level that removed the
  effect at 120 rad/s on dc_pmg132_v1 in s21_one_quadrant.md (decision S21-3).
"""

from __future__ import annotations

from functools import lru_cache

from driveflow.control.design_check import S21_LIGHT_LOAD_NM, s21_warning

DC_PLANTS = ("dc_perm_ex_v1", "dc_pmg132_v1")


def motor_reason(plant_config_id: str, motor_parameter_overrides: dict | None = None, dc_nominal_overrides: dict | None = None,
                 dc_converter_model: str | None = None) -> str | None:
    """Why this DC motor is exposed to S-21 (None = not covered by S21-3)."""
    from driveflow.datagen.runner import dc_default_load_parameter

    if plant_config_id not in DC_PLANTS or dc_converter_model == "dcm":
        return None
    if plant_config_id == "dc_pmg132_v1":
        return "dc_pmg132_v1 (scaled 2 % friction, round 0b D3)"
    if dc_default_load_parameter(motor_parameter_overrides, dc_nominal_overrides) is not None:
        return "changed motor with rated torque and speed: 0b-D1 friction (2 % of the rated torque)"
    return None


@lru_cache(maxsize=None)
def _stock(plant_config_id: str) -> tuple[float, float]:
    """(psi_e, supply V) of the plant's default motor."""
    from driveflow.datagen.runner import build_plant
    from driveflow.datagen.scenario import Scenario

    plant = build_plant(Scenario("s21_free_speed", plant_config_id=plant_config_id))
    return float(plant.electrical_motor.motor_parameter["psi_e"]), float(plant.supply.u_nominal)


def free_speed_rad_s(condition: dict) -> float:
    plant = condition.get("plant_config_id", "dc_perm_ex_v1")
    psi_stock, u_stock = _stock(plant)
    psi = float((condition.get("motor_parameter_overrides") or {}).get("psi_e", psi_stock))
    u = float(condition["dc_supply_v"]) if condition.get("dc_supply_v") is not None else u_stock
    return u / psi


def condition_warning(condition: dict) -> dict | None:
    """The S-21 warning for one experiment (Scenario keyword dict, SI values), or None."""
    from driveflow.datagen.scenario import Scenario

    plant = condition.get("plant_config_id", "dc_perm_ex_v1")
    reason = motor_reason(plant, condition.get("motor_parameter_overrides"), condition.get("dc_nominal_overrides"), condition.get("dc_converter_model"))
    if reason is None or condition.get("torque_ref_nm") is not None:
        return None
    omega_ref = float(condition.get("omega_ref_rad_s", Scenario.omega_ref_rad_s))
    load = float(condition.get("load_torque_nm") or 0.0)
    w_free = free_speed_rad_s(condition)
    if not (omega_ref < w_free and load < S21_LIGHT_LOAD_NM):
        return None
    return {**s21_warning(reason), "omega_ref_rad_s": omega_ref, "free_speed_rad_s": w_free, "load_torque_nm": load,
            "condition": f"speed reference {omega_ref:g} rad/s below the free speed {w_free:.1f} rad/s (U/psi_e) and load {load:g} N*m < {S21_LIGHT_LOAD_NM:g} N*m"}
