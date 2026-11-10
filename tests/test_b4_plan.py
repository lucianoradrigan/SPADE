"""B4 plan (decision D-B4-3, Luciano 2026-11-09): the healthy off-grid conditions added to the ESP32 training do not
coincide with the evaluation conditions: not at the held-out low speeds (50, 70 rad/s), not a grid cell, not an E3 condition."""

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN = yaml.safe_load((ROOT / "configs" / "b4_plan.yaml").read_text())


def _e3_conditions() -> list[dict]:
    spec = importlib.util.spec_from_file_location("rce", ROOT / "experiments" / "report_classifier_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [kw for kw, _scale, _part in mod.CONDITIONS.values()]


def test_offgrid_healthy_is_not_at_the_heldout_low_speeds():
    e4 = set(PLAN["e4_low_speed_heldout_rad_s"])
    assert e4 == {50, 70}
    assert not [c for c in PLAN["esp32_offgrid_healthy"] if c["omega_ref_rad_s"] in e4]


def test_heldout_low_speeds_are_not_trained_on():
    assert not set(PLAN["e4_low_speed_heldout_rad_s"]) & set(PLAN["grid_speeds_rad_s"])


def test_offgrid_healthy_conditions_are_not_grid_cells():
    for c in PLAN["esp32_offgrid_healthy"]:
        only_speed = set(c) == {"omega_ref_rad_s"}
        assert not (only_speed and c["omega_ref_rad_s"] in PLAN["grid_speeds_rad_s"]), c


def test_offgrid_healthy_conditions_are_not_e3_conditions():
    e3_speeds = {kw.get("omega_ref_rad_s") for kw in _e3_conditions() if set(kw) <= {"omega_ref_rad_s"}}
    for c in PLAN["esp32_offgrid_healthy"]:
        if set(c) == {"omega_ref_rad_s"}:
            assert c["omega_ref_rad_s"] not in e3_speeds, c


def test_sixty_runs_and_the_approved_thresholds():
    assert len(PLAN["esp32_offgrid_healthy"]) * PLAN["esp32_offgrid_healthy_seeds_per_condition"] == 60
    assert PLAN["criteria"]["int8_vs_keras"] == {"max_recall_drop_pts_per_class": 5.0, "max_macro_f1_drop_pts": 3.0}
