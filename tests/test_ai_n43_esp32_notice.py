"""AI-N43 (2026-11-16): the ESP32 low-speed notice quoted the B4 candidate's int8 numbers (esp32_dscnn/2026-11-10_run01,
never shipped: inner_race recall 0.35 vs float 1.00) as if they were the shipped model's. The shipped int8 run gets
inner_race 1.00 at 50/70 rad/s and misses outer_race instead (0.28), and its float model is as weak there.
The notice must name the shipped run and quote its own numbers from the B4 evaluation (simulated)."""

import json
from pathlib import Path

import yaml

from driveflow.viz.components.model_notices import ESP32_LOW_SPEED, ESP32_SHIPPED_RUN

ROOT = Path(__file__).resolve().parents[1]
B4 = json.loads((ROOT / "docs" / "audit" / "b4" / "b4_results.json").read_text(encoding="utf-8"))["tiers"]["ESP32"]


def test_the_notice_names_the_shipped_esp32_run():
    entries = yaml.safe_load((ROOT / "configs" / "registry.yaml").read_text(encoding="utf-8"))["entries"]
    assert entries["dc_motor/esp32/classifier"].endswith(ESP32_SHIPPED_RUN)
    assert f"`{ESP32_SHIPPED_RUN}`" in ESP32_LOW_SPEED


def test_the_notice_quotes_the_shipped_runs_own_low_speed_numbers():
    shipped = B4["sets"]["E4"]["promoted"]  # 'promoted' in B4 = the shipped run, the candidate's baseline
    keras = B4["keras_descriptive"]["E4"]["promoted"]
    assert f"recall {shipped['recall']['outer_race']:.2f}" in ESP32_LOW_SPEED
    assert f"{100 * shipped['abstention']:.0f} % of windows abstained" in ESP32_LOW_SPEED
    assert f"({keras['recall']['outer_race']:.2f})" in ESP32_LOW_SPEED
    assert shipped["recall"]["inner_race"] >= 0.99  # so 'misses inner-race faults' would be false for the shipped run


def test_the_candidates_numbers_are_not_attributed_to_the_shipped_run():
    cand = B4["int8_vs_keras"]["E4"]["int8"]["recall"]["inner_race"]
    assert f"{cand:.2f}" not in ESP32_LOW_SPEED and "inner-race" not in ESP32_LOW_SPEED
    assert "AI-N28" in ESP32_LOW_SPEED and "AI-N36" in ESP32_LOW_SPEED and "AI-N43" in ESP32_LOW_SPEED
