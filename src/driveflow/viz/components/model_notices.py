"""Short, visible notices about known limitations of a model (R1, 2026-11-15; decision R-D5).

The ESP32 dc_motor classifier is fragile at low speed (AI-N28, AI-N36): its int8 file misses inner-race faults below the
training grid on simulated data. No ESP32 round is open; the limitation is shown where that model appears."""

from __future__ import annotations

AUDIT_URL = "https://github.com/lucianoradrigan/SPADE/blob/main/docs/AI_TECHNICAL_AUDIT.md"
ESP32_LOW_SPEED = (
    "**Known limitation — ESP32 classifier at low speed (AI-N28, AI-N36).** The int8 model is fragile below its training "
    "grid: on simulated runs at 50–70 rad/s it misses most inner-race faults (int8 recall 0.35 where its own float model "
    f"gets 1.00). Treat its outputs below 80 rad/s as unreliable. [AI audit §13.2, §13.10]({AUDIT_URL})"
)


def is_esp32_classifier(tier: str | None, block: str | None = "classifier", domain: str | None = "dc_motor") -> bool:
    return tier == "esp32" and block == "classifier" and domain in (None, "dc_motor")


def esp32_low_speed_notice(st) -> None:
    st.warning(ESP32_LOW_SPEED, icon=":material/speed:")
