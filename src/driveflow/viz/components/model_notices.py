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


#: AI-N40 (2026-11-15): the speeds the shipped dc_motor classifiers were trained and evaluated on (the diagnosis grid,
#: simulated). Above it they were never evaluated before 2026-11-15 and can be confidently wrong there.
DC_GRID_RAD_S = (80.0, 220.0)
ABOVE_GRID = ("**Outside the classifier's training grid — speed {w:.0f} rad/s > {hi:.0f} rad/s (AI-N40).** This classifier was "
              "trained and evaluated on {lo:.0f}–{hi:.0f} rad/s (simulated). Above their training range the dc_motor classifiers can be "
              "confidently wrong: at 350 rad/s a healthy motor was read as 'cage' (RPi5 model, 1.00). SPADE shows no diagnosis there "
              f"(decision D1). [AI audit §13.13]({AUDIT_URL})")
BELOW_GRID = ("**Outside the classifier's training grid — speed {w:.0f} rad/s < {lo:.0f} rad/s (AI-N29).** This classifier was "
              "trained and evaluated on {lo:.0f}–{hi:.0f} rad/s (simulated). Below their training range the dc_motor classifiers can be "
              "confidently wrong (the 2026-11-04 PC model at 50 rad/s). SPADE shows no diagnosis there (decision D1). "
              f"[AI audit §13.3]({AUDIT_URL})")


def speed_grid_message(omega_ref_rad_s: float | None, envelope=None) -> str | None:
    """The notice for a dc_motor speed setpoint outside a classifier's training range (its declared trained envelope when
    given, AI-N40; otherwise the diagnosis grid), or None inside it (or unknown)."""
    if omega_ref_rad_s is None:
        return None
    lo, hi = (float(envelope[0]), float(envelope[1])) if envelope else DC_GRID_RAD_S
    w = float(omega_ref_rad_s)
    if w > hi:
        return ABOVE_GRID.format(w=w, lo=lo, hi=hi)
    if w < lo:
        return BELOW_GRID.format(w=w, lo=lo, hi=hi)
    return None


def speed_grid_notice(st, omega_ref_rad_s: float | None, envelope=None) -> None:
    msg = speed_grid_message(omega_ref_rad_s, envelope)
    if msg:
        st.warning(msg, icon=":material/speed:")
