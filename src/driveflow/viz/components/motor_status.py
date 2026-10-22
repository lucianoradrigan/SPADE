"""MotorStatusView: the motor illustration as an instrument, not a decoration (guide Sec. 4).

Inputs are a driveflow.monitoring.motor_status.MotorViewModel and nothing else -- colour, label
and motion all come from its fields, which are derived from backend evidence (ServerAgent active
alerts, evaluated/unevaluated, source mode).

Design decisions (guide Sec. 1.2 / 4.2):
- ONE stable base image (figures/motors/motor_nominal_cyan.png, resized to assets/motor_base.webp)
  for every state. The package's healthy/warning/critical renders are different generations with
  slightly different geometry and a status colour baked in; swapping them would make the motor
  "jump" and would bake meaning into pixels. State is an overlay: a status ring on the end bell, a
  soft halo, a text badge and a glyph.
- The rotation cue spins only when rpm > 0 from an available (non-offline) source, at a deliberately
  SLOWED rate that still scales with speed; the real rpm is printed as text next to it.
- Warning/critical halo pulses slowly (< 0.5 Hz, no flashing); healthy is steady; unknown/offline
  are desaturated and still. `prefers-reduced-motion: reduce` disables all motion.
- Accessible: role="img" with a full-text aria-label; status is also a text badge + glyph.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

from driveflow.monitoring.motor_status import MotorViewModel
from driveflow.viz.components.page_header import source_badge_html, status_badge_html
from driveflow.viz.theme.tokens import DARK, STATUS_META

ASSET = Path(__file__).resolve().parents[1] / "assets" / "motor_base.webp"
#: End-bell centre in the base image, as fractions of its width/height (measured on the asset).
_BELL_CX, _BELL_CY, _BELL_D = 22.0, 57.0, 33.0

_CSS = """
<style>
.spade-motor { border: 1px solid rgba(148,163,184,0.28); border-radius: 14px; padding: 0.7rem; background: rgba(148,163,184,0.05); }
.spade-motor-stage { position: relative; width: 100%; max-width: 440px; margin: 0 auto; aspect-ratio: 1 / 1; }
.spade-motor.is-compact .spade-motor-stage { max-width: 230px; }
.spade-motor-img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; z-index: 2; transition: filter .3s ease; }
.spade-motor[data-status="unknown"] .spade-motor-img { filter: grayscale(.45) brightness(.92); }
.spade-motor[data-status="offline"] .spade-motor-img { filter: grayscale(.9) brightness(.7); }
.spade-motor-halo { position: absolute; left: 6%; top: 18%; width: 88%; height: 70%; border-radius: 50%; z-index: 1; filter: blur(28px); opacity: .22; background: var(--tone); }
.spade-motor[data-status="unknown"] .spade-motor-halo, .spade-motor[data-status="offline"] .spade-motor-halo { opacity: 0; }
.spade-motor[data-status="warning"] .spade-motor-halo, .spade-motor[data-status="critical"] .spade-motor-halo { animation: spade-breathe 2.6s ease-in-out infinite; }
.spade-motor-frame { position: absolute; z-index: 3; left: calc(%(cx)s%% - %(r)s%%); top: calc(%(cy)s%% - %(r)s%%); width: %(d)s%%; height: %(d)s%%; transform: scaleX(.78) rotate(-6deg); pointer-events: none; }
.spade-motor-ring { position: absolute; inset: 4%%; border-radius: 50%%; border: 3px solid var(--tone); box-shadow: 0 0 14px var(--tone), inset 0 0 10px var(--tone); opacity: .85; }
.spade-motor[data-status="unknown"] .spade-motor-ring { border-style: dashed; box-shadow: none; opacity: .55; }
.spade-motor[data-status="offline"] .spade-motor-ring { border-style: dotted; box-shadow: none; opacity: .45; }
.spade-motor-rotor { position: absolute; inset: 18%%; border-radius: 50%%; border: 2px dashed rgba(226,232,240,.7); animation: spade-spin var(--period, 2s) linear infinite; animation-play-state: var(--play, paused); }
.spade-motor-rotor::after { content: ""; position: absolute; top: -5px; left: calc(50%% - 4px); width: 8px; height: 8px; border-radius: 50%%; background: #E2E8F0; }
.spade-motor-tags { position: absolute; z-index: 4; top: 4px; left: 4px; right: 4px; display: flex; gap: .35rem; flex-wrap: wrap; }
.spade-motor-meta { display: grid; grid-template-columns: auto 1fr; gap: .15rem .7rem; font-size: .84rem; margin-top: .45rem; }
.spade-motor-meta dt { opacity: .7; } .spade-motor-meta dd { margin: 0; font-variant-numeric: tabular-nums; }
.spade-motor-note { font-size: .74rem; opacity: .66; margin: .45rem 0 0 0; }
@keyframes spade-spin { to { transform: rotate(360deg); } }
@keyframes spade-breathe { 0%%, 100%% { opacity: .12; transform: scale(.96); } 50%% { opacity: .32; transform: scale(1.03); } }
@media (prefers-reduced-motion: reduce) { .spade-motor-rotor, .spade-motor-halo { animation: none !important; } }
</style>
""".replace("%(cx)s", str(_BELL_CX)).replace("%(cy)s", str(_BELL_CY)).replace("%(r)s", str(_BELL_D / 2)).replace("%(d)s", str(_BELL_D)).replace("%%", "%")


@lru_cache(maxsize=1)
def _image_data_uri() -> str:
    return "data:image/webp;base64," + base64.b64encode(ASSET.read_bytes()).decode()


def rotation_period_s(rpm: float | None) -> float | None:
    """Visible seconds per turn: monotonic in rpm, clamped to a calm 0.8-6 s range. None = still."""
    if rpm is None or rpm <= 0:
        return None
    return max(0.8, min(6.0, 3600.0 / rpm))


def motor_status_html(view: MotorViewModel, compact: bool = False) -> str:
    label, glyph, tone_key = STATUS_META[view.status]
    tone = DARK[tone_key]
    period = rotation_period_s(view.rpm) if view.animate else None
    play = "running" if period else "paused"
    rpm_txt = "—" if view.rpm is None else f"{view.rpm:,.0f} rpm"
    target_txt = "" if view.rpm_target is None else f" (target {view.rpm_target:,.0f})"
    aria = f"Motor {view.motor_id}: status {label}, source {view.source_mode}, speed {rpm_txt}. {view.status_reason}."
    meta = [("Status", f"{glyph} {label} — {view.status_reason}"), ("Speed", rpm_txt + target_txt), ("Source", view.source_mode.upper())]
    if view.timestamp_utc:
        meta.append(("Evaluated", view.timestamp_utc))
    if view.active_alerts:
        meta.append(("Active alerts", ", ".join(view.active_alerts)))
    if view.prediction is not None:
        conf = "" if view.confidence is None else f", p={view.confidence:.2f}"
        meta.append(("AI suggestion", f"{view.prediction}{conf} · {view.model_id or 'model n/a'} (not an alarm)"))
    meta_html = "" if compact else '<dl class="spade-motor-meta">' + "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in meta) + "</dl>"
    if period:
        real_turns_per_s = view.rpm / 60.0
        note = f"Illustrative render, not a measurement. Rotation cue slowed ~{real_turns_per_s * period:.0f}× relative to the real shaft speed."
    else:
        note = "Illustrative render, not a measurement. Rotation cue stopped (no valid positive speed from a simulation run or monitoring pass)."
    return (
        f'{_CSS}<div class="spade-root spade-motor{" is-compact" if compact else ""}" data-status="{view.status}" style="--tone:{tone}">'
        f'<div class="spade-motor-stage" role="img" aria-label="{escape(aria)}">'
        f'<div class="spade-motor-halo"></div><img class="spade-motor-img" src="{_image_data_uri()}" alt="">'
        f'<div class="spade-motor-frame" aria-hidden="true"><div class="spade-motor-ring"></div>'
        f'<div class="spade-motor-rotor" style="--period:{period or 2}s;--play:{play}"></div></div>'
        f'<div class="spade-motor-tags">{status_badge_html(view.status)}{source_badge_html(view.source_mode)}</div></div>'
        f'{meta_html}<p class="spade-motor-note">{escape(note)}</p></div>'
    )


def render_motor_status(view: MotorViewModel, compact: bool = False) -> None:
    st.html(motor_status_html(view, compact=compact))
