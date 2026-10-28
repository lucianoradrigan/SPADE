"""Scoped CSS for SPADE's own components (guide Sec. 3.3 / 5.3).

Rules:
- Only `.spade-*` classes defined by this app are styled -- no selectors on Streamlit/BaseWeb
  internals (`data-baseweb`, `stSidebar`, generated class names), which change between Streamlit
  releases. Native widgets are themed by .streamlit/config.toml instead.
- Theme-agnostic: text inherits the native theme's colour, surfaces/borders are translucent
  neutrals, and status colour is applied to borders/tints/glyphs -- never as the only carrier of
  meaning, and never as body-text colour (so contrast follows the active light/dark theme).
- Motion respects `prefers-reduced-motion`.
"""

import streamlit as st

from driveflow.viz.theme.tokens import DARK, active_palette

_CSS = """
<style>
.spade-root, .spade-root * { box-sizing: border-box; }
.spade-eyebrow { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: var(--spade-primary, %(primary)s); margin: 0 0 0.15rem 0; }
.spade-title { font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.25rem 0; }
.spade-desc { font-size: 0.98rem; opacity: 0.78; margin: 0 0 0.6rem 0; max-width: 72ch; }
.spade-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem 1.25rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
.spade-badges { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }

.spade-badge { --tone: %(unknown)s; display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.18rem 0.6rem; border-radius: 999px;
  border: 1px solid var(--tone); background: color-mix(in srgb, var(--tone) 14%%, transparent); font-size: 0.74rem; font-weight: 650; letter-spacing: 0.04em; white-space: nowrap; }
.spade-badge .spade-glyph { color: var(--tone); font-size: 0.8rem; line-height: 1; }

.spade-kpi { --tone: %(primary)s; border: 1px solid rgba(148,163,184,0.28); border-left: 3px solid var(--tone); border-radius: 10px; padding: 0.7rem 0.9rem; background: rgba(148,163,184,0.06); height: 100%%; }
.spade-kpi-label { font-size: 0.8rem; opacity: 0.75; margin-bottom: 0.15rem; }
.spade-kpi-value { font-size: 1.55rem; font-weight: 700; line-height: 1.15; font-variant-numeric: tabular-nums; }
.spade-kpi-unit { font-size: 0.9rem; font-weight: 500; opacity: 0.75; margin-left: 0.2rem; }
.spade-kpi-sub { font-size: 0.78rem; opacity: 0.7; margin-top: 0.15rem; }

.spade-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.5rem; margin: 0.3rem 0 0.8rem 0; }
.spade-step { --tone: rgba(148,163,184,0.45); display: flex; gap: 0.6rem; align-items: center; padding: 0.55rem 0.7rem; border-radius: 10px; border: 1px solid rgba(148,163,184,0.25); background: rgba(148,163,184,0.05); }
.spade-step.is-active { border-color: var(--spade-primary, %(primary)s); background: color-mix(in srgb, %(primary)s 10%%, transparent); }
.spade-step-num { flex: 0 0 auto; width: 1.9rem; height: 1.9rem; border-radius: 50%%; display: grid; place-items: center; font-weight: 700; border: 2px solid var(--tone); }
.spade-step.is-active .spade-step-num { border-color: %(primary)s; }
.spade-step.is-done .spade-step-num { border-color: %(healthy)s; }
.spade-step-title { font-weight: 650; font-size: 0.92rem; }
.spade-step-sub { font-size: 0.75rem; opacity: 0.7; }
@media (max-width: 900px) { .spade-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.spade-checks { display: flex; flex-direction: column; gap: 0.3rem; }
.spade-check { --tone: %(unknown)s; display: grid; grid-template-columns: 1.4rem 1fr auto; gap: 0.5rem; align-items: center; padding: 0.4rem 0.55rem; border-radius: 8px; border: 1px solid rgba(148,163,184,0.2); }
.spade-check-glyph { color: var(--tone); font-weight: 800; text-align: center; }
.spade-check-name { font-weight: 600; font-size: 0.88rem; }
.spade-check-detail { font-size: 0.78rem; opacity: 0.72; }

.spade-prov { width: 100%%; border-collapse: collapse; font-size: 0.85rem; }
.spade-prov td { padding: 0.28rem 0.4rem; border-bottom: 1px solid rgba(148,163,184,0.18); vertical-align: top; }
.spade-prov td:first-child { opacity: 0.72; width: 38%%; }
.spade-prov code { font-size: 0.8rem; }

.spade-flow { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: stretch; }
.spade-flow-node { flex: 1 1 9rem; border: 1px solid rgba(148,163,184,0.28); border-radius: 10px; padding: 0.55rem 0.7rem; background: rgba(148,163,184,0.05); }
.spade-flow-node b { display: block; font-size: 0.9rem; }
.spade-flow-node span { font-size: 0.76rem; opacity: 0.72; }
.spade-flow-arrow { align-self: center; opacity: 0.55; }

.spade-groupflow { display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: stretch; margin: 0.2rem 0 0.9rem 0; }
.spade-groupflow-node { --tone: %(primary)s; flex: 1 1 10rem; display: flex; gap: 0.6rem; align-items: center; padding: 0.6rem 0.75rem; border-radius: 12px;
  border: 1px solid color-mix(in srgb, var(--tone) 45%%, transparent); background: linear-gradient(135deg, color-mix(in srgb, var(--tone) 16%%, transparent), transparent 75%%); }
.spade-groupflow-num { flex: 0 0 auto; width: 2.1rem; height: 2.1rem; border-radius: 10px; display: grid; place-items: center; font-weight: 800; font-size: 0.85rem;
  border: 2px solid var(--tone); background: color-mix(in srgb, var(--tone) 18%%, transparent); font-variant-numeric: tabular-nums; }
.spade-groupflow-node b { display: block; font-size: 0.92rem; }
.spade-groupflow-node span { font-size: 0.75rem; opacity: 0.72; }
.spade-groupflow-arrow { align-self: center; font-size: 1.1rem; opacity: 0.5; }
@media (max-width: 900px) { .spade-groupflow-arrow { display: none; } }

.spade-group { --tone: %(primary)s; position: relative; border-radius: 10px; padding: 0.85rem 0.95rem 0.8rem 1.1rem; margin-bottom: 0.25rem; overflow: hidden;
  background: linear-gradient(120deg, color-mix(in srgb, var(--tone) 18%%, transparent), color-mix(in srgb, var(--tone) 3%%, transparent) 70%%); }
.spade-group::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--tone); }
.spade-group-top { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.spade-group-num { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; padding: 0.1rem 0.45rem; border-radius: 6px; border: 1px solid var(--tone); font-variant-numeric: tabular-nums; }
.spade-group-title { font-size: 1.15rem; font-weight: 700; flex: 1 1 auto; }
.spade-group-count { font-size: 0.72rem; font-weight: 650; letter-spacing: 0.04em; padding: 0.15rem 0.55rem; border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--tone) 60%%, transparent); background: color-mix(in srgb, var(--tone) 12%%, transparent); white-space: nowrap; }
.spade-group-about { font-size: 0.86rem; opacity: 0.8; margin-top: 0.4rem; line-height: 1.45; }
.spade-tool-desc { font-size: 0.78rem; opacity: 0.72; line-height: 1.4; margin: -0.35rem 0 0.35rem 0.15rem; padding-left: 0.6rem;
  border-left: 2px solid color-mix(in srgb, var(--tone, %(primary)s) 55%%, transparent); }

.spade-split { display: flex; height: 0.7rem; border-radius: 999px; overflow: hidden; margin: 0.35rem 0; }

:where(.spade-root) a:focus-visible, :where(.spade-root) [tabindex]:focus-visible { outline: 3px solid %(primary)s; outline-offset: 2px; }
</style>
"""


def inject_base_css() -> None:
    """Called once per script run by the app shell (driveflow.viz.app)."""
    st.html(_CSS % active_palette())
