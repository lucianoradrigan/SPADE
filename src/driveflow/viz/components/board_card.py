"""Edge board cards and section panels of the Edge Deployment page (the overview reuses the cards).

Icons are SVG data URIs used as a CSS mask over currentColor, because st.html strips <svg>
elements: they still follow the theme and need no icon font. Each board's colour is its BOARD_ACCENT identity; what the evidence says is carried by the badge's label and
glyph, not by the colour."""

import re
from html import escape
from urllib.parse import quote

import streamlit as st

from driveflow.viz.components.page_header import badge_html
from driveflow.viz.theme.tokens import BOARD_ACCENT, tone as tone_color

_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{}</svg>'
_CHIP = '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>'
ICONS = {
    "esp32": _SVG.format('<rect x="7" y="12" width="10" height="9" rx="1.5"/><path d="M10 21v-3M14 21v-3M8.5 8.5a5 5 0 0 1 7 0M5.5 5.5a9.2 9.2 0 0 1 13 0"/>'),
    "rpi5": _SVG.format('<rect x="3" y="5" width="18" height="14" rx="2"/><rect x="7" y="9" width="6" height="6" rx="1"/><path d="M16 9h2M16 12h2M16 15h2M6 19v2M18 19v2"/>'),
    "c2000": _SVG.format(_CHIP + '<path d="M12.8 8.8 10.5 12.4h3l-2.3 3.2"/>'),
    "chip": _SVG.format(_CHIP),
    "format": _SVG.format('<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 13h5M10 17h5"/>'),
    "runtime": _SVG.format('<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>'),
    "code": _SVG.format('<path d="m8 7-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>'),
    "target": _SVG.format('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6"/>'),
}



def _icon(name: str) -> str:
    return f'''<span class="spade-icon" aria-hidden="true" style="--icon:url('data:image/svg+xml,{quote(ICONS[name], safe="")}')"></span>'''


#: target -> what the page says about the board. "evidence" is (badge label, tone, glyph).
BOARDS = {
    "esp32": {"name": "ESP32", "tagline": "Sensor node", "evidence": ("EXPORTED · NOT RUN ON A BOARD", "warning", "▲"),
              "specs": [("chip", "Device", "Microcontroller, 2 cores up to 240 MHz"), ("format", "Model format", ".tflite int8"), ("runtime", "Runtime", "TensorFlow Lite Micro"),
                        ("code", "Languages", "C / C++ (ESP-IDF)"), ("target", "Best for", "Small sensor node")]},
    "rpi5": {"name": "Raspberry Pi 5", "tagline": "Edge gateway", "evidence": ("EXPORTED · NOT RUN ON A BOARD", "warning", "▲"),
             "specs": [("chip", "Device", "Linux computer, 4 cores at 2.4 GHz"), ("format", "Model format", ".tflite float16"), ("runtime", "Runtime", "LiteRT / TensorFlow Lite"),
                       ("code", "Languages", "Python, C++"), ("target", "Best for", "Gateway with larger models")]},
    "c2000": {"name": "TI C2000", "tagline": "Real-time controller", "evidence": ("MEASURED ON LAUNCHXL-F28379D", "healthy", "✓"),
              "specs": [("chip", "Device", "Real-time MCU, F28379D at 200 MHz"), ("format", "Model format", "Generated C, float32"), ("runtime", "Runtime", "None: plain C kernels"),
                        ("code", "Languages", "C (TI cl2000)"), ("target", "Best for", "Inside the power-electronics controller")]},
}


def _head(target: str, subtitle: str) -> str:
    board = BOARDS[target]
    return (f'<div class="spade-board-head"><span class="spade-board-icon">{_icon(target)}</span>'
            f'<div><b>{escape(board["name"])}</b><span>{escape(subtitle)}</span></div></div>')


def _evidence(target: str) -> str:
    label, tone, glyph = BOARDS[target]["evidence"]
    title = "Simulated windows replayed from flash, not a real signal." if target == "c2000" else None
    return badge_html(label, tone, glyph, title)


def board_cards(notes: dict | None = None) -> None:
    """One card per board: the full spec list, or `notes[target]` in its place (overview)."""
    cards = []
    for target, board in BOARDS.items():
        if notes:
            body = f'<p class="spade-board-note">{escape(notes[target])}</p>'
        else:
            body = "".join(f'<div class="spade-board-spec"><span class="spade-spec-icon">{_icon(icon)}</span><span>{escape(key)}</span><b>{escape(value)}</b></div>' for icon, key, value in board["specs"])
        cards.append(f'<div class="spade-board" style="--tone:{BOARD_ACCENT[target]}">{_head(target, board["tagline"])}{body}<div class="spade-badges">{_evidence(target)}</div></div>')
    st.html(f'<div class="spade-root spade-boards">{"".join(cards)}</div>')


def board_banner(target: str, text: str) -> None:
    board = BOARDS[target]
    st.html(f'<div class="spade-root spade-board-banner" style="--tone:{BOARD_ACCENT[target]}">{_head(target, board["tagline"])}<p>{escape(text)}</p><div class="spade-badges">{_evidence(target)}</div></div>')


def panel(title: str, items: list, tone: str, glyph: str) -> None:
    """A titled list with a glyph on every item, so its meaning does not rest on the colour."""
    rows = "".join(f'<li><span aria-hidden="true">{escape(glyph)}</span>{escape(item)}</li>' for item in items)
    st.html(f'<div class="spade-root spade-panel" style="--tone:{tone_color(tone)}"><b>{escape(title)}</b><ul>{rows}</ul></div>')


def inline_markup(text: str) -> str:
    """The `code` and **bold** of a tutorial line, as HTML; everything else is escaped."""
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", escape(text, quote=False))
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", html)


def tutorial_step(number: int, text: str, target: str) -> None:
    st.html(f'<div class="spade-root spade-tstep" style="--tone:{BOARD_ACCENT[target]}"><span class="spade-tstep-num">{number}</span><div>{inline_markup(text)}</div></div>')
