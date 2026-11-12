"""The assistant's avatar "PhD. Radrigan" as a Streamlit v2 component (task FE-3).

Ports the prototype (proposed_agent/phd_radrigan_assistant.html): the body and hand layers (hand
pivot at 60 % / 100 %), the chip glow at (60 %, 85.8 %), thinking dots, sweep ring, a beat per
streamed word, the greeting wave, voice bars, the progress bar and the neon name as CSS text.
No JS library and nothing from a CDN (FE-D4; SPADE works offline). The layers come from
``persona.avatar_dir`` (configs/assistant.yaml) as data URIs; without them a neutral silhouette is
shown and the page does not break (FE-2). The avatar never approves results (FE-D6): ``results_ready``
is a neutral state; the hop happens only in ``confirmed``.

    radrigan_avatar("thinking")                               # label defaults to the state's own
    radrigan_avatar("running", label="Simulation running: 3/5", progress=0.6)
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from pathlib import Path

import streamlit as st

from driveflow.viz.adapters.assistant_adapter import STATES

_DIR = Path(__file__).with_name("radrigan_avatar")


class UnknownAvatarState(ValueError):
    pass


@lru_cache(maxsize=1)
def _component():
    return st.components.v2.component(
        "radrigan_avatar",
        html=(_DIR / "avatar.html").read_text(),
        css=(_DIR / "avatar.css").read_text(),
        js=(_DIR / "avatar.js").read_text(),
        isolate_styles=True,
    )


@lru_cache(maxsize=4)
def _images(avatar_dir: str) -> dict | None:
    from driveflow.assistant.config import avatar_assets, load_config

    cfg = load_config()
    if avatar_dir:
        cfg = {**cfg, "persona": {**(cfg.get("persona") or {}), "avatar_dir": avatar_dir}}
    a = avatar_assets(cfg)
    if not a["ok"]:
        return None
    raw = {k: a["files"][f"{k}.webp"].read_bytes() for k in ("base", "hand")}
    return {"base": "data:image/webp;base64," + base64.b64encode(raw["base"]).decode(), "hand": "data:image/webp;base64," + base64.b64encode(raw["hand"]).decode(),
            "digest": hashlib.sha256(raw["base"] + raw["hand"]).hexdigest()[:12]}


def avatar_props(state: str, label: str | None = None, progress: float | None = None, compact: bool = False, beat: int = 0, wave: int = 0,
                 speaking: bool = False, avatar_dir: str = "", report_env: bool = False) -> dict:
    """The data sent to the component (pure; tested without a browser).

    report_env (AI-N39, 2026-11-12): the component reports {reduced_motion, state, has_assets} back to Python only when
    asked. Reporting it sets the component's state, and in a browser that requests a rerun; on the Assistant page every
    repaint is a new component, so each repaint during a turn restarted the script and the turn never finished. Only the
    avatar demo page (the FE-3 spike) asks for it."""
    if state not in STATES:
        raise UnknownAvatarState(f"unknown avatar state {state!r}; states: {list(STATES)}")
    if progress is not None and not 0.0 <= float(progress) <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    from driveflow.assistant.config import load_config

    persona = load_config().get("persona") or {}
    return {"state": state, "label": label, "progress": progress, "compact": bool(compact), "beat": int(beat), "wave": int(wave), "speaking": bool(speaking),
            "name": persona.get("name", "SPADE assistant"), "tagline": persona.get("tagline", "AI assistant · local model"), "images": _images(avatar_dir),
            "report_env": bool(report_env)}


def radrigan_avatar(state: str, label: str | None = None, progress: float | None = None, compact: bool = False, beat: int = 0, wave: int = 0,
                    speaking: bool = False, key: str = "radrigan_avatar", avatar_dir: str = "", report_env: bool = False):
    """Mount the avatar. Raises UnknownAvatarState for a state outside the 11 of ESPEC_FRONTEND §1.3.
    report_env: see avatar_props (AI-N39); False on the Assistant page."""
    return _component()(key=key, data=avatar_props(state, label, progress, compact, beat, wave, speaking, avatar_dir, report_env), on_env_change=lambda: None)
