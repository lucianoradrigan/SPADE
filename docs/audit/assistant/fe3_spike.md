# FE-3 — `radrigan_avatar` component: spike and checks (Streamlit 1.66.0)

- **Commands:** `PYTHONPATH=src python -m pytest -q tests/assistant/test_assistant_avatar_component.py` (15 tests: props for every state, unknown state, missing layers, `AppTest` of the test page in each of the 11 states and without layers / compact); browser checks with Google Chrome headless (`--headless=new --virtual-time-budget`, `--dump-dom`, `--screenshot`) against `streamlit run scripts/assistant_avatar_demo.py --theme.base dark` on a free port (8765 was in use by the user's running SPADE app, left untouched).
- **Code:** `src/driveflow/viz/components/radrigan_avatar.py` + `radrigan_avatar/{avatar.html, avatar.css, avatar.js}` — the prototype's CSS and JS ported (classes prefixed `rad-`, the stage only: layers base + hand with the hand pivot at 60 % / 100 %, chip glow at (60 %, 85.8 %), thinking dots, sweep ring, beat per streamed word, greeting wave, voice bars, progress bar, neon name as CSS text). Prototype fonts came from Google Fonts: replaced by system font stacks (no CDN; SPADE works offline). No JS library (FE-D4).
- **Test page:** `scripts/assistant_avatar_demo.py` (not in `PAGES`): the 11 states as buttons, `?state=` / `?compact=1` / `?no_assets=1` for screenshots, and a recorded event sequence replayed through `assistant_adapter.avatar_state`.

## Spike results

| Item | Result |
|---|---|
| (a) Data and resources into a v2 component | `st.components.v2.component(name, html=, css=, js=, isolate_styles=)` returns a mount function `(*, key, data, default, width, height, **on_<state>_change)`; the JS module exports `default function(component)` and receives `{data, parentElement, setStateValue, setTriggerValue}`. Files by path need an installed package with `asset_dir`, so the component passes its HTML/CSS/JS as strings and the two layers as `data:image/webp;base64` URIs inside `data` (≈ 61 KB per rerun; the JS sets `src` only when their digest changes). |
| (b) `isolate_styles=True` and leaving the column | Kept **inside its column** (the spec's fallback): with `isolate_styles=True` the markup lives in a shadow root inside Streamlit's element container; making it overlap the chat column would mean styling Streamlit's own containers from outside. Not attempted. |
| (c) `prefers-reduced-motion` | **Verified:** the component reports `reduced_motion=True` under `--force-prefers-reduced-motion` and `False` without it (`--dump-dom`, text written by the test page from the component's `setStateValue`); the CSS disables every animation and transition under it, and the JS skips wave and beat. |
| (d) Animations paused with the tab hidden | Implemented (`visibilitychange` toggles a `paused` class that sets `animation-play-state: paused`). **Browser check not executed:** a headless page cannot be hidden and shown again from the command line. |
| (e) Resources < 1 MB | `base.webp` 34 908 B + `hand.webp` 10 950 B = 45.9 KB. |

## Other checks

- **Chip contrast** (text on the chip background `#14121f`): normal text `#ebe8f7` 15.3:1, error text `#ffb547` 10.5:1, muted `#9f9ab8` 6.9:1 — all ≥ 4.5:1.
- **Unknown state:** `UnknownAvatarState: unknown avatar state 'celebrating'; states: [...]` (test).
- **Without layers:** neutral silhouette, the page does not break (test and `avatar_no_assets.png`).
- **FE-D6:** `results_ready` is a neutral state (no hop, no smile); the hop exists only in `confirmed`.
- **Screenshots** (dark theme): `docs/screenshots/assistant/avatar_<state>.png` for the 11 states, plus `avatar_no_assets.png` and `avatar_compact_thinking.png`. Animations are not visible in still images.
- **Known limitation (provisional resources, FE-D2):** in a light theme the provisional render loses its halo; the component keeps its own dark card. Not fixed with CSS, as instructed.
