# Frontend round 2 (R3, 2026-11-15/16): assistant page, for Luciano's review

Branch **`feat/FE-assistant-polish`** (not merged into `dev`, decision R-D4), up to date with `dev` (v0.2.1, R1 labels, AI-N38, AI-N40). Presentation only: the assistant's logic, tools, model, prompts, thresholds and safety texts are unchanged. Everything shown is **simulated**; the screenshots use the real local model (`qwen3.5:9b`).

## What to look at first (short list)

1. **Narrow screen during a turn:**
   - `round2/before_narrow_during_turn.png`: the visible avatar said "Ready" during the turn, and the message sat below the input;
   - `round2/after_narrow_during_turn.png`: "Thinking…", with the message above the input.
2. **The avatar's head in the assistant's bubbles:** `round2/after_desktop_after_turn.png`, against the default icon in `before_desktop_after_turn.png`.
3. **The visible response time** under each reply ("Answered in 18.4 s · local model"), and the coloured origin chips with their legend (round 1, kept).
4. **The real-browser test** `tests/assistant/test_assistant_browser_llm.py`: decide whether it should become part of the release gates (G5).
5. **Proposals that need your decision** (bottom of this page).

## Changes

| # | Element | From | Status | Screenshot (before → after) | Test |
|---|---|---|---|---|---|
| 1 | The shown response time is the real one (model + tools), measured **before** the 400 ms minimum of "thinking" | R3 (instructions) | **confirmed, unchanged code** | after_desktop_after_turn.png ("Answered in …") | `test_the_visible_time_is_the_real_time_without_the_thinking_minimum` (pad 1.5 s; shown < 1 s) |
| 2 | Both avatars follow a turn. The compact header avatar, the only one visible on narrow screens, used to stay "idle"; now both are repainted on every agent event | ESPEC §1.2–1.3 (feedback during a turn; narrow screens) | **done** | before/after_narrow_during_turn.png | `test_assistant_browser_llm.py`: both avatars seen as idle → thinking → awaiting_confirmation in a real browser |
| 3 | The new turn is drawn **above** the chat input (it was below it until the next run) | inventory (observed 2026-11-12) | **done** | before/after_narrow_during_turn.png, *_desktop_after_turn.png | `test_the_new_turn_is_drawn_above_the_input` |
| 4 | **A3:** the avatar's head in the assistant's chat bubbles, cropped at run time from the licensed `base.webp` (no new image file; replacing the layer replaces the head; default icon when the layers are missing) | ESPEC §1.1 (A3) | **done** | *_desktop_after_turn.png | `test_the_chat_head_is_a_crop_of_the_licensed_base_layer`, `test_assistant_bubbles_use_the_head` |
| 5 | Visible response time, 400 ms minimum of "thinking", coloured origin chips with legend, model digest in the header tooltip, link from job progress to Experiment Runs | round 1 (2026-11-12) | **kept** (merged with dev) | `docs/screenshots/frontend/polish/` | `test_assistant_page_polish.py` |
| 6 | One complete turn in a **real browser** with Ollama, with the avatar states recorded (the lesson of AI-N39) | R3 (instructions) | **done** | — | `test_assistant_browser_llm.py` (marker `llm`; Playwright via `importorskip`, not a dependency); passed: reply in 12 s |
| 7 | AI-N38 fixed card when a v1 plant is refused | R1 (in `dev`) | in `dev` | *_desktop_v1_request.png | `test_assistant_ai_n38.py` |

In the v1 screenshots the model chose the v2 plant directly, so no tool refused v1 and the fixed card did not appear. That is the expected behaviour: the card appears only when a tool refuses. The card is tested with recorded replies.

## Accessibility (priority 2): verified, nothing left to change on this page

- `prefers-reduced-motion` turns off every avatar animation (CSS and JS).
- The status chip is in an `aria-live="polite"` region, and the state is never communicated by animation alone.
- Contrast in the component, which has its own dark panel: text 15.3:1, amber 10.5:1, tagline 6.9:1. Origin chips ≥ 4.5:1 (tested).
- Focus: SPADE elements have a 3 px focus ring. The avatar is decorative and not focusable. Native widgets keep Streamlit's focus.
- Alt text: "PhD. Radrigan, SPADE assistant (AI)" on the figure; the hand layer is `alt=""` (decorative).

## Proposals (not implemented: they need assets, a flow change or your decision)

| Element | From | Why not here |
|---|---|---|
| Draft as a tab under the chat on narrow screens | ESPEC §1.2 | Restructures the three columns. Today the draft stacks below the chat (`after_narrow_after_turn.png`) |
| A4 poses (hand on chin, paper, wink) and A6 light-theme variant | ESPEC §1.1 | Need the final assets (FE-D2) |
| Bubble tail toward the figure; figure overlapping the chat column | ESPEC §1.2 | Not possible with `isolate_styles` (fe3_spike.md (b)) |
| Proposal **Edit** button | ESPEC §3 | Changes the assistant's flow |
| `results_summary` with Plotly charts | ESPEC §3 | Needs the results data path in the page; to be scoped with you |
| Stop the word-by-word text streaming under `prefers-reduced-motion` | accessibility | The page (Python) cannot know the browser preference without the component's report, which AI-N39 restricted. It would need a different channel |

**Harness note:** the round-2 screenshots open `/assistant` directly, so the sidebar shows the file-based page list. That is an artefact of the script (CLAUDE.md asks to open `/` first), the same before and after; it is not a change.
