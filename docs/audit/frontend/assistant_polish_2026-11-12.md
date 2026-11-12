# Assistant page polish (M6.3, 2026-11-12): changes to review

Branch **`feat/FE-assistant-polish`**. **Not merged into `dev`** (decision M-D6): this file and the screenshots are for Luciano's review. Presentation only. The assistant's logic, tools, model, thresholds and safety texts are unchanged. One file changed: `src/driveflow/viz/views/assistant.py`. Everything shown is **simulated**.

Taken from the inventory (`inventory_2026-11-12.md` §3): the "partial" or "missing" elements of the frontend spec that can be done without new assets and without changing the assistant's flow.

| # | Change | Spec | Before | After |
|---|---|---|---|---|
| 1 | **Visible response time** under every reply: "Answered in X.X s · local model". This is the time of the model and tools, not of the visual word-by-word streaming | M6.3 | not shown | caption under each reply, also kept in the history |
| 2 | **"thinking" stays at least 400 ms** (`MIN_THINKING_S`), so a fast reply does not flicker | ESPEC §1.3 | not enforced (the adapter delegated it to the UI) | enforced in the page after the turn |
| 3 | **Origin chips with colour and a legend** in the draft panel | ESPEC §3, §5 | plain text column | coloured cell (light background, dark text, contrast ≥ 4.5:1, tested), text kept, legend caption |
| 4 | **Model digest in the header's tooltip** | ESPEC §1.4 | tag and status only | tooltip: tag, digest, "runs on this machine; no data leave it" |
| 5 | **Link from job progress to Experiment Runs** | ESPEC §3 (`job_progress`) | progress bar only | `st.page_link` under the progress (a caption when the view runs outside the app) |

**Observed in the screenshots, not changed (layout, already in dev):** right after a turn, the chat input appears between the first message and the new turn, because the turn is drawn after the input in the same run. The next run puts it back at the bottom.

**Not done here, and why:**
- Draft as a tab under the chat on narrow screens: it needs restructuring the three columns and changes the page layout; better reviewed separately.
- A3 head crop, A4 poses, A6 light variant: they need assets (FE-D2).
- Bubble tail / figure overlapping the chat: not possible with `isolate_styles` (fe3_spike.md (b)).
- Proposal **Edit** button: it would change the assistant's flow.
- Results charts.
- Keyboard focus: already a 3 px focus ring on SPADE elements; the avatar is decorative and not focusable; native widgets keep Streamlit's focus.

**Tests:**
- `tests/assistant/test_assistant_page_polish.py` (4): response time shown and stored, minimum thinking time, origin colours and legend with the text kept, WCAG AA of the chip colours, digest in the tooltip.
- `tests/assistant/test_assistant_page.py` (3) passes unchanged.

**Screenshots:** `docs/screenshots/frontend/polish/`, before (dev `e4fed56`) and after (this branch, which includes dev's AI-N39 fix; without it no turn finishes in a browser), desktop 1440 px and narrow 390 px, after one turn with the real local model. Because they are real model replies, the text of the turn differs between before and after. The avatar in them is covered by `licenses/avatar/LICENSE`.
