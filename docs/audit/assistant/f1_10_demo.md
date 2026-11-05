# F1.10 + FE-4 — Assistant page and milestone "Demo interna del asistente"

- **Model:** `qwen3.5:9b` with `think`, pinned by **Luciano's decision** for the internal demo (`f1_1_decision.md`; the F1.1 round-3 rule was not met; status `internal-demo; F1.1 round-3 rule not met`).
- **Not a benchmark.** This is the milestone's descriptive evidence. The independent test is F1.11.

## Page

- `src/driveflow/viz/pages/assistant.py` → `views/assistant.py`, registered in `PAGES["Intelligence"]` as "Assistant". No other page was modified; `app.py` only gained the registry line.
- **Layout:**
  - avatar column, chat column (`st.chat_message` / `st.chat_input`, no attachments), and draft panel (fields with origin chips) with the **Run** button;
  - on narrow screens a compact avatar shows in the header (CSS on the keyed containers).
- **Loop and avatar:**
  - `on_event` repaints the avatar in an `st.empty()`, so `thinking` is painted before the model call;
  - the answer is shown word by word, with `beat` every 4 words (visual streaming; the Ollama reply arrives whole);
  - job progress refreshes in `st.fragment(run_every="2s")`.
- **What the user reads:**
  - the assistant's sentence (F1.9 post-processing);
  - cards (SPADE data verbatim with its source: plant, tunables, proposal, results, documentation);
  - the draft panel.
- **Confirmation:** set only by the page, through the Run button or a yes typed right after a proposal.
- **Fixed texts:** the first message "I'm SPADE's AI assistant. I set up and run simulations; my suggestions require your verification. All results are SIMULATED." and the footer "All results are SIMULATED.". The system prompt (F1.9) says the assistant never speaks in the first person as Luciano.
- **Ollama down or no model:** state `unavailable`, a warning with how to start it, and the page keeps working (test and smoke check 2c).
- **Jobs:** the page offers "Process queued jobs here (once)" for the demo; otherwise a worker runs separately (`python -m driveflow.jobs.worker`).

## Milestone criteria

| Criterion | Evidence | Result |
|---|---|---|
| Complete configuration conversation on the page with the chosen model: request → (questions) → proposal → "sí"/Run → job → results | Live run below, conversation A; ReplayClient AppTest `test_complete_configuration_conversation_with_the_run_button` | **Met** |
| Objective-mode conversation | Live run, conversation B (the assistant first asked which plant; then proposal → Run → tuning job → requirements check) | **Met** |
| Avatar in all its states | All 11 states reached from recorded events (FE-1/FE-3 tests: `test_assistant_events.py`, `test_assistant_avatar_component.py`). Live: idle, thinking, using_tool, awaiting_confirmation, confirmed, running, results_ready, error (A, a summarize call before the job ended); `unavailable` with Ollama down (page test, smoke 2c). `listening` and `extracting` (the datasheet step, phase 2) only from recorded events. | **Met** (as FE-4 defines it: "todos los estados se alcanzan desde eventos grabados") |
| Full suite green | `pytest` on the integrated stack: 1736 passed, 1 skipped, 9 xfailed (19:43) | **Met** |
| `smoke --launch` green | `python scripts/smoke_test.py --launch`: **7/7** (the 6 existing checks plus 2c, the Assistant page without Ollama) | **Met** |

## Live run (`python experiments/report_f1_10_demo.py`, AppTest of the real page, Ollama 0.40.13, RTX 5060 Ti; raw in `f1_10_demo.json`)

| Conversation | Turns | Total | Confirmations | Launches without confirmation |
|---|---|---|---|---|
| A configuration | request, Run, worker, "¿Cuáles son los resultados?" | 42 s | 1 (button, recorded) | 0 |
| B objective | request, the plant (the assistant asked), Run, worker, "¿Se cumplen los requisitos?" | 64 s | 1 (button, recorded) | 0 |

**Checks of what the assistant said:**
- **A:**
  - The plant description in its sentence matches the catalog.
  - "Coste ~1,6 s" matches the proposal card (1.58 s).
  - The draft numbers were removed by the post-processing.
  - The results numbers come from the results card.
- **B:** "se cumplen ambos requisitos con el candidato 'best' (150.18 rad/s, 1.04, 0.66)" was **verified** against the tuning report of the job: `check_requirements` gives "candidate 'best' meets all 2 requirements"; worst overshoot 4.87 % ≤ 5 %; settling 0.053 s ≤ 0.2 s.
- **No false statement found** in these two conversations.

**Observed and recorded (findings registered, scope frozen until the milestone):**
- In the objective proposal, removing the draft numbers left "Requisitos: ¿Confirmas…". The values are in the draft panel, but the sentence reads truncated.
- After Run, the assistant called `summarize_results` before the job ended. The turn ended through the final call without tools (state `error` shown briefly).
- The worker button does not refresh the assistant's text; the next question does.
