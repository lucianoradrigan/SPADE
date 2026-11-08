# Demo fixes: the three details seen in the milestone demo

- **Branch:** `feat/ASSIST-1-demo-fixes` (from `dev` = `47a1dfe`). **Decided by Luciano, 2026-11-08**, after the milestone "Demo interna del asistente" (`f1_10_demo.md`, "Observed and recorded").
- **Model:** `qwen3.5:9b`, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, think on the tool-choosing calls; status `internal-demo; F1.1 round-3 rule not met` (unchanged).
- **Runtime:** Ollama 0.40.13. **GPU:** NVIDIA GeForce RTX 5060 Ti, 16 GB.
- **Tests:** `tests/assistant/test_assistant_demo_fixes.py` (18 tests, scripted replies, no Ollama). Assistant tests: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/assistant` → 271 passed, 1 skipped.
- **Live check:** `PYTHONPATH=src .venv/bin/python experiments/report_demo_fixes.py` → `demo_fixes_live.json`. It runs the two milestone conversations on the Assistant page (AppTest), with one more question about the results WHILE the job is still queued.

## 1. Truncated text

**Seen:** the objective proposal read "Requisitos: ¿Confirmas…". The model had written a "Requisitos:" line followed by two indented lines with the thresholds. The filter removed the two lines and left the header.

**Fix (`assistant/answer.py`):**
- The filter removes whole units and never shortens one. A unit is a sentence or a line.
- A sentence ends only at `.` `!` `?` followed by an upper-case letter, `¿`, `¡` or a list marker, or at a line break. Common abbreviations (`e.g.`, `aprox.`, `etc.`, …) do not end a sentence, and decimals never split one.
- The kept text keeps its line breaks.
- A line ending in `:` whose content was removed, or that has none, goes with it.
- A markdown table whose body rows were all removed goes whole: header row and `|---|` line.
- The evaluations budget of the objective counts as a draft number, like the thresholds.

**The proposal card (`assistant/cards.py`)** now lists the plant, base motor, tuned parameters, objective, algorithm, evaluations (with origin) and seed. It shows one line per requirement, e.g. `requirement: overshoot_pct <= 5 %`, and the verification. Before, it was a generic dump cut at 12 lines.

**Live (B):** the proposal now reads "…con controlador PI: / - Objetivo… / - Algoritmo… / ¿Confirmas que deseas lanzar esta sintonización?". The requirements are in the "Tuning proposal" card.

| Check (`demo_fixes_live.json`) | A configuration | B objective |
|---|---|---|
| Texts filtered | 2 | 2 |
| … made only of whole sentences / lines of the model's text | 2 | 2 |
| Headers left without content | 0 | 0 |

## 2. Results too early

**Seen:** after Run, the model called `summarize_results` before the job ended. The turn ended through the final call, and the avatar showed `error` for a moment.

**Fix:**
- **Policy (`core/policy.py`):** `summarize_results` / `check_requirements` are visible only when a job of their kind is in a terminal state (`completed`, `partially_completed`, `failed`, `cancelled`). Before, they were visible as soon as a job existed.
- **Job refresh (`tools.refresh_jobs`):** the loop reads the job states from the job service before every model call.
- **Refusal text:** a call to a result tool while the job runs is refused with "the job … has not finished (state …)".
- **Avatar:** while a job runs, a refused or failed job tool (`get_job_status`, `summarize_results`, `check_requirements`) emits the job's state, not an error. A turn that stops (repeated refusals, a limit) while a job runs leaves the avatar in `running` (`viz/adapters/assistant_adapter.py`). An error of another tool, e.g. `update_draft`, still shows `error`, with the FE priority.

**Live:**

| Check | A | B |
|---|---|---|
| Model calls that offered a result tool between the launch and the end of the job | 0 | 0 |
| `error` events | 0 | 0 |
| Avatar states after the launch | confirmed → running → results_ready | confirmed → running → results_ready |

In A, the Run turn ended after 3 refused `get_job_status` repeats while the job was queued (`turn_ended`, `stopped: errors`). In an earlier run of the same script this showed `error`. That is what the adapter rule covers, and the test `test_a_turn_stopped_by_repeated_status_calls_leaves_the_avatar_running` reproduces it.

## 3. Text not refreshed

**Seen:** the "Process queued jobs" button did not update the assistant's text; the next question did.

**Fix (`Agent.job_updates`):**
- It reads the jobs, emits their events and, for a job that reached a terminal state, returns a fixed message in the language of the conversation. Example: "El job … terminó. Los resultados (simulados) están en la tarjeta."
- The message comes with the results card: `summarize_results` or `check_requirements`, the same tool the model would call. There is no model call, and the numbers are only in the card.
- The message is added to the conversation, so the model knows the job ended.
- The page calls it after the button and in its 2 s refresh. The CLI calls it after `/worker`.

**Live:** after the button, without a new question, A showed the message with "Results (SIMULATED · AI-PLANNED), run …" and B with "Requirements check (SIMULATED)". The avatar went to `results_ready`.

## Other observations (not changed)

- **0 launches without confirmation;** both confirmations came from the Run button.
- The model still restates the KPIs of the results in its own text. That is allowed by F1.9: only draft numbers are removed, and the KPIs are tool results. It is not part of these fixes.
- **Changed test:** `test_assistant_loop.py::test_tools_by_state` now checks the new rule. Before, it expected `check_requirements` visible with a non-terminal job, which is the behaviour Luciano asked to change.
