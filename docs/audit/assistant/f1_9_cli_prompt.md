# F1.9 — system prompt v1 and CLI, with the fixes decided after F1.1

**Model:** `qwen3.5:9b` with `think`, pinned by **Luciano's decision** for the internal demo (`f1_1_decision.md`; the F1.1 round-3 rule was not met).

## Built

- **`src/driveflow/assistant/prompts/system_v1.md`.** The prompt of plan §7, shortened and made explicit for a local model.
  - **Why shorter:** on the F1.1 bench the 9B models skipped tools, restated tool data wrongly and looped (`f1_1_model_choice*.md`).
  - **Removed:** the datasheet step (phase 2).
  - **Added** (FE-4): the assistant is not a person and never speaks in the first person as Luciano Radrigan.
  - **Added:** a section on what the interface shows, so the model must not repeat it.
- **CLI `python -m driveflow.assistant`** (`assistant/cli.py`). After every turn it prints:
  - the sentence;
  - the cards (SPADE data verbatim, with source);
  - the draft panel (each value with its origin chip: USER / SUGGESTED / DEFAULT / …).

  The confirmation is the user's own input (`sí` / `yes` right after a proposal, or `/run`), set by the CLI and never by the model. Other commands: `/draft`, `/worker`, `/help`, `/quit`; `--replay <cassette>` runs on a recording. If Ollama is down or the model is missing, the CLI says how to fix it and keeps going.

## Fixes (Luciano, 2026-11-04; F1.9/F1.10 work, no more selection rounds)

| Fix | Where | Test |
|---|---|---|
| Questions about SPADE: the answer is the card (verbatim snippet + source) plus **at most one** context sentence. The sentence is dropped, leaving only the card, if it has a number, finding ID, parameter name or negation that is not in the card (deterministic). | `answer.spade_answer`, loop `_postprocess`, guard `spade_answer_cards` | `test_spade_answer_is_the_card_plus_one_checked_sentence` |
| Draft values: the model does not repeat values or origins. Sentences of the final text with a draft number are removed; the draft panel shows them with their origin chip. | `answer.strip_draft_numbers`, `answer.draft_panel`, guard `strip_draft_numbers` | `test_draft_numbers_are_not_repeated_by_the_model` |
| Objective mode: the tunable parameters are a card generated from `describe_tunable`. | `cards.card_from_tool` | `test_objective_mode_shows_a_tunables_card` |
| Proposals (`propose_experiments` / `propose_tuning`) and results (`summarize_results` / `check_requirements`) are cards too, with the KPIs flattened. | `cards.py` | `test_complete_configuration_conversation` |
| Latency: `think: true` only on the calls that choose tools; the final call without tools uses `think: false`. | `OllamaClient.chat(think=)`, guard `final_think` | `test_final_answer_without_tools_has_thinking_off` |
| The extra call of the "brief" note is removed (replaced by the deterministic post-processing). | loop | (the round-3 brief test was removed) |
| At most 2 `update_draft` retries; then one final call without tools asks the user. | loop, guard `update_draft_retries` | `test_update_draft_retries_are_capped_then_the_user_is_asked` |
| The "record it" nudge no longer fires on turns that launched, checked a job or already proposed. Found by the CLI test. | loop `nudge_draft` | `test_complete_configuration_conversation` |
| Search cards limited to 3, snippets to 450 characters. Found in the live run: 6 cards, several irrelevant. | `cards.py` | — |

- **Tests:** `tests/assistant/test_assistant_cli.py` (complete configuration conversation on ReplayClient, a yes without proposal, Ollama down, prompt rules) and `test_assistant_guards.py`. 249 passed.
- **Library default:** `Agent(guards=None)` keeps every guard off; the CLI and the page pass `configs/assistant.yaml`.

## Live run (descriptive, not a benchmark)

[`f1_9_live_cli_transcript.txt`](f1_9_live_cli_transcript.txt): `qwen3.5:9b` on Ollama 0.40.13, RTX 5060 Ti, temporary workspace.

- **Conversation, 5 turns in 46 s:** request, proposal with cards, "sí" recorded by the CLI, job run with `/worker`, results card, question about S-21.
- **Observed:**
  - The draft numbers the model restated (100 rad/s, 0.5 s) were removed.
  - The S-21 answer kept its first sentence, which the card supports.
  - In the results turn the model listed the KPI values in prose. They are tool numbers, not draft numbers, so they are not removed; with the flattened card they can now be checked against it.
- This run was made before the two last fixes (search cards limited, KPIs flattened).
