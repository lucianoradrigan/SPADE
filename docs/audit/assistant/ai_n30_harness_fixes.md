# AI-N30, item 3(b): harness fixes after F1.11

- **Decision:** Luciano, 2026-11-08.
- **Declared:** every fix below was designed by reading the F1.11 conversations (`f1_11_benchmark.md`, `f1_11_review.md`). F1.11 can therefore no longer test them. **F1.12** (new requests, the same thresholds, committed before the requests) is the independent test.
- **Code:** `src/driveflow/assistant/guardrails.py`; `core/loop.py` (guard `harness_v2`, on in `configs/assistant.yaml`); `catalog.search_docs`; `cards.py`.
- **Tests:** `tests/assistant/test_assistant_harness_v2.py` (11 tests, scripted replies, no Ollama). Assistant tests: 299 passed, 1 skipped.

| # | Luciano's item | F1.11 evidence | Fix |
|---|---|---|---|
| 1 | Missing required fields asked by the interface | "Asks when missing" 60 %; m02's question was lost | `guardrails.missing_card`: when `draft.status()` has missing REQUIRED fields and nothing is proposed, a question card lists each field (unit, why, choices). It adds a suggestion only when `suggest_value` has an internal source (P5). The fixed sentence "Para seguir necesito los valores de la tarjeta." is added when the model did not ask. It does not depend on the model. |
| 2 | Suggestions and their sources in cards | Rounded tool numbers in the text (26 of the 40 "questions" failures) | `suggest_value` results are cards ("Suggestion for <field>": value, rule, source), or "No suggestion" when SPADE has no internal source. |
| 3 | Why "not documented" is 0 % | 0 / 12 | **The rule was mis-built.** `search_docs` scored a paragraph on ANY query word longer than two letters ("the", "que", "spade", "motor"), so it almost never came back empty. "Not documented after two empty searches" could not happen. Now a query keeps only its distinctive words (Spanish / English stopwords and SPADE's generic vocabulary removed). A paragraph is a hit only if it contains ≥ 75 % of them (all of them when there are two or fewer). A finding named by an existing ID is always returned. On the F1.11 queries the CAN/dSPACE and AI-N99 searches now return 0 hits, while S-21, AI-N29, dcm and the noise model are still found. |
| 4 | Bound the free sentences where the false statements and missing citations appeared | 5 false statements; citations 83 % | (a) **Card + one checked sentence only in question mode.** In configuration / objective mode a documentation card no longer replaces the assistant's question (m02). (b) **Capability claims:** the model's sentences about what SPADE can or cannot do (simulate, tune, train, support, document, with a limit or a negation) are removed. The tools' own words about the limit (refusals, not-simulable reasons, not-tunable notes) are shown in a card, with a fixed lead-in. This covers 4 of the 5 false statements (q05, q06, i04, i05). (c) **Launch claims** ("se está ejecutando", "launched", "is queued"…) are removed when no launch tool succeeded and no job exists, and a fixed sentence is added (l03). (d) **Question mode:** plants named by a common name (e.g. "PMSM") get their catalog card (q04's missing citation). (e) **Loop notes** go with the `system` role: the model answered them as the user's words (c06, c08, c10, m01). |

**Not changed:**
- the model, the system prompt, the thresholds;
- the 0-launches / 0-invented-numbers rules, which were met.
