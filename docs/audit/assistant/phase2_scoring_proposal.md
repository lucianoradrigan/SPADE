# Phase 2 scoring: proposal of criteria — **DRAFT, NOT APPROVED**

Written 2026-11-15 by Claude Code (R4, decision R-D6). **Luciano approves the criteria before Phase 2 starts.** Nothing here has been measured: no LLM and no extractor has been run on the golden set (R-D6). The scorer exists and is tested without any model.

## What is measured

The scorer is `experiments/score_golden_extraction.py`, with tests in `tests/assistant/test_score_golden_extraction.py` (50, no LLM).

- **Input:** one extraction output per document (format in the scorer's docstring).
- **Reference:** the hand-written expected files `tests/assistant/golden/expected/<id>.json`: 15 datasheets and Ingenium records, and 21 rating-plate photos.

| Metric | Definition | Kind |
|---|---|---|
| **values** | Share of printed, legible expected values given correctly. The number must be within the relative tolerance **and** unit, basis (phase / line-line) and waveform (rms / peak / dc_equivalent) must agree. Claiming a basis or waveform where the document says none is wrong. A converted value counts only if the output declares the rule of `configs/assistant_conversions.yaml` and the value reproduces it | share |
| **inventions** | Values given where the expected one is `null`, `"illegible"` or `"ambiguous: must ask"` | count (severe) |
| **must_ask** | Share of the expected questions the agent asked | share |
| **unneeded_questions** | Questions about a value that is printed and legible | count |
| **must_warn** | Share of the expected warnings the agent gave | share |
| **simulable** | Share of documents where the agent's "simulable today" conclusion is right | share |

## Proposed criteria (all must hold; DRAFT)

| # | Criterion | Proposed threshold | Why this number |
|---|---|---|---|
| C1 | inventions | **= 0** | An invented datasheet value enters a simulation as if it were read. The same zero-tolerance as the assistant's "invented numbers" in phase 1 |
| C2 | must_ask | **≥ 90 %** | Same level as the phase 1 criterion "asks when a required value is missing" |
| C3 | values | **≥ 85 %** | Leaves room for genuinely hard prints (photos, rotated tables) while catching systematic convention errors |
| C4 | unneeded_questions | **≤ 1 per document on average** | Asking for what is printed wastes the user's time but is not dangerous |
| C5 | must_warn | **≥ 80 %** | Warnings matter (temperatures, pole pairs vs poles), but missing one is less severe than a wrong value |
| C6 | simulable | **≥ 90 %** | Telling the user a machine is simulable when it is not (an induction motor) is a false statement |
| C7 | values on datasheets **and** on photos separately | the C3 threshold in each group | So that the photos (easier labels) cannot hide errors on datasheets, or the reverse |

## Parameters of the scorer (to approve with the criteria)

- **Relative tolerance for values:** `--rtol 0.02` (2 %). Prints round to 2–3 significant figures.
- **Text matching of questions and warnings:** an exact match after normalisation, or ≥ 60 % of the expected item's content words present in what the agent said (`--text-threshold 0.6`). This is lenient on purpose, because the agent paraphrases.
  - **Risk:** a long answer can match by vocabulary alone. Alternative to decide: score questions and warnings by hand on a sample, or use structured questions (a field name plus a reason) in the Phase 2 output format.
- **Repeated labels** on a plate pair by order of appearance.

## Not decided here (questions for Luciano)

1. **Independence:**
   - the golden set's expected values were written by Claude Code (`written_by` in each file);
   - should a second reader check a sample before it is used to accept Phase 2?
   - the set will also inform the extractor's design; a part should be held out (proposal: the 21 photos as development, D1–D7 and P1–P10 split half and half).
2. **Text matching vs structured questions** (see above).
3. **Documents still missing:** D6, P4, D7 and I2 (`golden_set.md`). Phase 2 would run on 36 documents without them.
