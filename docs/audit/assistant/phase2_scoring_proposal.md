# Phase 2 scoring: criteria — **APPROVED (Luciano, 2026-11-16, decision D8)**

Written 2026-11-15 by Claude Code (R4, decision R-D6). **Approved by Luciano on 2026-11-16 (D8)**, with two changes: questions are compared as structures (field and reason), and one third of the golden set is held out. Nothing here has been measured: no LLM and no extractor has been run on the golden set (R-D6). The scorer exists and is tested without any model.

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

## Criteria (all must hold; APPROVED 2026-11-16)

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
- **Questions (D8): structured.** Each expected question is `{field, reason}` with `reason` from a closed list (below). The output's `asked` must hold the same pair; there is no text matching for questions. Asking for a field that is printed and legible counts as an unneeded question.
- **Warnings:** still free text (normalised exact match, or ≥ 60 % of the expected item's content words, `--text-threshold 0.6`).
- **Repeated labels** on a plate pair by order of appearance.

## Held-out set (D8)

- `tests/assistant/golden/holdout.yaml`: **12 of the 36 documents**, stratified by style (A–F) and machine group, seed 20261116 (`experiments/select_golden_holdout.py`). It was committed (`bb7d70e`) before any Phase 2 iteration.
- Only `experiments/score_golden_extraction.py --final-eval` reads them. Without the flag the scorer refuses those IDs and scores the 24 development documents.
- `tests/assistant/test_golden_holdout_guard.py` checks the split, its reproducibility, the refusal, and that no other development code opens the expected files.
- **Declared:**
  - every expected file, including the 12 now held out, was written by Claude Code in round C;
  - all 36 were read again on 2026-11-16 to convert `must_ask` to the structured format, before the split was drawn;
  - the holdout therefore protects against tuning an extractor on those documents, not against the author of the expected values having seen them.
  - An independent second reader of a sample (question 1 below) would address the second point.

## Not decided here (questions for Luciano)

1. **Independence:**
   - the golden set's expected values were written by Claude Code (`written_by` in each file);
   - should a second reader check a sample before it is used to accept Phase 2?
   - the set will also inform the extractor's design; a part should be held out (proposal: the 21 photos as development, D1–D7 and P1–P10 split half and half).
2. **Text matching vs structured questions** (see above).
3. **Documents still missing:** D6, P4, D7 and I2 (`golden_set.md`). Phase 2 would run on 36 documents without them.

## Structured questions: closed list of reasons and the equivalence table (D8)

**Reasons:** `missing`, `variant_unspecified`, `ambiguous_basis`, `ambiguous_waveform`, `inconsistent`, `ambiguous_meaning`, `condition_unspecified`, `machine_type_unspecified`, `manufacturer_unnamed`.

The expected files changed **format only**:
- each former `must_ask` string is now one or more `{field, reason, item, text}` entries;
- `text` keeps the original sentence unchanged, and `item` is its former position;
- a sentence that names several fields became one entry per field: 30 sentences → 50 entries.

The 12 held-out documents are converted too; their table rows are not reproduced here, so the reserve is not read again. Their conversion is mechanical, in the commit that introduced this table.

| Doc | Former sentence (`text`) | → `{field, reason}` |
|---|---|---|
| D1 | which supply voltage / column (24, 36, 48, 60 or 72 V) | `voltage`/variant_unspecified |
| D1 | the load (not in the document) | `load_torque`/missing |
| D2 | the armature inductance (not printed) | `inductance`/missing |
| D3 | which of Ke and Kt to use: Ke 0.0225 V/rpm = 0.215 V s/rad but Kt = 0.192 N m/A (+12 %); never average them | `torque_constant`/inconsistent |
| D3 | armature resistance, inductance and inertia (not printed) | `resistance`/missing, `inductance`/missing, `inertia`/missing |
| D3 | the operating voltage within 12-72 V | `voltage`/variant_unspecified |
| P10 | the electrical constants, resistance, inductance, inertia and poles: not in this sheet; ask for the manufacturer's data sheet | `torque_constant`/missing, `voltage_constant`/missing, `resistance`/missing, `inductance`/missing, `inertia`/missing, `poles`/missing |
| P3 | the basis of R and L (not printed; with tau_el they do not agree, see warn) | `resistance`/ambiguous_basis, `inductance`/ambiguous_basis |
| P3 | the ambient temperature of 'Winding Resistance @ Ambient' | `resistance_ambient`/condition_unspecified |
| P5 | the basis of the back-EMF constant (Vrms/krpm, phase or line-line not printed) | `back_emf_constant`/ambiguous_basis |
| P6 | the back-EMF constant (not printed on this sheet, unlike the 24 V variant) | `back_emf_constant`/missing |
| P7 | which winding (6 windings x cooling) | `winding`/variant_unspecified |
| P7 | the convention of Kt and Ke: Kt matches 1.5 p psi without sqrt(2) (peak current) although it says Arms; Ke does not match psi as line-line rms | `torque_constant`/inconsistent, `ke_constant`/inconsistent |
| P9 | which model (rows) | `model`/variant_unspecified |
| P9 | the continuous current (not printed) | `current_continuous`/missing |
| P9 | the basis and waveform of Kt and back EMF (not printed) | `torque_constant`/ambiguous_basis, `torque_constant`/ambiguous_waveform, `back_emf`/ambiguous_basis, `back_emf`/ambiguous_waveform |
| W01 | the voltage and current pairs go with the connection Δ/Y: 230 V Δ / 1.67 A and 400 V Y / 0.97 A | `connection`/variant_unspecified |
| W07 | the kW value (13.2 or 15.2); 110 V x 120 A = 13.2 kW suggests 13.2 | `power`/ambiguous_meaning |
| W07 | whether it is a DC machine and how it is excited (not on the plate) | `machine_type`/machine_type_unspecified, `excitation`/missing |
| W10 | the machine type: the plate shows an excitation (Erregung 43 V / 425 A) but no 'synchronous' or 'asynchronous' | `machine_type`/machine_type_unspecified |
| W17 | the field excitation (series, shunt, separately excited or permanent magnet: not printed) | `excitation`/missing |
| W17 | armature resistance, inductance and inertia | `resistance`/missing, `inductance`/missing, `inertia`/missing |
| W20 | whether 115 is kW or kVA: 166 A at 400 V is sqrt(3) x 400 x 166 = 115 kVA, so 115 kW at cos φ 0,8 would need 207 A | `power`/inconsistent |
