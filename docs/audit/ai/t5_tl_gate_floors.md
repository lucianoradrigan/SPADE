# T5 (round 2026-10-22): D1 floors in the transfer-learning gate

**Decision D1** (user, 2026-10-22): the floors below are gate defaults. A reviewer can raise them, not lower them.

| Floor | Default | Applies to |
|---|---|---|
| `min_macro_f1` | 0.88 | every classification candidate: a comparable base, or a new class vocabulary |
| `min_per_class_recall` | 0.70 | every class with test windows, every classification candidate |

## Implementation

- **Floors** (`ai/transfer/validator.py`):
  - `resolve_floors()` applies the defaults and the reviewer's values. It raises `FloorError` with an actionable message when a value is lower than the default.
  - `floor_checks()` adds the two required checks.
  - The validation report carries `floors: {name: {value, origin}}`, where origin is "default D1 2026-10-22" or "reviewer".
- **Gate** (`ai/promotion.py`):
  - New required item `acceptance_floors`. It re-checks that the persisted floors exist and are not below D1, so an edited `validation_report.json` cannot lower them.
  - `gate_report.json` records `floors`.
- **GUI** (Transfer Learning):
  - Step 2 shows both floors as raise-only inputs.
  - Step 3 shows the table "Candidate vs. acceptance floors" (macro-F1 and every class's recall against its floor).
  - A lower value is refused with the FloorError message.

## Tests

Command: `pytest tests/test_transfer_learning/test_gate_floors.py` (9 passed).

| Case | Result |
|---|---|
| Macro-F1 ≈ 0.90 with one class at recall 0.5 | BLOCKED (`min_per_class_recall`) |
| Macro-F1 0.87 | BLOCKED (`min_macro_f1`) |
| Macro-F1 0.90 and every class at recall ≥ 0.70 | Floors PASS; the only blocking item is `human_approval` |
| A reviewer lowers either floor | `FloorError` |
| An edited report with a lower floor | BLOCKED (`acceptance_floors`) |

**Tests updated for D1.** Three assertions from the previous round expected the old behaviour, where no floor was proposed and the check was INCOMPLETE:
- `test_transfer_modes.py`: 2 assertions, and one test replaced by "a lower floor is refused";
- `test_ai_training_gui.py`: 1 assertion.

No existing measured threshold was changed.

## Documented candidates, re-evaluated

[`tl_gate_floors.md`](tl_gate_floors.md), from `python experiments/report_tl_gate_floors.py` (recorded metrics, nothing retrained):

| Candidate | Macro-F1 | Result |
|---|---|---|
| NEW_HEAD on the shifted condition | 0.556 | BLOCKED (macro-F1; per-class recall was not recorded) |
| NEW_HEAD adding `unbalance` | 1.000 | Floors PASS; needs a reviewer |
| GUI-trained 4-class model | 0.417 | Would be BLOCKED by both floors (it is not a TL candidate) |
