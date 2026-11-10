# AI-N33: the classified window and the default duration of AI Diagnostics (decision N-D1)

**Simulated.** Raw values: [`ai_n33.json`](ai_n33.json). **Verdict: CORRECTED.**

## Cause

- **Training windows** (`src/driveflow/models/common/windowing.py:215-216`, `acquisitions.window_starts` at `acquisitions.py:64-75`): every run is cut into non-overlapping windows **aligned to its start (t = 0)**. The first windows, inside the start-up, are kept.
- **The data scripts do not discard the start-up** (`experiments/generate_diagnosis_grid_dataset.py:67-100`, runs of 2.5 s): the start-up enters training, but only up to the grid's maximum of 220 rad/s.
- **AI Diagnostics before:** it classified the **last 2048 rows** of the sample, not a window of the training grid. With the 0.6 s default that was t = 0.40–0.60 s, the end of the acceleration to 300 rad/s, a transient no training run contains.

## Correction (only the page and the function that picks its window)

- **Window:** `ai_adapter.training_window_start` / `assess_training_window` pick the **last complete window of the training grid**, which is the training rule.
- **Default duration: 0.6 s → 0.82 s.** It is the shortest duration whose last grid window (0.6144–0.8192 s) lies entirely after t = 0.5 s. That is the start-up boundary with which the promoted models were evaluated (`report_classifier_evaluation`, `startup_transient`).
- **The grid rule alone at 0.6 s is not enough:** its window is 0.20–0.41 s, and there the faults come out as "normal" (see the JSON).
- **Unchanged:** `Scenario`, `datagen/`, the data scripts, the models, the thresholds and every byte-for-byte test.
- **The start-up warning stays** for anyone who shortens the duration.

## Before (0.6 s, last rows) and after (0.82 s, training grid)

| Case | Speed (rad/s) | Before: suggested (top, confidence) | After |
|---|---|---|---|
| healthy | 80 | normal (normal 0.999) | normal (normal 0.999) |
| healthy | 300 | cage (cage 1.000) | normal (normal 0.955) |
| healthy | 350 | cage (cage 1.000) | normal (normal 0.909) |
| outer_race | 300 | cage (cage 0.999) | outer_race (outer_race 0.950) |
| inner_race | 300 | cage (cage 1.000) | inner_race (inner_race 1.000) |
| ball | 300 | cage (cage 1.000) | ball (ball 1.000) |
| cage | 300 | cage (cage 1.000) | cage (cage 1.000) |

## Criterion (N1.5)

| Check | Result |
|---|---|
| A healthy motor gives `normal` (or abstains) at 80 rad/s, 300 rad/s and the page maximum (350 rad/s) | met |
| No fault that was detected before is worse | met |

- Before the change, every case at 300 rad/s came out "cage" at 1.00, including the healthy one. "Cage" was right only for the cage fault itself.
- After the change, all four faults are classified correctly.
- **No threshold or model was changed.**

**Tests:** `tests/test_ai_n33_startup_window.py`:
- the grid rule;
- that 0.82 s is the shortest steady duration;
- that the default healthy sample gives `normal` without a warning;
- that 0.6 s still shows the warning.
