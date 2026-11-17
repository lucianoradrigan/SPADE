# AI-N40 corrected: trained envelope per classifier (2026-11-16)

**All results are simulated** (AppTest on simulated samples; no measured data). Decisions D1–D3 (Luciano, 2026-11-16); single-speed runs accepted by Luciano on 2026-11-16 23:12.

## What changed

- `driveflow.ai.envelope`: a classifier's trained envelope (speed range, rad/s) is read from `trained_envelope` in its `audit_findings.json`; when a run has none, it is deduced read-only from its training dataset's `grid_speed` (`run_metadata.json` → `dataset_path`), and is unknown when the dataset is missing.
- **Every PC-tier run selectable in AI Diagnostics declares its envelope** (12 runs: the promoted one plus 11), so a clone without `data/` knows it (`test_every_selectable_classifier_has_a_known_envelope`). Only `trained_envelope` was added; existing findings unchanged (checked against `HEAD`). The three promoted classifiers (`esp32_dscnn/2026-10-18_run01`, `pc_server_5class_w2048/2026-11-04_run01`, `rpi5_resnet1d_se_5class_w2048/2026-11-04_run01`) declare 80–220 rad/s.
- **Rule (deterministic, in Python, before the model runs):** outside the envelope AI Diagnostics, the DC Motor AI card and Monitoring show no class, no bars and no confidence, only `Outside the trained range (lo–hi rad/s): no diagnosis`, together with the AI-N40/AI-N29 grid notice and the section's AI SUGGESTION label. The simulation slider is not limited.
- **D2:** AI Diagnostics defaults to 150 rad/s (was 300).

## Deviation from the letter of A1 (accepted by Luciano, 2026-11-16)

A1 asks for a prediction at 80/150/220 rad/s. `pc_server/2026-09-01_run01` and `pc_server/2026-10-13_run01` were trained at **150 rad/s only** (`AI_TECHNICAL_AUDIT.md` AI-N07; for 2026-10-13 also the `rpm` column of its dataset, median 150 rad/s). They declare 150–150 rad/s and **diagnose only at 150 rad/s**; at 80 and 220 rad/s they show the box. Every other selectable run gives a prediction at 80/150/220.

## Existing tests

- `tests/test_ai_n40_speed_grid_notice.py::test_ai_diagnostics_default_sample_shows_the_notice`: one line added, `set_value(300.0)` on the speed slider, because the default moved from 300 to 150 rad/s (D2). No assertion and no threshold changed.
- `tests/test_ai_dashboard.py::TestSampleRunControlsAreReal::test_dc_motor_exposes_fault_speed_and_severity_widgets`: the asserted AI Diagnostics default changed from 300.0 to 150.0 (D2). Nothing else.
- `tests/test_ai_n40_speed_grid_notice.py::test_dc_motor_ai_card_shows_the_notice_at_the_default_speed` and `tests/test_truth_labels.py`: **unchanged** (the outside branch keeps the notice and the AI label, and the box is a warning, not an error, so pages that must show no error at their defaults still pass).

## AI-N42 found on the way

The AI Diagnostics "Classifier" selector reset to the promoted model on every rerun (since 2026-10-21, v0.2.0 and v0.2.1); see `AI_TECHNICAL_AUDIT.md` §13.15. Fixed on this branch. Both sweeps below choose the run through the page's request key right before the evaluating run, which worked before and after the fix, and record the run named by the model badge: **in every row the evaluated run is the selected one** (`summary.evaluated_is_selected_*` in `ai_n40.json`).

## Sweep: every selectable classifier × 80/150/220/300/350 rad/s (healthy, AI Diagnostics defaults)

Command: `PYTHONPATH=src:. python /mnt/datos/spade/a1/sweep.py <out.json>`; before = `dev` at `5f4fbae` (clone `/mnt/datos/spade/fresh_dev`), after = `feat/AI-N40-trained-envelope`. Tables: `python /mnt/datos/spade/a1/report.py before.json after4.json ai_n40.json`. "normal" / "No class suggested" = the page's metric; "box" = the no-diagnosis box.

#### before

| run | 80 | 150 | 220 | 300 | 350 |
|---|---|---|---|---|---|
| `promoted` | normal | normal | normal | normal | normal |
| `pc_server_5class_w2048/2026-11-10_run01` | normal | normal | normal | No class suggested | ball |
| `pc_server/2026-11-10_run01` | normal | normal | normal | normal | normal |
| `pc_server/2026-11-04_run01` | normal | normal | normal | normal | normal |
| `pc_server/2026-10-25_run01` | normal | normal | normal | normal | normal |
| `pc_server_5class_w2048/2026-10-25_run01` | normal | normal | normal | normal | normal |
| `pc_server_5class_w2048/2026-10-20_run01` | normal | normal | normal | normal | normal |
| `pc_server_5class/2026-10-20_run01` | normal | normal | normal | normal | cage |
| `pc_server/2026-10-18_run02` | normal | normal | normal | normal | normal |
| `pc_server/2026-10-18_run01` | normal | normal | normal | normal | normal |
| `pc_server/2026-10-13_run01` | normal | normal | normal | normal | normal |
| `pc_server/2026-09-01_run01` | normal | normal | normal | normal | normal |

#### after

| run | 80 | 150 | 220 | 300 | 350 |
|---|---|---|---|---|---|
| `promoted` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server_5class_w2048/2026-11-10_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-11-10_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-11-04_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-10-25_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server_5class_w2048/2026-10-25_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server_5class_w2048/2026-10-20_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server_5class/2026-10-20_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-10-18_run02` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-10-18_run01` | normal | normal | normal | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-10-13_run01` | box: no diagnosis | normal | box: no diagnosis | box: no diagnosis | box: no diagnosis |
| `pc_server/2026-09-01_run01` | box: no diagnosis | normal | box: no diagnosis | box: no diagnosis | box: no diagnosis |


**Result after:** 60 cases, 0 exceptions; above 220 rad/s **24/24 show the box, 0 predictions**. Before: at 350 rad/s the B4 candidate said 'ball' and `pc_server_5class/2026-10-20_run01` said 'cage' on a healthy motor (reachable in the GUI only through the request key, because of AI-N42).

## Screenshots (real Chrome, Playwright; `/mnt/datos/spade/a1/shots.py`)

| | 150 rad/s | 350 rad/s |
|---|---|---|
| before, promoted | `docs/screenshots/ai_n40/before_promoted_150rads.png` | `before_promoted_350rads.png`: "normal 90.9 %" + yellow notice |
| before, B4 candidate chosen in the selector | `before_b4_selection_ignored_150rads.png` | `before_b4_selection_ignored_350rads.png`: the badge names the **promoted** run (AI-N42) |
| after, promoted | `after_promoted_150rads.png`: "normal" | `after_promoted_350rads.png`: the no-diagnosis box (yellow), no class |
| after, B4 candidate | `after_b4_candidate_150rads.png`: "normal 99.7 %", badge `…2026-11-10_run01` | `after_b4_candidate_350rads.png`: "Outside the trained range (40–220 rad/s): no diagnosis" |

## Tests

`tests/test_ai_n40_trained_envelope.py` (23), `tests/test_ai_n42_classifier_selection.py` (2, fail without the fix), `tests/test_ai_n40_speed_grid_notice.py`, `tests/test_truth_labels.py`; with `tests/test_ai_dashboard.py` and `tests/test_v1_rotor_inertia.py`: 87 passed (2026-11-16).
