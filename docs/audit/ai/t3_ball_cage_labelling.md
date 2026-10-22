# T3 (round 2026-10-22): ball/cage labelling

Required text: **"ball/cage: synthetic-only, uncalibrated amplitude"** (`driveflow.ai.audit_findings.SYNTHETIC_LABEL`).
No model was retrained or re-promoted; these are metadata and interface changes only.
Test: `pytest tests/test_ball_cage_labelling.py` (12 passed).

## Runs (audit_findings.json)

Command: `pytest tests/test_ball_cage_labelling.py -k every_ball_cage_run`.

| Run | Classes include ball/cage | Finding present before T3 |
|---|---|---|
| `pc_server_5class/2026-10-20_run01` | yes | yes |
| `pc_server_5class_w2048/2026-10-20_run01` (promoted PC) | yes | yes |
| `rpi5_resnet1d_se_5class/2026-10-20_run01` | yes | yes |
| `rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` (promoted RPi5) | yes | yes |
| `esp32_dscnn_5class/2026-10-20_run01` | yes | yes |
| `esp32_dscnn_5class_w2048/2026-10-20_run01` | yes | yes |

**Gap found in code that creates runs.** Three paths could create a ball/cage run without the finding:
- AI Diagnostics → Train a model;
- transfer-learning candidates (`workflow.save_candidate`);
- imported models.

All three now call `ensure_synthetic_class_finding()`, and the tests cover the import path and the training path.

## GUI

| Where | Before T3 | After T3 |
|---|---|---|
| AI Diagnostics, classifier panel | Warning only when ball/cage was the **suggested** class | Also when it is the top class but abstained or flagged unknown, with the exact label |
| DC Motor, "AI suggestion" KPI | "synthetic-only class, uncalibrated amplitude", decided on the **decision** label (missed abstained top classes) | Exact label, decided on the top class |
| Monitoring, motor card "AI suggestion" | No label (`synthetic_only` was computed from the decision, not the class) | The prediction reads "ball (ball/cage: synthetic-only, uncalibrated amplitude)" |
| Model Registry, model table | Only "⚠ N finding(s)" (the text appeared after selecting a row) | New column "class limitation" with the exact label for every ball/cage run |
