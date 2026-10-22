# D1 floors applied to the documented candidates (round 2026-10-22, T5)

Command: `python experiments/report_tl_gate_floors.py`. Source metrics: [`custom_training_tl.json`](custom_training_tl.json), recorded on 2026-10-21; nothing was retrained. Floors: macro-F1 ≥ 0.88 and recall ≥ 0.70 for every class (default D1 2026-10-22).

| Candidate | macro-F1 | min_macro_f1 | min_per_class_recall | Floors verdict |
|---|---|---|---|---|
| A · GUI-trained 4-class model (custom training, not a TL candidate) | 0.417 | FAIL | FAIL: worst class inner_race recall 0.000 vs floor 0.7 (default D1 2026-10-22); below: inner_race, outer_race, unbalance | BLOCKED |
| B · NEW_HEAD from the promoted 5-class model, shifted condition | 0.556 | FAIL | NOT RECORDED: per-class recall not stored in custom_training_tl.json | BLOCKED |
| C · NEW_HEAD adding 'unbalance' to pc_server/2026-10-18_run01 | 1.000 | PASS | PASS: worst class inner_race recall 1.000 vs floor 0.7 (default D1 2026-10-22) | PASS (still needs human approval) |

A is not a transfer-learning candidate (no gate applies to it); it is listed because the request asked for it. Under the same floors it would be blocked.
Every candidate also needs a named reviewer's approval: the floors never approve on their own.
