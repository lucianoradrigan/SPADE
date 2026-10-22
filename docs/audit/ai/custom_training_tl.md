# Custom classifier training and transfer-learning modes: evidence (2026-10-21)

Every number below comes from one run of `python experiments/report_custom_training_tl.py` (4 simulation processes), which also writes `custom_training_tl.json`. All data is SIMULATED. No model was promoted; the models were written to a temporary directory.

## A. AI Diagnostics → Train a model

- **Request:** classes normal, outer_race, inner_race, unbalance (unbalance = user-defined fault, order 1.0, mechanical severity 0.03). Controllers PI and MPC; ω_ref ∈ {100, 150, 200} rad/s; severity × {0.5, 1, 2}; noise up to 50 % in two thirds of the runs; ψ_e and J drift. Runs of 1.5 s with the first 0.5 s dropped; window 1024 samples; `pc_server` architecture; 15 epochs.
- **Data:** 72 runs (normal: 18, outer_race: 18, inner_race: 18, unbalance: 18), 0 failed.
- **Windows:** train 360 / validation 144 / test 144, split by run.
- **Test macro-F1 0.417**, accuracy 0.458. OOD detector fitted: True.
- **Recall per class:** inner_race 0.000, normal 1.000, outer_race 0.306, unbalance 0.528.
- **Time:** simulation 397.8 s for 108 simulated seconds with 4 processes (3.683 s per simulated second). The GUI's single-process estimate was 324 s. Training took 14.8 s.

Test confusion matrix (windows):

| true \ predicted | inner_race | normal | outer_race | unbalance |
|---|---|---|---|---|
| inner_race | 0 | 0 | 27 | 9 |
| normal | 0 | 36 | 0 | 0 |
| outer_race | 0 | 0 | 11 | 25 |
| unbalance | 0 | 0 | 17 | 19 |

## B. Full vs. feature transfer on a shifted condition (same classes)

- **Base:** `configs/classifiers/pc_server_5class_w2048/2026-10-20_run01`, the promoted 5-class model.
- **Condition:** MPC, 120-220 rad/s, 0-60 % noise (all runs noisy), 1.5 s runs, first 0.5 s dropped.
- **Data:** 36 acquisitions. Windows: train 72 / validation 36 / test 36. Classes present: inner_race, normal, outer_race. 10 epochs each.

| Strategy | Mode | Base macro-F1 (same test) | Candidate macro-F1 | Candidate accuracy | Failing checks | Base unchanged |
|---|---|---|---|---|---|---|
| full_finetune | full | 1.000 | 1.000 | 1.000 | none | True |
| feature_extractor | feature | 1.000 | 1.000 | 1.000 | none | True |
| new_head | feature | 1.000 | 0.556 | 0.667 | macro_f1_vs_base | True |

The base model keeps all 5 classes. The test set holds only the 3 classes listed, so its macro-F1 is computed over the classes present.

## C. A class the base model does not have (feature transfer, new output layer)

- **Base:** `configs/classifiers/pc_server/2026-10-18_run01`, classes inner_race, normal, outer_race.
- **Target classes:** inner_race, normal, outer_race, unbalance (new vocabulary: True).
- **Full fine-tuning was refused:** "the dataset has classes the base model does not know; only feature transfer with a new output head can learn them".
- **Windows:** train 456 / validation 228 / test 228.
- **NEW_HEAD candidate:** macro-F1 1.000; recall inner_race 1.000, normal 1.000, outer_race 1.000, unbalance 1.000.
- **Checks:** predictions_finite PASS, mode_collapse PASS, beats_majority_baseline PASS, macro_f1_vs_base NOT_APPLICABLE, min_macro_f1 PASS, convergence PASS.
- **min_macro_f1 floor:** 0.8 for this report only. The GUI proposes no value; the reviewer has to enter one.
- **Gate without a reviewer:** BLOCKED, blocking human_approval.

## Limits

- Simulated data only. The unbalance, ball and cage amplitudes are synthetic placeholders, not calibrated against measured machines.
- One seed per experiment. The differences between strategies in B are single runs, not a statistical comparison.
- The simulation-time estimate in the GUI is an order of magnitude measured on this laptop, not a guarantee for other machines.
