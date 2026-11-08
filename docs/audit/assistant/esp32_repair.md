# AI-N28 — repair round of the ESP32 int8 classifier

**Simulated.** Pre-registration: [`esp32_repair_preregistration.md`](esp32_repair_preregistration.md), approved by Luciano (decision E1, 2026-11-08) and committed before any export, generation or evaluation. Raw values: [`esp32_repair.json`](esp32_repair.json).

- **Commands:** `python experiments/report_esp32_repair.py prepare | export | select | test --work <scratch dir>`.
- **E2a′:** `python experiments/generate_ai_n27_heldout.py --grid 3class --rng-seed 20261108 --first-run-seed 70001 --set-name esp32_repair_E2a_prime --out data/esp32_repair_heldout_3class_dcm.parquet --workers 10` (45 runs, 2365 s of wall time).
- **Stage 2 trainings:** `experiments/train_model.py --config configs/classifiers/esp32_dscnn.yaml --dataset data/diagnosis_grid_dataset_dcm.parquet --epochs 30 --batch-size 32 --seed 42 --keep-rpm-level --cap-per-class 3000`, with:
  - K1: `--distill --distill-alpha 0.75 --teacher-run configs/classifiers/pc_server/2026-11-04_run01` → `esp32_dscnn/2026-11-08_run01`;
  - K2: no distillation → `2026-11-08_run02`.

## Verdict: **DOES NOT MEET THE CRITERION** (selected variant V11)

The promoted ESP32 (`esp32_dscnn/2026-10-18_run01`) stays, with its AI-N27 warning. AI-N28 stays open with this evidence. Nothing is retried in this round (pre-registration §6).

## Preconditions

| | Result |
|---|---|
| R1, reproduction | V0 re-export **byte-identical** to the committed candidate (SHA-256 `2932b1702d9f939f…`); int8 outputs identical on all validation windows |
| R2, per-tensor switch | TensorFlow 2.21.0: 1 scale per weight tensor with the switch, 12 per channel without |
| R3, split | 1811 validation windows = recorded; groups match |
| R4, E2a′ is new | seeds 70001–70045; 0 overlaps with the old E2a, the grid or the evaluation cache; 0 bit-identical plant acquisitions |
| P3, stage 2 split | K1 and K2 have the candidate's `split.group_hashes` |

## Selection on validation

1811 windows. The fixed rule ranks by minimum per-class recall, then macro-F1, then the lowest variant number.

| Variant | Keras | Calibration | Weights | Min recall | Macro-F1 | Abstention |
|---|---|---|---|---|---|---|
| V0 | K0 | C0 | per channel | 0.154 | 0.602 | 0.123 |
| V1 | K0 | C0 | per tensor | 0.149 | 0.602 | 0.114 |
| V2 | K0 | C1 | per channel | 0.126 | 0.588 | 0.120 |
| V3 | K0 | C1 | per tensor | 0.130 | 0.591 | 0.108 |
| V4 | K0 | C2 | per channel | 0.163 | 0.608 | 0.132 |
| V5 | K0 | C2 | per tensor | 0.141 | 0.598 | 0.128 |
| V6 | K1 | C0 | per channel | 0.345 | 0.703 | 0.257 |
| V7 | K1 | C0 | per tensor | 0.379 | 0.719 | 0.268 |
| V8 | K1 | C1 | per channel | 0.263 | 0.669 | 0.052 |
| V9 | K1 | C1 | per tensor | 0.255 | 0.665 | 0.052 |
| V10 | K1 | C2 | per channel | 0.783 | 0.857 | 0.223 |
| **V11** | K1 | C2 | per tensor | 0.819 | 0.867 | 0.210 |
| V12 | K2 | C0 | per channel | 0.757 | 0.866 | 0.114 |
| V13 | K2 | C0 | per tensor | 0.739 | 0.860 | 0.116 |
| V14 | K2 | C1 | per channel | 0.658 | 0.835 | 0.208 |
| V15 | K2 | C1 | per tensor | 0.650 | 0.832 | 0.209 |
| V16 | K2 | C2 | per channel | 0.744 | 0.863 | 0.119 |
| V17 | K2 | C2 | per tensor | 0.726 | 0.857 | 0.125 |

- **Keras models:** K0 is the 2026-11-04 candidate (alpha 0.5), K1 uses alpha 0.75, K2 has no distillation.
- **Calibration sets:** C0 is the current recipe's 1000 windows; C1 is the training split, stratified by class; C2 is the same, with noise-free acquisitions only.

## The one test: selected variant against the promoted ESP32 (int8, both)

| Set | Acq. | Windows | Macro-F1 prom. → V11 (Δ, 95 % CI) | Abstention prom. → V11 | Recall inner / normal / outer, prom. → V11 |
|---|---|---|---|---|---|
| **E1** (dcm test split) | 30 | 870 | 0.899 → 0.882 (-1.7, [-4.5, 1.0]) | 0.166 → 0.248 | 0.879 → 0.897 / 0.924 → 0.831 / 0.893 → 0.917 |
| **E2a′** (new held-out, affected regime) | 45 | 2160 | 0.815 → 0.856 (+4.1, [-3.0, 11.2]) | 0.232 → 0.271 | 0.953 → 0.972 / 0.949 → 0.879 / 0.560 → 0.717 |
| **E3** (generalization, 13 conditions) | 156 | 2844 | 0.990 → 0.882 (-10.7, [-15.3, -7.0]) | 0.039 → 0.229 | 0.987 → 0.987 / 0.994 → 0.687 / 0.988 → 0.989 |
| E2a old (descriptive only) | 45 | 2160 | 0.843 → 0.887 (+4.4, [-5.5, 14.0]) | 0.240 → 0.244 | 0.879 → 0.969 / 0.951 → 0.899 / 0.699 → 0.792 |

| Check (pre-registration §5.2) | Value | Pass |
|---|---|---|
| gate_D1_macro_f1 | 0.882 | yes |
| gate_D1_min_recall | 0.831 | yes |
| E1_delta_macro_f1 | -1.746 | **no** |
| E1_worst_delta_recall | -9.310 (normal) | **no** |
| E1_delta_abstention | 8.276 | **no** |
| E2a_delta_macro_f1 | 4.088 | yes |
| E2a_worst_delta_recall | -6.944 (normal) | **no** |
| E2a_delta_abstention | 3.889 | yes |
| E3_delta_macro_f1 | -10.745 | **no** |
| E3_worst_delta_recall | -30.696 (normal) | **no** |
| E3_delta_abstention | 18.952 | **no** |

## Reading (post hoc, descriptive; does not change the verdict)

- **The int8 collapse of AI-N28 does not appear in the selected variant.**
  - outer_race recall is 0.717 on E2a′ (promoted 0.560) and 0.989 on E3.
  - The variants of the old Keras model (K0, V0–V5) stay at a minimum recall of 0.13–0.16 on validation, whatever the calibration or the weight quantization.
  - That confirms §0 of the pre-registration: the problem was the float model, not only the quantization.
- **The new failure is the healthy class:**
  - normal recall falls to 0.687 on E3 (promoted 0.994) and to 0.831 on E1 (0.924);
  - abstention rises by 19 points on E3;
  - the candidate calls healthy windows faults, or abstains: false alarms.
- **E2a′ improves:** +4.1 points of macro-F1, CI [-3.0, 11.2]. But normal recall drops 6.9 points there too, past the −5 limit.
- **Validation was optimistic.** V11's validation normal recall was 0.819 and its abstention 0.210, so the selection rule (minimum recall) did see the weak class. The test shows the class is weaker still outside the grid's own runs (E3).

## Not done (pre-registration §6–§7)

No other variant tested, no other seed, alpha or threshold, no promotion, no C2000 export, no board measurement.

Possible next steps, each a new pre-registration and Luciano's decision:
- include the ESP32 in the consolidated model round E4 (grid below 80 rad/s and v2 data);
- a class-balanced loss or a normal-class threshold, chosen on validation;
- the 5-class ESP32 config.

**Candidates kept:** `esp32_dscnn/2026-11-08_run01` (K1) and `2026-11-08_run02` (K2). Neither is promoted, and their `audit_findings.json` says so.

**Variant files:** the 18 int8 variants were written to a scratch directory, not committed. V11's file is reproducible with `prepare` + `export`.
