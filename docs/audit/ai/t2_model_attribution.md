# T2 (round 2026-10-22): figures attributed to the run that produced them

**Scope:** `README.md`, `docs/VALIDATION_SUMMARY.md` and `docs/AI_TECHNICAL_AUDIT.md`. No model or registry entry was changed.

| Figure | Run that produced it | Where it was unattributed | Fix |
|---|---|---|---|
| AI-N07: 1.000 at 115 / 190 / 250 rad/s and severity ×0.6; 0.969 at 100 % noise | `pc_server/2026-10-18_run01` (3 classes, promoted 2026-10-18 → 2026-10-20) | VALIDATION_SUMMARY §2 row "Diagnosis classifiers", §3 row AI-N07; AUDIT §6 summary rows AI-N06/AI-N07; AUDIT §7.1 tables labelled "(promoted)" | Each mention names the run. The "promoted" labels now carry their dates. |
| AI-N14: 0.58 at 50 rad/s, 0.68 at 150 % noise | `pc_server/2026-10-18_run01` (`ood_detector.md`) | VALIDATION_SUMMARY §7.3; AUDIT §6 row and §8.1 | Run named. The promoted 5-class model was re-measured (below). |

**README:** it contains none of these figures, so nothing changed there.

## Equivalent evaluations of the promoted `pc_server_5class_w2048/2026-10-20_run01`

- **AI-N07 conditions:** they already existed, so nothing was rerun. Source: [`classifier_evaluation_5class_w2048.md`](classifier_evaluation_5class_w2048.md).
  - 1.000 at 115, 190 and 250 rad/s, at severity ×0.6 and at 100 % noise;
  - 0.963 at 100 rad/s;
  - 0.60 during start-up.
- **AI-N14 conditions:** they did not exist, so they were run now.
  - Command: `python experiments/report_ood_detector.py --run-dir configs/classifiers/pc_server_5class_w2048/2026-10-20_run01 --ood-set custom_orders --cache data/ood_eval_cache_custom.parquet --out-dir docs/audit/ai/promoted_pc_server_5class_w2048`.
  - Output: [`promoted_pc_server_5class_w2048/ood_detector_custom_orders.md`](promoted_pc_server_5class_w2048/ood_detector_custom_orders.md). It is written to a new directory, so no existing report was overwritten.
  - The cache is the one simulated for the A2 report on 2026-10-20 (`ood_eval_cache_custom.parquet`, not committed).

| Condition (accuracy on answered windows; abstention + Mahalanobis filter) | 3-class `pc_server/2026-10-18_run01` | Promoted 5-class `pc_server_5class_w2048/2026-10-20_run01` |
|---|---|---|
| 50 rad/s | 0.578 (n = 234; 42 % abstained or unknown) | 0.818 (n = 54; 59 % abstained or unknown) |
| 150 % noise | 0.675 (n = 234) | 1.000 (n = 54; none flagged) |

**Caveat.** The Mahalanobis filter in these numbers is fitted for the report and is **not** active in the GUI (AI-N17, decision D6).
