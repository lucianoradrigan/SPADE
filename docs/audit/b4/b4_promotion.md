# B4 promotion (R2, 2026-11-15): **not merged** — G1 and G6 fail

Decision R-D3 approved promoting `pc_server_5class_w2048/2026-11-10_run01` and `rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` after B4c, in a branch, merged into `dev` only if G1, G3 and G6 pass.

- **Done on `feat/B4-promote` (`83fcf1d`):** both `audit_findings.json` completed (cage at 45 rad/s; RPi5 E1 CI; ball/cage synthetic only; the abstention rule changed after B4 and confirmed on E4c; simulated, no board measurement), and the two promotions with `experiments/promote_run.py` (relative paths in `configs/registry.yaml` and its history).
- **Result: G1 and G6 fail, so the branch is not merged.** Nothing to roll back: `dev` and `main` were never changed. G3 was not run. Everything is **simulated**.

## G6: AI Diagnostics with the defaults and the new models

| Case (0.82 s, PI, seed 0) | 2026-11-04 PC (shipped) | B4 PC candidate |
|---|---|---|
| healthy, 80 rad/s | normal 0.999 | **normal 0.998** |
| healthy, 300 rad/s | normal 0.955 | **cage 0.619 (abstains)** |
| healthy, 350 rad/s | normal 0.909 | **ball 0.879** |
| outer_race, 300 rad/s | outer_race 0.950 | **outer_race 0.995** |
| inner_race, 300 rad/s | inner_race 1.000 | **inner_race 1.000** |
| ball, 300 rad/s | ball 1.000 | **ball 1.000** |
| cage, 300 rad/s | cage 1.000 | **cage 1.000** |

**Healthy at 350 rad/s (the page's maximum) becomes 'ball' 0.879 as a known condition: a confident false positive. The default sample (300 rad/s) goes from 'normal' to an abstention.** G6 requires normal (or an abstention) at all three speeds, so it fails. The faults at 300 rad/s are detected as before or better.

## Why B4 and B4c did not see it

Both evaluated only up to 220 rad/s, the top of the training grid: E1 and E3 on the grid, E4 and E4c at low speed. AI Diagnostics and the DC Motor page default to **300 rad/s** and go up to 350. **Above the grid neither the old nor the new models were ever evaluated.**

## All four models above the grid (`scripts/oneoff/r2_highspeed_old_vs_new.py`, raw `b4_promotion/highspeed_old_vs_new.json`)

| Case | PC old | PC new | RPi5 old (shipped) | RPi5 new |
|---|---|---|---|---|
| healthy, 80 | normal 0.999 | normal 0.998 | normal 1.000 | normal 0.999 |
| healthy, 220 | normal 0.973 | normal 0.993 | normal 0.998 | normal 0.997 |
| healthy, 250 | normal 0.967 | normal 0.985 | normal 0.999 | normal 0.996 |
| healthy, 300 | normal 0.955 | cage 0.619 (abstains) | cage 0.633 (abstains) | normal 0.995 |
| healthy, 350 | normal 0.909 | ball 0.879 | cage 1.000 | normal 0.985 |
| outer_race, 300 | outer_race 0.950 | outer_race 0.995 | ball 0.928 | ball 0.895 |
| inner_race, 300 | inner_race 1.000 | inner_race 1.000 | inner_race 0.994 | inner_race 0.534 (abstains) |
| ball, 300 | ball 1.000 | ball 1.000 | ball 1.000 | cage 0.857 |
| cage, 300 | cage 1.000 | cage 1.000 | cage 1.000 | cage 0.991 |

- **The RPi5 model shipped in v0.2.1 already fails above the grid:** healthy at 350 rad/s → **cage 1.00**; outer_race at 300 → ball 0.93. It can be chosen in AI Diagnostics.
- **New finding AI-N40 (High):** the dc_motor classifiers are confidently wrong above their training grid (> 220 rad/s), and the GUI defaults to 300 rad/s.
  - Mitigated in `dev` with a visible notice (stop rule: SPADE says something false). No default and no model changed.
  - See `docs/AI_TECHNICAL_AUDIT.md` §13.13.

## G1: 5 failures (raw `b4_promotion.json`)

| Test | Cause | What a fix needs |
|---|---|---|
| `test_registry_resolves_the_grid_trained_classifier` | pinned to the 2026-11-04 run (assert resolve(...) == .../2026-11-04_run01) | changing an existing test (not allowed this round) |
| `test_promoted_dc_motor_classifiers_carry_ai_n27` | requires an AI-N27 finding and entries[key].endswith('2026-11-04_run01') | an AI-N27 audit entry (allowed) AND changing the pinned run in the test (not allowed) |
| `test_model_registry_lists_it_with_the_run` | the new runs carry no 'AI-N27' audit entry | an AI-N27 entry in their audit_findings.json (allowed; not done since the promotion stops anyway) |
| `test_default_healthy_sample_is_normal_without_warning` | with the new PC model the page's default sample (healthy, 300 rad/s) is no longer 'normal': it abstains (cage 0.62) | nothing to fix in the test: a real behaviour change (see G6) |
| `test_quoted_metrics_match_the_promoted_runs` | the system diagram quotes the 2026-11-04 runs' test accuracy | updating the quoted numbers in views/system_diagram.py (allowed) |

Two failures are pinned to the 2026-11-04 run. Passing them requires changing existing tests, which this round forbids.

## Screenshots (branch, new models)

- `b4_promotion/ai_diagnostics_default_300rads.png`: the default sample now ends in "No class suggested" (cage 62 %).
- `b4_promotion/model_registry.png`.

## What would be needed (decision for Luciano)

1. Evaluate both candidates **above the grid** (250–350 rad/s), pre-registered on new data, as B4c did at low speed. Today the new PC model is worse there than the shipped one.
2. If they pass: allow updating the two pinned tests and the quoted metrics, add an AI-N27 entry to their audit files, and repeat G1, G3 and G6.
