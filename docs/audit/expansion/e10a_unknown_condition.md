# E10a: "Unknown condition" state in AI Diagnostics

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E10a. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §8.2.
Branch: `feat/E10a-unknown-condition` from `dev` (c619a0c). Date: 2026-10-21. Machine: Linux x86-64, Python 3.12.13, Streamlit 1.66.0.

This task does not implement or refit a detector: it uses `ai/ood.py` from A1 as it is. The optional ensemble / MDN candidate (Fund. `4_Neural_Network/Multi_Modal_Distribution/`) was not added.

## 1. What was built

### 1.1 `src/driveflow/ai/condition.py` (domain)

It decides four mutually distinguishable states, in this order (the spec's decision rule):

| State | GUI label | When |
|---|---|---|
| `input_invalid` | Input invalid | a model channel is missing, there are fewer rows than one window, or the window has NaN/±inf. Nothing is predicted. |
| `unknown` | Unknown condition — REQUIRES VERIFICATION | the run has a usable OOD detector and the window's score is above its threshold, whatever the top-class confidence |
| `uncertain` | Uncertain — low confidence | not OOD, but the calibrated top-class probability is below the run's abstention threshold |
| `known` | Known condition | both checks passed |

- The decision is `ai/calibration.decide()`, the same function the monitoring path uses, so the two cannot disagree.
- `ood_availability(metrics)` says whether the `unknown` state is reachable. A detector counts only when `metrics.json["ood"]` names a known method (`mahalanobis`/`energy`) with a finite threshold calibrated on the validation split (`fitted_on.thresholds == "val"`, as `fit_ood` writes it).
- Otherwise the assessment carries `ood_available = False` and the reason. A score passed for a run without a detector is ignored, so no OOD score is ever shown for it.

### 1.2 Adapter (`viz/adapters/ai_adapter.py`)

- `assess_last_window(domain, df)`: validates the last window, then runs the model and the detector. An invalid input never reaches the model.
- `ood_validation_report(run_dir, metrics)`: finds the A1 evaluation report (`docs/audit/ai/ood_detector*.json`) of *this* detector. It matches by run path **and** threshold, because several reports cover one run with different feature layers.
- `agent_hypothesis(domain, df, operating_point)`: the simulation-based agent (`driveflow.agents`, the same one Monitoring uses) as a secondary hint.
- `condition_evidence(...)`: the JSON behind "Export evidence".

### 1.3 Page (`viz/views/ai_diagnostics.py`, classifier panel)

- One state box per outcome, each with its own label, colour and icon:
  - Input invalid: error, with the reasons;
  - Unknown: error, with the OOD score against the threshold and the method; the top class is named but not suggested;
  - Uncertain: warning, with the spec's sentence "The classifier cannot distinguish reliably among its known classes.";
  - Known: green **only when the OOD check ran**. Without a detector it is a neutral box, "Known condition (confidence check only)", so a confident "normal" is never shown as healthy in green unless the input was also checked against the training distribution.
- The probability chart highlights the top class only in the `known` state.
- "**OOD detection unavailable** — <reason>" whenever the model has no usable detector.
- Expander "OOD method and validation": the method, how the threshold was fitted, and the matching A1 numbers (false-unknown rate on in-distribution test windows; unknown rate and AUROC for each OOD set), or a note that no report exists.
- "Secondary simulated-agent hypothesis": runs on demand (several physics simulations). It shows the closest hypothesis and the agent's heuristic score, labelled as not calibrated and not a confirmed diagnosis. It never changes the classifier state. For a generated run the known speed setpoint is passed as the operating point (AI-N09); for an uploaded file the agent estimates it.
- "Export evidence (JSON)": the state, scores, model, input provenance and, if it was run, the agent's hypothesis.

## 2. State of the promoted models today

- `dc_motor/pc/classifier` = `pc_server_5class_w2048/2026-10-20_run01` has **no** `ood` block, so the page shows *OOD detection unavailable*, and *Unknown condition* cannot occur with the promoted model. I checked this in Chrome (sample run: healthy, 300 rad/s → "Known condition (confidence check only)", normal 98.2 %; agent: closest hypothesis `healthy`, score 0.00).
- `vsc_dpc` has no promoted classifier.
- The A1 detector on the 3-class run (`ood_detector.md`) flags only 13 % of ball windows (target ≥ 90 %). Fitting a detector to the promoted model and deciding whether it is good enough to show is a separate decision. This task did not refit or promote anything.

## 3. Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| AppTest of the three states | `tests/test_ai_condition_page.py::test_known_condition`, `test_uncertain`, `test_unknown_condition` (the Unknown test also checks a 98 % "normal" is not shown as a result) | PASS |
| Input invalid | `test_input_invalid` (missing channel: no metric, no prediction) and `tests/ai/test_condition.py::test_window_problems` (NaN/inf, short input) | PASS |
| "Detector absent" | `test_detector_absent` (caption *OOD detection unavailable*, no OOD score, not green) and `test_unavailable_detector_never_yields_unknown` (4 cases: no block, unknown method, NaN threshold, threshold not fitted on val; a score of 1e9 is ignored) | PASS |
| Validation report belongs to the run's detector | `test_ood_validation_report_matches_run_and_threshold`, `test_real_a1_report_is_found_for_its_run` | PASS |

In those AppTests the promoted model is replaced by a stub with fixed probabilities and a fixed OOD score; the decision is the real `condition.assess`. The existing AI Diagnostics and UI-contract tests still pass. `test_backend_failure_is_reported_honestly` now also patches the adapter, because the classifier panel goes through it.

## 4. Tests

| File | Tests |
|---|---|
| `tests/ai/test_condition.py` | 10 |
| `tests/test_ai_condition_page.py` | 9 |

Full suite on the branch: **890 passed, 0 failed** (668 s; 871 after E6 + 19 new) — `e10a_branch_pytest.txt`.

## 5. Limits

- The states are only as good as the detector. With A1's Mahalanobis detector most ball windows would still come out as *Known* or *Uncertain*, not *Unknown*.
- The agent's score is the existing heuristic (`agents/base.py`); its 0.7 alert threshold is not calibrated (AUDIT_ANSWERS Q25).
