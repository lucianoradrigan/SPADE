# SPADE v0.2.2 — release notes

**All results are simulated.** SPADE ingests no measured drive data. No `.tflite` file has run on a Raspberry Pi 5 or an ESP32. The only board measurements are on a TI C2000 (LAUNCHXL-F28379D), replaying simulated windows from flash; PLECS itself has never been run here. **The assistant is experimental / internal demo** (AI-N30): it does not meet its benchmarks and can state false things in free text.

Follows v0.2.1 (`122ce2e`). **Upgrade if you use AI Diagnostics, the DC Motor AI card, Monitoring or the ESP32 model:** in v0.2.1 a classifier could give a confident, wrong diagnosis at speeds it was never trained on (AI-N40), the AI Diagnostics classifier selector silently fell back to the promoted model (AI-N42), and the ESP32 notice quoted another model's numbers (AI-N43). Evidence: `docs/VALIDATION_SUMMARY.md` and the files under `docs/audit/`.

## Fixed

- **AI-N40 (High): no diagnosis outside a classifier's trained range.**
  - Each classifier declares its **trained envelope** (the speed range of its training grid) in its `audit_findings.json`. The three promoted classifiers: 80–220 rad/s. Every other PC run selectable in AI Diagnostics declares its own too, so a clone without `data/` knows it.
  - AI Diagnostics, the DC Motor AI card and Monitoring check the simulated speed **before** running the model. Outside the envelope there is no class, no bars and no confidence, only *"Outside the trained range (X–Y rad/s): no diagnosis"*. The rule is deterministic Python; the simulation slider is not limited. The DC Motor page keeps its 300 rad/s default, so its AI card now shows this box until the speed is set inside 80–220 rad/s.
  - **AI Diagnostics now defaults to 150 rad/s** (was 300 rad/s, above the training grid).
  - The two old single-speed PC runs (`pc_server/2026-09-01_run01`, `2026-10-13_run01`) were trained at 150 rad/s only; they diagnose only there.
  - Before: on a healthy simulated motor at 350 rad/s, the B4 PC candidate said 'ball' and `pc_server_5class/2026-10-20_run01` said 'cage'. After: every selectable classifier shows the box above 220 rad/s (`docs/audit/release/ai_n40.md`).
- **AI-N42 (Medium, present since v0.2.0):** the AI Diagnostics "Classifier" selector reset to the promoted model on every rerun, so in a browser only the promoted model could be evaluated. Fixed; the release gate G6 of v0.2.1 had evaluated only the promoted model.
- **AI-N43 (Medium, present in v0.2.1):** the ESP32 low-speed notice quoted the int8 numbers of a retrained ESP32 that was never shipped ("misses most inner-race faults, recall 0.35"). The shipped `esp32_dscnn/2026-10-18_run01` gets inner-race right at 50/70 rad/s and misses most **outer-race** faults there (recall 0.28, 31 % abstained; its float model 0.22). The notice now quotes the shipped model's own numbers. Found by a claims audit.
- **AI-N41 (Low):** the assistant's glossary no longer drops a finding named by its ID.

## New

- **Phase 2 of the assistant (datasheet extraction): scoring criteria approved, not executed.** A third of the golden set is held out (stratified, fixed seed, committed before any iteration); the scorer reads it only with `--final-eval`, and a test enforces it. No extractor or LLM has been run on the golden set.

**The shipped models are the same as in v0.2.1.** Only metadata (`trained_envelope`, and an AI-N40 note on two old runs) was added to the runs' `audit_findings.json`.

## Open findings (read before using the results)

| ID | Severity | Status | What it means for you |
|---|---|---|---|
| **S-21** | Major | Corrected in v2 · **open in v1** | The v1 (GEM-faithful) plants keep the converter artifact when braking with a light load |
| **AI-N26** | Medium | Mitigated in software · board re-measurement pending | At 10 kHz the 512-sample ESP32 classifier does not keep up on the C2000 |
| **AI-N28, AI-N36** | High | **Open** (notice in the GUI) | The ESP32 int8 classifier is fragile, mostly at low speed |
| **AI-N29** | High | **Open** in this release | The shipped PC classifier is confidently wrong at 50 rad/s. Below 80 rad/s it is now outside its envelope and shows no diagnosis |
| **AI-N30** | High | **Open** | The assistant does not meet its benchmarks |

Full lists: `docs/SIMULATION_STABILITY_REPORT.md` and `docs/AI_TECHNICAL_AUDIT.md`.

## Compatibility

- **Python** 3.11 or 3.12.
- **No new dependencies.**

## Verification (gates G1–G8, `docs/audit/release/v0.2.2_gates.md`)

Run on `dev` at `54a7606`, 2026-11-16, Linux, Python 3.12, nothing heavy in parallel. The release commit adds only the version number, these notes and the gate evidence; its CI is in `v0.2.2_published.md`.

| Gate | Result |
|---|---|
| G1 full suite | 2132 passed, 1 skipped, 9 xfailed, 0 failed (each xfail cites an open finding) |
| G2 smoke `--launch` | 7/7 |
| G3 CI | 5/5 |
| G4 fresh clone from GitHub (no `data/` or `workspace/`, Ollama unreachable) | smoke 7/7; 18/18 pages open without exception; the assistant reports itself unavailable |
| G5 assistant with the real model | LLM test passes; **one real turn in Chrome: reply in 18 s** |
| G6 AI Diagnostics, **every selectable classifier** (D3) | 12 runs × 80/150/220/300/350 rad/s: 60/60 as expected; above 220 rad/s 24/24 show "no diagnosis" and no class. Default (150 rad/s, promoted PC): healthy → normal 0.922; each fault detected. In Chrome: default → prediction, 350 rad/s → no diagnosis |
| G7 byte-for-byte tests | unchanged since v0.2.1 and passing |
| G8 absolute paths | none |

The version in the package, the citation and the Windows installer is **0.2.2**.
