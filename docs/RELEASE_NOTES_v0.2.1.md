# SPADE v0.2.1 — release notes (hotfix)

**All results are simulated.** SPADE ingests no measured drive data. No `.tflite` file has run on a Raspberry Pi 5 or an ESP32. The only board measurements are on a TI C2000 (LAUNCHXL-F28379D), replaying simulated windows from flash; PLECS itself has never been run here. **The assistant is experimental / internal demo** (AI-N30): it does not meet its benchmarks and can state false things in free text.

Hotfix of v0.2.0 (`aa6c5fd`). **Upgrade if you use the Assistant page:** in v0.2.0 a turn never finished in a real browser with the local model running (AI-N39). The evidence for each item is in `docs/VALIDATION_SUMMARY.md` §14.7 and the files under `docs/audit/`.

## Fixed

- **AI-N39 (High, present in v0.2.0):** the Assistant page never finished a turn in a real browser with Ollama running.
  - **Cause:** the avatar reported its state back on every repaint, each report requested a rerun, and the rerun cancelled the turn after its first model call.
  - **Fix:** the component reports only when asked. A real turn in Chrome now gets its reply in 13–18 s.
  - The CLI and the MCP server were never affected.
- **AI-N38 (Low):** when the user asks for a frozen v1 DC plant, the Assistant page now shows a fixed card: why it is not used (S-21) and what is simulated instead.
- **Truthful labels (2026-11-15):**
  - **SIMULATED** on Transfer Learning, Model Registry and Experiment Runs;
  - the full **AI SUGGESTION — REQUIRES VERIFICATION** on DC Motor and Monitoring;
  - the Model Registry tile "Hardware verified" is now **"Board-measured"** ("no board measurements recorded yet");
  - "Max macro-F1 drop" is labelled as the fraction it is (0.02 = 2 pt);
  - a visible notice of the **ESP32 classifier's low-speed limitation** (AI-N28, AI-N36) wherever that model appears.
  - A new test walks the 18 pages: no "hardware verified", "production ready" or "live", except two justified occurrences; full AI label on prediction pages; SIMULATED on result pages.
- **S-24:** `Scenario` checks the DC voltage limit against the effective supply (the given one, or the stock 65 V).
- **AI-N37:** a distilled run's `run_metadata.json` records `teacher_run`, `teacher_source` and `distill_alpha`.
- **CI:** `tests.yml` runs on every push to `dev`; a newer run cancels the older one on `dev` and `feat/*` only, never on `main`.
- Already in v0.2.0, listed for completeness: **AI-N32** (`OLLAMA_HOST` is honoured) and **AI-N33** (AI Diagnostics classifies a steady-state window and defaults to 0.82 s).

## New

- **DC Motor and PMSM pages, Phase 0 options (V1):**
  - DC Motor: plant mode v2 or v1 (the GEM-faithful mode, with the S-21 warning), load torque in N·m, supply voltage, limits and nominal values;
  - PMSM: the supply voltage;
  - both pages: the rotor inertia `j_rotor`.

  Every value is validated by `Scenario`. No default changed, and the stock run is identical.
- **MCP server of the assistant's tools (V4):**
  - stdio, standard library only;
  - the same policies as the assistant;
  - a launch needs the user's confirmation, tied to the proposal hash.
- **Classifier evidence:**
  - B4 (`docs/audit/b4/b4_results.md`);
  - B4c (`docs/audit/b4/b4c_results.md`): the PC and RPi5 candidates pass a confirmatory low-speed evaluation pre-registered before its data;
  - the AI-N36 quantization diagnosis (`docs/audit/b4/ai_n36_quant_diagnosis.md`).

  **The shipped models are the same as in v0.2.0.** The promotion of the B4 candidates is outside this release.
- **C2000 visit kit** (`docs/audit/b4/c2000_visit_kit.md`): nothing measured on the board yet.

## Open findings (read before using the results)

| ID | Severity | Status | What it means for you |
|---|---|---|---|
| **S-21** | Major | Corrected in v2 · **open in v1** | The v1 (GEM-faithful) plants keep the converter artifact when braking with a light load |
| **AI-N26** | Medium | Mitigated in software · board re-measurement pending | At 10 kHz the 512-sample ESP32 classifier does not keep up on the C2000 (68.5 ms for a 51.2 ms window) |
| **AI-N28, AI-N36** | High | **Open** (notice in the GUI) | The ESP32 int8 classifier is fragile, mostly at low speed |
| **AI-N29** | High | **Open** in this release | The shipped PC classifier is confidently wrong at 50 rad/s; the B4 candidates that fix it are not in this release |
| **AI-N30** | High | **Open** | The assistant does not meet its benchmarks |

Also open, unchanged: AI-N24, AI-N25, S-22. AI-N34 and AI-N35 (the B4 candidates) are resolved by the B4c evaluation; their promotion comes after this release. Full lists: `docs/SIMULATION_STABILITY_REPORT.md` and `docs/AI_TECHNICAL_AUDIT.md`.

## Compatibility

- **Python** 3.11 or 3.12.
- **No new dependencies.**

## Verification (gates G1–G8, `docs/audit/release/v0.2.1_gates.md`)

Run on `dev` at `e804fc4`, 2026-11-15, Linux, Python 3.12, nothing heavy in parallel. The release commit adds only the version number, these notes and the gate evidence; its CI is in `v0.2.1_published.md`.

| Gate | Result |
|---|---|
| G1 full suite | 2045 passed, 1 skipped, 9 xfailed, 0 failed (each xfail cites an open finding) |
| G2 smoke `--launch` | 7/7 |
| G3 CI | 5/5 |
| G4 fresh clone from GitHub (no `data/` or `workspace/`, Ollama unreachable) | smoke 7/7; 18/18 pages open without exception; the assistant reports itself unavailable |
| G5 assistant with the real model | LLM test passes; **one real turn in Chrome: reply in 18 s** |
| G6 AI Diagnostics defaults | healthy → normal at 80, 300 and 350 rad/s (0.999 / 0.955 / 0.909); each fault at 300 rad/s detected |
| G7 byte-for-byte tests | unchanged since v0.2.0 and passing |
| G8 absolute paths | none |

The version in the package, the citation and the Windows installer is **0.2.1**.
