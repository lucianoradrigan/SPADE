# SPADE v0.2.1 — release notes (DRAFT, 2026-11-12; not tagged)

> **Draft.** Decision M-D7: there is no `v0.2.1` tag. The version number in the package, the citation and the Windows installer still says 0.2.0; bumping it is part of the release, when Luciano decides it.

**All results are simulated.** SPADE ingests no measured drive data. No `.tflite` file has run on a Raspberry Pi 5 or an ESP32. The only board measurements are on a TI C2000 (LAUNCHXL-F28379D), replaying simulated windows from flash; PLECS itself has never been run here. **The assistant is experimental / internal demo** (AI-N30).

Changes since `v0.2.0` (`aa6c5fd`, 2026-11-12). The evidence for each item is in `docs/VALIDATION_SUMMARY.md` §14.6–§14.7 and the files under `docs/audit/`.

## New

### DC Motor and PMSM pages: Phase 0 options in the GUI (V1)
- **DC Motor page:**
  - plant mode: v2 (default) or v1, the GEM-faithful mode, with the S-21 warning;
  - load torque in N·m;
  - supply voltage;
  - motor limits and nominal values.
- **PMSM page:** the supply voltage.
- **Rotor inertia** `j_rotor` on both pages (M6.2, 2026-11-12).
- Every value is validated by `Scenario`, and no default changed. The stock run is identical frame for frame. Evidence: `docs/audit/b3/v1_gui_phase0.md`.

### MCP server of the assistant's tools (V4)
- A local stdio server, standard library only.
- It applies the same policies as the assistant. A launch needs the user's confirmation through MCP elicitation, tied to the proposal hash.
- It was tried with Claude Code as a real client. `docs/assistant/MCP.md` says what leaves the machine with a cloud client.

### Classifier evidence (no model changed; nothing promoted)
- **B4** (`docs/audit/b4/b4_results.md`): pre-registered retraining on the v2 plants, with 40 and 60 rad/s added to the grids. No tier met its criterion:
  - the PC and RPi5 candidates failed only the two-sided abstention rule at low speed;
  - the ESP32 int8 candidate failed on its own merits (AI-N36).
- **B4c** (`docs/audit/b4/b4c_results.md`): confirmatory evaluation of the PC and RPi5 candidates on 150 new low-speed runs, with a one-sided abstention rule pre-registered before the data.
  - **Both pass:** macro-F1 0.994, min recall 0.970, selective risk 0.004 (the promoted PC gets 0.275).
  - **They are recommended for promotion and were not promoted.** The shipped models are those of v0.2.0.
- **AI-N36 diagnosis** (`docs/audit/b4/ai_n36_quant_diagnosis.md`, validation only):
  - int8 activation rounding shifts a fragile low-speed boundary of the small ESP32 model;
  - calibrating with low speeds does not fix it, and per-channel input scaling makes it worse.
- **C2000 visit kit** (`docs/audit/b4/c2000_visit_kit.md`): kits with `fs_hz` 10 kHz for `esp32_tiny` and the ESP32 classifier, and a step sheet. **Nothing measured on the board yet.**

## Fixed

- **AI-N39 (High, present in v0.2.0):** the Assistant page never finished a turn in a real browser with the local model running. The avatar's state report requested a rerun on every repaint and cancelled the turn. The component now reports only when asked. The CLI and the MCP server were not affected.

- **S-24:** `Scenario` checks the DC voltage limit against the effective supply, the given one or the stock 65 V. Before, it did so only when both were given explicitly.
- **AI-N37:** a distilled run's `run_metadata.json` now records `teacher_run`, `teacher_source` and `distill_alpha`. For a registry teacher, `metrics.json` had kept only the registry key.
- **CI:** `tests.yml` runs on every push to `dev`, and a newer run cancels the older one on `dev` and `feat/*` only, never on `main`.
- Already in v0.2.0, listed here for completeness: **AI-N32** (`OLLAMA_HOST` is honoured) and **AI-N33** (AI Diagnostics classifies a steady-state window and defaults to 0.82 s).

## Open findings (read before using the results)

| ID | Severity | Status | What it means for you |
|---|---|---|---|
| **S-21** | Major | Corrected in v2 · **open in v1** | The v1 (GEM-faithful) plants keep the converter artifact when braking with a light load |
| **AI-N26** | Medium | Mitigated in software · **board re-measurement pending** | At the real 10 kHz the 512-sample ESP32 classifier does not keep up on the C2000 (68.5 ms for a 51.2 ms window) |
| **AI-N28, AI-N36** | High | **Open** | The ESP32 int8 classifier is fragile, mostly at low speed |
| **AI-N29** | High | **Open** while the B4 candidates are not promoted | The promoted PC classifier is confidently wrong at 50 rad/s |
| **AI-N30** | High | **Open** | The assistant does not meet its benchmarks; it can state false things about SPADE in free text |
| **AI-N34, AI-N35** | Medium | Open: B4c passes; promotion is Luciano's decision | — |
| **AI-N38** | Low | **Open** | When the user asks for a frozen v1 plant, the assistant proposes v2 without saying why. The proposal card shows the plant actually used |

Also open, unchanged: AI-N24, AI-N25 and S-22. Full lists: `docs/SIMULATION_STABILITY_REPORT.md` and `docs/AI_TECHNICAL_AUDIT.md`.

## Compatibility

- **Python** 3.11 or 3.12.
- **No new dependencies.**

## Verification (dev stability check, `docs/audit/release/dev_stability_2026-11-12.md`)

First run on `dev` at `849721c`, then **re-run at `e4fed56`** after AI-N39 was found and fixed (2026-11-12, Linux, Python 3.12). Verdict: **stable at `e4fed56`**.

| Gate | Result |
|---|---|
| Full suite | 2019 passed, 1 skipped, 9 xfailed, 0 failed |
| Smoke `--launch` | 7/7 |
| CI | 5/5 (run 43250952418) |
| Fresh clone (no `data/`, Ollama unreachable) | smoke 7/7; 18/18 pages open, 18 screenshots; the assistant reports itself unavailable |
| Assistant with the real model | LLM test passes; end-to-end demo completes; **one real turn in Chrome: reply in 18 s** |
| AI Diagnostics defaults | healthy → normal at 80, 300 and 350 rad/s |
| Byte-for-byte tests | unchanged and passing |
| Absolute paths | none |

Frontend: the assistant page polish of M6.3 is on the branch `feat/FE-assistant-polish`, **not included** (pending Luciano's review).
