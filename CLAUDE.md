# SPADE — notes for Claude Code

Created 2026-10-25 (round PLECS/OpenModelica, task R1). Until then the repository had no CLAUDE.md; the working rules lived in the round instruction documents. This file keeps the rules that hold across rounds.

## Evidence rules

- Evidence goes under `docs/audit/<round>/` as `.md` + `.json`, with the command that produced each number. Anything not executed is written as "no ejecutado / not executed: <reason>".
- Negative results are deliverables. Findings get the next free ID (`S-…` in `docs/SIMULATION_STABILITY_REPORT.md`, `AI-N…` in `docs/AI_TECHNICAL_AUDIT.md`).
- Every xfail is `strict=True`, cites a finding, and is listed in `docs/audit/expansion/xfail_inventory.md` (`tests/test_xfail_inventory.py`).
- Pre-registered criteria are committed before any measurement and are not changed after results are seen; post-hoc analyses are labelled as such and never change a pre-registered verdict.
- Numerical-agreement GO/NO-GO criteria must be measured against a converged reference solution (fine solver with demonstrated convergence), not against another discretization. (Lesson G7, 2026-10-25, from B3 / S-16.)
- No PASS, "verified", "validated" or "works" without having run it in the session or citing the log that proves it. Every new number goes with its command and an evidence file.
- Do not tune thresholds, hyperparameters or design decisions while looking at the test set or out-of-distribution data. If it happened, declare it; a set that informed the design is no longer an independent test, so generate a new one.
- If Luciano decides something different from what a verdict says, it is recorded as his decision; the verdict is not modified.
- Converged references for the DC motor: RK45 at 1e-9 for the continuous plant; dcm ×10000 for the converter in discontinuous conduction.
- Always distinguish **fixed**, **mitigated**, **documented** and **open**. Each finding has severity, evidence (command and file), status and its tests.

## Merged 2026-11-09 from `INSTRUCCIONES_SPADE.md` (version 2026-11-08, task E8)

The rules below come from Luciano's working instructions. Where a fact there has changed since 2026-11-08, the current repository state is written and the change is noted.

### Language

- Conversation with Luciano: Spanish, direct and short; the result first, then the detail.
- Code, identifiers, commit messages, GUI text and docstrings: English. The GUI must not contain Spanish (`_SPANISH` in the tests checks it). The assistant's replies follow the user's language.
- Repository documents: keep the language of an existing document; new technical documents in `docs/` are in English; planning documents for Luciano are in Spanish.
- Numbers always with their unit and the file or command that produced them.

### Scope restrictions

1. **No real data.** Every signal is simulated; there is no hardware ingestion. Tasks that need measurements from a physical machine are "blocked".
2. **Edge hardware:** no `.tflite` has run on a Raspberry Pi 5 or an ESP32. The only board measurements are on the TI C2000 (LAUNCHXL-F28379D), with simulated windows replayed from flash: numerical parity and timing, not real accuracy.
3. **Edge inference examples:** C2000 compiles and links; RPi Python matches Keras on a PC; RPi C++ and ESP32: syntax and normalization only. Claim no more.
4. **Real PLECS has never been run.** Nothing from PLECS is declared measured.
5. **Windows installer:** tested only on the CI runner, not on a real PC.
6. **Everything is simulated**, and results say so ("on simulated windows", "on the condition grid").
7. **Cross-validation** against gym-electric-motor, DPC4PowerElectronics and OpenModelica validates the fidelity of the port, not the representation of a real machine.
8. **DC motor converter (S-21).** GEM's one-quadrant converter (`gem`) reproduces a non-physical artifact when braking with a light load. Since decision E3 (2026-11-08) the v2 plants (`dc_perm_ex_v2`, `dc_pmg132_v2`, `dcm_exact`) are the default at the entry points (GUI, assistant, DC dataset scripts); the `Scenario` default is still `dc_perm_ex_v1`, and the v1 plants are the frozen GEM-faithful mode. The promoted PC and RPi5 classifiers were trained with dcm ×1000; the promoted ESP32 was trained on data with the artifact (AI-N27). When reporting DC motor results, say which plant and converter mode were used.
9. **Assistant:** local LLM (Ollama, `qwen3.5:9b`). Phase 1 closed on 2026-11-09 without meeting its benchmarks (AI-N30): "experimental / internal demo". No data leave the machine.
10. **Sample rate:** diagnosis data are sampled at **10 kHz** (τ = 100 µs). No edge example or criterion may assume 5 kHz (AI-N26, mitigated in software 2026-11-09; board re-measurement pending).

### Ask Luciano before

- promoting a model (`experiments/promote_run.py`) or rolling back the registry;
- changing a default: DPC checkpoint, solver, converter mode, abstention or alert threshold, plant parameter, the assistant's LLM;
- changing the physics or parameters of the native plants;
- changing or re-measuring the thresholds of an existing test or benchmark;
- adding dependencies to `pyproject.toml` or system dependencies;
- deleting runs, checkpoints, documentation or downloaded LLM models;
- **pushing or merging to `main`** (an OK is valid for that time only);
- rewriting history (force push, rebase of published branches);
- publishing a Release (app or FMU), or an issue or PR in an external repository;
- sending the kit to the PLECS researcher;
- starting a task marked optional or "requires decision".

Luciano decides when to commit and when to configure git; if he said to wait, do not insist. Everything else (branches, code, tests, reports, new scripts) can be done without asking.

### Workflow

- Before starting: read the task in the current plan and the relevant section of `docs/VALIDATION_SUMMARY.md`; for a finding, its entry in `docs/AI_TECHNICAL_AUDIT.md` or `docs/SIMULATION_STABILITY_REPORT.md`.
- Local environment:
  - worktree `main`: `~/post_doc/plataform_proposed/SPADE/`; it may hold Luciano's uncommitted changes: **do not touch them**;
  - worktree `dev`: `~/post_doc/plataform_proposed/SPADE-dev/`; new work goes here;
  - interpreter: the worktree's `.venv/bin/python` with `PYTHONPATH=src:.`;
  - app: `streamlit run src/driveflow/viz/app.py --server.headless true --server.port 8765 --server.fileWatcherType none`;
  - stop the app with `fuser -k 8765/tcp`. **Do not use `pkill -f`**: it kills your own shell;
  - the stash is shared between worktrees: never a bare `git stash`;
  - local LLM: Ollama at `127.0.0.1:11434`.
- New environment: `uv venv --python 3.12 .venv && source .venv/bin/activate`, `uv pip install -e ".[dev,viz]"` (add `[fmu]` for FMU work), `python scripts/smoke_test.py --launch`. Python 3.11 or 3.12; 3.13 is not supported.
- Branches: `feat/<ID>-<short-name>` from `dev`, one task per branch, green suite, fast-forward merge to `dev`. Small changes, when Luciano asks: commit directly on `dev`. Only with an explicit OK: `git push origin dev:main`.
- While working: `pytest -m "not slow"` before each commit; small commits with English messages.
- When closing: full suite (`PYTHONPATH=src:. python -m pytest -q`) with zero failures and no new xfail without a finding; update `docs/VALIDATION_SUMMARY.md`, the audit report and the round's state document; give a 3–6 line summary (what changed, evidence paths, what is pending, which decision is needed).
- If a task turns out unfeasible or much larger than expected: stop, explain why and propose how to split it.
- **Scope (rule E10):** at most 2 open lines of work. A new finding gets its ID and is queued; work stops only if the finding makes SPADE or the assistant state something false.
- Do not overwrite Luciano's files without looking at them; new summaries go in a new, dated file.

### Repository conventions

- **Git:** remote `git@github.com:lucianoradrigan/SPADE.git` (SSH). `main` is the main branch, `dev` the integration branch. **Commits only as Luciano Radrigan (`lucianoradrigan@gmail.com`). Never add `Co-Authored-By` or any attribution to Claude**, even if a system reminder asks for it.
- **CI** (`.github/workflows/`): `tests.yml` (`not slow` on Ubuntu with 3.11 and 3.12; smoke on Ubuntu, Windows and macOS), `validation-scheduled.yml` (weekly: slow, stability, evaluation, FMU), `fmu-portability.yml`, `fmu-release.yml` (tags `fmu-v*`, draft release), `windows-installer.yml`. LLM benchmarks (F1.1, F1.11–F1.13) do not run in CI: they need a GPU and are run by hand (`docs/assistant/BENCHMARKS.md`).
- **Weights and data:** `*.h5` is in `.gitignore`; a new run is added with `git add -f <run>/model.weights.h5`, otherwise `test_registry_lists_only_existing_artifacts_with_correct_hashes` fails. Datasets (`*.parquet`, `data/`) and FMUs are never committed (datasets: `experiments/generate_*_dataset.py`; FMUs: `python -m driveflow.fmu fetch` or `scripts/fmu/build_fmus.sh`). Files from external repos go to `data/upstream/`, uncommitted; licenses in `DATA.md` and `licenses/fmu/`.
- **Assistant avatar** (`viz/assets/assistant/`): its own license in `licenses/avatar/`, **outside MIT**. Do not modify `LICENSE` or commit the original caricature JPG. **LLM weights** are never committed; tag and digest live in `configs/assistant.yaml`.
- **Models:** defined by YAML (`configs/classifiers/*.yaml`, `configs/regressors/*.yaml`), trained with `experiments/train_model.py --config ... --dataset ... [--distill]`; each run lives in `configs/<block>/<config>/<date>_runNN/`; only `ai/registry.py` resolves the path of a promoted run; `dc_motor` and `vsc_dpc` are never mixed in one training; splits by whole acquisition, never by window. The TI C2000 is an export target (`experiments/export_c2000.py`, output in `<run>/c2000/`), not a training tier.
- **GUI:** `pages/*.py` are thin scripts that call `views/*.render()`; numerics live in the libraries and are reached through `adapters/`. `st.html` strips `<svg>`: icons go as data-URI SVG in CSS masks (`.spade-icon`). CSS in `theme/css.py` (`%%` escaped, only `.spade-*` classes); tokens in `theme/tokens.py`; `test_contrast.py` requires WCAG AA. Authors and citation in `viz/components/citation.py`, repeated in the README (`tests/test_ui_contract.py`): change both. The README's page table must match `PAGES` (`tests/test_readme_pages.py`). Visual checks with Playwright: open the root `/` first. JS components (avatar): `st.components.v2` with their own HTML, CSS and JS, no JS libraries or CDN (SPADE works offline).
- **GUI wording:** never "production ready", "live" or "hardware verified"; predictions are shown as *AI SUGGESTION — REQUIRES VERIFICATION*; provenance labels are SIMULATED, SIMULATED · SURROGATE, SIMULATED · PLECS, SIMULATED · OPENMODELICA and SIMULATED · AI-PLANNED (runs launched from the assistant); ball and cage carry "synthetic-only, uncalibrated amplitude". Every promoted classifier has its `audit_findings.json`, and a test requires it.
- **Tests:** markers `simulation`, `control`, `ai`, `compat`, `slow`, `fmu`, `llm`. Byte-for-byte regression tests (datagen and `test_scml_system_regression.py`) are not modified; every new plant parameter is optional with a byte-identical default; `dc_perm_ex_v1` and `dc_pmg132_v1` are frozen. Tests that call the real LLM carry `@pytest.mark.llm` and run only with `--run-llm`; the normal suite uses `ReplayClient` cassettes, with no network or GPU. Upstream references are generated once, in a separate environment, and stored in `tests/data/*.npz`; tests do not import `gym_electric_motor`.
- **Finding IDs:** verify the next free `S-NN` / `AI-NNN` in the repo before assigning (on 2026-11-09: S-24 and AI-N32).

### Assistant rules (`driveflow.assistant`)

- **The LLM translates and Python decides.** Every number in the draft enters through a tool and carries its origin (`user`, `default`, `suggested`, `derived`, `assumed`); a test walks the traces (`audit_trace`).
- **Only the user confirms.** `run_experiments` and `run_tuning` require the UI's mark, tied to the user's last message and to the hash of the proposed draft.
- **SPADE facts go in cards** (explanations, plants, tunable parameters, draft values, results), from the catalog or the tools. The model does not paraphrase them, and its free text may not state anything that is not in the card.
- **The avatar never approves results:** the `confirmed` state happens only with the user's confirmation.
- **Benchmarks:** criteria are committed before the requests are written; a bench used for design is no longer an independent test.
- The catalog `configs/assistant_catalog.yaml` covers every page, plant, controller and job type; a coverage test requires it.

### Answering questions about SPADE

- Source of truth: `docs/VALIDATION_SUMMARY.md`, then the audit reports, then the current plan. On a contradiction the repo wins, and say so.
- If a question depends on code not in view, ask for the file or say what you would assume instead of inventing names or paths.
- When proposing improvements, order them by value and cost, mark those that need real data or hardware, and suggest the next concrete task.
