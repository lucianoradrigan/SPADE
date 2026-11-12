# `dev` stability verdict (2026-11-12): **stable at `e4fed56`** (revised; see §Revision)

Commit **`849721c`** (dev with M2, M3 and M4 integrated). Every result is **simulated**. The gates ran with nothing heavy in parallel (G1 09:22–09:43, then G4, G6 and G5 one after another). Raw: [`dev_stability_2026-11-12.json`](dev_stability_2026-11-12.json); screenshots and outputs in [`dev_stability_2026-11-12/`](dev_stability_2026-11-12/).

| # | Gate | Criterion | Result | Pass |
|---|---|---|---|---|
| G1 | Full suite | full suite on dev: 0 failures; no new xfail without a finding | 2009 passed, 1 skipped, 9 xfailed, 0 failures (20 min 59 s) | ✓ |
| G2 | Smoke --launch | smoke --launch all green | 7/7 | ✓ |
| G3 | CI on the last dev commit | CI on the last commit of dev 5/5 | run 43217225417 at 849721c: pytest 3.11 and 3.12, smoke Ubuntu, Windows, macOS: 5/5 success | ✓ |
| G4 | Fresh clone | fresh clone of dev from GitHub, new .venv, no data/ or workspace/, OLLAMA_HOST closed: smoke green; 18 pages open, each runs a simulation and leaves a screenshot; the assistant warns and does not crash | clone at 849721c, no data/ or workspace/, port 9 closed; smoke 7/7; 18/18 pages open without exception, 18 screenshots; an action ran on 9 pages (DC Motor, PMSM / FOC, VSC / DPC, DC-DC Converters, Tuning Studio, AI Diagnostics, Edge Deployment, Transfer Learning, Monitoring); 9 pages have no simulation by design (Overview, Experiment Runs, PLECS Plant Studio, RL Lab, Model Registry, Assistant, Alert History, Documentation, Algorithm Guide); the assistant shows 'Assistant unavailable ... http://127.0.0.1:9 ...' and survives a sent message; console: only 404 probes of Streamlit's frontend | ✓ |
| G5 | Assistant with real Ollama | assistant with real Ollama: the assistant smoke passes | pytest --run-llm -m llm: 1 passed (test_real_runtime_answers); end-to-end F1.10 demo with qwen3.5:9b (digest 56671c2a… as pinned): both conversations reach proposal -> Run -> job -> results with no exception (49 s and 61 s); output redirected to /mnt/datos/spade/m5/ so the committed f1_10_demo evidence is untouched; smoke 2c with the runtime up passes (G2) | ✓ |
| G6 | AI Diagnostics defaults | AI Diagnostics defaults: healthy -> normal at 80, 300 and max speed | defaults 0.82 s, window 0.6145-0.8192 s: normal 0.999 (80), 0.955 (300), 0.909 (350 = page max); faults at 300 rad/s: outer 0.95, inner 1.00, ball 1.00, cage 1.00 (identical to ai_n33.md) | ✓ |
| G7 | Byte-for-byte tests | byte-for-byte tests intact | tests/test_datagen.py, test_scml_system_regression.py, test_patch3_regression.py, test_r2_sensitivity_regression.py and tests/data unchanged since v0.2.0 (and df1950f); all pass in G1. datagen changed only in scenario.py (S-24) | ✓ |
| G8 | Absolute paths | no absolute paths outside logs | 3 matches of /home/, none a host path: scripts/fmu/om.sh (path inside the OpenModelica Docker container, twice) and two tests that check this rule (test_store.py, test_smoke.py) | ✓ |

## Verdict: **stable**

All eight gates are green. Two declarations:
- **G4:** 8 of the 18 pages have no simulation by design: Overview, Experiment Runs, PLECS Plant Studio, RL Lab, Model Registry, Assistant, Alert History, Documentation, Algorithm Guide. They open, render and leave a screenshot. The criterion 'each page runs a simulation' cannot apply to them.
- **G5** found **AI-N38** (low): a refused v1 plant request is not explained in the reply (`AI_TECHNICAL_AUDIT.md` §13.11). It is not a gate failure: the smoke and the end-to-end demo pass, and nothing false is stated. It is recorded, not fixed (assistant logic is out of scope this round).

## Commands

```bash
PYTHONPATH=src:. python -m pytest -q -rx                     # G1 (log /mnt/datos/spade/m5/g1_suite.txt)
python scripts/smoke_test.py --launch                         # G2
bash /mnt/datos/spade/m5/fresh_setup.sh && bash /mnt/datos/spade/m5/fresh_run.sh   # G4: clone of dev, uv venv, OLLAMA_HOST=127.0.0.1:9, Playwright over every page
pytest --run-llm -m llm; python experiments/report_f1_10_demo.py (OUT redirected)   # G5
python ai_n33_probe.py 0.82                                  # G6 (the probe of docs/audit/release/ai_n33.md)
grep -rn "/home/" --include=*.py --include=*.yaml --include=*.toml --include=*.sh   # G8
```

## Revision (12:25): AI-N39 found after the verdict → *not stable* at `849721c`/`09bd3bb`; fixed; gates re-run at `e4fed56`

- **Finding AI-N39 (High).** In a real browser with Ollama running, the Assistant page never finished a turn.
  - The avatar reported its state on every repaint, each report requested a rerun, and the rerun cancelled the turn after its first model call (813 runs in 3 min).
  - It was present in the published **v0.2.0** too. Evidence: [`../assistant/ai_n39_avatar_rerun.md`](../assistant/ai_n39_avatar_rerun.md).
- **Why the gates missed it:**
  - G5 drove the page through AppTest, which does not execute component JavaScript;
  - G4 used a real browser but with Ollama unreachable, so no turn started.
- **The first verdict ("stable") does not hold** for `849721c` and `09bd3bb`: those commits are **not stable**.
- **Fix:** `e4fed56` (small: the component reports its state only when asked; only the demo page asks). Merged by fast-forward.
- **G5 is extended:** it now also requires one real turn in a real browser.

**Gates re-run at `e4fed56`:**

| # | Result | Pass |
|---|---|---|
| G1 | 2019 passed, 1 skipped, 9 xfailed, 0 failures (same 9 xfails); run on the branch head that became `dev` | ✓ |
| G2 | smoke `--launch` 7/7 | ✓ |
| G3 | CI run 43250952418 at `e4fed56`: 5/5 success | ✓ |
| G4 | clean clone at `e4fed56` (no `data/` or `workspace/` at start, port 9 closed): smoke 7/7, 18/18 pages without exception | ✓ |
| G5 | `pytest --run-llm -m llm` 1 passed; **one real turn in Chrome with `qwen3.5:9b` in the clean clone: reply in 18 s** | ✓ |
| G6 | 0.999 / 0.955 / 0.909 normal at 80/300/350 rad/s; faults correct | ✓ |
| G7 | 0 diff lines in the byte-for-byte tests since v0.2.0; they pass in G1 | ✓ |
| G8 | the same 3 `/home/` matches, none a host path | ✓ |

**Final verdict: `dev` is stable at `e4fed56`** (all eight gates green after the AI-N39 fix, with G5 extended to a real browser turn).
