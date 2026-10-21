# T8 (E9): DAB — only the part that does not need PLECS

Round 2026-10-21, task T8 (decision D8). Branch `feat/E9-dab-prep` from `dev` (5288a76), worked in a parallel worktree. Machine: Linux x86-64, Python 3.12.13.

## Licence first

`DAB_sample1.plecs` was downloaded from Fund. to `data/upstream/fund/dab/` and is not committed.

- Fund.'s NOTICE licenses "code" (executable or functional components) under Apache 2.0 and "educational content" under CC BY-NC 4.0. A PLECS circuit fits neither definition clearly.
- Under D8, an unclear licence means: not committed, non-commercial use declared.
- `DATA.md` records the URL, size, SHA-256 and the reasoning. The licence does not forbid this use, so the task continued.

## What was built

| Item | Where | Notes |
|---|---|---|
| DAB profile and manifest for the E2 dataset route | `integrations/plecs/manifest.EXAMPLE_DAB`; template selector in PLECS Plant Studio → Import circuit | ModelVars declared by hand from the circuit's initialization commands: D1, D2, P, Vref and Ro (the notebook sets Ro = Vref²/P; Vin = 200 V, fs = 50 kHz, Lr = 50 µH and Cf = 160 µF are fixed in the file). **The circuit has no top-level outports** (it records with scopes), so `plecs.simulate` would return nothing over RPC. The template lists the outports to add (`v_out`, `i_Lr`), and the page says so. |
| DAB tab in DC-DC Converters | `viz/views/dcdc_converters.py::_dab_tab`, `viz/adapters/dab_adapter.py` | Strategy selector (SPS / DPS / EPS / TPS, which constrains D1 and D2) and a phase-shift diagram drawn from the chosen D0, D1, D2, labelled "not a simulation". The simulated-waveform and performance panels show "No simulation has been executed for this configuration." with the reason. The suggestion panel carries AI SUGGESTION — REQUIRES VERIFICATION and "No DAB performance model has been trained". "Compare suggested vs baseline" is disabled. |
| TPS modulation model in Keras, domain `dab`, YAML → `train_model.py` | `models/dab/{schemas,builder}.py`, `configs/dab/dab_tps_pc.yaml`, the `dab_performance` path in `experiments/train_model.py`, `ai/dab_modulation.py` | A performance regressor from (Vref, P, D1, D2) to (ipk2pk, irms), with the split grouped by operating point. `suggest()` grid-searches D1 and D2 for the lowest predicted objective next to the SPS baseline, and refuses operating points outside the training range. The idea comes from Fund. `one_stop_AI_DAB_modulation.ipynb` (only its CSV column names were read); no code or data was copied. |

## What is blocked

- **Generating the DAB dataset and training on real circuit data: not run (no PLECS).** No model is declared trained.
- The tests train only on a **synthetic** stress surface with a known minimum, to check the code path. That is not a DAB model.

## Tests

| File | Tests |
|---|---|
| `tests/dab/test_dab_model.py` | 5 (config guards, shipped config; slow: `train_model.py` on synthetic data, split by operating point, invalid rows dropped, the suggestion finds the synthetic optimum within 0.15, a point outside the training range is refused) |
| `tests/dab/test_dab_tab.py` | 8 (strategy constraints, waveform duty equals the inner shift, the tab's empty states and disabled comparison) |
| `tests/test_dcdc_page.py` | updated: the page now has the tabs Buck and DAB |

Full suite (`PYTHONPATH=src python -m pytest -q`): 1083 passed, 4 xfailed in 846 s — log in `t8_branch_pytest.txt`.
