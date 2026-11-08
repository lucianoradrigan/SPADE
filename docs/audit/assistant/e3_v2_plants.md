# Decision E3: v2 DC plants as the default of new simulations

- **Decision:** Luciano, 2026-11-08, approved with conditions after the variant-B evidence (`dc_perm_ex_v2_convergence.md`, `dc_perm_ex_v2_equivalence.md`).
- **Branch:** `feat/ASSIST-1-v2-plants`.
- **Simulated.**

## The four conditions

| Condition | How it is met |
|---|---|
| The `Scenario` dataclass default does not change | `Scenario.plant_config_id` still defaults to `dc_perm_ex_v1`. The v2 default is set where simulations start: the GUI (`viz/adapters/simulation_adapter.DC_PLANT` and the pages listed below), the assistant (`assistant/draft.ASSISTANT_PLANTS`, `objective.DC_PLANTS`) and the DC dataset scripts (`--plant`, default `dc_perm_ex_v2`). No byte-identity test was modified; `tests/test_datagen.py` and `tests/test_scml_system_regression.py` pass unchanged. |
| `dc_pmg132_v2` exists; the assistant and the GUI offer only v2 | Plants `dc_perm_ex_v2` and `dc_pmg132_v2` exist (`Scenario.DC_PLANTS`, `DC_V2_BASE`). The runner builds a v2 plant as its v1 plant with `dc_converter_model="dcm_exact"`, and the records keep the v2 id. The assistant refuses v1 ("frozen GEM-faithful plant … kept only to reproduce results"). The catalog marks v1 `assistant_can_drive: false`. |
| Wall time of a typical DC Motor page run, v1 vs v2 | `dc_motor_page_wall_time.md`: healthy, 300 rad/s, 1 s: **2.91 s vs 2.97 s (×1.02)**; outer_race: **3.05 s vs 3.13 s (×1.03)**. |
| S21-3 warning for v1 only; S-21 corrected in v2, open in the frozen v1 | `datagen/s21_warning.motor_reason` returns None for v2 at every level (catalog, design check, `validate_draft`, both proposal cards). The S-21 entry of `docs/SIMULATION_STABILITY_REPORT.md` now reads "Corrected in the v2 plants · open in the frozen v1 plants". |

## What the v2 plants are

- **Same as v1:** the motor, load, supply, limits, controllers (PI, MPC), fault and vibration models.
- **Converter:** `dc_converter_model` None means `"dcm_exact"`. `"dcm"` (substeps) is also accepted, as a study reference. `"gem"` is rejected: that is v1.
- **Not applicable on v2:** `dc_plant_solver` and `dc_plant_substeps`, because `dcm_exact` integrates each period in closed form.
- **Profiles:** `native:dc_perm_ex_v2` and `native:dc_pmg132_v2`, batch only, with no Gymnasium environment. `SPADE/DcMotor-v0` stays on v1.
- **Tuning:** the tuning bench (`validation.benches.build_dc_system`, `optim.objectives.BaseMotor.converter_model`) accepts `converter="dcm_exact"`. Tuning Studio and the assistant's objective mode tune on v2. The stock motor without the option makes exactly the old calls.
- **Datasets:** `DIAGNOSIS_PLANT_CONFIG_IDS` accepts `dc_perm_ex_v2` rows. The custom-training generator (GUI) produces v2.

## Entry points moved to v2

| Area | Entry points |
|---|---|
| GUI | DC Motor (single run and flow), Experiment Runs (DC sweep), Tuning Studio, Monitoring (DC), Transfer Learning (DC samples), AI Diagnostics (DC sample), training jobs' profile |
| Datasets | `generate_diagnosis_grid_dataset.py`, `generate_diagnosis_dataset.py`, `generate_first_dataset.py` (`--plant dc_perm_ex_v1` reproduces the datasets made before) |
| Assistant | plant recommendation, `describe_plant`, draft, objective mode |

**Not moved, declared:**
- the C2000 edge examples (`ai/c2000_export.py`), which reproduce a past export;
- `generate_ai_n27_heldout.py` and the evaluation-cache generator, which are pre-registered evaluation sets with their own explicit converter;
- the Gym environment.

## Existing tests changed (none byte-identity)

| Test | Change | Why |
|---|---|---|
| `tests/test_assistant_f0_plant_catalog.py` | `BASE` entries for the two v2 plants; v2 added to the list of plants whose controller default is PI | The coverage test iterates every catalog plant |
| `tests/test_pmg132_profile.py` | The two v2 profiles added to the expected set | New profiles |
| `tests/test_ai_n27_regen_generators.py::test_grid_cli_tags_every_job` | Also strips `plant_config_id` and checks it is `dc_perm_ex_v2` | The CLI default is v2 |
| `tests/test_tuning_studio_page.py::test_tuning_job_end_to_end` | Expected baseline evaluation-grid J **27.83 → 27.78** | Tuning Studio tunes on v2. The new `test_baseline_j_on_v1_and_v2` pins **27.83 on v1** (unchanged) and 27.78 on v2. |
| `tests/assistant/*` | Plant ids `_v1` → `_v2` | The assistant offers only v2 |
| `tests/assistant/test_assistant_s21_warning.py` | The v1 recorded-conversation test is removed; new tests check that v2 is never warned and v1 still is | The assistant now refuses v1, so the v1 conversation cannot be replayed. The cassette is kept as evidence in `docs/audit/assistant/s21_3_cassette_v1_pmg132.json`. |

## Observation

On the healthy case at 300 rad/s, the minimum current differs between v1 and v2 (84.0 A vs 80.6 A) although neither goes negative. v2 solves the armature equation exactly in each period, where v1 takes one Euler step per period. That is the discretization effect S-16 / B3 measured (2.6 % NRMS of i_a). It is not S-21.
