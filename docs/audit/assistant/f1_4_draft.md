# F1.4 — the assistant's draft with provenance

**No language model involved:** the patches below are written by hand, as a model would send them. Simulations are not run here (only `Scenario` validation).

- **Command:** `python experiments/report_f1_4_draft.py`. Raw values: [`f1_4_draft.json`](f1_4_draft.json). Tests: `tests/assistant/test_assistant_draft.py`, `tests/assistant/test_assistant_numbers.py`.
- **Code:** `src/driveflow/assistant/draft.py` (draft, ledger of evidence, atomic patches, `status()`, `expand()`, `hash()`, `audit_trace()`), `src/driveflow/assistant/numbers.py` (quantities in the user's text, unit table, comparison rule).

## 1. A configuration request

> Simula el motor DC de stock con PI a 1500 rpm, 2 s por corrida, con carga del 20 al 100 % cada 20 %.

| Field | Value | Unit | Origin | Source |
|---|---|---|---|---|
| `omega_ref_rad_s` | 1500 | rpm | user | user message 1: '1500 rpm' |
| `duration_s` | 2 | s | user | user message 1: '2 s' |
| `load_torque_nm` (varied) | 20.0 | % | user | user message 1: '20' |
| `load_torque_nm` (varied) | 40.0 | % | derived | range 20.0..100.0 step 20.0 |
| `load_torque_nm` (varied) | 60.0 | % | derived | range 20.0..100.0 step 20.0 |
| `load_torque_nm` (varied) | 80.0 | % | derived | range 20.0..100.0 step 20.0 |
| `load_torque_nm` (varied) | 100.0 | % | user | user message 1: '100 %' |

Load torque SPADE derives from the percentages: 3.2 N·m (20.0 % x 16 N*m), 6.4 N·m (40.0 % x 16 N*m), 9.6 N·m (60.0 % x 16 N*m), 12.8 N·m (80.0 % x 16 N*m), 16 N·m (100.0 % x 16 N*m); rated torque from: rated torque of the dc_perm_ex_v1 stock motor (default: describe_plant).
Status: ready **True**, 5 experiments, 10 simulated s, about 32 CPU s (configs/sim_cost.yaml dc_perm_ex_v1/PI, measured 2026-11-02T14:33:54-03:00 on one machine). Important fields left at their default (reported, not asked): motor_parameter_overrides, dc_supply_v, dc_limit_overrides.

## 2. Comparison rule (C3): explicit cases

| User wrote | Patch value | Expected | Result |
|---|---|---|---|
| a 3000 rpm | 3000 rpm → `omega_ref_rad_s` | accept | ok |
| a 3000 rpm | 314.159 rad/s → `omega_ref_rad_s` | accept | ok |
| a 3000 rpm | 314 rad/s → `omega_ref_rad_s` | refuse | ok |
| a 3000 rpm | 314.16 rad/s → `omega_ref_rad_s` | refuse | ok |
| a 300 rpm | 300 rad/s → `omega_ref_rad_s` | refuse | ok |
| a 1.500 rpm | 1500 rpm → `omega_ref_rad_s` | refuse | ok |
| a 1.500 rpm | 1.5 rpm → `omega_ref_rad_s` | refuse | ok |
| a 1.500,5 rpm | 1500.5 rpm → `omega_ref_rad_s` | accept | ok |
| alimentación de 220 V | 220 V → `dc_supply_v` | accept | ok |
| alimentación de 220 V | 230 V → `dc_supply_v` | refuse | ok |
| tres segundos | 3 s → `duration_s` | accept | ok |
| un motor | 1 s → `duration_s` | refuse | ok |
| del 20 al 100 % de carga | 60 % → `load_torque_nm` | refuse | ok |
| del 20 al 100 % de carga | 20 % → `load_torque_nm` | accept | ok |

14/14 cases behave as specified. Rules: SI conversion with one unit table, relative tolerance 1e-9, decimal comma or point, a lone separator before three digits ("1.500") is ambiguous and never a user value, "un"/"una" are articles, a range yields only its endpoints (intermediate values need the user's step and are then `derived`), Hz never converts to rad/s.

## 3. Untouched fields

A draft with only plant and controller expands to exactly `Scenario('assistant_000', controller_type='PI', plant_config_id='dc_perm_ex_v1')`: **True**. Defaults accepted with `use_defaults` are shown with origin `default` but not passed to `Scenario`, so the run stays the untouched one (passing the stock motor's values as overrides would take another code path, e.g. the 0b-D1 load rule).

## Not executed

- The no-invention check over recorded cassettes: no ejecutado todavía (no cassettes exist; they arrive with F1.2). `audit_trace()` and its test are in place and will run on every cassette.
- Origin `datasheet` (phase 2) and fault fields (phase 4) are refused by design.
