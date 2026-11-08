# S21-3: S-21 warning in the assistant (non-blocking)

- **Decision:** S21-3, approved by Luciano 2026-11-02 (`proposed_agent/DECISIONES_S21_F1_2026-11-02.md`); ordered 2026-11-08.
- **Branch:** `feat/ASSIST-1-s21-warning`. Physics, defaults and runs unchanged.
- **Model:** `qwen3.5:9b` @ `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`. **Runtime:** Ollama 0.40.13. **GPU:** NVIDIA GeForce RTX 5060 Ti, 16 GB.

## What was built

- **Rule** (`src/driveflow/datagen/s21_warning.py`):
  - **Motors:** `dc_pmg132_v1`, and `dc_perm_ex_v1` with a changed motor whose rated torque and speed are known (the 0b-D1 friction, `runner.dc_default_load_parameter` not None). The stock motor with its stock friction is not covered (`s21_one_quadrant.md`). With `dc_converter_model="dcm"` there is no warning.
  - **Experiments:** those motors in speed control (no `torque_ref_nm`) with a speed reference below the **free speed** and a load below 1 N·m.
    - The free speed is defined [CC] as U_supply/ψ_e, the no-load speed at duty 1, so a speed overshoot asks the converter to brake.
    - The load is `load_torque_nm`; 1 N·m is the level that removed the effect in `s21_one_quadrant.md`.
- **`controller_design_check(..., s21_reason=...)`:** a new `warnings` list (`id` S-21, `blocking: false`, fixed message "one-quadrant converter: light-load deceleration not physical (S-21)", evidence path). `status` is unchanged, so the existing test that the PI design check of `dc_pmg132_v1` is `ok` still holds.
- **`describe_plant`:**
  - In the catalog: top-level `warnings`, the default motor's.
  - In the assistant tool: also the draft's changed motor when the draft is on that plant. The plant card shows it.
- **`validate_draft` / `Draft.status()`:** `warnings` per experiment; `ready` ignores them.
- **Proposals:**
  - `propose_experiments` returns `warnings`, merged per ID with the experiment indices.
  - `propose_tuning` returns the motor-level warning for `dc_pmg132_v1` and 0b-D1 motors. The speed-step conditions of the tuning and evaluation grids carry no load (`optim.objectives.grids_for`).
  - Both cards show a `WARNING S-21: … (not blocking; see …)` line.
- **Card fix found while recording:** an experiment with no field set showed an empty line in the proposal card. It now says "no field set: every value is the Scenario default".

## Tests

- `tests/assistant/test_assistant_s21_warning.py`, 17 tests: the rule (motors, experiments, free speed), the design check, both `describe_plant`s, `validate_draft` not blocking, both proposal cards.
- **Recorded conversation** (`tests/assistant/cassettes/s21_warning_pmg132.json`), replayed strictly, so a change of prompt or tools breaks the test on purpose.
- Assistant tests 288 passed, 1 skipped (the `llm` test); related plant tests unchanged and passing.

## The recorded conversation

`PYTHONPATH=src .venv/bin/python experiments/report_s21_warning.py` → `s21_3_warning.json`, 10 model exchanges.

1. **User:** "Simula el motor dc_pmg132_v1 con un controlador PI a 120 rad/s durante 0,5 s, sin carga."
   - The plant card carries the warning.
   - The model tried a 0 N·m load, which "sin carga" does not write as a number, so the no-invention rule refused it. The model then asked the user. Its explanation ("lo considera ambiguo") is inexact: the reason is that the user did not write the number.
2. **User (fixed fallback):** "Sí, usa esos valores y propónmelo."
   - Proposal: 1 experiment on `dc_pmg132_v1`. The card shows `WARNING S-21: … dc_pmg132_v1 …; experiments 0 (not blocking; …)`, and the model's text also mentions the warning.
   - 0 launches; 0 `audit_trace` problems.

**Observed and not changed (model behaviour, not S21-3):** the model did not record 120 rad/s and 0.5 s in the draft, so the proposal uses the defaults (150 rad/s, 1 s). That is still below the free speed (359 rad/s) at no load, and the warning applies. A wrong plan like this is what F1.11's "correct plan" criterion measures.
