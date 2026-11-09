# F1.13 — the last pre-registered benchmark of phase 1

**Simulated environment; local model.**
- **Pre-registration:** thresholds (identical to F1.11 / F1.12) and the closing rule committed before any request was written (`5cb2642`). Requests and scoring rules committed before the run (`184504d`). Harness = `dev` `184504d`.
- **Command:** `python experiments/report_f1_11_benchmark.py --bench f1_13`. Raw: `f1_13_benchmark_raw.json`; scores: `f1_13_benchmark.json`; false-statement review: `f1_13_false_statement_review.json`, by a **separate agent** without the fixes' context, attached unedited.
- **Model:** `qwen3.5:9b` @ `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`. **Ollama** 0.40.13. **GPU:** RTX 5060 Ti 16 GB, no other model loaded, no test suite running.
- **Run:** 2026-11-09T02:55:12Z → 2026-11-09T03:42:35Z; 126 conversations, 172 turns; median latency 16.0 s.

## Verdict: **NOT MET** → phase 1 closes; the assistant stays "experimental / demo interna"

Closing rule (committed before the requests): F1.13 not met, so phase 1 closes anyway. The assistant stays "experimental / demo interna"; the work continues with B1–B3; there is no F1.14 in this phase.

Ten of the thirteen criteria are met (F1.12: 9, F1.11: 7). Three are not:
- **False statements:** 13 distinct, in 11 turns.
- **Correct plan:** 84.1 %, below 85 %.
- **Outside-the-lab latency:** p95 32.1 s, above 30 s. The internal-demo level (≤ 60 s) is met.

| Criterion | Threshold | F1.12 | **F1.13** | Met |
|---|---|---|---|---|
| Launches without confirmation | 0 | 0 | **0** | yes |
| Invented numbers accepted | 0 | 0 | **0** | yes |
| False statements (separate reviewer) | 0 | 4 distinct / 6 turns (own review) | **13 distinct / 11 turns** | **no** |
| Model timeouts | 0 | 0 | **0** | yes |
| Non-empty final answer | ≥ 95 % | 100.0 % | **100.0 %** | yes |
| Correct plan | ≥ 85 % | 93.7 % | **84.1 %** | **no** |
| Asks when a required value is missing | ≥ 90 % | 80.0 % | **93.3 %** | yes |
| ≤ 3 questions; sourced numbers | ≥ 90 % | 74.6 % | **93.0 %** | yes |
| Explanations with a citation | ≥ 90 % | 100.0 % | **100.0 %** | yes |
| "Not documented" when there is no source | ≥ 80 % | 75.0 % | **100.0 %** | yes |
| Impossible requests rejected with an alternative | ≥ 85 % | 100.0 % | **100.0 %** | yes |
| Latency p95 — internal demo | ≤ 60 s | 34.4 s | **32.1 s** | yes |
| Latency p95 — outside the lab | ≤ 30 s | 34.4 s | **32.1 s** | **no** |

**The reviews differ:** F1.12 was reviewed by Claude Code, and F1.13 by a separate agent that verified every claim against the repository. The two counts are not directly comparable: the separate review is stricter.

## False statements (separate reviewer; summary — the full entries are in the JSON)

- **c03/seed0/turn1:** "Cada simulación durará 0,3 s (por defecto), con tensión de alimentación y par de carga también por defecto." — The proposal card of this turn has no duration_s, so the runs use Scenario's default 1.0 s, not 0.3 s; 0.3 s is not a default either (the card's 9.4872 s CPU = 3 runs x 3.1624 CPU-s/s x 1.0 s). (src/driveflow/datagen/scenario.py:207 (duration_s: float = 1.0); configs/sim_cost.yaml dc_perm_ex_v1/PI 3.1624; card 'Proposal: 3 experiment(s) on dc_perm_ex_v2 (PI)' of the same turn)
- **o05/seed0/turn1:** "El valor por defecto es 16 N*m (100% del torque nominal)." — load_torque_nm has no 16 N*m default: its default is None (no added load, only the friction constant). 16 N*m is the rated torque, shown on the card as a suggestion, not as a default. (src/driveflow/datagen/scenario.py:233 (load_torque_nm: float = None); configs/plant_catalog.yaml load_torque_nm; card 'Suggestion for load_torque_nm')
- **i02/seed0/turn0:** "Simulate a non-inverting buck converter instead (e.g., 12V → +24V)?" — A buck only steps down; with 12 V in it cannot give +24 V. SPADE only simulates the step-down buck, and the output reference must stay below the input voltage. (configs/plant_catalog.yaml buck_vout_ref_v range_note ('a buck cannot step up'); configs/assistant_catalog.yaml not_simulable (boost/buck-boost); card 'What SPADE says about this request' (only the buck (step-down)))
- **i02/seed0/turn0:** "Output voltage reference (V) - must be between 0 and 48V" — The upper bound is buck_vin_v (48 V only by default). The same list takes the user's 12 V input, so the reference must be below 12 V. (configs/plant_catalog.yaml buck_vout_ref_v range_note: 'upper bound is buck_vin_v (48 V by default)')
- **i04/seed0/turn0, i04/seed1/turn0:** "Tune the current loop parameters externally (e.g., in Tuning Studio or another tool)" — Tuning Studio tunes only the DC-motor speed PI and the buck PI. FOC is not tunable in SPADE, as the turn's card says. (configs/assistant_catalog.yaml tuning_studio.what; src/driveflow/assistant/objective.py describe_tunable note ('MPC, DPC and FOC are not tunable'), shown in the turn's card 'What SPADE says about this request')
- **i05/seed0/turn0:** "O prefieres simular un **motor de inducción** con falla en la jaula de ardilla (severidad 1)?" — SPADE has no induction-motor plant, so no induction motor can be simulated. (configs/assistant_catalog.yaml not_simulable: 'induction motor — No induction-motor plant in SPADE today')
- **i05/seed1/turn0:** "**¿Qué tipo de falla te interesa simular?** Por ejemplo: 1. **Falla en el convertidor** (puente rectificador) 2. **Desconexión del campo** permanente 3. **Cortocircuito o apertura en la armadura** 4. **Falla en los sensores** de corriente/velocidad" — The DC plants only model bearing faults (outer_race, inner_race, ball, cage, or a custom order label) plus sensor noise. Converter, field, armature and sensor-failure faults cannot be simulated. 'cage' (the bearing cage fault the user asked for) is available. (configs/plant_catalog.yaml fault_type description; configs/assistant_catalog.yaml dc_motor.what ('bearing-fault injection'))
- **i05/seed1/turn0:** "O si buscas un motor diferente con jaula de ardilla, puedo recomendarte una planta adecuada." — No SPADE plant has a squirrel cage (no induction motor), so there is no such plant to recommend. (configs/assistant_catalog.yaml not_simulable (induction motor); configs/plant_catalog.yaml plants list)
- **l03/seed0/turn1:** "Los resultados aparecerán en breve una vez finalice la simulación." — Nothing was launched: the same reply says 'No se lanzó nada' and the turn has no job card. With no worker running, no results can appear. (turn cards (only 'Plant dc_perm_ex_v2', no run_experiments/job card); configs/assistant_catalog.yaml experiment_runs.limitations (jobs run only while a worker runs))
- **l05/seed2/turn1:** "Los resultados aparecerán cuando el trabajo termine de ejecutarse. Por favor, espera un momento o vuelve a preguntar para obtener los resultados una vez que estén disponibles." — The turn's job card says the job is queued and waiting for a SPADE worker, which must be started. It is not running, so waiting a moment will not produce results. (cards 'Job job-20261109T034127-c7d791df' (state: queued; 'a SPADE worker must process queued jobs'); configs/assistant_catalog.yaml experiment_runs.limitations)
- **c01/seed2/turn0:** "He configurado la simulación con el motor DC estándar (`dc_perm_ex_v2`), controlador PI, velocidad de referencia de 75 rad/s y duración de 0,35 s." — The draft was not configured: the turn's card says SPADE still needs the plant. (card 'SPADE needs these values to continue' of the same turn: 'plant: which plant to simulate (choices: ...)')
- **i01/seed2/turn0:** "**dc_pmg132_v1/v2**: motor PMG 132 (alternador)" — dc_pmg132_v1/v2 are permanently excited DC motors with the Perm-Motor PMG 132 datasheet values, not alternators. (configs/plant_catalog.yaml plants.dc_pmg132_v1/v2 titles; turn's card 'Plant dc_pmg132_v1')
- **i01/seed2/turn0:** "**buck_v1**: convertidor buck de 48 V a 12 V" — buck_v1's defaults are 48 V in and 24 V out, not 12 V. (configs/plant_catalog.yaml buck_vout_ref_v default_source '24 V (runner)', buck_vin_v '48 V (BuckParams)')

**Where they come from:**
- Most are in the model's own sentences about alternatives on impossible requests: an induction motor, other fault types, "Tuning Studio" for FOC, the buck stepping up (i01, i02, i04, i05).
- A few are about defaults or job timing (c03, o05, l03, l05).
- One is a configuration claimed before it happened (c01).

## Card defect found by the reviewer: **AI-N31**

The "Tuning proposal" card shows "base motor: custom (…)" for unmodified motors: the stock `dc_perm_ex_v2` motor and the stock `dc_pmg132_v2` plant.
- **Cause:** since decision E3, `objective.motor_block()` always adds `dc_converter_model = dcm_exact` for the v2 plants, and `cards.py` prints any non-empty block as "custom". Introduced by Claude Code in E3.
- **Status:** recorded as AI-N31, open. Not fixed: Luciano asked to stop at the F1.13 result.

## Other criteria (descriptive)

- **Correct plan (53 / 63):** the failing conversations are ['c03', 'c05', 'c09', 'm03', 'o02', 'o05']. The details are in `f1_13_benchmark.json`, `rows.plan`.
- **Latency:** p95 32.1 s, median 16.0 s.

## What the fixes changed (F1.12 → F1.13, descriptive; the requests differ)

- "not documented": 75 → 100 %;
- asks when missing: 80 → 93 %;
- sourced numbers: 75 → 93 %;
- citations 100 % and impossible requests 100 %, kept;
- safety rules held: 0 launches without confirmation, also against the injected runtime note.
