# Demo validation scenarios (for the demonstration video)

There are five reproducible experiments, A–E, that show what SPADE does, including where it fails. For each one this guide gives:
- the parameters;
- the GUI steps, plus an equivalent script so the result can be checked outside the GUI;
- the metrics to read;
- the **expected result**;
- the evidence to capture;
- the screen to show.

> **Expected results are test hypotheses, not experimental results.** Where a number is given, it comes from the offline measurement cited next to it, run on 2026-10-18 on Linux x86_64 with Python 3.11.15 and TF 2.21.0. Re-check every number on the recording machine before narrating it. If the GUI shows something else, record what it shows and say so.

Common setup: install as in `docs/INSTALL_*.md`, then run `streamlit run src/driveflow/viz/app.py`. Unless stated otherwise:
- all GUI runs use seed 0 and the default plant (`dc_perm_ex_v1`: 65 V supply, PolynomialStaticLoad a=0.01, b=0.05, J_load=0.0025);
- every value on screen is **simulated** (the sidebar reads "data source: simulation only").

---

## Scenario A — Motor operating conditions: healthy vs. bearing fault

**Goal.** Show how a bearing fault changes the time signals and the spectrum, and where the change appears in frequency.

| Parameter | Value |
|---|---|
| Page | **Simulation → DC Motor**, mode *Single simulation* |
| Controller | PI |
| Control mode | Speed (ω_ref), **150 rad/s** |
| Duration | 1.0 s |
| Run 1 | Fault condition **normal** |
| Run 2 | Fault condition **outer_race**; electrical severity 8 N·m, mechanical severity default (0.05) |
| Run 3 (optional) | **inner_race** (mechanical severity default 0.02) |

**Steps.**
1. Set the parameters for Run 1 and press **Generate**. Show the **Time signals** tab.
2. Open **Spectrum (FFT)**. Select *Armature current i_a*, then *Vibration acc_x*.
3. Repeat for Run 2 (and Run 3).

**Metrics.** Steady-state current level and ripple. FFT peak positions and amplitudes. Shaft frequency f_r = 150/(2π) = 23.87 Hz.

**Expected (hypothesis).**
- **Healthy.** The current settles at **45.5 A**, which balances the friction torque: (0.01 + 0.05·150)/0.165 = 45.515 A. This analytic value is checked by `tests/simulation/test_dc_motor_numerics.py`. The current spectrum has no fault line.
- **outer_race.** Current and vibration show a line at **BPFO = 73.3 Hz** (6203 geometry at 150 rad/s), plus its harmonics.
- **inner_race.** The line is at **BPFI = 117.7 Hz**.

The frequencies come from `bearing_frequencies.fault_frequencies_hz`. The existing test `tests/test_datagen.py::test_fault_frequency_visible_in_current_and_vibration_spectra` checks they are visible.

**Evidence to capture.** Screenshots of both FFTs with the fault line marked. The *Parameters*/lineage panel showing the configuration hash and seed. Caveat to state on screen: acc_y/acc_z use **assumed** gains, and only the x axis was calibrated against real data (AUDIT_ANSWERS Q29).

**Script equivalent.**
```bash
python -c "
from driveflow.datagen import Scenario, run_scenario
from driveflow.analysis import amplitude_spectrum, dominant_peak
import numpy as np
for ft in (None, 'outer_race', 'inner_race'):
    r = run_scenario(Scenario(scenario_id='A', fault_type=ft, duration_s=1.0))
    x = np.array([q['current_r'] for q in r])[5000:]
    f, a = amplitude_spectrum(x - x.mean(), 1e4)
    m = (f > 30) & (f < 400)
    print(ft, 'mean i = %.3f A' % x.mean(), 'dominant line in 30-400 Hz: %.1f Hz' % f[m][np.argmax(a[m])])
"
```

---

## Scenario B — Controller performance under identical conditions

**Goal.** Compare PI and MPC on the same plant, with the same reference and the same disturbance, using the metrics that fit a speed loop.

| Parameter | Value |
|---|---|
| Page | **Simulation → DC Motor** |
| Runs | PI and MPC, Speed mode, **150 rad/s**, 0.6 s, normal (healthy), seed 0 |
| Disturbance (script only) | +10 N·m load-torque step at t = 0.5 s |

**Steps.**
1. Generate with **Controller = PI**, then with **Controller = MPC**. Show the speed and current traces for each.
2. Run the battery for the numbers and figures:
   ```bash
   python experiments/run_stability_battery.py
   ```
   It writes `docs/audit/stability/results.md` and `.json`, plus `dc_load_step.png` and `dc_small_step_vs_linear.png`.

**Metrics.** Rise time (10–90 %), settling time (±2 %), overshoot, steady-state error, RMSE, max |e|, peak current. For the load step: maximum speed dip, recovery time into ±0.5 %, and residual offset.

**Expected (hypothesis; from `docs/audit/stability/results.md`).**
- **0 → 150 rad/s step.** Both controllers are **current-limited**:
  - PI reaches the physical rise-time bound of **0.1385 s** while holding the 168 A cap (0.8 × 210 A);
  - MPC is slightly faster and peaks near 179 A, since its current limit is a soft penalty at 0.85 × 210 A.
- **Steady-state error.** PI: about 0, thanks to integral action. MPC: a small residual offset of **~0.06 rad/s**, because it has no integrator or disturbance model.
- **Load step.** PI dips **~1.8 rad/s**, matching its linear design model at 1.78 rad/s, and fully recovers. MPC dips less but keeps a small offset.
- **Small 2 rad/s step.** The PI overshoots **~15 %**, which agrees with the linear design model. This is the PI zero, not a bug, but it contradicts the old "no overshoot" docstring (finding S-04).

**Evidence to capture.** Side-by-side speed plots, the metrics table from `results.md`, and `dc_load_step.png`.

**Screen.** DC Motor → *Time signals* for each controller, then the figure files.

---

## Scenario C — AI fault diagnosis, including the model's errors

**Goal.** Contrast the true condition with the AI suggestion, and show that the model can be wrong. Show an error case on purpose.

| Parameter | Value |
|---|---|
| Page | **Simulation → DC Motor** (tab **AI diagnosis**) and **Intelligence → AI Diagnostics** |
| Runs | 150 rad/s, 1.0 s, seed 0: normal, outer_race, inner_race |
| Model | the promoted `dc_motor/pc/classifier` (`pc_server/2026-09-01_run01`) |

**Steps.**
1. Generate each condition. Open **AI diagnosis** and note the predicted class and its (uncalibrated) confidence.
2. Open **AI Diagnostics**. Show the audit-finding banner attached to the promoted model.
3. Run the full offline evaluation. It also covers the not-promoted candidate `2026-10-13_run01`:
   ```bash
   python experiments/report_classifier_evaluation.py     # ~2 min with 8 workers
   ```
   It writes `docs/audit/ai/classifier_evaluation.md`.

**Metrics.**
- Per-class recall, balanced accuracy, macro-F1, and the confusion matrix.
- False-alarm rate (healthy reported as faulty) and missed-detection rate (faulty reported as healthy).
- ECE (calibration), and confidence on correct vs. incorrect predictions.

**Expected (hypothesis; from `docs/audit/ai/classifier_evaluation.md`).**
- **Promoted model.** normal → normal, outer_race → outer_race, but **inner_race → outer_race**. Its inner_race recall is **0.00** (finding AUDIT_ANSWERS Q15 / AI-N06). Confidences are around 0.5.
- **Candidate at the training point.** 100 % on fresh runs. It still fails when conditions change:
  - **100 rad/s:** inner_race is predicted as outer_race with confidence **≈1.0** (AI-N07);
  - **half fault severity:** inner_race is reported as **normal**, a missed detection;
  - **unseen fault type (ball):** reported as outer_race with confidence 0.99.
- **Message to state:** the confidence is uncalibrated and cannot flag unknown conditions.

**Evidence to capture.** The AI-diagnosis panel showing the wrong inner_race prediction, the evaluation table, and the confusion matrix.

**Screen.** DC Motor → *AI diagnosis*, then AI Diagnostics.

---

## Scenario D — Monitoring and alerts: detection, acknowledgement, resolution

**Goal.** Introduce an abnormal condition, then show its detection, logging, acknowledgement (seen) and resolution (gone) as three separate events.

| Parameter | Value |
|---|---|
| Page | **Operations → Monitoring**, domain **VSC / DPC** |
| Abnormal condition | Load resistance **R = 2.0 Ω** via *Custom value*. This is below R* ≈ 3.37 Ω, where the plant itself is open-loop unstable. |
| Normal condition | R = 8.0064 Ω (the default) |
| Rule | `dpc_load_resistance_divergence`: R < 3.3707 Ω, severity high, 2 s hysteresis (`monitoring/rules/vsc_dpc.yaml`) |

**Steps.**
1. Set R = 2.0 Ω and press **Run monitoring**.
   - **Expected:** no rule alert yet. In the GUI the rule's 2 s hysteresis is measured in wall-clock time between clicks (finding AI-N02).
2. Wait at least 2 s and press **Run monitoring** again.
   - **Expected:** the rule fires and the domain shows **CRITICAL**.
3. Press **Acknowledge** on the active alert.
   - **Expected:** it shows "acknowledged by operator", and the domain **stays CRITICAL**, because acknowledgement is not resolution.
4. Set R = 8.0064 Ω and run monitoring again.
   - **Expected:** the alert is resolved, and the domain returns to HEALTHY (if no other agent alert is active).
5. Open **Operations → Alert History** and filter *Event* by fired, acknowledged and resolved.
   - **Expected:** the three events appear in order.

**Metrics.** Number of active alerts per stage, and the domain health per stage: UNKNOWN → CRITICAL → CRITICAL (acknowledged) → HEALTHY.

**Script equivalent (deterministic, simulation clock).**
```bash
python -c "
from driveflow.monitoring.rules.schema import load_ruleset
from driveflow.monitoring.agents.agent_gateway import GatewayAgent
from driveflow.monitoring.agents.agent_server import ServerAgent
g = GatewayAgent(load_ruleset('src/driveflow/monitoring/rules/vsc_dpc.yaml')); s = ServerAgent()
for t, r in [(0, 2.0), (2, 2.0), (3, 8.0064)]:
    for e in g.step({'load_resistance_ohm': r}, t): s.record_alert('vsc_dpc', e)
    if t == 2: s.acknowledge('vsc_dpc', 'dpc_load_resistance_divergence', by='operator'); print('after ack:', s.domain_health('vsc_dpc'))
    print(t, r, s.domain_health('vsc_dpc'))
print([(e['action'], e['resolved']) for e in s.alert_history()])
"
```

**Evidence to capture.** Screenshots at each stage, plus the Alert History table.

**Note to state.** All monitoring telemetry is SIMULATED; no device is connected.

---

## Scenario E — Model validation and acceptance criteria

**Goal.** Show how a candidate model is evaluated and why it is blocked or approved.

| Parameter | Value |
|---|---|
| Page | **Intelligence → Transfer Learning**, then **Model Registry** |
| Base model | dc_motor PC classifier |
| Data | *Generate a sample run*, at least 3 acquisitions per class, 1 s each |
| Thresholds | the defaults: max macro-F1 drop vs. base 0.02. They are labelled *REQUIRE RESEARCHER CONFIRMATION*. |

**Steps.**
1. **Prepare dataset.** Show the split manifest. Splits are grouped by acquisition, and no window is shared between splits.
2. **Train candidate** with a short run, e.g. 2 epochs.
3. Show the gate report. Every required item must be PASS before promotion is allowed. Missing evidence gives INCOMPLETE, which blocks promotion.
4. Show **Model Registry → Metrics / Lineage / Deployment** for the promoted model. Deployment shows *hardware verification: none recorded*.
5. Offline, show the extended evaluation from Scenario C (`docs/audit/ai/classifier_evaluation.md`): the acceptance metrics beyond accuracy.

**Metrics.** Macro-F1 of base vs. candidate on the same test windows, per-class recall, convergence check, and base-weights fingerprint unchanged.

**Expected (hypothesis).**
- A 2-epoch candidate is usually **BLOCKED**, either by convergence or by the macro-F1 criterion (observed in the earlier audit, `AUDIT_ANSWERS.md` Q13).
- Promotion without approval is refused by the backend (`registry.promote` re-checks the persisted evidence).

**Evidence to capture.** The gate report showing the blocking reasons, and the lineage panel.

**Screen.** Transfer Learning (steps *Prepare → Train → Validate → Promote*), then Model Registry.

---

## What not to claim in the video

- No real (measured) data, and no hardware execution. The `.tflite` models were only **exported** and checked for parity on the host CPU.
- No calibrated probabilities, and no out-of-distribution abstention.
- The classifiers were trained at a single speed and severity. Their accuracy does not transfer to other operating points (Scenario C).
- The DPC tracks well only near its training load, R = 8 Ω (`docs/SIMULATION_STABILITY_REPORT.md`, S-05).
