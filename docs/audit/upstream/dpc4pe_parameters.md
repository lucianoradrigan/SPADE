# DPC4PowerElectronics: parameters in the original Simulink models vs. SPADE

TODO_CLAUDE_CODE.md V1. Date: 2026-10-20.

## Source and how the values were obtained

**Source.** https://github.com/aipoweraau/DPC4PowerElectronics, commit `cebbe8683a33bf4daca102502bcdf470070dd78f` (MIT licence). It was cloned into `data/upstream/dpc4pe/`, which is not committed.

**Command.**

```bash
python experiments/extract_dpc4pe_slx_params.py --upstream data/upstream/dpc4pe \
    --out docs/audit/upstream/dpc4pe_parameters.json
```

**What the script reads.** Each `.slx` is opened as a zip archive. The script reads:
- the model `InitFcn` callback (`simulink/blockdiagram.xml`);
- every block's dialog parameters (`simulink/systems/system_*.xml`), including whether the block or an ancestor is commented out;
- the source of every MATLAB Function block (`simulink/stateflow/chart_*.xml`);
- the solver settings (`simulink/configSet0.xml`);
- the load and reference columns of `Data4train.mat`.

MATLAB was not used. The full machine-readable output, with the SHA-256 of every upstream file, is in [`dpc4pe_parameters.json`](dpc4pe_parameters.json).

**Signal tracing.** The wiring of the active controller (Saturation blocks and Delay feedback) was traced by hand from the `<Line>` entries of `CCS_MPC.slx` `simulink/systems/system_136.xml` (the `PRN_controller1` subsystem). Because it is a manual step, the trace is recorded here and not in the JSON.

**Limits of the extraction.**
- `DPCvalidation.slx` has **no `InitFcn`**. Its workspace variables (`Rload`, `Lload`, `Vdc`, `Ts`, `Vref`, `Td`, `Rl`, …) are defined nowhere in the repository. They are reported as **not found**, not assumed equal to `CCS_MPC.slx`.
- The trained network the validation model loads is `trainedNet1nDPCv3.mat`, at a local Windows path (`C:\Users\DU47TW\...`). It is not in the repository.

## Parameter table

| Parameter | Value in the upstream model | Where (file → block) | SPADE value | Match? |
|---|---|---|---|---|
| Filter topology | One three-phase **RL** branch (`inductor`) and one three-phase **C** branch (`capacitor`) inside the subsystem `Inverter_LC_filter1` | `CCS_MPC.slx`, `system_2.xml` → `inductor`, `capacitor`; same in `DPCvalidation.slx` | 2-state L-C model (`sim/vsc_params.py`) | **Yes.** It is an LC filter, not LCL. V0 corrected the 7 "LCL" mentions. |
| Filter inductance L | `L = 5.02e-3` (+ `errorL = 0`) | `CCS_MPC.slx` InitFcn; `inductor.Inductance = L+errorL` | 5.02 mH | **Yes** |
| Filter resistance R_f | `Rl = 2` in `CCS_MPC.slx`; **`0`** in `DPCvalidation.slx` | `CCS_MPC.slx` InitFcn and `inductor.Resistance = Rl`; `DPCvalidation.slx` `inductor.Resistance = 0` | 2.0 Ω | Yes for the data-collection model. **No** for the validation model → **S-12** |
| Filter capacitance C | `C = 15e-6` (+ `errorC = 0`) | `CCS_MPC.slx` InitFcn; `capacitor.Capacitance = C+errorC` | 15 µF | **Yes** |
| Controller sample period Ts | `Ts = 1e-4` | `CCS_MPC.slx` InitFcn; `PRN_controller1.SystemSampleTime = Ts` | `TAU = 1e-4` | **Yes.** Now stated directly, beyond the V0 identification. |
| Plant solver | Fixed-step discrete, step 1e-7 s, stop time 1.1 s (switched model) | `CCS_MPC.slx` `configSet0.xml` | Averaged discrete model ADF/BDF, one step per Ts | **Modelling difference** (see below) |
| PWM | SVPWM, `Tpwm = 1e-4` (10 kHz) | `PRN_controller1/SVPWM Controller4/Subsystem` → `Tpwm` | none; v_o is applied as an average | Modelling difference |
| Dead time | `Td = 4e-6` | `CCS_MPC.slx` InitFcn; `DPCvalidation.slx` `Subsystem1/On Delay*` (`delay = Td`, value not found there) | none | Modelling difference |
| DC-link voltage V_dc | `Vdc = 200` | `CCS_MPC.slx` InitFcn; `Inverter_LC_filter1/DC.Amplitude = Vdc` | not modelled | **No** → **S-09** |
| Converter voltage limit | Active MPC output clamped to **±134 V per α/β axis** (≈ 2·Vdc/3 = 133 V) | `PRN_controller1/Saturation9` and `Saturation6` (active), fed by `MATLAB Function4` output 1 via `Demux26` | **none**: the DPC output is never clipped | **No** → **S-09** |
| Validation clamp | ±467 V per axis on the DPC output | `DPCvalidation.slx` `PRN_controller1/Saturation2`, `Saturation4` | none | No (wider than V_dc allows, so inactive in practice) |
| Reference frequency | `wref = 2*pi*50` | InitFcn; MPC source | 2π·50 rad/s | **Yes** |
| Reference magnitude | `Vref = 50`, `Vref1 = 50`; the Step at 0.1 s switches between them | InitFcn; `Voltage reference1/Voltage_reference_creator/Step` | default 50 V; dashboard 10–150 V | Default yes. Range: **see S-09** |
| Loads (data collection) | `Load` R = 10 Ω, always connected. `Linear load3` R = 40 Ω behind an externally switched breaker (Step at 0.6 s). | `CCS_MPC.slx` `Linear Load1/*` | 8.0064 Ω nominal | **Yes:** 10 Ω ∥ 40 Ω = 8 Ω. `Data4train`'s 8.0064 Ω is this parallel value, measured as \|v\|/\|i\|. |
| Loads in `Data4train.mat` | Exactly two values: **8.0064 Ω** (5000 rows) and **39.9984 Ω** (5000 rows) | `Data4train.mat` column 7 | 8.0064 Ω (`datagen/runner.py` `_VSC_R_OHM`) | See **S-11** |
| Loads (validation) | `Linear load` R = `Rload` (not found), `Linear load1` R = 120 Ω, `Linear load2` R = `Rload`, breakers switched at 0.04 s and 0.07 s | `DPCvalidation.slx` `Linear Load1/*` | — | Not comparable (`Rload` not found) |
| Identified model ADF/BDF | `Adf`, `Bdf` literals | `CCS_MPC.slx` `MATLAB Function4` (`chart_361.xml`) | `control/dpc/loss.py` | **Yes.** Max abs difference 4.3e-8, the float32 rounding of SPADE's constants. |
| Data-collection controller | Unconstrained condensed MPC: horizon **N = 5**, weight **Q = diag(2, 1)** on [i_f, v_c] per axis, closed-form solution `U = (-H \ f')/2`, one-step delay compensation, references extrapolated with `3x_k − 3x_{k−1} + x_{k−2}`. The only constraint is the downstream ±134 V clamp. | `CCS_MPC.slx` `PRN_controller1/MATLAB Function4` (the only active controller; `MATLAB Function`, `Function1`, `Function2` and all `DPC-CCS-MPC*` / `PRN_step*` networks are commented out) | none (V2 ports it) | → **S-10** |
| Training-loss horizon and weights | Horizon 5; voltage and current weights 1.0 / 1.0 | `Custom_Loss.txt` | `HORIZON = 5`; weights 1 / 1 (`control/dpc/loss.py`) | **Yes** (already pinned by `tests/test_dpc_loss.py`) |

### Modelling difference: switched vs. averaged plant

The upstream models simulate a switched three-phase inverter:
- six IGBT/diodes (R_on = 1 mΩ);
- SVPWM at 10 kHz;
- a 4 µs dead time (validation model);
- a 1e-7 s fixed step.

SPADE simulates the identified averaged model ADF/BDF: one update per 100 µs, with v_o applied as its average. This is the model the DPC loss itself uses, so SPADE is internally consistent. Its consequences:
- SPADE results contain no switching ripple and no dead-time distortion;
- the absolute RMSE and THD numbers are therefore not comparable to the paper's Simulink figures.

## Findings

### S-09: no converter voltage limit in SPADE (open)

**Upstream.** V_dc = 200 V, and the active MPC output is clamped at ±134 V per axis. 134 V is the SVPWM hexagon vertex, 2·V_dc/3 = 133 V; the linear-modulation circle is V_dc/√3 = 115 V.

**SPADE.** It never limits v_o. The voltage the converter needs in steady state follows from the L-C model:

```
i_f = v_c/R + jωC·v_c
v_o = v_c + (R_f + jωL)·i_f
```

| R (Ω) | \|v_ref\| = 50 V | 120 V | 150 V |
|---|---|---|---|
| 4 | 77 V | **186 V** | **232 V** |
| 8.0064 | 63 V | **151 V** | **189 V** |
| 20 | 55 V | 132 V | **164 V** |
| 1000 | 50 V | 119 V | **149 V** |

Bold marks values above the 134 V clamp.

**Consequence.**
- **At 150 V**, every load needs more than the upstream converter can deliver.
- **At 120 V**, loads of 8 Ω and below need more.
- **Which SPADE results this concerns:**
  - the dashboard's |v_ref| range of 10–150 V;
  - the |v_ref| = 120 V and 150 V points of `tests/test_dpc_robustness_grid.py`;
  - the |v_ref| = 120 V and 150 V points of `MAGNITUDES_EVAL` in `experiments/finetune_dpc_r_range.py`.
- **Why they are not comparable:** these points describe an idealized converter with unlimited voltage, not the upstream hardware.

**Status.** Documented here. No limit is added and no default is changed without the user's confirmation; this is a modelling decision. V2 runs the MPC both with and without the clamp.

### S-10: the data-collection controller is a 5-step MPC, not a one-step explicit MPC (open, documentation)

**What upstream says.** The upstream README says the DPC "learns from a single-prediction horizon explicit model predictive control algorithm".

**What the active block does.** It solves an unconstrained 5-step condensed QP in closed form, with Q = diag(2, 1). The commented-out `MATLAB Function` is the only single-step controller: an exhaustive 100 V-grid search over ±467 V, with different ADF/BDF.

**Data4train columns.** The training data come from `MATLAB Function4`'s second output, which is logged by `To Workspace5`. That output holds:
- the **delay-compensated one-step prediction** of the state (`iftd`, `vcd`), not the raw measurement;
- the references **3 to 7 steps ahead**, `v_ref3x`…`v_ref7x`, built by quadratic extrapolation.

**Status.** V2 can therefore port the original controller rather than reconstruct it. V3 checks whether SPADE's closed loop feeds the network the same quantities. Today SPADE feeds the measured state and references 0 to 4 steps ahead (`control/dpc/controller.py`).

### S-11: the base DPC was trained mostly at 40 Ω (informational)

**How the data split.** Main.txt trains on rows 1–6000, which hold 5000 rows at 39.9984 Ω and 1000 rows at 8.0064 Ω. The holdout, rows 6001–10000, is entirely 8.0064 Ω.

**Is SPADE's comment wrong?** No. SPADE's comment that 8.0064 Ω is "the constant R found in Data4train.mat's holdout split" is correct for the holdout.

**What it changes.** The base network v1 saw mostly 40 Ω. This is relevant to C1: the nominal-point accuracy SPADE wants at 8 Ω comes from the v2/v3 closed-loop fine-tunes, not from the original training.

**Reference magnitude.** Every `Data4train` row has |v_ref| = 50 V.

### S-12: the validation model's filter differs from the identified model (informational)

**The difference.** In `DPCvalidation.slx`, the inductor branch has R = 0 Ω. In `CCS_MPC.slx` and in the identified ADF/BDF it is R = 2 Ω.

**Missing workspace.** The validation model has no initialization script, so its remaining parameters cannot be recovered from the repository.

**What it changes.** The upstream closed-loop validation therefore ran on a slightly different plant from the one the network's loss assumes. SPADE validates on the identified model itself.

## Not found in the repository

- **Workspace of `DPCvalidation.slx`:** `Rload`, `Lload`, `Vdc`, `Ts`, `Vref`, `Td`, `Rl`.
- **`Lload` in `CCS_MPC.slx`:** it is used by `Linear load3`, but that block is a pure R branch, so the inductance is unused.
- **`trainedNet1nDPCv3.mat`:** the trained network loaded by `DPCvalidation.slx/PRN_controller1/Predict1` (see V3).
- **Commented-out arrays in the InitFcn:** `CCS_MPC.slx`'s InitFcn contains commented-out arrays `weights_fc1..3` and `bias_fc1..3` (15 → 16 → 16 → 10). Their provenance and date are not stated (see V3).

