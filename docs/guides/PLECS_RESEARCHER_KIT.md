# PLECS researcher kit

This kit lets you run SPADE's "route a" on your own computer, where PLECS is installed:

> your PLECS circuit → a small, declared sweep → a dataset → a fast surrogate plant (a Gymnasium environment) → a comparison with the PLECS runs it never saw

You run **one command** and send back **one zip file**. We ingest it on our side without needing to
be present. Nothing in this kit changes your circuit file: it only reads it and hashes it.

---

## 1. Requirements

| Item | Version / setting |
|---|---|
| PLECS Standalone | Any version with the XML-RPC interface. Enable it in **Preferences → General → RPC interface**, port **1080**. The RPC server is local and disabled by default. |
| Python | **3.11 or 3.12**, 64-bit |
| OS | Windows 10/11 or Linux (x86_64) |
| Disk | ~3 GB for SPADE and TensorFlow, plus a few MB per sweep |

Keep PLECS **open** while the kit runs: the kit drives it over RPC.

## 2. Install SPADE

These are the README's steps. Run them in order.

### Windows (PowerShell)

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
# close and reopen PowerShell
mkdir C:\dev -Force; cd C:\dev
git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
python scripts\smoke_test.py          # every line [PASS]
```

- If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- If `pip install` fails with `WinError 206`, enable long paths (README, Windows step 2) and reboot.

### Linux (Ubuntu)

```bash
sudo apt update && sudo apt install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
git clone https://github.com/lucianoradrigan/SPADE.git && cd SPADE
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,viz]"
python scripts/smoke_test.py          # every line [PASS]
```

## 3. Prepare your circuit (Buck)

The surrogate learns how the **duty cycle** drives the converter. A sweep of constant parameters
only gives step responses, so the duty must **move** during each run, following a profile that SPADE
can reproduce exactly. Run the buck **open loop**: no voltage controller in the circuit.

### 3.1 ModelVars

Declare these in **Simulation → Simulation parameters → Initialization**, with any values; the kit
overrides them run by run:

```
Vin = 48; Rload = 6;
prbs_seed = 5; D_min = 0.1; D_max = 0.8; T_hold = 1e-3;
```

Use `Vin` for the input source and `Rload` for the load resistor.

### 3.2 The duty profile (C-Script)

Add a **C-Script** block with no input and one output (the duty), discrete sample time `T_hold`.
Give it parameters `prbs_seed, D_min, D_max` and feed its output to your PWM modulator, or to the
averaged switch.

```c
/* Code declarations */
static unsigned s;

/* Start function code */
s = (unsigned) ParamRealData(0, 0);       /* prbs_seed, 1..127 */

/* Output function code (discrete, sample time = T_hold) */
unsigned bit = ((s >> 6) ^ (s >> 5)) & 1u;
s = ((s << 1) | bit) & 0x7Fu;
Output(0) = bit ? ParamRealData(2, 0) : ParamRealData(1, 0);   /* D_max : D_min */
```

The macro names (`ParamRealData`, `Output`) are those of the PLECS C-Script block. This code was **not run in PLECS** by SPADE's authors, because no PLECS licence was available. If your PLECS version names them differently, keep the arithmetic and tell us.

This is PRBS7 (x⁷ + x⁶ + 1), the same as SPADE's reference in
`src/driveflow/integrations/plecs/excitation.py`. SPADE compares the duty your circuit actually
applied with that reference, and reports any mismatch in the evidence.

### 3.3 Outports

Add three **top-level Outport** blocks, in this exact order:

| Port | Signal | Unit |
|---|---|---|
| 1 | `v_out`, the output voltage | V |
| 2 | `i_L`, the inductor current | A |
| 3 | `duty`, the duty actually applied (the C-Script output) | – |

PLECS returns one signal per outport, in port order. If the number of signals differs, the run is
rejected; nothing is guessed.

### 3.4 Template

Copy `configs/plecs/manifest_templates/buck_example.yaml`.
- Edit the ranges if your circuit needs others. Keep `sample_s: 0.00002`: the surrogate steps every 20 µs.
- The declared sweep is 3 × 3 operating points × 2 PRBS seeds = 18 runs of 20 ms. On a normal PC it takes a few minutes.

## 4. Run the kit

With PLECS open and its RPC interface enabled:

```bash
python experiments/plecs_researcher_run.py --circuit path/to/my_buck.plecs \
    --manifest configs/plecs/manifest_templates/buck_example.yaml --out results/
```

The kit:
1. tests the RPC connection and records whatever version PLECS reports (PLECS has no documented RPC
   version command, so it may say "not reported");
2. computes the SHA-256 of your circuit;
3. runs the 18-run sweep as a job;
4. fits the surrogate and checks the environment;
5. compares the surrogate with the PLECS runs it was not fitted on;
6. writes `results/plecs_evidence_<UTC>.zip`.

## 5. What to send back

**Only** the file `results/plecs_evidence_<UTC>.zip`. It contains:

| File | Content |
|---|---|
| `evidence.json` | versions (SPADE, Python, OS, PLECS if reported), circuit SHA-256, every step's result, and the failure cause if any |
| `log.txt` | the run log |
| `template.yaml` | the template you used |
| `dataset/` | one Parquet file per completed run, `conditions.csv` with every condition's status and error, `manifest.json` |
| `surrogate.json` | the fitted surrogate and its per-run errors (buck only) |
| `hashes.json` | the SHA-256 of every file, checked on ingestion |

Your `.plecs` file is **not** included, only its hash. If a step fails, the zip is written anyway,
with the cause: send it too.

## 6. Optional: the DAB (completes task E9)

- **Source:** `DAB_sample1.plecs` from Xinze Li's repository *Fundamentals_of_AI_for_PE*
  (`8_PE_Simulation_Automation/PlecsAutomation/`).
- **Licence:** its content is **CC BY-NC 4.0** (non-commercial). SPADE does not distribute it: use
  your own downloaded copy.
- **Changes needed:** the file records with scopes and has **no top-level outports**. Add two, in this
  order: `v_out` (V) and `i_Lr`, the leakage-inductor current (A).
- **ModelVars:** `Vin, Vref, P, D1, D2, Ro`, where the kit sets `Ro = Vref^2 / P`.

```bash
python experiments/plecs_researcher_run.py --circuit DAB_sample1.plecs \
    --manifest configs/plecs/manifest_templates/dab_example.yaml --out results/
```

This returns a dataset only: the DAB has no surrogate template. Training the modulation model is a
separate round.

**Do not reuse that repository's acquisition notebook** to produce data. It has two defects:
- `writerow` calls `get_a_param_set(prelim)` without an index, so every row carries the parameters
  of combination 0;
- its URL `'http://localhost:1080/RPC2"'` has a stray quote.

The kit writes each condition's own parameters on every row, and a test checks that.

## 7. Troubleshooting

| Symptom in `evidence.json` / `log.txt` | Fix |
|---|---|
| `Cannot reach RPC endpoint` | PLECS is not running, or its RPC interface is disabled or on another port (`--rpc-url`) |
| `Backend running on a different host` | Run the kit on the same computer as PLECS. The RPC server is local |
| `[bad_result] … signals` | The number or order of top-level outports does not match the template |
| `samples must be exactly every 2e-05 s` | Keep `sample_s: 0.00002`. If your solver cannot output at that rate, tell us |
| `[timeout]` | Raise `time_limit_s` in the template |
| `action consistency` above 0 in `surrogate.json` | The C-Script does not follow the PRBS7 of §3.2 |
