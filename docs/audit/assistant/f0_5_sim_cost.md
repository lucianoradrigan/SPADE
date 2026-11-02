# F0.5 — simulation cost per plant

**Measured on the machine below; not a guarantee. Another CPU, background load or library versions change these numbers.**

- **Command:** `python experiments/measure_sim_cost.py` (2026-11-02T14:33:54-03:00). Raw values: [`f0_5_sim_cost.json`](f0_5_sim_cost.json); machine-readable copy for the future `estimate_cost`: `configs/sim_cost.yaml`.
- **Machine:** AMD Ryzen 5 5600X 6-Core Processor, 12 logical CPUs, Linux-7.0.0-38-generic-x86_64-with-glibc2.43; Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, TensorFlow 2.21.0.
- **Method:** 5 runs × 0.5 simulated s per plant through `run_scenario`, `time.process_time()`, one untimed warm-up run first. The machine was otherwise idle (no test suite running).

| Plant / controller | Median CPU s per simulated s | Range (5 runs) | CPU time for 1 simulated s |
|---|---|---|---|
| `dc_perm_ex_v1/PI` | **3.16** | 3.13–3.29 | ≈ 3.2 s |
| `dc_perm_ex_v1/MPC` | **5.41** | 5.39–5.42 | ≈ 5.4 s |
| `pmsm_v1/FOC_MTPA` | **1.14** | 1.13–1.15 | ≈ 1.1 s |
| `vsc_dpc_v1/DPC` | **13.75** | 13.43–13.82 | ≈ 13.8 s |
| `buck_v1/PI` | **0.67** | 0.67–0.69 | ≈ 0.7 s |

Every native plant runs one control period per 100 µs (10 000 steps per simulated second) in a Python loop; the DC plant also runs the vibration model, the PMSM integrates with 10 Euler substeps, the DPC calls a Keras network per step.

## Not executed

- Other machines (Windows, macOS, the Raspberry Pi): no ejecutado. The table is valid for this machine only.
- PLECS and FMU plants: no ejecutado (not native; their cost depends on the external tool and the model).
